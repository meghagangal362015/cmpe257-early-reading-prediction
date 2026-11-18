"""Robustness extension: does MaskMLP help when data is incomplete or labels are scarce? Not part of the paper.

MaskMLP is motivated by missing inputs and unlabelled students, but the paper's benchmark evaluates students
who have all eight reading assessments, with about 600-750 labelled students per training fold. This script
tests the situations the method was designed for, with every model trained and evaluated the same way:

- Experiment A, missing values at test time: models are trained as usual; then a fixed share of each test
  student's observed inputs (0, 10, 25, 40, 50 %) is hidden at random before prediction.
- Experiment B, scarce labels: pretraining still uses all students without labels, but the supervised part
  uses only a share (10, 25, 50, 100 %) of each fold's training and validation students. Test students are
  always the full test fold.
- Experiment C, whole tests missing at test time: as when a child is absent on a test day, every score from
  1 or 2 of the four start-of-year tests (WRMT, DIBELS NWF, ORF, SAT) is hidden for each test student.

Hidden cells, hidden tests and label subsets are drawn once per seed and fold, so every model sees exactly the
same data, and they are nested (everything hidden at 25 % is also hidden at 40 %).

Models:
- logistic regression and XGBoost (same-fold references) and the three plain MLPs;
- MaskMLP as in the paper (cosine loss), and MaskMLP_triplet, pretrained with the triplet loss (one of the
  paper's Table 4 losses) because the cosine loss collapses in our runs;
- our MaskMLP variants, which change only the fine-tuning stage:
  MaskMLP_maskft  hides 25 % of observed inputs in every fine-tuning batch too, as in pretraining;
  MaskMLP_frozen  keeps the pretrained encoder fixed and trains only the output layer;
  MaskMLP_lowlr   fine-tunes the pretrained encoder with a 10x smaller learning rate;
- MLP_mean_maskft, a control: MLP (Mean) trained with the same hidden-input batches as MaskMLP_maskft, to tell
  apart the effect of pretraining from the effect of training on incomplete inputs.

All settings are the main script's defaults and are fixed before any test result is seen: nothing is tuned on
the test folds. This file only imports from Supriya_Neural_network_implementation.py and changes nothing in it.
Outputs go to results/robustness_extension/ by default.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Supriya_Neural_network_implementation import (
    COLORS, FEATURES, INK, INK_2, MLP, TARGETS, available_models, evaluate, fit_predict, fit_scaler, load_unfiltered,
    make_folds, mask_observed, matrix, new_figure, numeric, observed_mean, pretrained_maskmlp, probabilities, save,
    scale, set_seed, traditional_model, verify_unfiltered,
)

TRADITIONAL = ("LogisticRegression", "XGBoost")
PLAIN = ("MLP_zero", "MLP_mean", "MLP_indicator")
PAPER_MASKMLP = ("MaskMLP", "MaskMLP_triplet")
VARIANTS = ("MaskMLP_maskft", "MaskMLP_frozen", "MaskMLP_lowlr")
CONTROLS = ("MLP_mean_maskft",)
MASKED = (*PAPER_MASKMLP, *VARIANTS)
MODELS = (*TRADITIONAL, *PLAIN, *CONTROLS, *MASKED)
TESTS = {"WRMT": ["RMwidRS", "RMwdaRS"], "DIBELS NWF": ["NWFcls", "NWFwrc"], "ORF": ["ORFwc"],
         "SAT": ["SAwrS", "SAsrS", "SAtoS"]}
STYLE = {
    "LogisticRegression": ("#eb6834", "-"), "XGBoost": ("#b54a17", "-"),
    "MLP_zero": ("#8db8f5", "-"), "MLP_mean": ("#4f86d6", "-"), "MLP_indicator": ("#1f5fbf", "-"),
    "MLP_mean_maskft": ("#4f86d6", ":"),
    "MaskMLP": (COLORS["MaskMLP"], "-"), "MaskMLP_triplet": ("#0e6b4a", "--"),
    "MaskMLP_maskft": ("#7a3fb0", "-"), "MaskMLP_frozen": ("#c79a00", "-"), "MaskMLP_lowlr": ("#c0398a", "-"),
}
EXPERIMENTS = {
    "missing_at_test": ("Experiment A: random inputs hidden at test time",
                        "Share of each test student's known inputs hidden", "{:.0%}"),
    "label_fraction": ("Experiment B: fewer labelled students", "Share of labelled training students used", "{:.0%}"),
    "missing_tests": ("Experiment C: whole tests missing at test time",
                      "Start-of-year tests missing per test student (of 4)", "{:.0f}"),
}


def family(model: str) -> str:
    return ("traditional" if model in TRADITIONAL else "maskmlp" if model in MASKED
            else "control" if model in CONTROLS else "plain_mlp")


def hide_cells(frame: pd.DataFrame, rate: float, scores: np.ndarray) -> pd.DataFrame:
    """Hide round(rate x observed count) of each student's observed inputs (at least one when two or more are
    observed and rate > 0). `scores` fixes which cells go first, so masks are nested across rates."""
    if rate <= 0:
        return frame
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    observed = np.isfinite(values)
    count = observed.sum(axis=1, keepdims=True)
    n_hide = np.floor(count * rate + 0.5)
    n_hide = np.where(count >= 2, np.maximum(n_hide, 1), 0)
    ranked = np.where(observed, scores, np.inf).argsort(axis=1).argsort(axis=1)
    hide = (ranked < n_hide) & observed
    masked = frame.copy()
    masked[FEATURES] = np.where(hide, np.nan, values)
    return masked


def hide_tests(frame: pd.DataFrame, k: int, order: np.ndarray) -> pd.DataFrame:
    """Hide every score of the first k tests in each student's random test order (nested across k)."""
    if k <= 0:
        return frame
    masked = frame.copy()
    names = list(TESTS)
    for i, row in enumerate(masked.index):
        for test in (names[j] for j in order[i, :k]):
            masked.loc[row, TESTS[test]] = np.nan
    return masked


def label_subset(frame: pd.DataFrame, fraction: float, order: np.ndarray) -> pd.DataFrame:
    """First `fraction` of a fixed random order (nested across fractions); both classes always kept."""
    if fraction >= 1:
        return frame
    n = max(int(np.ceil(fraction * len(frame))), 10)
    for extra in range(len(frame)):
        part = frame.iloc[np.sort(order[: n + extra])]
        if part["_label"].nunique() == 2:
            return part
    return frame


def fine_tune(model, x, y, xv, yv, args, device, prepare=None, freeze=False, encoder_lr_scale=1.0):
    """BCE fine-tuning with early stopping on validation loss, as in the main script, with three options:
    `prepare` transforms each training batch (used to hide inputs), `freeze` trains only the output layer, and
    `encoder_lr_scale` lowers the encoder's learning rate."""
    if freeze:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        groups = [{"params": model.classifier.parameters(), "lr": args.learning_rate}]
    else:
        groups = [{"params": model.encoder.parameters(), "lr": args.learning_rate * encoder_lr_scale},
                  {"params": model.classifier.parameters(), "lr": args.learning_rate}]
    optimizer = torch.optim.Adam(groups)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=args.batch_size, shuffle=True)
    xv_tensor, yv_tensor = torch.from_numpy(xv).to(device), torch.from_numpy(yv).to(device)
    best, best_state, patience = float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        for batch, labels in loader:
            batch, labels = batch.to(device), labels.to(device)
            if prepare is not None:
                batch = prepare(batch)
            loss = criterion(model(batch), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(xv_tensor), yv_tensor).item()
        if valid_loss < best - 1e-6:
            best, best_state, patience = valid_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def train(model: str, train_df, valid_df, test_df, pretrain_records, args, device, seed):
    """Fit one model and return a function that predicts probabilities for any (possibly masked) frame."""
    set_seed(seed)
    ytrain = train_df["_label"].to_numpy(np.float32)
    yvalid = valid_df["_label"].to_numpy(np.float32)
    if model in TRADITIONAL:
        outer = pd.concat([train_df, valid_df])
        fitted = traditional_model(model, seed).fit(numeric(outer, FEATURES), outer["_label"].to_numpy(int))
        return lambda frame: fitted.predict_proba(numeric(frame, FEATURES))[:, 1]
    if model in PLAIN or model == "MaskMLP_triplet" or model == "MaskMLP":
        method, variant = model, args
        if model == "MaskMLP_triplet":
            method, variant = "MaskMLP", copy.copy(args)
            variant.pretrain_loss = "triplet"
        _, _, (net, encode) = fit_predict(method, FEATURES, train_df, valid_df, test_df, pretrain_records,
                                          variant, device, seed)
        return lambda frame: probabilities(net, encode(frame), device)
    if model == "MLP_mean_maskft":
        scaler = fit_scaler(matrix(train_df, FEATURES))
        mean = torch.from_numpy(observed_mean(scale(matrix(train_df, FEATURES), scaler)).astype(np.float32)).to(device)
        fill = lambda batch: torch.where(batch == -1, mean.expand_as(batch), batch)
        encode = lambda frame: scale(matrix(frame, FEATURES), scaler)
        net = MLP(len(FEATURES), args.hidden_size, args.layers).to(device)
        prepared = lambda frame: fill(torch.from_numpy(encode(frame)).to(device)).cpu().numpy()
        fine_tune(net, encode(train_df), ytrain, prepared(valid_df), yvalid, args, device,
                  prepare=lambda batch: fill(mask_observed(batch, args.mask_rate, args.mask_mode)))
        return lambda frame: probabilities(net, prepared(frame), device)
    # MaskMLP variants: the paper's pretrained encoder (cosine loss), different fine-tuning
    net, scaler, _ = pretrained_maskmlp(FEATURES, pretrain_records, args, device, seed)
    encode = lambda frame: scale(matrix(frame, FEATURES), scaler)
    set_seed(seed + 10_000)
    options = {"MaskMLP_maskft": {"prepare": lambda batch: mask_observed(batch, args.mask_rate, args.mask_mode)},
               "MaskMLP_frozen": {"freeze": True},
               "MaskMLP_lowlr": {"encoder_lr_scale": 0.1}}[model]
    fine_tune(net, encode(train_df), ytrain, encode(valid_df), yvalid, args, device, **options)
    return lambda frame: probabilities(net, encode(frame), device)


def record(rows, base, ytest, tx, p):
    for subset, selected in {"all": np.ones(len(ytest), dtype=bool), "intervention": tx}.items():
        if selected.any():
            rows.append({**base, "subset": subset, **evaluate(ytest[selected], p[selected])})


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset)
    args.pretrain_pool = None
    if args.raw_dataset:
        pool = load_unfiltered(args.raw_dataset)
        if verify_unfiltered(df, pool)["verified"]:
            args.pretrain_pool = pool
        else:
            print("WARNING: raw records do not match --dataset; pretraining on --dataset instead", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    models = available_models(args.models)
    fractions = sorted(set(args.label_fractions) | {1.0})
    print(f"Robustness extension: seeds {args.seeds}; models {models}; mask rates {args.mask_rates}; "
          f"tests missing {args.tests_missing}; label fractions {fractions}; device {device}", flush=True)

    rows = []
    for seed in args.seeds:
        args.seed = seed
        for group in args.groups:
            for target in args.targets:
                for fold, train_df, valid_df, test_df, pretrain_records in make_folds(df, group, target, args):
                    rng = np.random.default_rng([seed, fold, TARGETS.index(target), ("school", "student").index(group)])
                    train_order, valid_order = rng.permutation(len(train_df)), rng.permutation(len(valid_df))
                    cell_scores = rng.random((len(test_df), len(FEATURES)))
                    test_order = np.argsort(rng.random((len(test_df), len(TESTS))), axis=1)
                    ytest = test_df["_label"].to_numpy(int)
                    tx = pd.to_numeric(test_df["Tx"], errors="coerce").to_numpy() == 1
                    test_sets = [("missing_at_test", rate, hide_cells(test_df, rate, cell_scores))
                                 for rate in args.mask_rates]
                    test_sets += [("missing_tests", k, hide_tests(test_df, k, test_order)) for k in args.tests_missing]
                    for fraction in fractions:
                        tr = label_subset(train_df, fraction, train_order)
                        va = label_subset(valid_df, fraction, valid_order)
                        for model in models:
                            predict = train(model, tr, va, test_df, pretrain_records, args, device, seed + fold)
                            base = {"seed": seed, "group": group, "target": target, "fold": fold, "model": model,
                                    "family": family(model), "train_n": len(tr) + len(va)}
                            record(rows, {**base, "experiment": "label_fraction", "condition": fraction},
                                   ytest, tx, predict(test_df))
                            if fraction == 1.0:
                                for experiment, cond, frame in test_sets:
                                    record(rows, {**base, "experiment": experiment, "condition": cond},
                                           ytest, tx, predict(frame))
                        print(f"seed {seed} | {group} | {target} | fold {fold} | labels {fraction:.0%} done", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    summary, verdict = summarise(results)
    summary.to_csv(out / "summary.csv", index=False)
    verdict.to_csv(out / "verdict.csv", index=False)
    config = {k: v for k, v in vars(args).items() if k not in ("pretrain_pool", "seed")}
    for key in ("dataset", "raw_dataset", "output_dir"):
        if config.get(key):
            config[key] = Path(config[key]).name
    config["pretrain_pool_size"] = len(df) if args.pretrain_pool is None else len(args.pretrain_pool)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for experiment in EXPERIMENTS:
        plot(summary, experiment, out / f"{experiment}.png")
    print(verdict.round(3).to_string(index=False))
    print(f"Saved robustness results to {out}", flush=True)


def summarise(results: pd.DataFrame, metric: str = "balanced_accuracy"):
    """Mean over seeds, splits, targets and folds per condition; margins of each MaskMLP model and paired counts.
    The verdict uses balanced accuracy by default: a model that predicts the majority class for everyone keeps
    a plain accuracy of about 0.55 here however much input is hidden, but its balanced accuracy is 0.5."""
    keys = ["experiment", "condition", "subset", "model"]
    summary = results.groupby(keys)[["accuracy", "balanced_accuracy", "pr_auc"]].mean().reset_index()
    summary["units"] = results.groupby(keys).size().to_numpy()
    rows = []
    unit = ["seed", "group", "target", "fold"]
    for (experiment, cond, subset), part in results.groupby(["experiment", "condition", "subset"]):
        wide = part.pivot_table(index=unit, columns="model", values=metric)
        means = wide.mean()
        others = [m for m in (*TRADITIONAL, *PLAIN) if m in wide]
        plain = [m for m in PLAIN if m in wide]
        trad = [m for m in TRADITIONAL if m in wide]
        for masked in [m for m in MASKED if m in wide]:
            row = {"experiment": experiment, "condition": cond, "subset": subset, "model": masked,
                   "metric": metric, "value": means[masked],
                   "rank_among_reference_models": int((means[others] > means[masked]).sum() + 1)}
            if plain:
                best_plain = means[plain].idxmax()
                row.update(best_plain_mlp=best_plain, minus_best_plain_mlp=means[masked] - means[best_plain],
                           units_ahead_of_best_plain=int((wide[masked] > wide[best_plain]).sum()),
                           units_behind_best_plain=int((wide[masked] < wide[best_plain]).sum()))
            if trad:
                best_trad = means[trad].idxmax()
                row.update(best_traditional=best_trad, minus_best_traditional=means[masked] - means[best_trad])
            if "MLP_mean_maskft" in wide:
                row["minus_MLP_mean_maskft"] = means[masked] - means["MLP_mean_maskft"]
            row["units"] = len(wide)
            rows.append(row)
    return summary, pd.DataFrame(rows)


def plot(summary: pd.DataFrame, experiment: str, path: Path, metric: str = "balanced_accuracy") -> None:
    data = summary[summary["experiment"] == experiment]
    if data.empty:
        return
    title, xlabel, fmt = EXPERIMENTS[experiment]
    fig, axes = new_figure(1, 2, (13, 5.2))
    for ax, subset in zip(axes[0], ("all", "intervention")):
        part = data[data["subset"] == subset]
        for model in [m for m in MODELS if m in set(part["model"])]:
            line = part[part["model"] == model].sort_values("condition")
            color, style = STYLE[model]
            width = 2.6 if model in MASKED else 1.6
            ax.plot(line["condition"], line[metric], style, marker="o", color=color, linewidth=width,
                    markersize=5, label=model)
        if experiment == "label_fraction":
            ax.set_xscale("log")
            ax.minorticks_off()
        ticks = sorted(part["condition"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([fmt.format(t) for t in ticks])
        ax.axhline(0.5, color=INK_2, linestyle=":", linewidth=1)
        ax.set_xlabel(xlabel, color=INK_2)
        ax.set_ylabel(f"{metric.replace('_', ' ').capitalize()} (mean over seeds, splits, targets, folds)", color=INK_2)
        ax.set_title("All students" if subset == "all" else "Intervention group", color=INK, fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(f"{title} (extension, not in the paper)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description="Robustness extension: missing inputs at test time and scarce labels")
    parser.add_argument("--dataset", default="data/ECRIDataSet.csv")
    parser.add_argument("--raw-dataset", default=None,
                        help="ECRIRawData.csv: pretrain MaskMLP on all 6,916 students, as in the main run")
    parser.add_argument("--output-dir", default="results/robustness_extension")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--groups", nargs="+", choices=["school", "student"], default=["school", "student"])
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--mask-rates", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.4, 0.5],
                        help="share of each test student's observed inputs hidden (experiment A)")
    parser.add_argument("--tests-missing", type=int, nargs="+", default=[1, 2],
                        help="number of the four start-of-year tests hidden per test student (experiment C)")
    parser.add_argument("--label-fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0],
                        help="share of labelled training and validation students used (experiment B)")
    parser.add_argument("--eval-rows", choices=["assessments", "all", "complete"], default="assessments")
    parser.add_argument("--pretrain-scope", choices=["all", "train"], default="all")
    parser.add_argument("--pretrain-loss", default="cosine", help=argparse.SUPPRESS)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-rate", type=float, default=0.25,
                        help="share of observed inputs hidden in MaskMLP pretraining and in masked fine-tuning")
    parser.add_argument("--mask-mode", choices=["exact", "bernoulli"], default="exact")
    parser.add_argument("--contrastive-margin", type=float, default=0.0)
    parser.add_argument("--triplet-margin", type=float, default=1.0)
    parser.add_argument("--collapse-ratio", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.select_on_test = False
    run(args)


if __name__ == "__main__":
    main()
