"""k-nearest neighbours (kNN) as an exploratory extension model. Not part of the paper.

Two kNN variants are evaluated on exactly the same folds and test students as
Supriya_Neural_network_implementation.py (same seed, same splits, same evaluation code):

- kNN_raw: kNN on the 16 input features. Min-max scaled on the training fold, missing values
  filled with the training-fold mean, plus one missing-value indicator per feature (as MLP_indicator).
- kNN_MaskMLP_embedding: kNN on the embeddings of the pretrained MaskMLP encoder (before
  fine-tuning), with cosine distance. It shows how much neighbour information the
  self-supervised pretraining keeps, which links to the embedding-collapse finding.

For each fold, k is chosen on the grouped validation split (by validation accuracy, ties go to
the smaller k); the model is then refit on training + validation students and scored on the test fold.

The results are compared with the six models in an existing run of the main script
(default results/neural_network/main), so that run's seed must match --seed. If that run was not
available, --compare-only rebuilds the comparison later from the saved kNN results without retraining.

This file only imports from Supriya_Neural_network_implementation.py and changes nothing in it.
Outputs go to results/knn_extension/ by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier

from Supriya_Neural_network_implementation import (
    COLORS, FEATURES, INK, INK_2, METRICS, TARGET_NAMES, TARGETS, embeddings, evaluate, family,
    fit_scaler, load_unfiltered, loss_spec, make_folds, matrix, new_figure, observed_mean, pretrained_maskmlp,
    save, scale, set_seed, verify_unfiltered,
)

KNN_MODELS = ("kNN_raw", "kNN_MaskMLP_embedding")
KNN_COLOR = "#8c5bb5"
K_GRID = (3, 5, 7, 11, 15, 21, 31, 41, 51, 75, 101)
REFERENCE_ORDER = ("LogisticRegression", "XGBoost", "MLP_zero", "MLP_mean", "MLP_indicator", "MaskMLP")


def raw_encoder(train: pd.DataFrame):
    """Scale on the training fold; fill missing with the training-fold mean and add missing indicators."""
    xtrain = matrix(train, FEATURES)
    scaler = fit_scaler(xtrain)
    mean = observed_mean(scale(xtrain, scaler))

    def encode(frame: pd.DataFrame) -> np.ndarray:
        x = scale(matrix(frame, FEATURES), scaler)
        missing = x == -1
        return np.column_stack([np.where(missing, mean, x), missing]).astype(np.float32)
    return encode


def embedding_encoder(pretrain_records: pd.DataFrame, args, device, seed: int):
    """Pretrained (not fine-tuned) MaskMLP encoder, pretrained with the same settings as in the main script."""
    model, scaler, info = pretrained_maskmlp(FEATURES, pretrain_records, args, device, seed)
    return (lambda frame: embeddings(model, scale(matrix(frame, FEATURES), scaler), device)), info


def choose_k(xtrain, ytrain, xvalid, yvalid, metric: str):
    scores = []
    for k in K_GRID:
        if k > len(xtrain):
            break
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(xtrain, ytrain)
        scores.append((k, float((knn.predict(xvalid) == yvalid).mean())))
    best = max(scores, key=lambda item: (item[1], -item[0]))
    return best, scores


def fit_predict_knn(model: str, train, valid, test, pretrain_records, args, device, seed):
    info = {}
    if model == "kNN_raw":
        encode, metric = raw_encoder(train), "euclidean"
    else:
        encode, info = embedding_encoder(pretrain_records, args, device, seed)
        metric = "cosine"
    ytrain, yvalid = train["_label"].to_numpy(int), valid["_label"].to_numpy(int)
    (k, valid_accuracy), _ = choose_k(encode(train), ytrain, encode(valid), yvalid, metric)
    outer = pd.concat([train, valid])
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(encode(outer), outer["_label"].to_numpy(int))
    return knn.predict_proba(encode(test))[:, 1], k, valid_accuracy, info


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.compare_only:
        write_comparison(out, args)
        return
    set_seed(args.seed)
    df = pd.read_csv(args.dataset)
    args.pretrain_pool = None
    if args.raw_dataset:
        pool = load_unfiltered(args.raw_dataset)
        if verify_unfiltered(df, pool)["verified"]:
            args.pretrain_pool = pool
        else:
            print("WARNING: raw records do not match --dataset; pretraining on --dataset instead", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pool_size = len(df) if args.pretrain_pool is None else len(args.pretrain_pool)
    print(f"kNN extension: {len(df)} students; pretraining pool {pool_size}; seed {args.seed}; device {device}", flush=True)

    rows, chosen = [], []
    for group in ("school", "student"):
        for target in TARGETS:
            for fold, train, valid, test, pretrain_records in make_folds(df, group, target, args):
                ytest = test["_label"].to_numpy(int)
                tx = pd.to_numeric(test["Tx"], errors="coerce").to_numpy() == 1
                for model in KNN_MODELS:
                    p, k, valid_accuracy, info = fit_predict_knn(model, train, valid, test, pretrain_records,
                                                                 args, device, args.seed + fold)
                    chosen.append({"group": group, "target": target, "fold": fold, "model": model, "k": k,
                                   "validation_accuracy": valid_accuracy,
                                   "embedding_spread": info.get("embedding_spread", np.nan)})
                    for subset, selected in {"all": np.ones(len(test), dtype=bool), "intervention": tx}.items():
                        if selected.any():
                            rows.append({"record_type": "fold", "group": group, "target": target, "model": model,
                                         "family": "extension", "subset": subset, "fold": fold,
                                         "train_n": len(train) + len(valid), **evaluate(ytest[selected], p[selected])})
                    print(f"{group} | {target} | fold {fold} | {model} | k={k} | "
                          f"test accuracy={rows[-2 if tx.any() else -1]['accuracy']:.3f}", flush=True)

    fold_df = pd.DataFrame(rows)
    summary = fold_df.groupby(["group", "target", "model", "family", "subset"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.insert(0, "record_type", "summary")
    pd.concat([fold_df, summary], ignore_index=True, sort=False).to_csv(out / "results.csv", index=False)
    pd.DataFrame(chosen).to_csv(out / "selected_k.csv", index=False)
    config = {k: v for k, v in vars(args).items() if k != "pretrain_pool"}
    for key in ("dataset", "raw_dataset", "output_dir", "reference_run"):
        if config.get(key):
            config[key] = Path(config[key]).name
    config.update(k_grid=list(K_GRID), pretrain_pool_size=pool_size)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    write_comparison(out, args, fold_df)
    print(f"Saved kNN results to {out}", flush=True)


def write_comparison(out: Path, args, fold_df: pd.DataFrame | None = None) -> None:
    if fold_df is None:
        saved = out / "results.csv"
        if not saved.exists():
            raise SystemExit(f"{saved} not found; run the kNN models first (without --compare-only)")
        config = out / "config.json"
        if config.exists():
            args.seed = json.loads(config.read_text()).get("seed", args.seed)
        fold_df = pd.read_csv(saved)
        fold_df = fold_df[fold_df["record_type"] == "fold"]
    comparison = compare(fold_df, Path(args.reference_run), args.seed)
    if comparison is not None:
        comparison.to_csv(out / "comparison.csv", index=False)
        plot(comparison, out / "knn_comparison.png")
        print(comparison.query("subset == 'all'").groupby("model")[["accuracy", "balanced_accuracy", "pr_auc"]]
              .mean().sort_values("accuracy", ascending=False).round(3).to_string())
        print(f"Comparison written to {out}", flush=True)


def compare(fold_df: pd.DataFrame, reference_run: Path, seed: int):
    """Mean over folds for the kNN models next to the six models of the reference run (same seed, same folds)."""
    results, config = reference_run / "results.csv", reference_run / "config.json"
    if not results.exists():
        print(f"NOTE: {results} not found; skipping the comparison with the other models", flush=True)
        return None
    if config.exists() and json.loads(config.read_text()).get("seed") != seed:
        print(f"NOTE: {reference_run} used a different seed, so its folds differ; skipping the comparison", flush=True)
        return None
    reference = pd.read_csv(results)
    reference = reference[reference["record_type"] == "fold"]
    keys = ["group", "target", "model", "subset"]
    columns = ["accuracy", "balanced_accuracy", "pr_auc"]
    both = pd.concat([reference[keys + columns], fold_df[keys + columns]], ignore_index=True)
    table = both.groupby(keys)[columns].mean().reset_index()
    table["family"] = table["model"].map(lambda m: "extension" if m in KNN_MODELS else family(m))
    return table


def plot(table: pd.DataFrame, path: Path) -> None:
    """Accuracy of every model, one panel per split and target (all students), kNN highlighted."""
    data = table[table["subset"] == "all"]
    order = [m for m in (*REFERENCE_ORDER, *KNN_MODELS) if m in set(data["model"])]
    fig, axes = new_figure(2, 2, (11, 7.5))
    for row, group in enumerate(("school", "student")):
        for col, target in enumerate(TARGETS):
            ax = axes[row, col]
            part = data[(data["group"] == group) & (data["target"] == target)].set_index("model").reindex(order)
            y = np.arange(len(order))[::-1]
            colors = [KNN_COLOR if m in KNN_MODELS else COLORS["MaskMLP"] if m == "MaskMLP"
                      else COLORS[family(m)] for m in order]
            ax.scatter(part["accuracy"], y, s=70, color=colors, zorder=3)
            for yi, value in zip(y, part["accuracy"]):
                ax.annotate(f"{value:.3f}", (value, yi), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK)
            ax.axvline(0.5, color=INK_2, linestyle="--", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels(order)
            ax.set_xlim(0.45, max(0.7, float(part["accuracy"].max()) + 0.04))
            ax.set_ylim(-0.7, len(order) - 0.3)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("Accuracy (mean over 5 folds)", color=INK_2)
            ax.set_title(f"{TARGET_NAMES[target]} · {group} split", color=INK, fontsize=11)
    fig.suptitle("kNN (purple, exploratory extension) vs the reproduced models · all students",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description="Exploratory kNN extension on the same folds as the main experiments")
    parser.add_argument("--dataset", default="data/ECRIDataSet.csv")
    parser.add_argument("--raw-dataset", default=None,
                        help="ECRIRawData.csv: pretrain the MaskMLP encoder on all 6,916 students, as in the main run")
    parser.add_argument("--output-dir", default="results/knn_extension")
    parser.add_argument("--reference-run", default="results/neural_network/main",
                        help="run of the main script to compare against (must use the same --seed)")
    parser.add_argument("--compare-only", action="store_true",
                        help="rebuild comparison.csv and knn_comparison.png from an existing --output-dir without retraining")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-rows", choices=["complete", "all"], default="all")
    parser.add_argument("--pretrain-scope", choices=["all", "train"], default="all")
    parser.add_argument("--pretrain-loss", type=loss_spec, default="cosine",
                        help="loss for the MaskMLP encoder used by kNN_MaskMLP_embedding (e.g. cosine, simsiam)")
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-rate", type=float, default=0.25)
    parser.add_argument("--contrastive-margin", type=float, default=0.0)
    parser.add_argument("--triplet-margin", type=float, default=1.0)
    parser.add_argument("--collapse-ratio", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
