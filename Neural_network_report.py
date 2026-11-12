"""Paper comparison, robustness and validation tables for Neural_network_implementation.py outputs.

Scans --outputs for run folders (any folder with results.csv) and writes to <outputs>/report/:

- table1_comparison.csv / paper_vs_ours_*.png: our results next to the paper's Table 1
  (accuracy, and balanced accuracy against the paper's AUC column) for the primary setting.
- auc_interpretation.csv: checks the paper's AUC against (specificity + sensitivity) / 2
  using its own Table 5 values.
- sensitivity.csv: every setting found (pretraining loss, pretraining scope, evaluation rows).
- robustness.csv / maskmlp_wins.csv: seed-to-seed spread and how often MaskMLP beats each
  comparison model across seeds, folds, splits and targets (settings with 2+ seeds).
- collapse.csv: MaskMLP embedding spread before pretraining, after one epoch and at the end.

Every table marks each pretraining loss as "paper" or "extension" (e.g. simsiam, which is not in
the paper). The primary Table 1 comparison only ever uses a paper loss.

Each run's setting is read from the config.json that Neural_network_implementation.py writes. Partial runs
(fewer models, one target, one split, one seed) are handled: each table uses what is present
and anything that cannot be computed is skipped with a note.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Neural_network_implementation import (COLORS, EXTENSION_LOSSES, INK, INK_2, SURFACE, TARGET_NAMES, TRADITIONAL, family,
                             new_figure, save)

MODELS = ("LogisticRegression", "XGBoost", "MLP_zero", "MLP_mean", "MLP_indicator", "MaskMLP")
PLAIN_MLPS = ("MLP_zero", "MLP_mean", "MLP_indicator")
SETTING = ["loss", "scope", "rows"]
METRICS = ["accuracy", "balanced_accuracy", "pr_auc"]
# Paper Table 1 columns: (target, subset, metric) in published order.
TABLE1_COLUMNS = [(t, s, m) for t in ("RMwidRS_target", "RMwdaRS_target") for s in ("all", "intervention")
                  for m in ("accuracy", "auc")]
TABLE1 = {
    "school": {
        "LogisticRegression": [0.6119, 0.6121, 0.6214, 0.5850, 0.5923, 0.5591, 0.6311, 0.5130],
        "XGBoost": [0.6503, 0.6515, 0.6504, 0.6536, 0.6090, 0.5817, 0.6376, 0.5628],
        "MLP_zero": [0.6725, 0.6670, 0.7410, 0.7319, 0.6535, 0.6303, 0.7346, 0.6369],
        "MLP_mean": [0.6695, 0.6640, 0.7281, 0.7152, 0.6371, 0.5849, 0.7476, 0.6236],
        "MLP_indicator": [0.6710, 0.6607, 0.7539, 0.7394, 0.6698, 0.6321, 0.7412, 0.6403],
        "MaskMLP": [0.6726, 0.6693, 0.7704, 0.7633, 0.6697, 0.6545, 0.7704, 0.6869],
    },
    "student": {
        "LogisticRegression": [0.6207, 0.6201, 0.6474, 0.6394, 0.6281, 0.5666, 0.6796, 0.5503],
        "XGBoost": [0.6607, 0.6603, 0.6699, 0.6659, 0.6119, 0.5743, 0.6731, 0.5849],
        "MLP_zero": [0.6756, 0.6745, 0.7120, 0.7002, 0.6756, 0.6368, 0.7638, 0.6956],
        "MLP_mean": [0.6652, 0.6627, 0.7186, 0.7171, 0.6563, 0.6214, 0.7572, 0.6799],
        "MLP_indicator": [0.6889, 0.6873, 0.7411, 0.7198, 0.6637, 0.6202, 0.7670, 0.6851],
        "MaskMLP": [0.6919, 0.6899, 0.7766, 0.7537, 0.6741, 0.6485, 0.7865, 0.7333],
    },
}
# Paper Table 5 (MaskMLP, school split): (target, dropped feature, specificity, sensitivity, AUC).
TABLE5 = [
    ("RMwidRS_target", "Gender", 0.6284, 0.6802, 0.6543), ("RMwidRS_target", "Tx", 0.6055, 0.7257, 0.6656),
    ("RMwidRS_target", "Age1b", 0.6147, 0.7503, 0.6825), ("RMwidRS_target", "Tier2_N", 0.5903, 0.7220, 0.6562),
    ("RMwidRS_target", "grp_rate", 0.6674, 0.6604, 0.6639), ("RMwidRS_target", "ORFwc", 0.7278, 0.5745, 0.6511),
    ("RMwdaRS_target", "Gender", 0.4938, 0.7260, 0.6099), ("RMwdaRS_target", "Tx", 0.4026, 0.8068, 0.6047),
    ("RMwdaRS_target", "Age1b", 0.3356, 0.8573, 0.5965), ("RMwdaRS_target", "NWFcls", 0.4847, 0.7734, 0.6291),
]


def run_config(folder: Path) -> dict | None:
    path = folder / "config.json"
    if not path.exists():
        return None
    config = json.loads(path.read_text())
    return {"loss": config.get("pretrain_loss", "cosine"), "scope": config.get("pretrain_scope", "all"),
            "rows": config.get("eval_rows", "all"), "seed": int(config.get("seed", 42)),
            "pool": config.get("pretrain_pool_size")}


def load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for results in sorted(root.rglob("results.csv")):
        if "report" in results.parts:
            continue
        config = run_config(results.parent)
        if config is None:
            print(f"NOTE: skipping {results.parent} (no config.json; re-run it with the current Neural_network_implementation.py)")
            continue
        df = pd.read_csv(results)
        df = df[df["record_type"] == "fold"].assign(**config, run=results.parent.relative_to(root).as_posix())
        frames.append(df)
    if not frames:
        raise SystemExit(f"No run folders with results.csv and config.json found under {root}")
    return pd.concat(frames, ignore_index=True)


def loss_origin(loss) -> str:
    """'extension' for losses that are not in the paper (reported separately from the reproduction)."""
    return "extension" if set(str(loss).split("+")) & set(EXTENSION_LOSSES) else "paper"


def label(setting) -> str:
    loss, scope, rows = setting
    return f"loss={loss}, pretrain={scope}, rows={rows}"


def pick_primary(folds: pd.DataFrame, requested: str) -> tuple:
    wanted = tuple(requested.split(","))
    available = folds.groupby(SETTING)["seed"].nunique()
    if wanted in available.index:
        return wanted
    paper = available[[loss_origin(s[0]) == "paper" for s in available.index]]
    fallback = (paper if not paper.empty else available).idxmax()
    print(f"NOTE: primary setting {label(wanted)} not found; using {label(fallback)}")
    return fallback


def table1_comparison(primary: pd.DataFrame) -> pd.DataFrame:
    """Ours (mean and seed-to-seed std of the 5-fold mean) next to the paper's Table 1, where present."""
    per_seed = primary.groupby(["group", "target", "subset", "model", "seed"])[METRICS].mean()
    stats = per_seed.groupby(["group", "target", "subset", "model"]).agg(["mean", "std"])
    stats.columns = [f"ours_{m}_{s}" for m, s in stats.columns]
    rows = []
    for (group, target, subset, model), values in stats.iterrows():
        if group not in TABLE1 or model not in TABLE1[group]:
            continue
        paper = dict(zip(TABLE1_COLUMNS, TABLE1[group][model]))
        row = {"split": group, "target": target, "subset": subset, "model": model,
               "paper_accuracy": paper[(target, subset, "accuracy")], "paper_auc": paper[(target, subset, "auc")],
               **values.to_dict()}
        row["accuracy_gap"] = row["ours_accuracy_mean"] - row["paper_accuracy"]
        row["balanced_accuracy_vs_paper_auc_gap"] = row["ours_balanced_accuracy_mean"] - row["paper_auc"]
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["seeds"] = primary["seed"].nunique()
    # Ranks only among the models present in our run, for both sources.
    for source, column in (("paper", "paper_accuracy"), ("ours", "ours_accuracy_mean")):
        table[f"{source}_rank"] = table.groupby(["split", "target", "subset"])[column].rank(ascending=False, method="min")
    return table


def auc_interpretation() -> pd.DataFrame:
    table = pd.DataFrame(TABLE5, columns=["target", "dropped_feature", "specificity", "sensitivity", "paper_auc"])
    table["balanced_accuracy"] = (table["specificity"] + table["sensitivity"]) / 2
    table["difference"] = table["paper_auc"] - table["balanced_accuracy"]
    return table


def margins(means: pd.DataFrame) -> dict:
    """MaskMLP minus the best plain MLP and the best traditional model, for each metric (when present)."""
    result = {}
    if "MaskMLP" not in means.index:
        return result
    for name, group in (("plain_mlp", PLAIN_MLPS), ("traditional", TRADITIONAL)):
        present = [m for m in group if m in means.index]
        for metric in METRICS:
            result[f"maskmlp_minus_best_{name}_{metric}"] = (means.loc["MaskMLP", metric] - means.loc[present, metric].max()
                                                             if present else np.nan)
    return result


def sensitivity(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    allrows = folds[folds["subset"] == "all"]
    for setting, part in allrows.groupby(SETTING):
        means = part.groupby("model")[METRICS].mean()
        common = {"pretrain_loss": setting[0], "loss_origin": loss_origin(setting[0]),
                  "pretrain_scope": setting[1], "eval_rows": setting[2],
                  "seeds": part["seed"].nunique(), "splits": ",".join(sorted(part["group"].unique())),
                  "targets": ",".join(sorted(part["target"].unique()))}
        for model, values in means.iterrows():
            rows.append({**common, "model": model, **values.to_dict()})
        rows.append({**common, "model": "(margins)", **margins(means)})
    return pd.DataFrame(rows)


def robustness(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, wins = [], []
    allrows = folds[folds["subset"] == "all"]
    for setting, part in allrows.groupby(SETTING):
        if part["seed"].nunique() < 2:
            continue
        common = {"pretrain_loss": setting[0], "loss_origin": loss_origin(setting[0]),
                  "pretrain_scope": setting[1], "eval_rows": setting[2], "seeds": part["seed"].nunique()}
        per_seed = part.groupby(["model", "seed"])[METRICS].mean()
        stats = per_seed.groupby("model").agg(["mean", "std", "min", "max"])
        stats.columns = [f"{m}_{s}" for m, s in stats.columns]
        summary.append(stats.reset_index().assign(**common))
        units = part.pivot_table(index=["seed", "group", "target", "fold"], columns="model", values="accuracy")
        if "MaskMLP" not in units:
            continue
        for other in [m for m in units.columns if m != "MaskMLP"]:
            diff = (units["MaskMLP"] - units[other]).dropna()
            wins.append({**common, "compared_to": other, "units": len(diff), "maskmlp_wins": int((diff > 0).sum()),
                         "ties": int((diff == 0).sum()), "maskmlp_losses": int((diff < 0).sum()),
                         "mean_accuracy_difference": diff.mean(), "std_accuracy_difference": diff.std()})
    if not summary:
        print("NOTE: robustness needs a setting with at least two seeds; none found.")
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(summary, ignore_index=True), pd.DataFrame(wins)


def collapse(folds: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    mask = folds[(folds["model"] == "MaskMLP") & (folds["subset"] == "all")]
    for run, part in mask.groupby("run"):
        row = {"source": run, "pretrain_loss": part["loss"].iloc[0], "pretrain_scope": part["scope"].iloc[0],
               "seed": part["seed"].iloc[0], "spread_before": part["embedding_spread_initial"].mean(),
               "spread_after": part["embedding_spread"].mean(), "maskmlp_accuracy": part["accuracy"].mean()}
        log = root / run / "pretrain_log.csv"
        if log.exists():
            history = pd.read_csv(log)
            if "spread" in history:
                row["spread_after_epoch_1"] = history.loc[history["epoch"] == 1, "spread"].mean()
        rows.append(row)
    for ablation in root.rglob("loss_ablation.csv"):
        table = pd.read_csv(ablation)
        for loss, part in table.groupby("pretrain_loss", sort=False):
            rows.append({"source": f"{ablation.parent.relative_to(root).as_posix()} (loss ablation)",
                         "pretrain_loss": loss, "spread_after": part["embedding_spread"].mean(),
                         "maskmlp_accuracy": part["accuracy"].mean()})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    before = table["spread_before"].dropna().mean() if "spread_before" in table else np.nan
    table.insert(2, "loss_origin", table["pretrain_loss"].map(loss_origin))
    table["collapsed"] = table["spread_after"] < 0.1 * before if np.isfinite(before) else np.nan
    return table


def plot_paper_vs_ours(table: pd.DataFrame, figures: Path) -> None:
    for subset, part in table.groupby("subset"):
        splits = [s for s in ("school", "student") if s in set(part["split"])]
        targets = [t for t in ("RMwidRS_target", "RMwdaRS_target") if t in set(part["target"])]
        fig, axes = new_figure(len(splits), len(targets), (5.5 * len(targets), 3.8 * len(splits)))
        for row, split in enumerate(splits):
            for col, target in enumerate(targets):
                ax = axes[row, col]
                cell = part[(part["split"] == split) & (part["target"] == target)].set_index("model")
                cell = cell.loc[[m for m in MODELS if m in cell.index]]
                y = np.arange(len(cell))[::-1]
                for yi, (model, r) in zip(y, cell.iterrows()):
                    color = COLORS["MaskMLP"] if model == "MaskMLP" else COLORS[family(model)]
                    ax.errorbar(r["ours_accuracy_mean"], yi, xerr=np.nan_to_num(r["ours_accuracy_std"]), fmt="o",
                                markersize=8, color=color, elinewidth=2, capsize=0, zorder=3)
                    ax.scatter(r["paper_accuracy"], yi, s=64, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=4)
                ax.set_yticks(y)
                ax.set_yticklabels(cell.index)
                ax.grid(axis="y", visible=False)
                ax.set_xlabel("Accuracy")
                ax.set_title(f"{TARGET_NAMES[target]} · {split} split", color=INK, fontsize=11)
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=INK_2, label="Ours (mean ± std over seeds)"),
                   plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=SURFACE,
                              markeredgecolor=INK_2, markeredgewidth=2, label="Paper, Table 1")]
        top = 1 - 0.75 / (3.8 * len(splits))
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, top + 0.035))
        fig.suptitle(f"Accuracy: paper vs our reimplementation · {'all students' if subset == 'all' else 'intervention group'}",
                     y=0.995, fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, top - 0.02))
        save(fig, figures / f"paper_vs_ours_{subset}.png")


def write(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        print(f"NOTE: nothing to write for {path.name}")
        return
    table.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Paper comparison and robustness report for Neural_network_implementation.py outputs")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--primary", default="cosine,all,all",
                        help="setting compared with Table 1: pretrain_loss,pretrain_scope,eval_rows")
    args = parser.parse_args()
    root = Path(args.outputs)
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    folds = load_runs(root)
    print("Runs found:", ", ".join(sorted(folds["run"].unique())))
    primary = pick_primary(folds, args.primary)
    primary_runs = folds[(folds["loss"] == primary[0]) & (folds["scope"] == primary[1]) & (folds["rows"] == primary[2])]
    table = table1_comparison(primary_runs)
    write(table, report / "table1_comparison.csv")
    write(auc_interpretation(), report / "auc_interpretation.csv")
    write(sensitivity(folds), report / "sensitivity.csv")
    summary, wins = robustness(folds)
    write(summary, report / "robustness.csv")
    write(wins, report / "maskmlp_wins.csv")
    write(collapse(folds, root), report / "collapse.csv")
    if not table.empty:
        plot_paper_vs_ours(table, report)
    print(f"Report written to {report} (primary setting: {label(primary)})")


if __name__ == "__main__":
    main()
