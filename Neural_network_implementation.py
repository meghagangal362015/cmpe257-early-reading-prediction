"""MLP / MaskMLP reimplementation for ECRI reading-outcome prediction (Shangguan et al., NeurIPS 2024).

Trains the paper's three-layer, 64-unit MLP with Zeros / Mean / Indicator missing-value baselines and
MaskMLP (cosine-loss pretraining on masked inputs, then BCE fine-tuning). It evaluates them with five-fold
school- and student-grouped cross-validation. Logistic regression and XGBoost run on the same folds as a
controlled reference. Optional analyses: t-SNE, feature importance, the paper's Table 4 loss ablation,
accuracy by progress quintile, and embedding-collapse diagnostics.

Design decisions and their justification are documented in README.md and REPORT.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, average_precision_score, auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

FEATURES = [
    "Gender", "Tx", "Age1b", "Tier2_N", "grp_rate", "rcmistot",
    "gnrl_fid", "TKPctCrct", "NWFcls", "NWFwrc", "ORFwc", "SAwrS",
    "SAsrS", "SAtoS", "RMwidRS", "RMwdaRS",
]
TARGETS = ["RMwidRS_target", "RMwdaRS_target"]
TARGET_NAMES = {"RMwidRS_target": "Word identification", "RMwdaRS_target": "Word attack"}
TARGET_SCORES = {"RMwidRS_target": "RMwidRS", "RMwdaRS_target": "RMwdaRS"}
NEURAL = ("MLP_zero", "MLP_mean", "MLP_indicator", "MaskMLP")
TRADITIONAL = ("LogisticRegression", "XGBoost")
MODEL_SETS = {
    "all": TRADITIONAL + NEURAL,
    "neural": NEURAL,
    "traditional": TRADITIONAL,
    "core": ("LogisticRegression", "XGBoost", "MLP_zero", "MaskMLP"),
}
METRICS = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "average_precision", "sensitivity", "specificity"]
PAPER_LOSSES = ("cosine", "mse", "contrastive", "triplet")
# Extension, not in the paper: Eq. 1 with a predictor and stop-gradient. Excluded from the Table 4 ablation.
EXTENSION_LOSSES = ("simsiam",)
LOSSES = PAPER_LOSSES + EXTENSION_LOSSES
# Paper Table 4 rows: each loss alone, every pair, and all four (terms averaged with equal weights).
ABLATION_LOSSES = (*PAPER_LOSSES, "cosine+mse", "cosine+contrastive", "cosine+triplet", "mse+contrastive",
                   "mse+triplet", "contrastive+triplet", "cosine+mse+contrastive+triplet")
PROGRESS_LABELS = ["Large regression", "Slight regression", "No change", "Slight improvement", "Large improvement"]
PRETRAIN_CACHE: dict = {}

COLORS = {"traditional": "#eb6834", "neural": "#2a78d6", "MaskMLP": "#1baf7a"}
SERIES = {"LogisticRegression": "#6250d6", "XGBoost": "#eb6834", "MLP_zero": "#2a78d6", "MaskMLP": "#1baf7a"}
NEGATIVE, POSITIVE = "#a3a29d", "#2a78d6"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def family(model: str) -> str:
    return "traditional" if model in TRADITIONAL else "neural"


def available_models(names) -> list[str]:
    if XGBClassifier is None and "XGBoost" in names:
        print("NOTE: skipping XGBoost; install xgboost to include it.", flush=True)
        return [m for m in names if m != "XGBoost"]
    return list(names)


def loss_spec(value: str) -> str:
    parts = value.split("+")
    unknown = [p for p in parts if p not in LOSSES]
    if unknown or len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError(f"use one or more of {LOSSES} joined by '+', e.g. cosine+mse")
    return value


def validate_data(df: pd.DataFrame) -> None:
    required = ["StuID", "SchlID", *FEATURES, *TARGETS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required dataset columns missing: {missing}")
    if df["StuID"].isna().any() or df["StuID"].duplicated().any():
        raise ValueError("StuID must be unique and nonmissing (one row per student).")
    if df["SchlID"].isna().any():
        raise ValueError("SchlID must be nonmissing.")
    for target in TARGETS:
        values = set(pd.to_numeric(df[target], errors="raise").dropna().unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"{target} must contain binary 0/1 labels or NA.")
    for col in FEATURES:
        converted = pd.to_numeric(df[col], errors="coerce")
        if (df[col].notna() & converted.isna()).any():
            raise ValueError(f"{col} contains nonnumeric observed values; encode consistently first.")
        if (converted.dropna() < 0).any():
            raise ValueError(f"{col} has negative observed values, incompatible with -1 sentinel.")


def numeric(df: pd.DataFrame, features) -> np.ndarray:
    return df[list(features)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)


def matrix(df: pd.DataFrame, features) -> np.ndarray:
    values = numeric(df, features)
    return np.where(np.isfinite(values), values, -1.0).astype(np.float32)


def fit_scaler(x: np.ndarray):
    observed = np.where(x == -1, np.nan, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanmin(observed, axis=0), np.nanmax(observed, axis=0)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    span = np.where(np.isfinite(hi) & (hi > lo), hi - lo, 1.0)
    return lo, span


def scale(x: np.ndarray, scaler) -> np.ndarray:
    # Min-max to [0, 1] on observed values; values below the training minimum
    # are clipped to 0 so -1 stays an unambiguous missing marker.
    lo, span = scaler
    scaled = np.clip((x - lo) / span, 0.0, None)
    return np.where(x == -1, -1.0, scaled).astype(np.float32)


def observed_mean(x: np.ndarray) -> np.ndarray:
    observed = np.where(x == -1, np.nan, x)
    count = np.sum(np.isfinite(observed), axis=0)
    return np.divide(np.nansum(observed, axis=0), count, out=np.zeros(x.shape[1]), where=count > 0)


def impute_baseline(x: np.ndarray, method: str, mean: np.ndarray) -> np.ndarray:
    missing = x == -1
    if method == "MLP_zero":
        return np.where(missing, 0.0, x).astype(np.float32)
    filled = np.where(missing, mean, x).astype(np.float32)
    if method == "MLP_indicator":
        filled = np.column_stack([filled, missing.astype(np.float32)])
    return filled


def traditional_model(name: str, seed: int):
    if name == "LogisticRegression":
        return make_pipeline(SimpleImputer(strategy="mean", add_indicator=True), StandardScaler(),
                             LogisticRegression(max_iter=5000))
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=1)
    raise ValueError(name)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        blocks, width = [], input_dim
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x)).squeeze(-1)


def batched(model: MLP, x: np.ndarray, device: torch.device, fn) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            batch = torch.from_numpy(x[start:start + 1024]).to(device)
            chunks.append(fn(batch).cpu().numpy())
    return np.concatenate(chunks)


def probabilities(model: MLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    return batched(model, x, device, lambda b: torch.sigmoid(model(b)))


def embeddings(model: MLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    return batched(model, x, device, model.encoder)


def embedding_spread(model: MLP, x: np.ndarray, device: torch.device) -> float:
    """Mean per-dimension std of L2-normalised embeddings; near 0 means collapse."""
    e = embeddings(model, x, device)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    return float(e.std(axis=0).mean())


def pretrain_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, spec: str, args,
                  predictor: nn.Module | None = None) -> torch.Tensor:
    """Paper Table 4: the original input is the anchor, its masked version the positive,
    and another randomly drawn student the negative; combined terms are averaged."""
    cos = nn.functional.cosine_similarity
    terms = []
    for name in spec.split("+"):
        if name == "simsiam":  # Eq. 1 through a predictor, with stop-gradient on the target branch
            terms.append(((1 - cos(predictor(positive), anchor.detach(), dim=1)).mean()
                          + (1 - cos(predictor(anchor), positive.detach(), dim=1)).mean()) / 2)
        elif name == "cosine":  # Eq. 1
            terms.append((1 - cos(anchor, positive, dim=1)).mean())
        elif name == "mse":
            terms.append(nn.functional.mse_loss(anchor, positive))
        elif name == "contrastive":  # cosine embedding loss over one positive and one negative pair
            negative_term = torch.clamp(cos(anchor, negative, dim=1) - args.contrastive_margin, min=0)
            terms.append(((1 - cos(anchor, positive, dim=1)) + negative_term).mean() / 2)
        elif name == "triplet":
            terms.append(nn.functional.triplet_margin_loss(anchor, positive, negative, margin=args.triplet_margin))
    return torch.stack(terms).mean()


def pretrain(model: MLP, x: np.ndarray, args, device: torch.device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=args.batch_size, shuffle=True)
    hidden = model.classifier.in_features
    predictor = (nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, hidden)).to(device)
                 if "simsiam" in args.pretrain_loss.split("+") else None)
    parameters = list(model.encoder.parameters()) + (list(predictor.parameters()) if predictor else [])
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    initial_spread = embedding_spread(model, x, device)
    history, spreads = [], []
    for _ in range(args.pretrain_epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            choose = (torch.rand_like(batch) < args.mask_rate) & (batch != -1)
            masked = torch.where(choose, torch.full_like(batch, -1.0), batch)
            anchor = model.encoder(batch)
            positive = model.encoder(masked)
            negative = anchor.roll(1, dims=0)  # batches are shuffled, so this is a random other student
            loss = pretrain_loss(anchor, positive, negative, args.pretrain_loss, args, predictor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(x))
        spreads.append(embedding_spread(model, x, device))
    return history, spreads, initial_spread, spreads[-1] if spreads else initial_spread


def fit(model: MLP, x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, args, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    xv_tensor = torch.from_numpy(xv).to(device)
    yv_tensor = torch.from_numpy(yv).to(device)
    best_loss = float("inf")
    best_state = None
    patience = 0
    for _ in range(args.epochs):
        model.train()
        for batch, labels in loader:
            batch, labels = batch.to(device), labels.to(device)
            loss = criterion(model(batch), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(xv_tensor), yv_tensor).item()
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    positives, negatives = (y == 1).sum(), (y == 0).sum()
    result = {"n": len(y), "accuracy": accuracy_score(y, pred),
              "sensitivity": ((pred == 1) & (y == 1)).sum() / positives if positives else np.nan,
              "specificity": ((pred == 0) & (y == 0)).sum() / negatives if negatives else np.nan}
    if len(np.unique(y)) == 2:
        precision, recall, _ = precision_recall_curve(y, p)
        result.update(balanced_accuracy=roc_auc_score(y, pred), pr_auc=auc(recall, precision),
                      average_precision=average_precision_score(y, p), roc_auc=roc_auc_score(y, p))
    else:
        result.update(balanced_accuracy=np.nan, pr_auc=np.nan, average_precision=np.nan, roc_auc=np.nan)
    return result


def inner_split(frame: pd.DataFrame, group_col: str, seed: int):
    for attempt in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + attempt)
        left, right = next(splitter.split(frame, groups=frame[group_col]))
        train, valid = frame.iloc[left], frame.iloc[right]
        if train["_label"].nunique() == 2 and valid["_label"].nunique() == 2:
            return train, valid
    raise ValueError("Could not create two-class grouped validation partition.")


def make_folds(df: pd.DataFrame, group: str, target: str, args):
    """Yield (fold, train, valid, test, pretrain_records); deterministic for a given dataset and seed."""
    group_col = "SchlID" if group == "school" else "StuID"
    labeled = df[df[target].notna()].copy()
    if args.eval_rows == "complete":
        labeled = labeled[labeled[FEATURES].notna().all(axis=1)]
    labeled["_label"] = labeled[target].astype(int)
    if labeled[group_col].nunique() < 5:
        raise ValueError("Insufficient groups for five folds")
    cv = GroupKFold(n_splits=5)
    for fold, (itrain, itest) in enumerate(cv.split(labeled, labeled["_label"], labeled[group_col]), 1):
        outer_train = labeled.iloc[itrain]
        test = labeled.iloc[itest]
        train, valid = inner_split(outer_train, group_col, args.seed + fold)
        # Pretraining pool: the released dataset, or every student rebuilt from ECRIRawData.csv.
        pool = getattr(args, "pretrain_pool", None)
        pool = df if pool is None else pool
        if args.pretrain_scope == "all":
            # Paper: "employing all data samples during pre-training ... no labels are used".
            pretrain_records = pool
        else:
            withheld = set(valid[group_col]) | set(test[group_col])
            pretrain_records = pool[~pool[group_col].isin(withheld)]
            assert not set(pretrain_records[group_col]).intersection(withheld)
        yield fold, train, valid, test, pretrain_records


def pretrained_maskmlp(features, pretrain_records, args, device, seed):
    """Pretrain a MaskMLP encoder, reusing it when the pretraining data and settings are identical.
    With --pretrain-scope all the encoder does not depend on the split or target, so it is shared."""
    xpre = matrix(pretrain_records, features)
    key = (tuple(features), seed, args.pretrain_loss, args.pretrain_epochs, args.mask_rate, args.learning_rate,
           args.batch_size, args.hidden_size, args.layers, args.contrastive_margin, args.triplet_margin, len(xpre))
    scaler = fit_scaler(xpre)
    model = MLP(len(features), args.hidden_size, args.layers).to(device)
    if args.pretrain_scope == "all" and key in PRETRAIN_CACHE:
        state, info = PRETRAIN_CACHE[key]
        model.encoder.load_state_dict(state)
        return model, scaler, info
    set_seed(seed)
    history, spreads, initial, spread = pretrain(model, scale(xpre, scaler), args, device)
    info = {"pretrain_n": len(xpre), "pretrain_loss": history, "spread_history": spreads,
            "embedding_spread_initial": initial, "embedding_spread": spread}
    if spread < args.collapse_ratio * initial:
        print(f"WARNING: MaskMLP embeddings collapsed during pretraining with '{args.pretrain_loss}' "
              f"(spread {initial:.2e} -> {spread:.2e}).", flush=True)
    if args.pretrain_scope == "all":
        PRETRAIN_CACHE[key] = (copy.deepcopy(model.encoder.state_dict()), info)
    return model, scaler, info


def fit_predict(method, features, train, valid, test, pretrain_records, args, device, seed):
    """Train one model on one fold. Returns test probabilities, info, and (model, encode) for neural models."""
    set_seed(seed)
    if method in TRADITIONAL:
        # Traditional models need no early stopping, so they train on the full outer training fold.
        outer = pd.concat([train, valid])
        model = traditional_model(method, seed)
        model.fit(numeric(outer, features), outer["_label"].to_numpy(int))
        return model.predict_proba(numeric(test, features))[:, 1], {}, None
    info = {}
    if method == "MaskMLP":
        model, scaler, info = pretrained_maskmlp(features, pretrain_records, args, device, seed)
        encode = lambda frame: scale(matrix(frame, features), scaler)
        set_seed(seed + 10_000)  # fine-tuning randomness is independent of whether pretraining was reused
    else:
        xraw = matrix(train, features)
        scaler = fit_scaler(xraw)
        mean = observed_mean(scale(xraw, scaler))
        encode = lambda frame: impute_baseline(scale(matrix(frame, features), scaler), method, mean)
        model = MLP(encode(train.iloc[:1]).shape[1], args.hidden_size, args.layers).to(device)
    fit(model, encode(train), train["_label"].to_numpy(np.float32),
        encode(valid), valid["_label"].to_numpy(np.float32), args, device)
    return probabilities(model, encode(test), device), info, (model, encode)


def cross_validate(method, features, folds, args, device) -> dict:
    scores = []
    for fold, train, valid, test, pretrain_records in folds:
        p, _, _ = fit_predict(method, features, train, valid, test, pretrain_records, args, device, args.seed + fold)
        scores.append(evaluate(test["_label"].to_numpy(int), p))
    return pd.DataFrame(scores)[["accuracy", "specificity", "sensitivity", "balanced_accuracy", "pr_auc"]].mean().to_dict()


def paired_tests(fold_df: pd.DataFrame, reference: str = "MaskMLP") -> pd.DataFrame:
    rows = []
    for (group, target, subset), part in fold_df.groupby(["group", "target", "subset"]):
        for metric in ("accuracy", "balanced_accuracy", "pr_auc"):
            table = part.pivot(index="fold", columns="model", values=metric)
            if reference not in table:
                continue
            for other in table.columns.drop(reference):
                pair = table[[reference, other]].dropna()
                if len(pair) < 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    t, p = ttest_rel(pair[reference], pair[other])
                rows.append({"group": group, "target": target, "subset": subset, "metric": metric,
                             "model": reference, "compared_to": other, "family": family(other),
                             "mean_difference": (pair[reference] - pair[other]).mean(),
                             "t": t, "p_value": p, "folds": len(pair)})
    return pd.DataFrame(rows)


def tsne_coordinates(model_encoders: dict, frame: pd.DataFrame, device, seed: int) -> pd.DataFrame:
    rows = []
    for method, (model, encode) in model_encoders.items():
        e = embeddings(model, encode(frame), device)
        xy = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=seed).fit_transform(e)
        rows.append(pd.DataFrame({"model": method, "x": xy[:, 0], "y": xy[:, 1],
                                  "label": frame["_label"].to_numpy(int), "StuID": frame["StuID"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def feature_importance(df: pd.DataFrame, args, device) -> pd.DataFrame:
    """Paper Fig. 4 / Table 5: drop each input in turn, retrain, and measure the change in performance."""
    models = available_models(["MaskMLP", "XGBoost"])
    groups = ("school", "student") if args.importance_group == "both" else (args.importance_group,)
    rows = []
    for group in groups:
        for target in TARGETS:
            folds = list(make_folds(df, group, target, args))
            for dropped in [None, *FEATURES]:
                features = [f for f in FEATURES if f != dropped]
                for method in models:
                    scores = cross_validate(method, features, folds, args, device)
                    rows.append({"group": group, "target": target, "model": method,
                                 "dropped_feature": dropped or "(none)", **scores})
                    print(f"importance | {group} | {target} | drop {dropped or '(none)'} | {method} | "
                          f"acc={scores['accuracy']:.4f}", flush=True)
    result = pd.DataFrame(rows)
    full = result[result["dropped_feature"] == "(none)"].set_index(["group", "target", "model"])
    keys = pd.MultiIndex.from_frame(result[["group", "target", "model"]])
    for metric in ("accuracy", "balanced_accuracy", "pr_auc"):
        result[f"{metric}_drop"] = full[metric].reindex(keys).to_numpy() - result[metric].to_numpy()
    return result


def loss_ablation(df: pd.DataFrame, args, device) -> pd.DataFrame:
    """Paper Table 4: MaskMLP with each pretraining loss and combination, on word identification."""
    rows = []
    for group in ("school", "student"):
        folds = list(make_folds(df, group, args.ablation_target, args))
        for spec in ABLATION_LOSSES:
            variant = copy.copy(args)
            variant.pretrain_loss = spec
            scores = cross_validate("MaskMLP", FEATURES, folds, variant, device)
            spreads = [PRETRAIN_CACHE[k][1]["embedding_spread"] for k in PRETRAIN_CACHE if k[2] == spec]
            rows.append({"group": group, "target": args.ablation_target, "pretrain_loss": spec,
                         "embedding_spread": np.mean(spreads) if spreads else np.nan, **scores})
            print(f"ablation | {group} | {spec} | acc={scores['accuracy']:.4f} balanced_accuracy={scores['balanced_accuracy']:.4f}", flush=True)
    return pd.DataFrame(rows)


def load_unfiltered(raw_path: str) -> pd.DataFrame:
    """Start-of-year inputs for every student in ECRIRawData.csv (one row per student at the first time point)."""
    raw = pd.read_csv(raw_path)
    start = raw[raw["t"] == raw["t"].min()]
    if start["StuID"].duplicated().any():
        raise SystemExit(f"{raw_path}: some students have more than one start-of-year row; cannot rebuild inputs")
    start = start.copy()
    start["Tx"] = start["Tx"].astype(str).str.split(":").str[0].astype(float)
    return start[["StuID", "SchlID", *FEATURES]].reset_index(drop=True)


def verify_labels(df: pd.DataFrame, raw_path: str) -> dict:
    """Rebuild each outcome label from ECRIRawData.csv and compare it with ECRIDataSet.csv.

    Label = 1 when a student's Woodcock gain (score at t=1 minus score at t=0) exceeds the mean gain
    of control-group (Tx = 0) students. This is an independent check of the released labels; the
    models always train on the labels in ECRIDataSet.csv.
    """
    raw = pd.read_csv(raw_path)
    start = raw["t"].min()
    tx = raw[raw["t"] == start].set_index("StuID")["Tx"].astype(str).str.split(":").str[0]
    report = {"label_definition": "gain(t=1 minus t=0) > mean gain of Tx=0 students"}
    for target, score in TARGET_SCORES.items():
        wide = raw.pivot_table(index="StuID", columns="t", values=score, aggfunc="first")
        gain = (wide[1.0] - wide[start]).dropna()
        control_mean = gain[tx.reindex(gain.index) == "0"].mean()
        rebuilt = (gain > control_mean).astype(int)
        released = pd.to_numeric(df.set_index("StuID")[target], errors="coerce").dropna().astype(int)
        shared = released.index.intersection(rebuilt.index)
        report[target] = {"released_labels": int(len(released)), "rebuilt_and_compared": int(len(shared)),
                          "mismatched_labels": int((rebuilt[shared] != released[shared]).sum()),
                          "control_mean_gain": float(control_mean)}
    report["verified"] = all(report[t]["rebuilt_and_compared"] == report[t]["released_labels"]
                             and report[t]["mismatched_labels"] == 0 for t in TARGET_SCORES)
    return report


def verify_unfiltered(df: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """Check that the rebuilt records reproduce ECRIDataSet.csv for every student the two files share."""
    merged = df[["StuID", "SchlID", *FEATURES]].merge(pool, on="StuID", suffixes=("_dataset", "_raw"))
    mismatched = int((merged["SchlID_dataset"] != merged["SchlID_raw"]).sum())
    for f in FEATURES:
        a = pd.to_numeric(merged[f"{f}_dataset"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(merged[f"{f}_raw"], errors="coerce").to_numpy(float)
        mismatched += int((~np.isclose(a, b, equal_nan=True)).sum())
    report = {"dataset_students": int(df["StuID"].nunique()), "rebuilt_students": int(len(pool)),
              "shared_students": int(len(merged)),
              "dataset_students_missing_from_raw": int(len(set(df["StuID"]) - set(pool["StuID"]))),
              "cells_compared": int(len(merged) * (len(FEATURES) + 1)), "mismatched_cells": mismatched}
    report["verified"] = report["dataset_students_missing_from_raw"] == 0 and mismatched == 0
    return report


def load_gains(raw_path: str) -> pd.DataFrame:
    """Start-to-end-of-year Woodcock gains per student from ECRIRawData.csv (paper Fig. 2)."""
    raw = pd.read_csv(raw_path)
    gains = pd.DataFrame(index=pd.Index(raw["StuID"].unique(), name="StuID"))
    for target, score in TARGET_SCORES.items():
        wide = raw.dropna(subset=[score]).pivot_table(index="StuID", columns="t", values=score, aggfunc="first")
        gains[f"gain_{target}"] = wide[wide.columns.max()] - wide[wide.columns.min()]
    return gains.reset_index()


def progress_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    """Accuracy within five quantile groups of actual score gain (paper Fig. 2)."""
    rows = []
    for (group, target), part in predictions.groupby(["group", "target"]):
        part = part.dropna(subset=["gain"]).copy()
        students = part.drop_duplicates("StuID").set_index("StuID")["gain"]
        if len(students) < 5:
            continue
        quintile = pd.qcut(students.rank(method="first"), 5, labels=PROGRESS_LABELS)
        part["progress"] = part["StuID"].map(quintile)
        part["correct"] = (part["p"] >= 0.5).astype(int) == part["y"]
        for (model, progress), cell in part.groupby(["model", "progress"], observed=True):
            rows.append({"group": group, "target": target, "model": model, "progress": progress,
                         "n": len(cell), "accuracy": cell["correct"].mean(),
                         "gain_min": cell["gain"].min(), "gain_max": cell["gain"].max()})
    return pd.DataFrame(rows)



def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(rows: int, cols: int, size):
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2})
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, facecolor=SURFACE)
    for ax in axes.flat:
        style_axes(ax)
    return fig, axes


def save(fig, path: Path) -> None:
    try:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    except OSError as error:  # e.g. Windows paths over 260 characters; results CSVs are already written
        print(f"WARNING: could not save figure {path.name}: {error}", flush=True)
    finally:
        plt.close(fig)


def model_color(model: str) -> str:
    return COLORS["MaskMLP"] if model == "MaskMLP" else COLORS[family(model)]


def family_legend(fig) -> None:
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=COLORS[key], label=label)
               for key, label in (("traditional", "Traditional ML"), ("neural", "Neural baseline"), ("MaskMLP", "MaskMLP"))]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))


def plot_comparison(fold_df: pd.DataFrame, predictions: pd.DataFrame, figures: Path) -> None:
    order = [m for m in (*TRADITIONAL, *NEURAL) if m in set(fold_df["model"])]
    targets = [t for t in TARGETS if t in set(fold_df["target"])]
    metrics = (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("pr_auc", "PR-AUC"))
    for (group, subset), part in fold_df.groupby(["group", "subset"]):
        fig, axes = new_figure(len(metrics), len(targets), (5.2 * len(targets), 0.4 * len(order) * len(metrics) + 3))
        for col, target in enumerate(targets):
            stats = part[part["target"] == target].groupby("model")[[m for m, _ in metrics]].agg(["mean", "std"])
            positive_rate = predictions.query("group == @group and target == @target and (@subset == 'all' or Tx == 1)")["y"].mean()
            for row, (metric, label) in enumerate(metrics):
                ax = axes[row, col]
                models = [m for m in order if m in stats.index]
                y = np.arange(len(models))[::-1]
                means = stats.loc[models, (metric, "mean")].to_numpy()
                stds = stats.loc[models, (metric, "std")].fillna(0).to_numpy()
                for yi, m, mu, sd in zip(y, models, means, stds):
                    ax.errorbar(mu, yi, xerr=sd, fmt="o", markersize=8, color=model_color(m), elinewidth=2, capsize=0)
                    if m == "MaskMLP":
                        ax.annotate(f"{mu:.3f}", (mu, yi), xytext=(0, 9), textcoords="offset points",
                                    ha="center", fontsize=8, color=INK)
                reference = {"pr_auc": ("no-skill", positive_rate), "balanced_accuracy": ("chance", 0.5)}.get(metric)
                if reference and np.isfinite(reference[1]):
                    ax.axvline(reference[1], color=INK_2, linestyle="--", linewidth=1)
                    ax.annotate(reference[0], (reference[1], y.max() + 0.4), fontsize=8, color=INK_2, ha="left",
                                xytext=(3, 0), textcoords="offset points")
                ax.set_yticks(y)
                ax.set_yticklabels(models)
                ax.set_ylim(-0.7, len(models) - 0.1)
                ax.grid(axis="y", visible=False)
                ax.set_xlabel(f"{label} (mean ± std, 5 folds)")
                if row == 0:
                    ax.set_title(TARGET_NAMES[target], color=INK, fontsize=11)
        fig.suptitle(f"{group.title()} split · {'all students' if subset == 'all' else 'intervention group'}",
                     y=0.95, fontsize=12, color=INK)
        family_legend(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        save(fig, figures / f"comparison_{group}_{subset}.png")


def plot_pr_curves(fold_df: pd.DataFrame, predictions: pd.DataFrame, figures: Path) -> None:
    allrows = fold_df[fold_df["subset"] == "all"]
    targets = [t for t in TARGETS if t in set(allrows["target"])]
    for group, part in allrows.groupby("group"):
        fig, axes = new_figure(1, len(targets), (5.2 * len(targets), 4.4))
        for col, target in enumerate(targets):
            ax = axes[0, col]
            means = part[part["target"] == target].groupby("model")["pr_auc"].mean()
            picks = [("MaskMLP", COLORS["MaskMLP"])] if "MaskMLP" in means else []
            for fam in ("neural", "traditional"):
                candidates = means[[m for m in means.index if family(m) == fam and m != "MaskMLP"]]
                if len(candidates):
                    picks.append((candidates.idxmax(), COLORS[fam]))
            subset = predictions.query("group == @group and target == @target")
            for model, color in picks:
                p = subset[subset["model"] == model]
                precision, recall, _ = precision_recall_curve(p["y"], p["p"])
                ax.plot(recall, precision, color=color, linewidth=2, label=f"{model} (PR-AUC {means[model]:.3f})")
            ax.axhline(subset["y"].mean(), color=INK_2, linestyle="--", linewidth=1, label="No-skill")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(TARGET_NAMES[target], color=INK, fontsize=11)
            ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle(f"Precision-recall, out-of-fold predictions · {group} split · best per family",
                     fontsize=12, color=INK)
        fig.tight_layout()
        save(fig, figures / f"pr_curves_{group}.png")


def plot_pretraining(log: pd.DataFrame, fold_df: pd.DataFrame, figures: Path) -> None:
    panels = [("loss", "Pretraining loss", "linear")]
    if "spread" in log:
        panels.append(("spread", "Embedding spread (log scale; near 0 = collapse)", "log"))
    fig, axes = new_figure(1, len(panels), (6.4 * len(panels), 4.2))
    maskmlp = fold_df[fold_df["model"] == "MaskMLP"]
    initial = maskmlp["embedding_spread_initial"].mean()
    for ax, (column, label, yscale) in zip(axes[0], panels):
        for group, color in (("school", COLORS["neural"]), ("student", COLORS["traditional"])):
            part = log[log["group"] == group]
            if part.empty:
                continue
            curve = part.groupby("epoch")[column].agg(["mean", "min", "max"])
            ax.plot(curve.index, curve["mean"], color=color, linewidth=2, label=f"{group.title()} split")
            ax.fill_between(curve.index, curve["min"], curve["max"], color=color, alpha=0.15, linewidth=0)
        if column == "spread" and np.isfinite(initial):
            ax.axhline(initial, color=INK_2, linestyle="--", linewidth=1)
            ax.annotate("before pretraining", (ax.get_xlim()[1], initial), xytext=(-4, 4), textcoords="offset points",
                        ha="right", fontsize=8, color=INK_2)
        ax.set_yscale(yscale)
        ax.set_xlabel("Pretraining epoch")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"MaskMLP pretraining · band: range over folds/targets · spread {initial:.2e} before → "
                 f"{maskmlp['embedding_spread'].mean():.2e} after", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, figures / "pretraining_loss.png")


def plot_tsne(coords: pd.DataFrame, figures: Path, caption: str) -> None:
    models = list(dict.fromkeys(coords["model"]))
    fig, axes = new_figure(1, len(models), (5 * len(models), 4.8))
    for ax, model in zip(axes[0], models):
        part = coords[coords["model"] == model]
        for value, color, label in ((0, NEGATIVE, "Insufficient progress (0)"), (1, POSITIVE, "Sufficient progress (1)")):
            points = part[part["label"] == value]
            ax.scatter(points["x"], points["y"], s=14, color=color, alpha=0.75, linewidths=0, label=label)
        ax.set_title(model, color=INK, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"t-SNE of encoder embeddings · {caption}", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, figures / "tsne_mlp_vs_maskmlp.png")


def plot_importance(importance: pd.DataFrame, figures: Path) -> None:
    for (group, target), part in importance.groupby(["group", "target"]):
        drops = part[part["dropped_feature"] != "(none)"]
        models = list(dict.fromkeys(drops["model"]))
        fig, axes = new_figure(1, len(models), (4.6 * len(models), 5.6))
        for ax, model in zip(axes[0], models):
            bars = drops[drops["model"] == model].sort_values("accuracy_drop")
            ax.barh(bars["dropped_feature"], bars["accuracy_drop"], height=0.7, color=model_color(model),
                    edgecolor=SURFACE, linewidth=2)
            ax.axvline(0, color=INK_2, linewidth=1)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("Decrease in accuracy when the feature is removed")
            full = part[(part["model"] == model) & (part["dropped_feature"] == "(none)")]["accuracy"].iloc[0]
            ax.set_title(f"{model} (all features: {full:.3f})", color=INK, fontsize=11)
        fig.suptitle(f"Leave-one-feature-out importance · {TARGET_NAMES[target]}, {group} split",
                     fontsize=11, color=INK)
        fig.tight_layout()
        save(fig, figures / f"feature_importance_{group}_{TARGET_SCORES[target]}.png")


def plot_loss_ablation(ablation: pd.DataFrame, figures: Path) -> None:
    metrics = (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"))
    groups = list(dict.fromkeys(ablation["group"]))
    fig, axes = new_figure(len(metrics), len(groups), (5.2 * len(groups), 9))
    for col, group in enumerate(groups):
        part = ablation[ablation["group"] == group].set_index("pretrain_loss").loc[list(ABLATION_LOSSES)]
        y = np.arange(len(part))[::-1]
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            ax.scatter(part[metric], y, s=64, color=COLORS["MaskMLP"], zorder=3, label="MaskMLP")
            ax.scatter(part.loc[["cosine"], metric], y[:1], s=64, facecolor=SURFACE, edgecolor=INK, linewidth=2,
                       zorder=4, label="Paper's loss (cosine)")
            ax.set_yticks(y)
            ax.set_yticklabels(part.index)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(f"{label}, mean over 5 folds")
            if row == 0:
                ax.set_title(f"{group.title()} split", color=INK, fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"MaskMLP pretraining loss ablation · {TARGET_NAMES[ablation['target'].iloc[0]]}",
                 y=0.995, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, figures / "loss_ablation.png")


def plot_progress(breakdown: pd.DataFrame, figures: Path) -> None:
    models = [m for m in SERIES if m in set(breakdown["model"])]
    width = 0.8 / len(models)
    for group, part in breakdown.groupby("group"):
        targets = [t for t in TARGETS if t in set(part["target"])]
        fig, axes = new_figure(1, len(targets), (6.4 * len(targets), 4.6))
        for ax, target in zip(axes[0], targets):
            cells = part[part["target"] == target].set_index(["model", "progress"])["accuracy"]
            x = np.arange(len(PROGRESS_LABELS))
            for i, model in enumerate(models):
                values = [cells.get((model, label), np.nan) for label in PROGRESS_LABELS]
                ax.bar(x - 0.4 + width * (i + 0.5), values, width=width, color=SERIES[model],
                       edgecolor=SURFACE, linewidth=2, label=model)
            ax.set_xticks(x)
            ax.set_xticklabels([label.replace(" ", "\n") for label in PROGRESS_LABELS], fontsize=8)
            ax.set_ylim(0, 1)
            ax.grid(axis="x", visible=False)
            ax.set_ylabel("Accuracy")
            ax.set_title(TARGET_NAMES[target], color=INK, fontsize=11)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, 0.935))
        fig.suptitle(f"Accuracy by student progress quintile (actual score gain) · {group} split",
                     y=0.99, fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        save(fig, figures / f"progress_breakdown_{group}.png")


def make_plots(out: Path) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(out / "results.csv")
    fold_df = results[results["record_type"] == "fold"]
    predictions = pd.read_csv(out / "predictions.csv")
    plot_comparison(fold_df, predictions, figures)
    plot_pr_curves(fold_df, predictions, figures)
    if (out / "pretrain_log.csv").exists():
        plot_pretraining(pd.read_csv(out / "pretrain_log.csv"), fold_df, figures)
    if (out / "tsne.csv").exists():
        tsne = pd.read_csv(out / "tsne.csv")
        plot_tsne(tsne, figures, tsne["caption"].iloc[0])
    if "gain" in predictions and predictions["gain"].notna().any():
        breakdown = progress_breakdown(predictions)
        breakdown.to_csv(out / "progress_breakdown.csv", index=False)
        plot_progress(breakdown, figures)
    if (out / "importance.csv").exists():
        plot_importance(pd.read_csv(out / "importance.csv"), figures)
    if (out / "loss_ablation.csv").exists():
        plot_loss_ablation(pd.read_csv(out / "loss_ablation.csv"), figures)
    print(f"Figures written to {figures}", flush=True)



def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        make_plots(out)
        return
    set_seed(args.seed)
    df = pd.read_csv(args.dataset)
    validate_data(df)
    gains = load_gains(args.raw_dataset) if args.raw_dataset else None
    use_raw = args.pretrain_source == "raw" or (args.pretrain_source == "auto" and args.raw_dataset)
    if use_raw and not args.raw_dataset:
        raise SystemExit("--pretrain-source raw needs --raw-dataset ECRIRawData.csv")
    args.pretrain_pool, verification = None, None
    if use_raw:
        pool = load_unfiltered(args.raw_dataset)
        verification = verify_unfiltered(df, pool)
        verification["labels"] = verify_labels(df, args.raw_dataset)  # reported only; does not gate pretraining
        (out / "raw_verification.json").write_text(json.dumps(verification, indent=2))
        print(f"Raw-data check: {verification}", flush=True)
        if verification["verified"]:
            args.pretrain_pool = pool
        elif args.pretrain_source == "raw":
            raise SystemExit("Records rebuilt from --raw-dataset do not match --dataset; see raw_verification.json")
        else:
            warnings.warn("Records rebuilt from --raw-dataset do not match --dataset; pretraining on --dataset instead")
            use_raw = False
    pool_size = len(args.pretrain_pool) if use_raw else len(df)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if not args.skip_main:
        config = {k: v for k, v in vars(args).items() if k != "pretrain_pool"}
        # File names only, so configs can be shared without local (personal) paths.
        for key in ("dataset", "raw_dataset", "output_dir"):
            if config.get(key):
                config[key] = Path(config[key]).name
        config["pretrain_records"] = "ECRIRawData.csv (unfiltered)" if use_raw else "ECRIDataSet.csv"
        config["pretrain_pool_size"] = pool_size
        config["raw_verification"] = verification
        (out / "config.json").write_text(json.dumps(config, indent=2))
    if args.skip_main:
        # Add the slow analyses to an existing --output-dir without re-running the main comparison.
        if args.importance:
            feature_importance(df, args, device).to_csv(out / "importance.csv", index=False)
        if args.loss_ablation:
            loss_ablation(df, args, device).to_csv(out / "loss_ablation.csv", index=False)
        make_plots(out)
        return
    methods = available_models(MODEL_SETS[args.models])
    groups_to_run = ("school", "student") if args.group == "both" else (args.group,)
    targets = TARGETS if args.target == "both" else (args.target,)
    rows, predictions, folds, pretrain_log, tsne = [], [], [], [], None
    print(f"Dataset: {len(df)} student rows; pretraining pool: {pool_size} students; device: {device}; "
          f"targets: {targets}; evaluating {args.eval_rows} rows; pretraining loss: {args.pretrain_loss}; "
          f"models: {methods}", flush=True)
    if not use_raw:
        print("NOTE: pass --raw-dataset ECRIRawData.csv to pretrain MaskMLP on all 6,916 students "
              "(ECRIDataSet.csv holds 6,803).", flush=True)
    for group in groups_to_run:
        for target in targets:
            for fold, train, valid, test, pretrain_records in make_folds(df, group, target, args):
                for split, frame in (("train", train), ("valid", valid), ("test", test)):
                    folds.append(pd.DataFrame({"group": group, "target": target, "fold": fold, "split": split,
                                               "StuID": frame["StuID"].to_numpy(), "SchlID": frame["SchlID"].to_numpy()}))
                ytest = test["_label"].to_numpy(int)
                tx = pd.to_numeric(test["Tx"], errors="coerce").to_numpy() == 1
                gain = (test["StuID"].map(gains.set_index("StuID")[f"gain_{target}"]).to_numpy()
                        if gains is not None else np.full(len(test), np.nan))
                encoders = {}
                for method in methods:
                    p, info, neural = fit_predict(method, FEATURES, train, valid, test, pretrain_records,
                                                  args, device, args.seed + fold)
                    if neural and method in ("MLP_zero", "MaskMLP"):
                        encoders[method] = neural
                    predictions.append(pd.DataFrame({"group": group, "target": target, "model": method,
                                                     "family": family(method), "fold": fold,
                                                     "StuID": test["StuID"].to_numpy(), "Tx": tx.astype(int),
                                                     "y": ytest, "p": p, "gain": gain}))
                    for epoch, (loss, spread) in enumerate(zip(info.get("pretrain_loss", []),
                                                               info.get("spread_history", [])), 1):
                        pretrain_log.append({"group": group, "target": target, "fold": fold, "epoch": epoch,
                                             "loss": loss, "spread": spread})
                    for subset, selected in {"all": np.ones(len(test), dtype=bool), "intervention": tx}.items():
                        if selected.any():
                            rows.append({"record_type": "fold", "group": group, "target": target,
                                         "model": method, "family": family(method), "subset": subset, "fold": fold,
                                         "train_n": len(train) + (len(valid) if method in TRADITIONAL else 0),
                                         "pretrain_n": info.get("pretrain_n", 0),
                                         "embedding_spread_initial": info.get("embedding_spread_initial", np.nan),
                                         "embedding_spread": info.get("embedding_spread", np.nan),
                                         **evaluate(ytest[selected], p[selected])})
                    print(f"{group} | {target} | fold {fold} | {method} | AP={average_precision_score(ytest, p):.4f}", flush=True)
                if (args.tsne and tsne is None and fold == 1 and group == groups_to_run[0]
                        and target == targets[0] and len(encoders) == 2):
                    # The paper's t-SNE compares MLP and MaskMLP embeddings; fold-1 models embed every evaluated student.
                    evaluated = pd.concat([train, valid, test])
                    tsne = tsne_coordinates(encoders, evaluated, device, args.seed)
                    tsne["caption"] = f"{TARGET_NAMES[target]}, {group} split, fold-1 models, all evaluated students"
    fold_df = pd.DataFrame(rows)
    summary = fold_df.groupby(["group", "target", "model", "family", "subset"], dropna=False)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.insert(0, "record_type", "summary")
    pd.concat([fold_df, summary], ignore_index=True, sort=False).to_csv(out / "results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out / "folds.csv", index=False)
    paired_tests(fold_df).to_csv(out / "significance.csv", index=False)
    if pretrain_log:
        pd.DataFrame(pretrain_log).to_csv(out / "pretrain_log.csv", index=False)
    if tsne is not None:
        tsne.to_csv(out / "tsne.csv", index=False)
    print(f"Saved {len(fold_df)} fold rows and {len(summary)} summary rows to {out}", flush=True)
    print(summary[["group", "target", "model", "subset", "accuracy_mean", "balanced_accuracy_mean", "pr_auc_mean"]].to_string(index=False))
    if args.importance:
        feature_importance(df, args, device).to_csv(out / "importance.csv", index=False)
    if args.loss_ablation:
        loss_ablation(df, args, device).to_csv(out / "loss_ablation.csv", index=False)
    make_plots(out)


def main():
    parser = argparse.ArgumentParser(description="ECRI neural network (MLP, MaskMLP) vs traditional ML experiments")
    parser.add_argument("--dataset", default="ECRIDataSet.csv")
    parser.add_argument("--raw-dataset", default=None,
                        help="ECRIRawData.csv: pretrains MaskMLP on all 6,916 students (after checking the rebuilt records "
                             "match --dataset) and enables the accuracy-by-progress-quintile breakdown (paper Fig. 2)")
    parser.add_argument("--output-dir", default="member3_outputs")
    parser.add_argument("--group", choices=["school", "student", "both"], default="both")
    parser.add_argument("--target", choices=[*TARGETS, "both"], default="both")
    parser.add_argument("--models", choices=list(MODEL_SETS), default="all",
                        help="all: neural models plus logistic regression and XGBoost as a same-fold reference; "
                             "neural: MLPs and MaskMLP only")
    parser.add_argument("--eval-rows", choices=["complete", "all"], default="all",
                        help="all: every labeled student, consistent with paper Table 1 where the Zeros/Mean/Indicator "
                             "MLPs differ; complete: only students with all 16 features, as the paper's text states")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-rate", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--pretrain-scope", choices=["all", "train"], default="all",
                        help="all: pretrain on every student without labels, as in the paper; "
                             "train: exclude validation/test-fold students from pretraining")
    parser.add_argument("--pretrain-source", choices=["auto", "dataset", "raw"], default="auto",
                        help="records for MaskMLP pretraining: raw = all 6,916 students rebuilt from --raw-dataset "
                             "(the paper's count); dataset = the 6,803 in ECRIDataSet.csv; auto = raw when "
                             "--raw-dataset is given")
    parser.add_argument("--pretrain-loss", type=loss_spec, default="cosine",
                        help="cosine (paper Eq. 1), mse, contrastive, triplet, or a '+'-joined combination "
                             "averaged with equal weights, e.g. cosine+mse (paper Table 4). simsiam is an extension, not "
                             "in the paper: Eq. 1 with a predictor and stop-gradient")
    parser.add_argument("--contrastive-margin", type=float, default=0.0)
    parser.add_argument("--triplet-margin", type=float, default=1.0)
    parser.add_argument("--collapse-ratio", type=float, default=0.1,
                        help="warn when embedding spread falls below this fraction of its untrained value")
    parser.add_argument("--no-tsne", dest="tsne", action="store_false")
    parser.add_argument("--importance", action="store_true",
                        help="leave-one-feature-out importance for MaskMLP and XGBoost on both targets (paper Fig. 4, Table 5)")
    parser.add_argument("--importance-group", choices=["school", "student", "both"], default="school")
    parser.add_argument("--loss-ablation", action="store_true",
                        help="MaskMLP with every pretraining loss and combination (paper Table 4)")
    parser.add_argument("--ablation-target", choices=TARGETS, default="RMwidRS_target")
    parser.add_argument("--plots-only", action="store_true", help="redraw figures from an existing --output-dir")
    parser.add_argument("--skip-main", action="store_true",
                        help="only run --importance / --loss-ablation into an existing --output-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
