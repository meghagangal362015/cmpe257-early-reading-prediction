"""Shared data loader for the CMPE 257 early reading prediction project.

Everyone imports from here so all models use the same data, features and splits.

Usage (from a notebook in notebooks/):
    import sys; sys.path.append("..")
    from src.data import load_labeled, FALL_FEATURES, get_cv
"""
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

DATA_DIR = "../data"

# Measured in the fall, before prediction time: safe to use.
FALL_FEATURES = [
    "Gender", "Tier", "LEP", "SpEd", "Tx", "Age1b", "Tier2_N",
    "TKPctCrct", "TYearsTeaching", "NWFcls", "NWFwrc", "ORFwc",
    "SAwrS", "SAsrS", "SAtoS", "RMwidRS", "RMwdaRS",
]

# Measured DURING the school year: possible leakage. Use only for comparison.
IN_YEAR_FEATURES = ["grp_rate", "rcmistot", "gnrl_fid", "Tier2MinRd"]

# Yes/no targets from the paper, and the matching score columns.
TARGETS = {"Word Identification": "RMwidRS", "Word Attack": "RMwdaRS"}


def load_labeled(data_dir=DATA_DIR):
    """Return the ~946 labeled students with fall features, targets and gain columns.

    For each test in TARGETS:
      <score>_target : 1 if the student grew more than the median, else 0
      <score>_gain   : spring score minus fall score (for regression)
    """
    df = pd.read_csv(f"{data_dir}/ECRIDataSet.csv")
    raw = pd.read_csv(f"{data_dir}/ECRIRawData.csv", low_memory=False)

    # Spring (Time 2) scores are used ONLY to build the gain target, never as features.
    spring = (raw[raw["Time"] == 2.0]
              .set_index("StuID")[list(TARGETS.values())]
              .add_suffix("_spring"))
    df = df.join(spring, on="StuID")
    df = df[df["RMwidRS_target"].notna()].reset_index(drop=True)

    for score in TARGETS.values():
        df[f"{score}_gain"] = df[f"{score}_spring"] - df[score]
    return df


def get_cv(level="school", n_splits=5):
    """Return (cv, uses_groups).

    level="school"  : test folds contain whole schools the model never saw.
    level="student" : random students, schools can appear in train and test.
    Pass groups=df["SchlID"] to cross-validation only when uses_groups is True.
    """
    if level == "school":
        return GroupKFold(n_splits=n_splits), True
    if level == "student":
        return KFold(n_splits=n_splits, shuffle=True, random_state=42), False
    raise ValueError("level must be 'school' or 'student'")
