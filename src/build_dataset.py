"""Build the combined ECRI dataset (Woodcock + Stanford targets).

Run from the repo root:
    python src/build_dataset.py

Reads the 3 original CSVs from data/ and writes data/processed/ecri_combined.csv.

What this script does:
  - one row per student, fall (start-of-year) features only
  - joins school-level info from MetaData.csv
  - adds MID-YEAR (winter) fluency scores as *_mid columns, for the mid-year scenario only
  - adds spring scores and gains, used ONLY to build targets
  - adds the Stanford growth target (gain >= median), like the paper's Woodcock targets
  - keeps students with at least one target
  - adds a fixed train/validation/test split (70/15/15) that keeps whole schools together

What it does NOT do (do these inside cross-validation to avoid leakage):
  - fill missing values, scale features, or select features
"""
import pandas as pd
import numpy as np

RAW_DIR = "data"
OUT = "data/processed/ecri_combined.csv"

df = pd.read_csv(f"{RAW_DIR}/ECRIDataSet.csv")
raw = pd.read_csv(f"{RAW_DIR}/ECRIRawData.csv", low_memory=False)
meta = pd.read_csv(f"{RAW_DIR}/MetaData.csv")

# Drop columns that are the same for every row (no information)
df = df.drop(columns=["Time", "t", "Keep"])

# School-level information
meta = meta.rename(columns={"SchoolID": "SchlID"}).add_prefix("Schl_").rename(columns={"Schl_SchlID": "SchlID"})
df = df.merge(meta, on="SchlID", how="left")

# Mid-year (Time 1.5, winter) fluency scores.
# Use ONLY in the "mid-year prediction" scenario. For fall prediction they are leakage.
mid = (raw[raw["Time"] == 1.5].set_index("StuID")[["NWFcls", "NWFwrc", "ORFwc"]]
       .add_suffix("_mid"))
df = df.join(mid, on="StuID")

# Spring (Time 2) scores: used ONLY to build targets, never as features
spring = (raw[raw["Time"] == 2.0].set_index("StuID")[["SAtoS", "RMwidRS", "RMwdaRS"]]
          .add_suffix("_spring"))
df = df.join(spring, on="StuID")

# Gains (spring minus fall)
for score in ["SAtoS", "RMwidRS", "RMwdaRS"]:
    df[f"{score}_gain"] = df[f"{score}_spring"] - df[score]

# Stanford growth target: 1 if gain >= median gain, else 0 (missing if no gain)
median_gain = df["SAtoS_gain"].median()
df["SAtoS_target"] = (df["SAtoS_gain"] >= median_gain).astype("float")
df.loc[df["SAtoS_gain"].isna(), "SAtoS_target"] = float("nan")

# Keep students with at least one target
targets = ["RMwidRS_target", "RMwdaRS_target", "SAtoS_target"]
df = df[df[targets].notna().any(axis=1)].reset_index(drop=True)

# Fixed 70 / 15 / 15 split by SCHOOL: every school goes entirely to one set,
# so validation and test scores show how models do on schools they have never seen.
# Schools are shuffled (seed 42), then added to test and validation until each
# reaches 15% of students without going over; the remaining schools are train.
school_sizes = df.groupby("SchlID").size()
order = np.random.default_rng(42).permutation(school_sizes.index.values)
cap = 0.15 * len(df)
school_split, filled = {}, {"test": 0, "validation": 0}
for school in order:
    n = school_sizes[school]
    if filled["test"] + n <= cap:
        school_split[school] = "test"; filled["test"] += n
    elif filled["validation"] + n <= cap:
        school_split[school] = "validation"; filled["validation"] += n
    else:
        school_split[school] = "train"
df["split"] = df["SchlID"].map(school_split)

df.to_csv(OUT, index=False)
for name in ["train", "validation", "test"]:
    df[df["split"] == name].to_csv(f"data/processed/{name}.csv", index=False)
print(f"Wrote {OUT}: {len(df)} students, {df.shape[1]} columns")
print(f"Stanford median gain: {median_gain}")
print(df[targets].notna().sum().to_string())
print()
for name in ["train", "validation", "test"]:
    part = df[df["split"] == name]
    print(f"{name}: {len(part)} students ({len(part) / len(df):.1%}), {part['SchlID'].nunique()} schools, "
          f"Stanford {part['SAtoS_target'].notna().sum()}, Woodcock {part['RMwidRS_target'].notna().sum()}")
assert df.groupby("SchlID")["split"].nunique().max() == 1, "a school is in more than one set"
