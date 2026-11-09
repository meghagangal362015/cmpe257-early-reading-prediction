# Combined ECRI Dataset Summary

Nov 3, 2026 · Anupama Singh

## Overview

The `anupama` branch of [cmpe257-early-reading-prediction](https://github.com/meghagangal362015/cmpe257-early-reading-prediction) holds the baseline models (pushed in commit `fd80850`) and, in the following commit, a new combined dataset and this report.

The combined dataset puts all ECRI students with a target into one table of 5,679 students and 46 columns. It adds the Stanford growth target, mid-year fluency scores and a fixed 70/15/15 train/validation/test split by school, so the whole team trains and tests on the same data.

## Files on the branch

13 files were pushed in commit `fd80850`; the dataset files and this report are added in the next commit.

| File | What it is | Status |
| --- | --- | --- |
| `notebooks/01_baselines.ipynb` | Logistic Regression and Decision Tree classifiers (AUC, accuracy) | Pushed |
| `notebooks/02_regression_baselines.ipynb` | Linear Regression and Decision Tree Regressor on score gain (MAE, RMSE) | Pushed |
| `notebooks/03_evaluation.ipynb` | School vs student split, leakage check, two plots | Pushed |
| `src/data.py`, `src/__init__.py` | Shared loader for the 946 Woodcock students and CV splits | Pushed |
| `results/` (5 CSVs, 2 PNGs) | Result tables and plots from notebooks 01-03 | Pushed |
| `.gitignore`, `requirements.txt` | Setup files; original ECRI CSVs are ignored | Pushed |
| `src/build_dataset.py` | Builds the combined dataset and the split from the 3 original ECRI files | Added with this report |
| `data/processed/ecri_combined.csv` | All 5,679 students, 46 columns, with a `split` column | Added with this report |
| `data/processed/train.csv` | 3,993 students | Added with this report |
| `data/processed/validation.csv` | 843 students | Added with this report |
| `data/processed/test.csv` | 843 students | Added with this report |
| `data/processed/README.md` | Column guide, split rules, license credit | Added with this report |
| `reports/dataset_summary.md` | This report | Added with this report |

The repo README does not have the Colab links yet.

## Dataset at a glance

`ecri_combined.csv` has 5,679 students (one row each) from 44 schools and 46 columns. Every student has at least one of three targets.

| Target column | Meaning | Students |
| --- | --- | --- |
| `SAtoS_target` | Stanford total score: 1 if fall-to-spring gain is 60 points or more (the median) | 5,673 |
| `RMwidRS_target` | Woodcock Word Identification: 1 if gain is above the median (from the paper) | 946 |
| `RMwdaRS_target` | Woodcock Word Attack: 1 if gain is above the median (from the paper) | 946 |

All three targets measure growth, not reading level. 940 students have all three. The Woodcock students are almost all Tier 2 (at-risk); the Stanford students cover all three tiers. 1,124 of the original 6,803 students have no target and are left out.

## Columns added and removed

The original `ECRIDataSet.csv` has 30 columns. The combined file removes 3 and adds 19, giving 46.

| Columns | Count | Source | Use as features? |
| --- | --- | --- | --- |
| `Schl_FreeLunch`, `Schl_AmericanIndian`, `Schl_Asian`, `Schl_Hispanic`, `Schl_AfricanAmerican`, `Schl_White`, `Schl_NativeHawaiian`, `Schl_MoreRaces` | 8 added | `MetaData.csv`, joined on school | Yes (school-level) |
| `NWFcls_mid`, `NWFwrc_mid`, `ORFwc_mid` | 3 added | `ECRIRawData.csv`, mid-year (winter) rows | Mid-year scenario only |
| `SAtoS_spring`, `RMwidRS_spring`, `RMwdaRS_spring` | 3 added | `ECRIRawData.csv`, spring rows | Never (they define the targets) |
| `SAtoS_gain`, `RMwidRS_gain`, `RMwdaRS_gain` | 3 added | Spring score minus fall score | Never (regression targets) |
| `SAtoS_target` | 1 added | Stanford gain of 60 or more = 1 | Never (target) |
| `split` | 1 added | `train`, `validation` or `test` | Never |
| `Time`, `t`, `Keep` | 3 removed | Same value in every row | — |

The other 27 original columns are unchanged, including the IDs, the fall features and the two Woodcock targets. Every fall value matches the raw file exactly.

## Train / validation / test split

Students are split 70 / 15 / 15 by school: every school sits entirely in one set, so validation and test scores show how a model does on schools it has never seen.

| Set | Students | Share | Schools | Stanford target | Woodcock targets |
| --- | --- | --- | --- | --- | --- |
| train | 3,993 | 70.3% | 30 | 3,989 | 700 |
| validation | 843 | 14.8% | 8 | 841 | 141 |
| test | 843 | 14.8% | 6 | 843 | 105 |

- Schools were shuffled with seed 42, then added to test and validation until each reached 15% of students without going over. Shares are close to, not exactly, 70/15/15 because schools differ in size.
- The Stanford target is about 50/50 in every set.
- The Woodcock test set is small (105 students, 62% labeled 1 vs 52% in train), so Woodcock test scores will be noisy.

## Mid-year scenario

Adding the winter fluency scores raises Woodcock AUC by about 0.11 to 0.13, but barely changes the Stanford target.

The three `*_mid` columns exist for 2,848 students: 94% of Tier 2, 50% of Tier 1 and almost no Tier 3. They cover 878 of the 946 Woodcock students.

| Target | Fall only | Fall + mid-year | Change |
| --- | --- | --- | --- |
| Stanford total | 0.779 | 0.782 | +0.003 |
| Word Identification | 0.685 | 0.793 | +0.108 |
| Word Attack | 0.617 | 0.747 | +0.130 |

AUC from Logistic Regression trained on train and scored on validation. These columns are measured after the fall prediction point, so they are leakage for the main fall results. Report the two scenarios in separate tables.

## Preprocessing status

The files are cleaned and structured but not model-ready: missing values, scaling and outliers are still in.

| Step | Status |
| --- | --- |
| Text values converted to numbers (Tier, intervention group, cohort) | Done (by the ECRI authors) |
| One row per student, fall features only | Done |
| Constant columns removed | Done |
| School info, spring scores, gains and targets added | Done |
| Train / validation / test split | Done |
| Missing values filled | Not done: fit `SimpleImputer` inside the model pipeline |
| Scaling | Not done: fit `StandardScaler` inside the pipeline |
| One-hot encoding of `Tier` | Not done (optional) |
| Outlier check | Not done: Stanford gain ranges from −92 to +205 points |

Filling and scaling are left out on purpose. Fitting them on all students before splitting would leak test information into training.

## Rules and next steps

**How to use the data:**

1. Fit models on `train.csv`.
2. Compare models and tune settings on `validation.csv` as often as needed.
3. Score `test.csv` once, at the end, for every model. Never change a model after seeing its test score.
4. For the main fall results, never use the `*_mid`, `*_spring`, `*_gain` or `split` columns as features.

**Next steps:**

- [ ] Add the three Colab links to the repo README
- [ ] Decide as a team how to handle negative Stanford gains
- [ ] Rerun notebooks 01-03 on the new train / validation / test split
- [ ] Share the dataset README with the team

## Source

Derived from the [Enhanced Core Reading Instruction (ECRI) dataset](https://github.com/ecri-data/ECRI-Dataset/tree/main/ECRI), licensed CC BY-NC-SA 4.0. Shangguan et al. (2024), [Scalable Early Childhood Reading Performance Prediction](https://papers.neurips.cc/paper_files/paper/2024/hash/862170d83a17b7c2ef3c5a58e7f1992d-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS Datasets and Benchmarks Track.
