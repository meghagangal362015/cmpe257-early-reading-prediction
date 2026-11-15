# ECRI Combined Dataset

`ecri_combined.csv`: **5,679 students, 46 columns**, one row per student. The `split` column marks each student as `train`, `validation` or `test`.

`train.csv`, `validation.csv` and `test.csv` contain the same rows, already separated.

Built by `src/build_dataset.py` from the original ECRI files. To rebuild it, put the 3 original CSVs in `data/` and run `python src/build_dataset.py` from the repo root.

## Targets

| Column | Meaning | Students with a value |
|---|---|---|
| `RMwidRS_target` | Woodcock Word Identification: 1 if fall-to-spring gain is above the median (from the paper) | 946 |
| `RMwdaRS_target` | Woodcock Word Attack: 1 if gain is above the median (from the paper) | 946 |
| `SAtoS_target` | Stanford total score: 1 if gain >= 60 points (median), else 0 | 5,673 |

A blank target means that student did not take that test. Train each model only on the rows where its target has a value. The 946 Woodcock students are almost all Tier 2 (at-risk); the Stanford students cover all tiers.

## Train / validation / test split (70 / 15 / 15)

| Set | Students | Share | Schools | Stanford target | Woodcock targets |
|---|---|---|---|---|---|
| train | 3,993 | 70.3% | 30 | 3,989 | 700 |
| validation | 843 | 14.8% | 8 | 841 | 141 |
| test | 843 | 14.8% | 6 | 843 | 105 |

- Split by **school**: every school is entirely in one set, so validation and test scores show how a model does on schools it has never seen.
- Fixed with seed 42, so everyone gets the same split. Rebuild with `python src/build_dataset.py`.
- Schools differ in size, so the shares are close to 70/15/15 rather than exact.
- The Woodcock test set is small (105 students), so expect noisy Woodcock test scores.

**Rules for the team:**
1. **train:** fit models.
2. **validation:** compare models and tune settings (depth, C, learning rate). Use it as often as you need.
3. **test:** use **once**, at the end, to report final scores for every model. Never change a model after seeing its test score.

## Columns

**IDs (do not use as features):** `StuID`, `SchlID`, `TeachID`

**Fall features (safe to use):**
- Student: `Gender`, `Tier`, `LEP` (English learner), `SpEd` (special education), `Tx` (intervention group), `Cohort`, `Age1b` (age in months; mostly missing outside the Woodcock group)
- Fall test scores: `NWFcls`, `NWFwrc` (Nonsense Word Fluency), `ORFwc` (Oral Reading Fluency), `SAwrS`, `SAsrS`, `SAtoS` (Stanford), `RMwidRS`, `RMwdaRS` (Woodcock; mostly missing outside the Woodcock group)
- Teacher and class: `Tier2_N`, `TKPctCrct`, `TYearsTeaching`
- School: `Schl_FreeLunch` and `Schl_*` demographic shares

**Mid-year (winter) scores, for the mid-year scenario ONLY:** `NWFcls_mid`, `NWFwrc_mid`, `ORFwc_mid`. Available for 2,848 students (94% of Tier 2, 50% of Tier 1, almost no Tier 3; 878 of the 946 Woodcock students). For fall prediction these are leakage.

**Measured during the school year (leakage risk; use only for comparison):** `grp_rate`, `rcmistot`, `gnrl_fid`, `Tier2MinRd`

**Spring scores and gains (never use as features; they define the targets):** `*_spring`, `*_gain`

## Two prediction scenarios

| Scenario | Features | Question it answers |
|---|---|---|
| **Fall prediction** (main) | Fall features only | Can we flag students at the start of the year? |
| **Mid-year prediction** | Fall features + `*_mid` columns | How much better can we predict by winter? |

Quick check (Logistic Regression, trained on train, scored on validation, AUC):

| Target | Fall only | Fall + mid-year |
|---|---|---|
| Stanford total | 0.779 | 0.782 |
| Word Identification | 0.685 | 0.793 |
| Word Attack | 0.617 | 0.747 |

Mid-year scores help the Woodcock targets a lot and barely change the Stanford target. Keep the two scenarios in separate result tables; never mix them.

## Preprocessing status

Done here: one row per student, constant columns removed, school info joined, targets built.

**Not done here, on purpose:** filling missing values, scaling, encoding, and feature selection. Fit these inside cross-validation (for example in an sklearn `Pipeline`), so information from the test fold does not leak into training.

## Source and license

Derived from the Enhanced Core Reading Instruction (ECRI) dataset, https://github.com/ecri-data/ECRI-Dataset, licensed CC BY-NC-SA 4.0. This derived file is shared under the same license.

Shangguan, Z., Huang, Z., Ohn-Bar, E., Ozernov-Palchik, O., Kosty, D., Stoolmiller, M., & Fien, H. (2024). *Scalable Early Childhood Reading Performance Prediction.* NeurIPS Datasets and Benchmarks Track.
