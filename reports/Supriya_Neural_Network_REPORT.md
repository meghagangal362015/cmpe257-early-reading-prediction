# MaskMLP reimplementation and comparison with traditional ML on ECRI

Reimplementation of the neural-network part of Shangguan et al., *Scalable Early Childhood Reading Performance Prediction* (NeurIPS 2024 Datasets & Benchmarks), and comparison with traditional ML on the released ECRI dataset.

**Scope (member 3):** implement the MLP / MaskMLP model, visualise model performance, and compare traditional ML with neural-network approaches.

**Code:** `notebooks/Supriya_Neural_network_implementation.py` (experiments) and `notebooks/Supriya_Neural_network_report.py` (paper comparison and validation tables). The neural-network sections of the repository's `README.md` cover setup and design decisions.

**Results:** `results/neural_network/`. Short paths in this report such as `main/` and `report/` are relative to that folder.

**Traditional models:** logistic regression and XGBoost are owned by members 1 and 2. They are included here only as a controlled reference, trained on exactly the same folds, rows and evaluation code as the neural models. They are not presented as member 3's implementation, and `--models neural` leaves them out. They handle missing values in the usual way for each model rather than with the −1 marker: logistic regression fills them with the training-fold mean and adds one missing-value flag per feature, and XGBoost uses its built-in handling of missing values. The paper assigns −1 to missing variables during training, possibly for every model; since our traditional results are within about 0.02 of Table 1, this has little effect.

**Data:** `ECRIDataSet.csv` (labelled records) and `ECRIRawData.csv` (unfiltered records, used for pretraining) from github.com/ecri-data/ECRI-Dataset. That repository contains the data and figures only, so the authors' implementation was not available to check against.

**Primary reproduction:** the paper's protocol as written:
- the cosine pretraining loss (Eq. 1), with exactly 25 % of each student's observed features masked;
- pretraining on all 6,916 unfiltered students;
- fine-tuning and evaluation on the 936 students per target that remain after the paper's exclusion of students with missing reading assessments.

It is run with 5 seeds (42–46). Every other setting is a separate sensitivity or investigation run and is labelled as such.

**Where the numbers come from:** every number in this report is computed from the 28 runs in `results/neural_network/`, the kNN run in `results/knn_extension/` and the robustness run in `results/robustness_extension/`, all produced on Google Colab's CPU runtime with `notebooks/Supriya_Neural_Network.ipynb` (setup in `README.md`).

---

## 1. Summary

- **Every component in scope is implemented and runs end to end.** That covers the Table 1 models and evaluation, the Fig. 2–4 analyses, the Table 4 loss ablation and the Table 5 feature importance.
- **MaskMLP's reported advantage was not reproduced under our documented training and evaluation choices.** In our runs, MaskMLP does not outperform the plain MLPs, and logistic regression and XGBoost score highest.
  - Across the 10 single-seed runs of the two pretraining scopes (seeds 42–46 each), MaskMLP is below the best plain MLP in 7 (the other three are +0.001 to +0.003) and below the best traditional model in all 10.
  - The same holds in the intervention group: logistic regression (0.630) and XGBoost (0.628) lead, the plain MLPs follow (0.620–0.622) and MaskMLP is lowest (0.613).
  - The authors' training code is not available, so these choices are our own (Section 7).
  - This result does not show that the published finding is invalid. It shows that the finding did not appear under our implementation of the described method.
- **Our traditional models match the paper closely; our neural models do not.** Mean accuracy gap to Table 1, all students:

  | Model family | Mean gap | Range |
  |---|---|---|
  | Traditional | −0.018 | −0.051 to +0.020 |
  | Neural | −0.080 | −0.108 to −0.049 |

- **Checkpoint selection explains most of the neural gap.** In a diagnostic run that chooses each neural model's epoch on the test fold (not a valid result, Section 3.5), the neural models rise from 0.579–0.596 to 0.640–0.648 accuracy and the mean neural gap to the paper shrinks from −0.080 to −0.025. The traditional models are unchanged. Even then, MaskMLP is not ahead of the plain MLPs.
- **Cosine pretraining caused embedding collapse in our implementation.**
  - This happens in every cosine run (22 of 22).
  - Embedding spread drops about 94 % after the first epoch, from about 0.036 to about 0.0023.
  - By epoch 100 it is near 0.00001.
  - Two collapse-resistant losses (the paper's contrastive loss and a SimSiam-style extension) keep the embeddings spread out, but neither improves MaskMLP's accuracy. Collapse therefore does not explain MaskMLP's shortfall in our runs.
  - Shorter pretraining (5 epochs, single seed) gave MaskMLP 0.593 instead of 0.579, still below the best plain MLP (Section 3.6).
- **Pretraining on training groups only gives about the same result as pretraining on all students.** MaskMLP accuracy was 0.583 with all students and 0.585 with training groups only, over 5 seeds each; the difference is smaller than the seed-to-seed spread.
  - The default follows the paper and is transductive: unlabelled test-student inputs are seen during pretraining.
  - The training-only runs are therefore the stronger evidence about generalisation to unseen students.
- **MaskMLP does not help where it was designed to help either (Section 10).** With test inputs or whole tests missing, or with only 10–25 % of the labels, no MaskMLP version is the best model, and fine-tuning fixes (masked fine-tuning, a frozen encoder, a lower encoder learning rate) leave it near chance. The plain MLP with mean filling copes best with missing values, and logistic regression with missing tests and scarce labels.
- **Several secondary findings do reproduce:**
  - The start-of-year score on the same skill is the most important input for XGBoost on both tasks, and for MaskMLP on word identification.
  - Accuracy is higher for students who improved than for those who regressed.
  - The paper's AUC values are reproduced exactly as balanced accuracy.
  - The paper's exclusion rule gives 936 students per target, and the classroom features still missing among them explain why its three missing-value MLPs differ.

## 2. Scope implemented

Every component in the "What Is Implemented" table of `README.md` is complete and runs end to end. The performance figures (comparison, PR curves, pretraining loss and spread, t-SNE, progress quintiles, feature importance, loss ablation, paper vs ours) are in `main/figures/` and `report/`. The kNN extension is in Section 9.

**How the missing-value baselines are read.** The paper names the three plain MLPs (Zeros, Mean, Indicator) but does not define them further. We use:
- **Zeros:** missing values set to 0 after min-max scaling, which equals each feature's training minimum.
- **Mean:** missing values set to the training-fold mean of the observed values.
- **Indicator:** mean filling plus one binary missing-value column per feature (32 inputs), the standard missing-indicator method. The paper's wording could also mean the −1 marker itself, which is the input MaskMLP receives.

## 3. Methodological checks

### 3.1 The paper's AUC definition (verified)

The paper's text describes its AUC as the area under the precision–recall curve. Its numbers say otherwise. In all 10 Table 5 rows checked, the reported AUC equals (specificity + sensitivity) / 2 to four decimals; the largest difference is 0.00005, which is rounding (`report/auc_interpretation.csv`).

That quantity is balanced accuracy. It equals ROC-AUC computed from 0/1 predictions. Table 1 is consistent with this too: in the word attack task, where classes are imbalanced, the paper's AUC is often well below its accuracy.

**Handling in the code:** no metric is labelled "the paper's AUC". The script reports:
- `balanced_accuracy`: the reading that matches the paper's tables. It is used when comparing against Table 1 AUC values.
- `pr_auc`: area under the PR curve from probabilities, as the paper's text describes.
- `roc_auc`: ROC-AUC from probabilities.

### 3.2 Pretraining scope: all records or training groups only (verified)

**What the paper says.** The paper states that it employs "all data samples during pre-training", and that no labels are used. Our default (`--pretrain-scope all`) follows that. This setting is transductive: MaskMLP sees the unlabelled inputs of validation and test students during pretraining. `--pretrain-scope train` removes every validation and test student of the fold from the pretraining pool.

**What we ran.** Both settings were run with 5 seeds on the unfiltered pool. Values are means over splits, targets and folds, all students:

| Pretraining scope | MaskMLP accuracy | Balanced accuracy | PR-AUC | MaskMLP − best plain MLP (accuracy) | MaskMLP − best traditional (accuracy) |
|---|---|---|---|---|---|
| all students (paper, transductive) | 0.583 ± 0.010 | 0.552 | 0.635 | −0.009 | −0.026 |
| training groups only (strict) | 0.585 ± 0.012 | 0.553 | 0.634 | −0.007 | −0.024 |

**Conclusion.** Including test students in pretraining gives MaskMLP no measurable advantage: the 0.002 difference is smaller than the seed-to-seed standard deviation and goes the other way. In our implementation the cosine encoder collapsed either way (Section 3.4), so it carries almost no information about any student, test students included. The paper's protocol is therefore kept as the primary setting, and the strict setting confirms that it does not inflate results here.

### 3.3 Unfiltered data and evaluation rows

**Unfiltered pretraining pool.**
- `ECRIDataSet.csv` contains 6,803 students. `ECRIRawData.csv` contains 6,916, the number the paper reports.
- We rebuilt the 16 input features from the raw file's start-of-year rows (`load_unfiltered()`).
- **Feature verification (`verify_unfiltered()`).** This check compares inputs only.
  - The raw file has exactly one start-of-year row for each of its 6,916 students, so no rows are merged or guessed.
  - All 6,803 students in `ECRIDataSet.csv` are present.
  - All 115,651 compared cells (16 features plus school ID for 6,803 students) are identical.
  - If it fails, the script pretrains on `ECRIDataSet.csv` instead, or stops if `--pretrain-source raw` was given.
- **Label verification (`verify_labels()`).** This is a separate check.
  - Each outcome label is rebuilt from the raw scores: label = 1 when the Woodcock gain from t = 0 to t = 1 exceeds the mean gain of control-group (Tx = 0) students.
  - The rebuilt labels are compared with the released labels. All 946 labels match for word identification, and all 946 match for word attack.
  - This check is only reported. The models always train on the released labels, and the check does not affect pretraining.
- **Where the results are kept.** Both results are saved in `results/neural_network/main/raw_verification.json`, and every run's `config.json` records them too (all 28 runs used `--raw-dataset`). Both checks passed in every run.

**Evaluation rows.** The paper (Sec. 3) excludes students with missing data from fine-tuning and evaluation, and it measures missingness over the eight reading assessments. Applying that rule (`--eval-rows assessments`, the default) leaves 936 of the 946 labelled students per target (426 in the intervention group). 230 of them still miss at least one classroom-level feature (mostly the teacher knowledge score, `TKPctCrct`), which is why the paper's Zeros, Mean and Indicator MLPs differ in Table 1. Two other row settings were run as sensitivity checks with 3 seeds each:
- all 946 labelled students (`--eval-rows all`);
- only students with all 16 features observed (`--eval-rows complete`, 706 students). This is stricter than the paper and matches the 706 students used in the team's Random Forest, XGBoost and Naive Bayes experiments.

| Evaluation rows | Students per target | Seeds | MaskMLP (scope all) | MaskMLP (scope train) | Best plain MLP | Logistic regression | XGBoost |
|---|---|---|---|---|---|---|---|
| paper's rule (default) | 936 | 5 | 0.583 | 0.585 | 0.592 | 0.609 | 0.600 |
| all labelled | 946 | 3 | 0.583 | — | 0.589 | 0.616 | 0.602 |
| all 16 features complete | 706 | 3 | 0.588 | 0.587 | 0.600 | 0.623 | 0.630 |

In every setting the traditional models are best and MaskMLP is below the best plain MLP. With complete rows only, the Zeros and Mean MLPs become identical, as expected, because nothing is left to fill. The choice of rows changes MaskMLP's accuracy by at most 0.005, so it does not explain the gap to the paper's results.

### 3.4 MaskMLP embedding collapse (investigated separately)

**How collapse is measured.** Spread is the mean per-dimension standard deviation of L2-normalised embeddings. It is recorded before pretraining and after every epoch (`pretrain_log.csv`, `report/collapse.csv`, right panel of `main/figures/pretraining_loss.png`).

**What happened with the cosine loss (Eq. 1) in our implementation.** Cosine pretraining caused embedding collapse in all 22 cosine runs: the 19 standard runs across both scopes, all three row settings and all seeds, plus the diagnostic and the two shorter-pretraining runs. This was under our training choices (Adam, learning rate 1e-3, batch size 64, 100 epochs, min-max inputs); the authors' settings may differ.

| | Spread (19 standard cosine runs) |
|---|---|
| Before pretraining | 0.033–0.038 |
| After epoch 1 | 0.0021–0.0026 |
| After epoch 100 | 0.000004–0.000013 |

The encoder maps every student to almost the same direction.

**Likely cause.** Eq. 1 only pulls a student's original and masked embeddings together. Nothing pushes different students apart, so a constant embedding minimises the loss. MSE collapsed in our runs as well, which fits that explanation.

**The paper's cosine loss stays the primary reproduction.**
- The contrastive loss is one of the paper's Table 4 losses.
- The SimSiam-style loss is an **extension that is not in the paper**. It is excluded from the Table 4 ablation (`main/loss_ablation.csv` holds the paper's 11 losses only) and is labelled `loss_origin = extension` in every report table.
- Both are reported here as a separate investigation (`results/neural_network/simsiam`, `results/neural_network/contrastive`), not as a replacement:

| Pretraining loss | Spread after 100 epochs | Collapsed | MaskMLP accuracy (seeds 42–44) | MaskMLP − MLP Zeros |
|---|---|---|---|---|
| cosine (paper Eq. 1) | 0.000004–0.000005 | yes | 0.587 | −0.004 |
| contrastive (Eq. 1 + push away a random other student) | 0.026–0.029 | no | 0.588 | −0.004 |
| SimSiam-style, extension (Eq. 1 + predictor + stop-gradient) | 0.071 | no | 0.583 | −0.008 |

**Findings from the investigation:**
- **Avoiding collapse does not improve MaskMLP in our runs.** Both alternative losses keep the embeddings spread out, but neither raises fine-tuned accuracy above the collapsed cosine encoder on the same seeds.
  - Contrastive: 0.588, level with cosine (0.587).
  - SimSiam-style: 0.583, slightly below cosine. It is 0.010 below the best plain MLP and beats MLP Zeros in 26 of 60 paired units (8 ties, 26 losses).
  - Both trail the traditional models by 0.022–0.026 accuracy.
- **Loss ablation (Table 4).** On word identification (seed 42, both splits), the 11 paper losses range from 0.564 to 0.615 accuracy. Cosine ranks 8th (0.593) and triplet is highest (0.615). These are single-seed results, so differences of this size are within noise (`main/loss_ablation.csv`).
- **Conclusion.** In our runs, the collapse does not explain MaskMLP's shortfall against the plain MLPs, the traditional models or the paper's numbers. One likely reading is that supervised fine-tuning largely overrides the pretrained encoder, so MaskMLP ends up close to a plain MLP whichever pretraining loss is used.

**Diagnostics in the code.** The script warns whenever spread falls below 10 % of its starting value. `--pretrain-loss` accepts any loss or combination for further study.

### 3.5 Checkpoint selection (diagnostic only)

The paper does not say how the fine-tuning epoch is chosen. Our models stop early on a separate grouped validation split, so test labels are never used. To measure how much this choice matters, one run (`diagnostic_test_selection`, seed 42, `--select-on-test`) picks each neural model's epoch by its accuracy on the test fold. That uses test labels, so it is **not a valid result** and is never pooled with the other runs. It only shows how far optimistic checkpoint selection can move the numbers.

Accuracy, all students, seed 42, mean over splits, targets and folds:

| Model | Validation-based selection (`main`) | Test-fold selection (diagnostic) |
|---|---|---|
| Logistic regression | 0.609 | 0.609 |
| XGBoost | 0.604 | 0.604 |
| MLP (Mean) | 0.596 | 0.648 |
| MLP (Indicator) | 0.586 | 0.646 |
| MLP (Zeros) | 0.591 | 0.641 |
| MaskMLP | 0.579 | 0.640 |

- Test-fold selection lifts every neural model by 0.050–0.061 and leaves the traditional models unchanged, because they have no epochs to choose.
- The mean neural gap to Table 1 on all students shrinks from −0.080 to −0.025 (range −0.058 to +0.001). In the intervention group it shrinks from −0.132 to −0.086.
- The neural models then rank above the traditional ones, as in the paper, but MaskMLP is still 0.007 below the best plain MLP. It ranks 1st in only 1 of the 8 Table 1 settings, against 6 in the paper.

This is consistent with checkpoint selection being the main difference between the paper's neural results and ours. It does not confirm what the authors did.

### 3.6 Pretraining length

The paper does not give the number of pretraining epochs. Because the cosine loss collapses within the first epochs, two shorter runs were added (seed 42, `pretrain_epochs/e5`, `e10`):

| Pretraining epochs | Spread at the end | MaskMLP accuracy, all students | Intervention group | MaskMLP − best plain MLP |
|---|---|---|---|---|
| 5 | 0.00068 | 0.593 | 0.633 | −0.003 |
| 10 | 0.00037 | 0.585 | 0.616 | −0.011 |
| 100 (primary) | 0.000004 | 0.579 | 0.617 | −0.017 |

Shorter pretraining keeps a little more spread and gives MaskMLP slightly higher accuracy, but even after 5 epochs the spread has fallen below 10 % of its starting value, and MaskMLP is still below the best plain MLP. The 0.014 difference between 5 and 100 epochs is a single-seed result and about 1.4 seed standard deviations, so it is suggestive only.

## 4. Comparison with the paper's Table 1

Figures: `report/paper_vs_ours_all.png` and `report/paper_vs_ours_intervention.png`. Full table: `report/table1_comparison.csv`. Our values are means over 5 seeds × 5 folds in the primary setting.

**Accuracy gap (ours − paper):**

| Model family | All students | Intervention group |
|---|---|---|
| Traditional ML (logistic regression, XGBoost) | −0.018 (−0.051 to +0.020) | −0.022 (−0.051 to −0.002) |
| Neural (MLPs, MaskMLP) | −0.080 (−0.108 to −0.049) | −0.132 (−0.186 to −0.086) |

**MaskMLP's rank among the 6 models across 8 settings** (2 splits × 2 targets × 2 subsets):

| Source | Ranks |
|---|---|
| Paper | 1st in 6 settings, 2nd in 2 |
| Ours | 5th–6th in all 8 settings |

**The paper's own MaskMLP margin is small.** On all students, the paper reports MaskMLP within ±0.003 accuracy of the best plain MLP. Its advantage appears mainly in the intervention group (+0.016 to +0.035). In our runs MaskMLP is below the best plain MLP in all 8 settings, by 0.004 to 0.026.

**Likely reasons for the neural gap.** None can be confirmed without the authors' code.

1. **Checkpoint selection.** The paper does not describe a validation set. Choosing each neural model's epoch on the test fold closes most of the neural gap on all students (Section 3.5). That would only inflate the neural models, because the traditional models have no epochs to pick. Our early stopping uses a separate grouped validation split, so no test labels are used for model selection.
2. **Less labelled data for the neural models.** Our neural models hold back a grouped 20 % of each training fold for early stopping, so they train on about 600 students per fold, while logistic regression and XGBoost, which need no early stopping, train on the whole fold (about 750; the per-fold count is `train_n` in `results.csv`). If the paper trained its MLPs on the whole fold, they had about 25 % more labelled data than ours.
3. **Small labelled set.** Only 936 students per target are used, and a plain MLP underfits at this size.
4. **Pretraining gives no benefit in our setup.** Cosine pretraining collapsed (Section 3.4), and neither the collapse-resistant losses nor shorter pretraining (Section 3.6) made MaskMLP better than a plain MLP.
5. **Unreported training details.** Input scaling, learning rate, batch size and number of epochs are not given in the paper.

## 5. Robustness

**Seeds.** The primary and strict-scope settings use 5 seeds (42–46). The 946-row, complete-rows, SimSiam and contrastive settings use 3 seeds (`report/robustness.csv`). The seeds change the weight initialisation, the grouped validation split, the masking and XGBoost's subsampling, but not the test folds, because `GroupKFold` assigns groups deterministically. The seed-to-seed spread therefore measures training randomness only, which is why logistic regression, with no random component, shows ± 0.000.

| Model | Accuracy, mean ± std over 5 seeds | Balanced accuracy | PR-AUC |
|---|---|---|---|
| Logistic regression | 0.609 ± 0.000 | 0.593 | 0.663 |
| XGBoost | 0.600 ± 0.002 | 0.588 | 0.671 |
| MLP (Mean) | 0.592 ± 0.003 | 0.570 | 0.651 |
| MLP (Indicator) | 0.590 ± 0.007 | 0.561 | 0.643 |
| MLP (Zeros) | 0.590 ± 0.003 | 0.564 | 0.646 |
| MaskMLP (cosine) | 0.583 ± 0.010 | 0.552 | 0.635 |

MaskMLP has the largest seed-to-seed spread of any model.

**Paired outcomes.** Each of the 100 units is one seed × fold × split × target combination, and compares MaskMLP's accuracy with another model's (`report/maskmlp_wins.csv`). Primary setting:

| MaskMLP vs | Wins / ties / losses | Mean accuracy difference |
|---|---|---|
| MLP (Zeros) | 39 / 19 / 42 | −0.007 |
| MLP (Mean) | 39 / 12 / 49 | −0.009 |
| MLP (Indicator) | 36 / 21 / 43 | −0.007 |
| Logistic regression | 33 / 3 / 64 | −0.026 |
| XGBoost | 39 / 5 / 56 | −0.017 |

MaskMLP is level with or slightly below the plain MLPs. Logistic regression beats it in 64 of 100 units and XGBoost in 56.

**Significance tests.** The paper compares MaskMLP with the MLP using a paired t-test over the five folds and reports t = 3.876, p = 0.009. With five folds (4 degrees of freedom), t = 3.876 gives p = 0.009 one-sided or 0.018 two-sided, so the paper's value is the one-sided p. We ran the same paired test for every split, target and subset against each plain MLP (seed 42, `main/significance.csv`). None of the 24 comparisons is significant (smallest two-sided p = 0.060), and MaskMLP is ahead in 9, level in 1 and behind in 14.

**Sensitivity.** Pretraining scope × evaluation rows × pretraining loss × pretraining length is covered in Sections 3.2–3.6 (`report/sensitivity.csv`). In every valid combination, the traditional models remain best.

## 6. Conclusions

**Agrees with the paper:**
- The MLP depth and width, the masking scheme and the pipeline can be implemented as described, and all pieces run end to end.
- The start-of-year score on the same skill is the most important input (feature importance, Fig. 4, school split) for XGBoost on both tasks and for MaskMLP on word identification. Dropping it costs XGBoost 0.053 accuracy for word identification and 0.028 for word attack, and MaskMLP 0.048 for word identification. For MaskMLP on word attack, the nonsense-word reading score (`NWFwrc`, 0.023) matters most.
- Models are more accurate for students who made large gains (0.73–0.81) than for those who regressed strongly (0.35–0.55) (Fig. 2 pattern).
- Our logistic regression and XGBoost results are close to Table 1.
- The paper's reported AUC values are reproduced exactly as balanced accuracy, although the paper's text calls the metric PR-AUC.
- The paper's description of its evaluation rows checks out against the data: excluding students with missing reading assessments leaves 936 per target, and the remaining missing classroom features explain why its three missing-value MLPs differ.

**Differs from the paper (under our implementation):**
- MaskMLP does not outperform the plain MLPs. With the paper's loss it is slightly below them.
- Neural models do not outperform traditional ML. On all students, logistic regression is best: 0.017 accuracy above the best neural model (MLP Mean) and 0.026 above MaskMLP. XGBoost is second. In the intervention group, both traditional models are also ahead of every neural model.
- MaskMLP does not show an advantage with missing inputs, missing tests or scarce labels, the situations it was designed for (Section 10).
- Cosine pretraining caused embedding collapse in our implementation, and avoiding it did not improve MaskMLP. Our results are therefore not consistent with the paper's t-SNE reading of "well-defined clusters", although the paper's own training setup may have avoided the collapse.

**Overall:** MaskMLP's reported advantage was not reproduced under our documented training and evaluation choices on the released data. Under those choices, neither MaskMLP nor the plain MLPs outperformed traditional ML, and logistic regression is the most accurate model. The paper's neural results are about 0.08 accuracy higher than ours, while its traditional baselines reproduce within about 0.02. That points to a difference in the neural training or evaluation procedure. The checkpoint-selection diagnostic closes most of that gap, which makes it the most likely explanation, but we cannot confirm what the authors did. Testing MaskMLP in the situations it was designed for (Section 10) did not reveal an advantage either.

## 7. Limitations

- The authors' code is unavailable, so unreported choices (scaling, optimiser settings, epochs, checkpoint selection) are our own and documented in the neural-network sections of `README.md`. The conclusions apply to these choices, and a different implementation of the same method could behave differently.
- The default pretraining is transductive (unlabelled test inputs are used), as in the paper. Fully held-out evidence comes from the training-only runs (`results/neural_network/train`).
- The loss ablation, feature importance, checkpoint-selection diagnostic and pretraining-length runs are single-seed runs (seed 42), so differences of a few points there are within noise.
- The 946-row, complete-rows, SimSiam and contrastive settings use 3 seeds rather than 5.
- The neural models train on 80 % of each training fold (the rest is used for early stopping), while the traditional models use the whole fold (Section 4).
- Results depend slightly on the software environment and settings. An earlier local run of an earlier version of the code (all 946 rows, per-feature random masking) with older package versions (Python 3.9.12, scikit-learn 1.0.2, torch 1.13.0, xgboost 1.7.1) gave accuracies within about 0.015 of those reported here and the same main conclusions.
- The kNN extension (Section 9) is exploratory and single-seed (seed 42), so it is compared with the seed-42 main run only.
- The robustness extension (Section 10) uses 3 seeds rather than 5. Its hidden values are chosen at random, and real missingness may follow other patterns.
- TabNet, GRAPE, VIME, SCARF and LightGBM are out of scope for this member.
- Subgroup bias analysis (Table 2) is out of scope for this member.

## 8. Output layout and reproducing

| Folder | Setting | Seeds |
|---|---|---|
| `results/neural_network/main` | Primary run (paper protocol) + feature importance + loss ablation + all figures | 42 |
| `results/neural_network/seeds/s43…s46` | Primary run, extra seeds | 43–46 |
| `results/neural_network/train/s42…s46` | Pretraining on training groups only | 42–46 |
| `results/neural_network/all946/s4x` | All 946 labelled students | 42–44 |
| `results/neural_network/complete/all-s4x`, `train-s4x` | Complete-feature rows only (706), scope all / train | 42–44 |
| `results/neural_network/simsiam/s4x` | Collapse investigation: SimSiam-style loss | 42–44 |
| `results/neural_network/contrastive/s4x` | Collapse investigation: contrastive loss | 42–44 |
| `results/neural_network/pretrain_epochs/e5`, `e10` | 5 and 10 pretraining epochs | 42 |
| `results/neural_network/diagnostic_test_selection` | Diagnostic only: epochs chosen on the test fold (Section 3.5) | 42 |
| `results/neural_network/report` | Paper comparison, sensitivity, robustness and collapse tables | — |
| `results/knn_extension` | Extension, not in the paper: kNN on the seed-42 folds, compared with `results/neural_network/main` (Section 9) | 42 |
| `results/robustness_extension` | Extension, not in the paper: missing values, missing tests and scarce labels (Section 10) | 42–44 |

Section 5 of `notebooks/Supriya_Neural_Network.ipynb` runs all 28 runs. The files kept in each folder, the data, the environment and the commands for the main run, the report tables and kNN are in the "Neural Network Files and Outputs" and "Running the Neural Network Experiments" sections of `README.md`. The commands below reproduce the additional runs:

```bash
# Extra seeds (repeat for 44, 45, 46)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/seeds/s43 --seed 43 --no-tsne

# Training-only pretraining (repeat for seeds 42-46)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/train/s42 --pretrain-scope train --seed 42 --no-tsne

# All 946 labelled students (repeat for seeds 42-44)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/all946/s42 --eval-rows all --seed 42 --no-tsne

# Complete-feature rows (add --pretrain-scope train for the train-s4x folders)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/complete/all-s42 --eval-rows complete --seed 42 --no-tsne

# Collapse investigation (repeat for seeds 42-44)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/simsiam/s42 --pretrain-loss simsiam --seed 42 --no-tsne
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/contrastive/s42 --pretrain-loss contrastive --seed 42 --no-tsne

# Pretraining length (repeat with 10)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/pretrain_epochs/e5 --pretrain-epochs 5 --seed 42 --no-tsne

# Checkpoint-selection diagnostic (uses test labels; not a valid result)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/diagnostic_test_selection --select-on-test --seed 42 --no-tsne
```

## 9. Extension: k-nearest neighbours (exploratory, not in the paper)

**Why.** kNN predicts from the most similar students, with no training beyond storing the data. It is a simple, very different model to compare against, and applied to MaskMLP's pretrained embeddings it tests directly how much information about similar students the self-supervised pretraining keeps.

**Setup** (`notebooks/Supriya_kNN_extension.py`, run in section 6 of `notebooks/Supriya_Neural_Network.ipynb`):
- The same folds and test students as the seed-42 main run (936 students per target), and the same evaluation code.
- `kNN_raw`: the 16 features, min-max scaled on the training fold, missing values filled with the training-fold mean plus one missing-value indicator per feature; Euclidean distance.
- `kNN_MaskMLP_embedding`: the pretrained MaskMLP encoder (cosine loss, all 6,916 unfiltered students, same settings as the main run), before fine-tuning; cosine distance.
- k is chosen per fold from 3 to 101 by accuracy on the grouped validation split, then the model is refit on training + validation students. Test labels are never used to choose k. The chosen k ranged from 3 to 101 for both variants (median 21 for `kNN_raw`, 26 for the embedding variant).

**Results** (`results/knn_extension/comparison.csv`, `knn_comparison.png`). Means over both splits and both targets, 5 folds each; the six models are from the seed-42 main run on the same folds:

| Model | Accuracy, all students | Balanced accuracy | PR-AUC | Accuracy, intervention group |
|---|---|---|---|---|
| Logistic regression | 0.609 | 0.593 | 0.663 | 0.630 |
| XGBoost | 0.604 | 0.593 | 0.674 | 0.635 |
| MLP (Mean) | 0.596 | 0.576 | 0.655 | 0.627 |
| MLP (Zeros) | 0.591 | 0.568 | 0.647 | 0.625 |
| MLP (Indicator) | 0.586 | 0.562 | 0.647 | 0.623 |
| MaskMLP | 0.579 | 0.547 | 0.638 | 0.617 |
| **kNN_raw** | **0.572** | 0.546 | 0.616 | 0.624 |
| **kNN_MaskMLP_embedding** | **0.515** | **0.504** | 0.575 | 0.523 |

**Findings:**
- **kNN on the raw features is the weakest of the models trained on those features on all students.** It ranks 7th of 8, 0.037 below logistic regression and 0.007 below MaskMLP. In the intervention group it does slightly better (0.624), above MaskMLP (0.617) and level with MLP (Indicator) (0.623). With 16 features and only about 750 training students per fold, a student's nearest neighbours are not much more similar to them than students further away. kNN also leans towards predicting improvement (sensitivity 0.75, specificity 0.34).
- **kNN on MaskMLP's pretrained embeddings is at chance** (balanced accuracy 0.504). This is independent evidence for the collapse in Section 3.4: in this run the encoder's embedding spread fell from about 0.038 to 0.000003–0.00002, so the embeddings no longer separate students and their neighbours carry no information about the outcome. MaskMLP still reaches 0.579 after fine-tuning, which supports the reading that fine-tuning, not pretraining, does the work.

## 10. Extension: robustness to missing data and scarce labels (not in the paper)

**Why.** MaskMLP is motivated by incomplete records and unlabelled students, but the paper's benchmark, and our reproduction of it, evaluates students who have all eight reading assessments, with about 600–750 labelled students per training fold. This extension tests MaskMLP in the situations it was designed for, to see whether its advantage appears there.

**Setup** (`notebooks/Supriya_robustness_extension.py`, section 7 of `notebooks/Supriya_Neural_Network.ipynb`; results in `results/robustness_extension/`):
- **Experiment A, values missing at test time.** Models are trained as usual; then 10, 25, 40 or 50 % of each test student's known inputs are hidden at random before prediction.
- **Experiment B, scarce labels.** Pretraining still uses all 6,916 students without labels, but fine-tuning uses only 10, 25, 50 or 100 % of each fold's labelled training and validation students.
- **Experiment C, whole tests missing at test time.** As when a child is absent on a test day, every score from 1 or 2 of the four start-of-year tests (WRMT, DIBELS NWF, ORF, SAT) is hidden for each test student.
- Hidden values, hidden tests and label subsets are drawn once per seed and fold, so every model sees exactly the same data, and they are nested across levels.
- **Models.** The six models of the main comparison; MaskMLP_triplet (pretrained with the triplet loss, one of the paper's Table 4 losses, because cosine collapses); three of our MaskMLP variants that change only fine-tuning: masked fine-tuning (25 % of inputs hidden in every fine-tuning batch), a frozen encoder, and a 10× smaller encoder learning rate; and a control, MLP (Mean) trained with the same hidden-input batches, to separate the effect of pretraining from that of training on incomplete inputs.
- All settings are the main script's defaults, fixed before any test result was seen. 3 seeds (42–44) × 2 splits × 2 targets × 5 folds = 60 units per condition. Run on Google Colab's CPU runtime, like the rest of the report; on complete data the six reference models match Section 5 closely (accuracy 0.587–0.609 here, 0.583–0.609 there).

**The verdict uses balanced accuracy.** About 55 % of students make sufficient progress, so a model that predicts "progress" for everyone keeps about 0.55 plain accuracy however much input is hidden. The frozen, low-learning-rate and masked-fine-tuning variants nearly do that: on complete data they catch 96–98 % of the students who progress but only 2–5 % of those who do not. Their plain accuracy (about 0.55) therefore looks stable, while their balanced accuracy is 0.50–0.51, chance. Plain MaskMLP drifts the same way as inputs disappear (0.529 accuracy but 0.506 balanced accuracy with half the inputs hidden).

**Results** (balanced accuracy, all students; 0.50 = chance):

| Scenario | Log. reg. | XGBoost | MLP (Mean) | MLP (Mean), masked training | MaskMLP | MaskMLP, masked fine-tuning | Frozen encoder | Low encoder LR |
|---|---|---|---|---|---|---|---|---|
| Complete data | **0.593** | 0.589 | 0.574 | 0.564 | 0.556 | 0.506 | 0.500 | 0.505 |
| 25 % of values hidden | 0.549 | 0.547 | **0.551** | 0.549 | 0.511 | 0.508 | 0.499 | 0.503 |
| 50 % of values hidden | 0.524 | 0.526 | **0.538** | 0.536 | 0.506 | 0.513 | 0.499 | 0.502 |
| 1 test missing | **0.578** | 0.558 | 0.562 | 0.551 | 0.510 | 0.506 | 0.500 | 0.501 |
| 2 tests missing | **0.565** | 0.535 | 0.553 | 0.541 | 0.504 | 0.506 | 0.499 | 0.499 |
| 10 % of labels | **0.556** | 0.545 | 0.505 | 0.503 | 0.501 | 0.500 | 0.500 | 0.500 |
| 25 % of labels | **0.573** | 0.557 | 0.518 | 0.519 | 0.505 | 0.499 | 0.501 | 0.501 |

**Findings:**
- **No MaskMLP version is the best model in any scenario.** Across all 22 condition and subset combinations, every MaskMLP version is below the best traditional model (by at least 0.013). Against the best plain MLP it is level at best: the triplet variant is 0.001 ahead with 25 % of the labels in the intervention group, and behind in every other combination. Its best rank among the six reference models is 3rd (`verdict.csv`).
- **Missing values:** the plain MLP with mean filling degrades least. With half of each student's inputs hidden it loses 0.036 balanced accuracy, against 0.063–0.069 for logistic regression and XGBoost, and from 25 % hidden it is the best model on all students; MaskMLP falls to near chance.
- **Missing whole tests:** logistic regression, with mean filling and missing-value flags, stays best on all students (0.578 and 0.565).
- **Scarce labels:** logistic regression degrades least. Every neural model, MaskMLP included, is close to chance with 10 % of the labels, so pretraining on 6,916 unlabelled students does not compensate for the missing labels.
- **Fine-tuning fixes make MaskMLP worse, not better.** The more a variant relies on the pretrained encoder (frozen, low learning rate), the closer it stays to chance. The control MLP (Mean) with masked training is far better than masked-fine-tuned MaskMLP (0.564 vs 0.506 on complete data), so training on incomplete inputs is not the problem; the collapsed encoder is (Section 3.4; kNN on the same embeddings is also at chance, Section 9).
- **Interpretation.** In the situations MaskMLP was designed for, our implementation does not show the advantage either. This is consistent with the collapse: the cosine-pretrained encoder carries almost no information about students, so it cannot help when data or labels are scarce. It does not rule out that the authors' unreported training settings avoid the collapse.
