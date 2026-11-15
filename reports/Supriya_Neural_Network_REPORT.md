# MaskMLP reimplementation and comparison with traditional ML on ECRI

Reimplementation of the neural-network part of Shangguan et al., *Scalable Early Childhood Reading Performance Prediction* (NeurIPS 2024 Datasets & Benchmarks), and comparison with traditional ML on the released ECRI dataset.

**Scope (member 3):** implement the MLP / MaskMLP model, visualise model performance, and compare traditional ML with neural-network approaches.

**Code:** `notebooks/Supriya_Neural_network_implementation.py` (experiments) and `notebooks/Supriya_Neural_network_report.py` (paper comparison and validation tables). The neural-network sections of the repository's `README.md` cover setup and design decisions.

**Results:** `results/neural_network/`. Short paths in this report such as `main/` and `report/` are relative to that folder.

**Traditional models:** logistic regression and XGBoost are owned by members 1 and 2. They are included here only as a controlled reference, trained on exactly the same folds, rows and evaluation code as the neural models. They are not presented as member 3's implementation, and `--models neural` leaves them out.

**Data:** `ECRIDataSet.csv` (labelled records) and `ECRIRawData.csv` (unfiltered records, used for pretraining) from github.com/ecri-data/ECRI-Dataset. That repository contains the data and figures only, so the authors' implementation was not available to check against.

**Primary reproduction:** the paper's protocol as written. That means the cosine pretraining loss (Eq. 1), pretraining on all 6,916 unfiltered students, and evaluation on all labelled students. It is run with 5 seeds (42–46). Every other setting is a separate sensitivity or investigation run and is labelled as such.

**Where the numbers come from:** every number in this report is computed from the 22 runs in `results/neural_network/`, which were produced on Google Colab's CPU runtime (setup in `README.md`).

---

## 1. Summary

- **Every component in scope is implemented and runs end to end.** That covers the Table 1 models and evaluation, the Fig. 2–4 analyses, the Table 4 loss ablation and the Table 5 feature importance.
- **MaskMLP's reported advantage was not reproduced under our documented training and evaluation choices.** In our runs, MaskMLP does not outperform the plain MLPs, and logistic regression and XGBoost score highest on all students.
  - This holds under both pretraining scopes and both evaluation-row settings. Across the 10 single-seed runs of the two scopes (seeds 42–46 each), MaskMLP is below the best plain MLP in 9 (the tenth is +0.001) and below the best traditional model in all 10.
  - In the intervention group the neural models close the gap: MaskMLP and MLP (Mean) (both 0.623) are level with XGBoost (0.620), but below logistic regression (0.639).
  - The authors' training code is not available, so these choices are our own (Section 7).
  - This result does not show that the published finding is invalid. It shows that the finding did not appear under our implementation of the described method.
- **Our traditional models match the paper closely; our neural models do not.** Mean accuracy gap to Table 1, all students:

  | Model family | Mean gap | Range |
  |---|---|---|
  | Traditional | −0.014 | −0.057 to +0.020 |
  | Neural | −0.083 | −0.106 to −0.047 |

- **Cosine pretraining caused embedding collapse in our implementation.**
  - This happens in every run (16 of 16).
  - Embedding spread drops about 94 % after the first epoch, from about 0.036 to about 0.0023.
  - By epoch 100 it is near 0.00001.
  - The investigation was kept separate from the primary results. Two collapse-resistant losses (the paper's contrastive loss and a SimSiam-style extension) keep the embeddings spread out, but neither improves MaskMLP's accuracy. Collapse therefore does not explain MaskMLP's shortfall in our runs.
- **Pretraining on training groups only gives about the same result as pretraining on all students.** MaskMLP accuracy was 0.580 with all students and 0.577 with training groups only, over 5 seeds each; the difference is smaller than the seed-to-seed spread.
  - The default follows the paper and is transductive: unlabelled test-student inputs are seen during pretraining.
  - The training-only runs are therefore the stronger evidence about generalisation to unseen students.
- **Several secondary findings do reproduce:**
  - The start-of-year score on the same skill is the most important input for XGBoost on both tasks, and for MaskMLP on word identification.
  - Accuracy is higher for students who improved than for those who regressed.
  - The paper's AUC values are reproduced exactly as balanced accuracy.

## 2. Scope implemented

Every component in the "What Is Implemented" table of `README.md` is complete and runs end to end. The performance figures (comparison, PR curves, pretraining loss and spread, t-SNE, progress quintiles, feature importance, loss ablation, paper vs ours) are in `main/figures/` and `report/`. The kNN extension is in Section 9.

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
| all students (paper, transductive) | 0.580 ± 0.009 | 0.546 | 0.624 | −0.011 | −0.036 |
| training groups only (strict) | 0.577 ± 0.008 | 0.544 | 0.624 | −0.014 | −0.039 |

**Conclusion.** Including test students in pretraining gives MaskMLP no measurable advantage: the 0.003 difference is smaller than the seed-to-seed standard deviation. In our implementation the cosine encoder collapsed either way (Section 3.4), so it carries almost no information about any student, test students included. The paper's protocol is therefore kept as the primary setting, and the strict setting confirms that it does not inflate results here.

### 3.3 Unfiltered data and incomplete labelled records

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
- **Where the results are kept.** Both results are saved in `results/neural_network/main/raw_verification.json`, and every run's `config.json` records them too (all 22 runs used `--raw-dataset`). Both checks passed in every run.

**Evaluation rows.** The paper (Sec. 3) excludes students with missing data from fine-tuning and evaluation, and it measures missingness over the eight reading assessments. Applying that rule leaves 936 labelled students per target, and 230 of them still miss at least one classroom-level feature (mostly the teacher knowledge score, `TKPctCrct`). That is why the paper's Zeros, Mean and Indicator MLPs differ in Table 1, so the paper's text and table are consistent. Our default uses all 946 labelled students, only 10 more than the paper's rule. The complete-rows setting (all 16 features observed, 706 students) is stricter than the paper and matches the team's paper-aligned experiments. Both settings were run (3 seeds each for the complete-rows setting):

| Evaluation rows | Students per target | MaskMLP (scope all) | MaskMLP (scope train) | Best plain MLP | Logistic regression | XGBoost |
|---|---|---|---|---|---|---|
| all labelled (default) | 946 | 0.580 | 0.577 | 0.591 | 0.616 | 0.602 |
| all 16 features complete (stricter than the paper) | 706 | 0.582 | 0.584 | 0.600 | 0.623 | 0.630 |

In both settings the traditional models are best and MaskMLP is the lowest-scoring model. With complete rows only, the Zeros and Mean MLPs become identical, as expected, because nothing is left to fill. The 10-student difference between our default and the paper's rule is too small to explain the gap to the paper's results.

### 3.4 MaskMLP embedding collapse (investigated separately)

**How collapse is measured.** Spread is the mean per-dimension standard deviation of L2-normalised embeddings. It is recorded before pretraining and after every epoch (`pretrain_log.csv`, `report/collapse.csv`, right panel of `main/figures/pretraining_loss.png`).

**What happened with the cosine loss (Eq. 1) in our implementation.** Cosine pretraining caused embedding collapse in all 16 cosine runs, across both scopes, both row settings and all seeds. This was under our training choices (Adam, learning rate 1e-3, batch size 64, 100 epochs, min-max inputs); the authors' settings may differ.

| | Spread |
|---|---|
| Before pretraining | 0.033–0.038 |
| After epoch 1 | 0.0020–0.0026 |
| After epoch 100 | 0.000005–0.000011 |

The loss reaches about 0 within 10 epochs. The encoder maps every student to almost the same direction.

**Likely cause.** Eq. 1 only pulls a student's original and masked embeddings together. Nothing pushes different students apart, so a constant embedding minimises the loss. MSE collapsed in our runs as well, which fits that explanation.

**The paper's cosine loss stays the primary reproduction.**
- The contrastive loss is one of the paper's Table 4 losses.
- The SimSiam-style loss is an **extension that is not in the paper**. It is excluded from the Table 4 ablation (`main/loss_ablation.csv` holds the paper's 11 losses only) and is labelled `loss_origin = extension` in every report table.
- Both are reported here as a separate investigation (`results/neural_network/simsiam`, `results/neural_network/contrastive`), not as a replacement:

| Pretraining loss | Spread after 100 epochs | Collapsed | MaskMLP accuracy (seeds 42–44) | MaskMLP − MLP Zeros |
|---|---|---|---|---|
| cosine (paper Eq. 1) | 0.000005–0.000006 | yes | 0.582 | −0.005 |
| contrastive (Eq. 1 + push away a random other student) | 0.023–0.026 | no | 0.571 | −0.017 |
| SimSiam-style, extension (Eq. 1 + predictor + stop-gradient) | 0.070 | no | 0.581 | −0.007 |

**Findings from the investigation:**
- **Avoiding collapse does not improve MaskMLP in our runs.** Both alternative losses keep the embeddings spread out, but neither raises fine-tuned accuracy above the collapsed cosine encoder on the same seeds.
  - Contrastive: 0.571, below cosine (0.582).
  - SimSiam-style: 0.581, level with cosine. It is 0.009 below the best plain MLP and beats MLP Zeros in 23 of 60 paired units (8 ties, 29 losses).
  - Both trail the traditional models by about 0.035–0.045 accuracy.
- **Conclusion.** In our runs, the collapse does not explain MaskMLP's shortfall against the plain MLPs, the traditional models or the paper's numbers. One likely reading is that supervised fine-tuning largely overrides the pretrained encoder, so MaskMLP ends up close to a plain MLP whichever pretraining loss is used.

**Diagnostics in the code.** The script warns whenever spread falls below 10 % of its starting value. `--pretrain-loss` accepts any loss or combination for further study.

## 4. Comparison with the paper's Table 1

Figures: `report/paper_vs_ours_all.png` and `report/paper_vs_ours_intervention.png`. Full table: `report/table1_comparison.csv`. Our values are means over 5 seeds × 5 folds in the primary setting.

**Accuracy gap (ours − paper):**

| Model family | All students | Intervention group |
|---|---|---|
| Traditional ML (logistic regression, XGBoost) | −0.014 (−0.057 to +0.020) | −0.022 (−0.065 to +0.019) |
| Neural (MLPs, MaskMLP) | −0.083 (−0.106 to −0.047) | −0.130 (−0.184 to −0.088) |

**MaskMLP's rank among the 6 models across 8 settings** (2 splits × 2 targets × 2 subsets):

| Source | Ranks |
|---|---|
| Paper | 1st in 6 settings, 2nd in 2 |
| Ours | 5th–6th in all 4 all-students settings; 2nd–5th in the 4 intervention-group settings |

**The paper's own MaskMLP margin is small.** On all students, the paper reports MaskMLP within ±0.003 accuracy of the best plain MLP. Its advantage appears mainly in the intervention group (+0.016 to +0.035). In our runs MaskMLP is never clearly ahead of the best plain MLP: its best cells are word attack in the intervention group, +0.001 (school split) and +0.000 (student split).

**Likely reasons for the neural gap.** None can be confirmed without the authors' code.

1. **Checkpoint selection.** The paper does not describe a validation set. In an earlier diagnostic run (not part of `results/neural_network/`), picking each MLP's best epoch on the test fold lifted our MLP's accuracy from 0.601 to 0.659, close to the paper's values. That would only inflate the neural models, because the traditional models have no epochs to pick. Our early stopping uses a separate grouped validation split, so no test labels are used for model selection.
2. **Small labelled set.** Only 946 students per target have end-of-year scores. That leaves about 450–600 training students per fold, and a plain MLP underfits at this size.
3. **Pretraining gives no benefit in our setup.** Cosine pretraining collapsed (Section 3.4), and the collapse-resistant losses did not help either, so in our implementation pretraining adds nothing over a plain MLP.
4. **Unreported training details.** Input scaling, learning rate, batch size and number of epochs are not given in the paper.

## 5. Robustness

**Seeds.** The primary and strict-scope settings use 5 seeds (42–46). The complete-rows, SimSiam and contrastive settings use 3 seeds (`report/robustness.csv`).

| Model | Accuracy, mean ± std over 5 seeds | Balanced accuracy | PR-AUC |
|---|---|---|---|
| Logistic regression | 0.616 ± 0.000 | 0.600 | 0.670 |
| XGBoost | 0.602 ± 0.002 | 0.591 | 0.668 |
| MLP (Mean) | 0.591 ± 0.008 | 0.567 | 0.658 |
| MLP (Indicator) | 0.589 ± 0.007 | 0.561 | 0.641 |
| MLP (Zeros) | 0.586 ± 0.007 | 0.557 | 0.651 |
| MaskMLP (cosine) | 0.580 ± 0.009 | 0.546 | 0.624 |

MaskMLP has the largest seed-to-seed spread of any model.

**Paired outcomes.** Each of the 100 units is one seed × fold × split × target combination, and compares MaskMLP's accuracy with another model's (`report/maskmlp_wins.csv`). Primary setting:

| MaskMLP vs | Wins / ties / losses | Mean accuracy difference |
|---|---|---|
| MLP (Zeros) | 35 / 24 / 41 | −0.006 |
| MLP (Mean) | 37 / 17 / 46 | −0.011 |
| MLP (Indicator) | 37 / 25 / 38 | −0.009 |
| Logistic regression | 24 / 4 / 72 | −0.036 |
| XGBoost | 35 / 3 / 62 | −0.021 |

MaskMLP is level with or slightly below the plain MLPs. Logistic regression beats it in 72 of 100 units and XGBoost in 62.

**Sensitivity.** Pretraining scope × evaluation rows × pretraining loss is covered in Sections 3.2–3.4 (`report/sensitivity.csv`). In every combination, the traditional models remain best.

## 6. Conclusions

**Agrees with the paper:**
- The MLP depth and width, the masking scheme and the pipeline can be implemented as described, and all pieces run end to end.
- The start-of-year score on the same skill is the most important input (feature importance, Fig. 4) for XGBoost on both tasks and for MaskMLP on word identification. For MaskMLP on word attack, no single feature stands out (largest drop 0.015). For XGBoost, dropping it costs 0.048 accuracy for word identification and 0.024 for word attack, the largest drop for each task.
- Models are more accurate for students who made larger gains than for those who regressed (Fig. 2 pattern).
- Our logistic regression and XGBoost results are close to Table 1.
- The paper's reported AUC values are reproduced exactly as balanced accuracy, although the paper's text calls the metric PR-AUC.
- The paper's description of its evaluation rows checks out against the data: excluding students with missing reading assessments leaves 936 per target, and the remaining missing classroom features explain why its three missing-value MLPs differ.

**Differs from the paper (under our implementation):**
- MaskMLP does not outperform the plain MLPs. With the paper's loss it is slightly below them.
- Neural models do not outperform traditional ML. On all students, logistic regression is best: 0.025 accuracy above the best neural model (MLP Mean) and 0.036 above MaskMLP. XGBoost is second. In the intervention group the best neural models are level with XGBoost but still below logistic regression.
- Cosine pretraining caused embedding collapse in our implementation, and avoiding it did not improve MaskMLP. Our results are therefore not consistent with the paper's t-SNE reading of "well-defined clusters", although the paper's own training setup may have avoided the collapse.

**Overall:** MaskMLP's reported advantage was not reproduced under our documented training and evaluation choices on the released data. Under those choices, neither MaskMLP nor the plain MLPs outperformed traditional ML: logistic regression is the most accurate model on all students and in the intervention group. The paper's neural results are about 0.08 accuracy higher than ours, while its traditional baselines reproduce within about 0.015. That points to a difference in the neural training or evaluation procedure. Checkpoint selection is one plausible candidate, but we cannot confirm it.

## 7. Limitations

- The authors' code is unavailable, so unreported choices (scaling, optimiser settings, epochs, checkpoint selection) are our own and documented in the neural-network sections of `README.md`. The conclusions apply to these choices, and a different implementation of the same method could behave differently.
- The default pretraining is transductive (unlabelled test inputs are used), as in the paper. Fully held-out evidence comes from the training-only runs (`results/neural_network/train`).
- The loss ablation and feature importance are single-seed runs (seed 42), so differences of a few points there are within noise.
- The complete-rows, SimSiam and contrastive settings use 3 seeds rather than 5.
- The paper's exact evaluation rows (936 students per target) were not run as a separate setting; our default includes 10 more students.
- Results depend slightly on the software environment. An earlier local run of the same code with older package versions (Python 3.9.12, scikit-learn 1.0.2, torch 1.13.0, xgboost 1.7.1) gave accuracies within about 0.015 of those reported here and the same main conclusions. One finding differed: there, SimSiam-style pretraining put MaskMLP level with the best plain MLP (−0.0003), whereas here it does not help. The SimSiam result should therefore be treated as inconclusive.
- The kNN extension (Section 9) is exploratory and single-seed (seed 42), so it is compared with the seed-42 main run only.
- TabNet, GRAPE, VIME, SCARF and LightGBM are out of scope for this member.
- Subgroup bias analysis (Table 2) is out of scope for this member.

## 8. Output layout and reproducing

| Folder | Setting | Seeds |
|---|---|---|
| `results/neural_network/main` | Primary run (paper protocol) + feature importance + loss ablation + all figures | 42 |
| `results/neural_network/seeds/s43…s46` | Primary run, extra seeds | 43–46 |
| `results/neural_network/train/s42…s46` | Pretraining on training groups only | 42–46 |
| `results/neural_network/complete/all-s4x`, `train-s4x` | Complete-feature rows only, scope all / train | 42–44 |
| `results/neural_network/simsiam/s4x` | Collapse investigation: SimSiam-style loss | 42–44 |
| `results/neural_network/contrastive/s4x` | Collapse investigation: contrastive loss | 42–44 |
| `results/neural_network/report` | Paper comparison, sensitivity, robustness and collapse tables | — |
| `results/knn_extension` | Extension, not in the paper: kNN on the seed-42 folds, compared with `results/neural_network/main` (Section 9) | 42 |

The files kept in each folder, the data, the environment and the commands for the main run, the report tables and kNN are in the "Neural Network Files and Outputs" and "Running the Neural Network Experiments" sections of `README.md`. The commands below reproduce the additional runs:

```bash
# Extra seeds (repeat for 44, 45, 46)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/seeds/s43 --seed 43 --no-tsne

# Training-only pretraining (repeat for seeds 42-46)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/train/s42 --pretrain-scope train --seed 42 --no-tsne

# Complete-feature rows (add --pretrain-scope train for the train-s4x folders)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/complete/all-s42 --eval-rows complete --seed 42 --no-tsne

# Collapse investigation (repeat for seeds 42-44)
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/simsiam/s42 --pretrain-loss simsiam --seed 42 --no-tsne
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/contrastive/s42 --pretrain-loss contrastive --seed 42 --no-tsne
```

## 9. Extension: k-nearest neighbours (exploratory, not in the paper)

**Why.** kNN predicts from the most similar students, with no training beyond storing the data. It is a simple, very different model to compare against, and applied to MaskMLP's pretrained embeddings it tests directly how much information about similar students the self-supervised pretraining keeps.

**Setup** (`notebooks/Supriya_kNN_extension.py`, run in `notebooks/Supriya_knn.ipynb`):
- The same folds and test students as the seed-42 main run, and the same evaluation code.
- `kNN_raw`: the 16 features, min-max scaled on the training fold, missing values filled with the training-fold mean plus one missing-value indicator per feature; Euclidean distance.
- `kNN_MaskMLP_embedding`: the pretrained MaskMLP encoder (cosine loss, all 6,916 unfiltered students, same settings as the main run), before fine-tuning; cosine distance.
- k is chosen per fold from 3 to 101 by accuracy on the grouped validation split, then the model is refit on training + validation students. Test labels are never used to choose k. The chosen k ranged from 7 to 75 for `kNN_raw` (median 21) and from 3 to 101 for the embedding variant (median 15).

**Results** (`results/knn_extension/comparison.csv`, `knn_comparison.png`). Means over both splits and both targets, 5 folds each; the six models are from the seed-42 main run on the same folds:

| Model | Accuracy, all students | Balanced accuracy | PR-AUC | Accuracy, intervention group |
|---|---|---|---|---|
| Logistic regression | 0.616 | 0.600 | 0.670 | 0.639 |
| XGBoost | 0.600 | 0.590 | 0.670 | 0.623 |
| MLP (Zeros) | 0.596 | 0.570 | 0.662 | 0.629 |
| MLP (Mean) | 0.590 | 0.566 | 0.663 | 0.632 |
| MaskMLP | 0.586 | 0.556 | 0.630 | 0.635 |
| MLP (Indicator) | 0.583 | 0.554 | 0.637 | 0.626 |
| **kNN_raw** | **0.567** | 0.548 | 0.623 | 0.612 |
| **kNN_MaskMLP_embedding** | **0.524** | **0.502** | 0.572 | 0.538 |

**Findings:**
- **kNN on the raw features is the weakest of the models trained on those features.** On average it ranks 7th of 8 on all students, 0.049 below logistic regression and 0.016 below the lowest neural model. Its best cell is word identification in the student split (0.595), level with MaskMLP (0.594). With 16 features and only about 750 training students per fold, a student's nearest neighbours are not much more similar to them than students further away. kNN also leans towards predicting improvement (sensitivity 0.74, specificity 0.35).
- **kNN on MaskMLP's pretrained embeddings is at chance** (balanced accuracy 0.502). This is independent evidence for the collapse in Section 3.4: in this run the encoder's embedding spread fell from about 0.03 to 0.000003–0.000007, so the embeddings no longer separate students and their neighbours carry no information about the outcome. MaskMLP still reaches 0.586 after fine-tuning, which supports the reading that fine-tuning, not pretraining, does the work.
