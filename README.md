# CMPE 257-01: Early Prediction of Reading Performance

## Project Overview

This machine learning project focuses on predicting future reading performance in young learners. The goal is to identify students who may experience reading difficulties early enough so that teachers and reading specialists can provide additional support.

The model uses students’ previous assessment results, demographic information, learning environment details, and intervention data. The system serves as a decision-support tool and does not replace professional judgment or formal assessments.

## Dataset

We use the Enhanced Core Reading Instruction (ECRI) dataset.

The dataset includes information from:

- 6,916 students
- 44 schools
- 172 teachers
- Student assessment scores
- Demographic information
- Special education status
- Intervention information
- Classroom and school-level characteristics

Dataset:  
https://github.com/ecri-data/ECRI-Dataset/tree/main/ECRI

Research paper:  
https://papers.neurips.cc/paper_files/paper/2024/hash/862170d83a17b7c2ef3c5a58e7f1992d-Abstract-Datasets_and_Benchmarks_Track.html

## Project Goals

- Predict future reading performance.
- Identify students who may need additional support.
- Handle missing student information.
- Prevent data leakage by using only information available before prediction.
- Compare traditional machine-learning and neural-network models.
- Analyze which features are most important for prediction.
- Evaluate model performance across different schools.

## Data Preprocessing

Following the paper, the supervised experiments use 706 complete labeled students. The dataset contains no missing values for the selected 16 features and the two reading targets.

The 16 paper features include:

- Eight initial reading assessment features
- Gender
- Intervention group
- Age
- Number of at-risk students in the class
- Teacher and classroom assessment features

Students are grouped by school. The models use five-fold cross-validation grouped by school to prevent data leakage. This ensures that students from the same school do not appear in both the training and validation groups.

## Models

The paper-aligned experiments compare:

- Logistic Regression and Decision Tree baseline models
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)
- MaskMLP-inspired neural network

Additional extension experiments include:

- Gaussian Naive Bayes
- Random Forest hyperparameter tuning
- XGBoost hyperparameter tuning
- Feature-group ablation
- Feature-importance analysis

The Random Forest and XGBoost paper-aligned experiments use five-fold cross-validation grouped by school. Gaussian Naive Bayes was added as an extension model using the same evaluation setup.

## Evaluation

The project predicts two binary reading-progress targets:

- Word Identification
- Word Attack

The main evaluation metrics are:

- Accuracy
- Area under the Precision–Recall curve (PR-AUC)
- Standard deviation across the five folds
- Feature-importance analysis

The paper-aligned experiments do not use a fixed 70/15/15 train-validation-test split. Unlabeled students may be used for MaskMLP pretraining, while supervised models use labeled students only.

## Paper-Aligned Results

| Target | Model | Mean Accuracy | Accuracy SD | Mean PR-AUC | PR-AUC SD |
|---|---|---:|---:|---:|---:|
| Word Identification | Random Forest | 0.6375 | 0.0614 | 0.6920 | 0.0929 |
| Word Identification | XGBoost | 0.6177 | 0.0385 | 0.6835 | 0.1030 |
| Word Attack | Random Forest | 0.5863 | 0.0551 | 0.6686 | 0.0992 |
| Word Attack | XGBoost | 0.6146 | 0.0699 | 0.6786 | 0.1024 |

Random Forest performed slightly better for Word Identification, while XGBoost performed slightly better for Word Attack.

Feature-importance results are available in:

`results/random_forest_feature_importance.csv`

## Extended Model Analysis

Random Forest and XGBoost were further evaluated through hyperparameter tuning and feature-group ablation.

The tuning experiments used the same five-fold cross-validation grouped by school. The tested parameters included:

- Random Forest number of estimators
- Random Forest maximum depth
- Random Forest minimum samples per leaf
- XGBoost number of estimators
- XGBoost maximum depth
- XGBoost learning rate

A feature-group ablation study was performed by removing:

- Reading assessment features
- Student features
- Classroom features

The full-feature model was used as the baseline. A negative PR-AUC change means that removing the feature group reduced performance.

Detailed results are available in:

- `results/random_forest_tuning_results.csv`
- `results/xgboost_tuning_results.csv`
- `results/tuned_model_comparison.csv`
- `results/feature_group_ablation_results.csv`

## Extension: Naive Bayes Comparison

Gaussian Naive Bayes was tested as an additional model beyond the paper-aligned experiments. It used the same 706 students, 16 features, and five-fold cross-validation grouped by school.

| Target | Mean Accuracy | Accuracy SD | Mean PR-AUC | PR-AUC SD |
|---|---:|---:|---:|---:|
| Word Identification | 0.6274 | 0.0400 | 0.6434 | 0.1041 |
| Word Attack | 0.5933 | 0.0602 | 0.6660 | 0.0830 |

Naive Bayes worked successfully and completed all five cross-validation folds. However, it did not improve the overall model performance.

For Word Identification, Naive Bayes performed lower than Random Forest and XGBoost. For Word Attack, its performance was close to Random Forest but lower than XGBoost.

This may be because Naive Bayes assumes that the features are independent. In this dataset, many reading assessment features are related to one another. Therefore, the tree-based models were better suited to learning the relationships among the features.

Naive Bayes results are available in:

- `results/naive_bayes_cv_results.csv`
- `results/naive_bayes_summary.csv`
- `results/model_comparison_with_naive_bayes.csv`

## Overall Findings

- Random Forest performed best for Word Identification.
- XGBoost performed best for Word Attack.
- Naive Bayes worked successfully but did not improve the results.
- Tree-based models were better suited to the relationships among the reading features.
- Model performance varied across school-based folds.
- The models should support educators and should not replace teacher or specialist judgment.

## Neural Network Models: MLP and MaskMLP

This part reproduces the neural-network experiments of Shangguan et al., *Scalable Early Childhood Reading Performance Prediction* (NeurIPS 2024 Datasets & Benchmarks): a three-layer MLP with three missing-value strategies and the self-supervised MaskMLP. It visualises their performance and compares them with traditional machine-learning models. The full results, methodological checks and conclusions are in `reports/Supriya_Neural_Network_REPORT.md`.

Unlike the paper-aligned setup above, it uses all 946 labelled students per target (706 complete students as a sensitivity check), evaluates both school-grouped and student-grouped five-fold cross-validation, and pretrains MaskMLP without labels on all 6,916 students. The reasons are under Design Decisions below.

### What Is Implemented

| Component | Paper reference | Notes |
|---|---|---|
| Three-layer, 64-unit MLP; Zeros / Mean / Indicator missing-value baselines | Table 1 | Missing values are encoded as −1 |
| MaskMLP | Eq. 1 | 25 % of observed features masked to −1, cosine embedding loss, then BCE fine-tuning |
| Five-fold school- and student-grouped evaluation, all students and intervention group | Table 1, Sec. 5.1 | Paired t-tests over folds |
| t-SNE of MLP vs MaskMLP embeddings | Fig. 3 | |
| Leave-one-feature-out importance | Fig. 4, Table 5 | |
| Accuracy by progress quintile | Fig. 2 | Needs `--raw-dataset` |
| Pretraining loss ablation | Table 4 | The paper's 11 losses and combinations only |
| Embedding-collapse diagnostics | Not in the paper | Spread recorded every epoch; warning on collapse |

Logistic Regression and XGBoost run on exactly the same folds and rows as a controlled reference. Members 1 and 2 own the traditional-model implementations, so the reference results below can differ slightly from the Random Forest, XGBoost and baseline results above, which use their own settings. Use `--models neural` to run the neural models only.

### Neural Network Design Decisions

These choices are not specified in the paper, or the paper is ambiguous about them. All are configurable.

- **AUC.** The paper's text calls its AUC the area under the precision–recall curve. However, every Table 5 value equals (specificity + sensitivity) / 2. The script therefore reports three metrics:
  - `balanced_accuracy`, the reading that matches the paper's tables
  - `pr_auc`, from probabilities
  - `roc_auc`, from probabilities
- **Pretraining data.** The paper pretrains on "all data samples" without labels. This is transductive: it includes validation and test students' inputs.
  - Default `--pretrain-scope all` follows the paper.
  - `--pretrain-scope train` excludes each fold's validation and test students.
- **Unfiltered pretraining pool.** With `--raw-dataset`, MaskMLP pretrains on all 6,916 students, rebuilt from the start-of-year rows of `ECRIRawData.csv`. `ECRIDataSet.csv` has 6,803 of them.
  - Before use, the rebuilt features and school IDs are checked cell by cell against `ECRIDataSet.csv` (`verify_unfiltered()`).
  - The outcome labels are also rebuilt from the raw scores and compared with the released labels (`verify_labels()`). This check is only reported and never changes the training data.
  - Both results are saved to `raw_verification.json`.
  - If the check fails, the script falls back to `ECRIDataSet.csv`, or stops if `--pretrain-source raw` was given.
- **Evaluation rows.** The paper excludes students with missing reading assessments from fine-tuning and evaluation. That leaves 936 labelled students per target, 230 of whom still miss a classroom-level feature, which is why the paper's Zeros, Mean and Indicator MLPs differ.
  - Default `--eval-rows all` uses every labelled student (946, only 10 more than the paper's rule).
  - `--eval-rows complete` keeps only students with all 16 features (706). This is stricter than the paper and matches the paper-aligned experiments above.
- **Training details.**
  - Inputs are min-max scaled, so −1 stays the only negative value.
  - A grouped 20 % validation split is used for early stopping (patience 15).
  - Optimiser: Adam, learning rate 1e-3, batch size 64.
  - Pretraining and fine-tuning each run for up to 100 epochs.
- **Embedding collapse.** The cosine loss only pulls positive pairs together, so the encoder can map every student to the same embedding. The script records embedding spread every epoch and warns when it falls below 10 % of its initial value.
- **Extension, not in the paper.** `--pretrain-loss simsiam` is Eq. 1 with a predictor and stop-gradient. It is used only to investigate the collapse: it is excluded from the Table 4 ablation and labelled `extension` in every report table.

### Neural Network Results

School-grouped split, all 946 labelled students per target, mean over 5 seeds × 5 folds. The standard deviation is across the 25 fold results.

| Target | Model | Mean Accuracy | Accuracy SD | Mean PR-AUC | PR-AUC SD |
|---|---|---:|---:|---:|---:|
| Word Identification | Logistic Regression (reference) | 0.6324 | 0.0281 | 0.6512 | 0.0696 |
| Word Identification | XGBoost (reference) | 0.5938 | 0.0238 | 0.6315 | 0.0711 |
| Word Identification | MLP (Zeros) | 0.5822 | 0.0470 | 0.6279 | 0.0988 |
| Word Identification | MLP (Mean) | 0.5855 | 0.0447 | 0.6393 | 0.1076 |
| Word Identification | MLP (Indicator) | 0.5924 | 0.0520 | 0.6286 | 0.0923 |
| Word Identification | MaskMLP | 0.5743 | 0.0628 | 0.6057 | 0.0951 |
| Word Attack | Logistic Regression (reference) | 0.5813 | 0.0545 | 0.6707 | 0.0325 |
| Word Attack | XGBoost (reference) | 0.5874 | 0.0290 | 0.6704 | 0.0429 |
| Word Attack | MLP (Zeros) | 0.5640 | 0.0547 | 0.6432 | 0.0580 |
| Word Attack | MLP (Mean) | 0.5712 | 0.0495 | 0.6474 | 0.0596 |
| Word Attack | MLP (Indicator) | 0.5648 | 0.0540 | 0.6326 | 0.0603 |
| Word Attack | MaskMLP | 0.5640 | 0.0483 | 0.6218 | 0.0723 |

With the 706 complete students used by the paper-aligned experiments above (school-grouped split, mean over 3 seeds × 5 folds, SD across the 15 fold results):

| Target | Model | Mean Accuracy | Accuracy SD | Mean PR-AUC | PR-AUC SD |
|---|---|---:|---:|---:|---:|
| Word Identification | Logistic Regression (reference) | 0.6375 | 0.0643 | 0.6906 | 0.0842 |
| Word Identification | XGBoost (reference) | 0.6219 | 0.0399 | 0.6741 | 0.1038 |
| Word Identification | MLP (Zeros) | 0.5988 | 0.0724 | 0.6454 | 0.0879 |
| Word Identification | MLP (Mean) | 0.5988 | 0.0724 | 0.6454 | 0.0879 |
| Word Identification | MLP (Indicator) | 0.5629 | 0.0728 | 0.6023 | 0.0987 |
| Word Identification | MaskMLP | 0.5491 | 0.0728 | 0.5876 | 0.0855 |
| Word Attack | Logistic Regression (reference) | 0.6005 | 0.0407 | 0.6750 | 0.1079 |
| Word Attack | XGBoost (reference) | 0.6057 | 0.0536 | 0.6737 | 0.1023 |
| Word Attack | MLP (Zeros) | 0.5872 | 0.0687 | 0.6564 | 0.0725 |
| Word Attack | MLP (Mean) | 0.5872 | 0.0687 | 0.6564 | 0.0725 |
| Word Attack | MLP (Indicator) | 0.5694 | 0.0880 | 0.6461 | 0.0793 |
| Word Attack | MaskMLP | 0.5825 | 0.0677 | 0.6370 | 0.0733 |

With complete students only, the Zeros and Mean MLPs are identical, as expected, because there are no missing values to fill. Student-grouped results, the intervention group, seed robustness and the comparison with the paper's Table 1 are in the report.

Main neural-network findings:

- The neural models did not outperform traditional machine learning. Over both splits and both targets, Logistic Regression scored 0.616 accuracy and XGBoost 0.602, against 0.586–0.591 for the MLPs and 0.580 for MaskMLP.
- MaskMLP's reported advantage over the plain MLPs was not reproduced under our documented training and evaluation choices.
- Cosine pretraining caused embedding collapse in all 16 cosine runs. Two collapse-resistant losses avoided the collapse but did not improve MaskMLP.
- Pretraining only on training students gave about the same MaskMLP accuracy (0.577) as pretraining on all students (0.580).
- The paper's reported AUC values are reproduced exactly as balanced accuracy.
- The start-of-year score on the same skill is the most important input for XGBoost on both tasks.

Detailed results are available in:

- `reports/Supriya_Neural_Network_REPORT.md`
- `results/neural_network/report/table1_comparison.csv`
- `results/neural_network/report/robustness.csv`
- `results/neural_network/report/sensitivity.csv`
- `results/neural_network/report/collapse.csv`
- `results/neural_network/main/importance.csv`
- `results/neural_network/main/loss_ablation.csv`
- `results/neural_network/main/figures/`

## Extension: k-Nearest Neighbours

k-nearest neighbours (kNN) was tested as an exploratory model beyond the paper. It uses exactly the same folds and test students as the seed-42 neural-network run, in two variants:

- `kNN_raw`: the 16 features, scaled on the training fold, with missing values filled and flagged; Euclidean distance.
- `kNN_MaskMLP_embedding`: the pretrained MaskMLP encoder's embeddings, before fine-tuning; cosine distance. This tests how much information about similar students the pretraining keeps.

For each fold, k (3 to 101) is chosen on the grouped validation split, so test labels are never used to choose it.

School-grouped split, all 946 labelled students, seed 42, mean over 5 folds:

| Target | Model | Mean Accuracy | Accuracy SD | Mean PR-AUC | PR-AUC SD |
|---|---|---:|---:|---:|---:|
| Word Identification | kNN_raw | 0.5795 | 0.0400 | 0.6058 | 0.0701 |
| Word Identification | kNN_MaskMLP_embedding | 0.4981 | 0.0274 | 0.5539 | 0.0380 |
| Word Attack | kNN_raw | 0.5391 | 0.0531 | 0.6152 | 0.0973 |
| Word Attack | kNN_MaskMLP_embedding | 0.5315 | 0.0265 | 0.5991 | 0.0737 |

kNN worked but did not improve the results. Over both splits and both targets, `kNN_raw` reached 0.567 accuracy, 7th of the 8 models compared. kNN on the MaskMLP embeddings was at chance level (balanced accuracy 0.502), which is direct evidence that the collapsed pretrained embeddings no longer distinguish students.

kNN results are available in:

- `results/knn_extension/results.csv`
- `results/knn_extension/selected_k.csv` (k chosen per fold)
- `results/knn_extension/comparison.csv` and `results/knn_extension/knn_comparison.png` (kNN next to the six models of `results/neural_network/main`)
- `results/knn_extension/config.json`

## Neural Network Files and Outputs

| File | Purpose |
|---|---|
| `notebooks/Supriya_Neural_network_implementation.py` | Experiments: models, cross-validation, analyses, figures |
| `notebooks/Supriya_Neural_network_report.py` | Paper comparison, sensitivity, robustness and collapse tables. It imports constants and plotting helpers from `Supriya_Neural_network_implementation.py`, so keep both scripts in the same folder. |
| `notebooks/Supriya_Neural_Network.ipynb` | The Colab notebook the reported runs were made with |
| `notebooks/Supriya_kNN_extension.py` | Exploratory extension, not in the paper: k-nearest neighbours on the same folds. It imports from `Supriya_Neural_network_implementation.py` and changes nothing in it. |
| `notebooks/Supriya_knn.ipynb` | The Colab notebook the kNN run was made with |
| `reports/Supriya_Neural_Network_REPORT.md` | Results, methodological checks and conclusions |
| `requirements.txt` | Python dependencies (minimum versions) |
| `results/neural_network/` | The experimental evidence behind the report (see below) |
| `results/knn_extension/` | kNN extension results (see Extension: k-Nearest Neighbours) |

The repository keeps the outputs behind every number in the report. For each of the 22 runs in `results/neural_network/` it keeps:

| File | Contents |
|---|---|
| `config.json` | Every setting, the seed and the size of the pretraining pool. Data paths are stored as file names only. |
| `results.csv` | Per-fold and summary metrics for every model, split, target and subset |
| `significance.csv` | Paired t-tests over folds |
| `pretrain_log.csv` | MaskMLP pretraining loss and embedding spread, per epoch |
| `progress_breakdown.csv` | Accuracy by progress quintile |

Also kept:

- `results/neural_network/main/importance.csv`, `results/neural_network/main/loss_ablation.csv` and `results/neural_network/main/raw_verification.json`
- `results/neural_network/main/figures/`
- `results/neural_network/report/`, the consolidated tables and charts

Left out of Git to keep the repository small (about 7 MB of outputs, 156 files, are tracked):

- the per-student `predictions.csv` and `folds.csv` files (about 50 MB)
- the per-run figure copies in the seed and sensitivity folders (about 20 MB)

They are kept in the `results.zip` downloaded from Colab and can be regenerated with the commands below.

## Running the Neural Network Experiments

The data is not included. Download `ECRIDataSet.csv` and `ECRIRawData.csv` from the authors' repository, [ecri-data/ECRI-Dataset](https://github.com/ecri-data/ECRI-Dataset), and place them in `data/`. Git ignores CSV files there.

The reported experiments were run on Google Colab's CPU runtime, which has every dependency preinstalled. Elsewhere, install them with `pip install -r requirements.txt`, which sets minimum versions only. Different library versions run the same code but can shift accuracies slightly: an earlier local run with older versions (Python 3.9.12, scikit-learn 1.0.2, torch 1.13.0, xgboost 1.7.1) differed by up to about 0.015 (report, Section 7).

All commands and paths are relative to the repository's top folder.

```bash
pip install -r requirements.txt   # not needed on Colab

# Complete experiment: primary reproduction + feature importance + loss ablation
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/main --importance --loss-ablation

# Quick check (reduced epochs, one split and target); delete results/neural_network/smoke afterwards
python notebooks/Supriya_Neural_network_implementation.py --dataset data/ECRIDataSet.csv --raw-dataset data/ECRIRawData.csv --output-dir results/neural_network/smoke --models neural --group school --target RMwidRS_target --epochs 5 --pretrain-epochs 5 --no-tsne

# Comparison tables (works on any subset of runs)
python notebooks/Supriya_Neural_network_report.py --outputs results/neural_network

# Extension: kNN on the same folds (needs results/neural_network/main for the comparison)
python notebooks/Supriya_kNN_extension.py --raw-dataset data/ECRIRawData.csv

# Rebuild only the kNN comparison table and figure from saved kNN results (no retraining)
python notebooks/Supriya_kNN_extension.py --compare-only
```

The commands for the seed, scope, evaluation-row and collapse-investigation runs are listed in the report, Section 8. Run `python notebooks/Supriya_Neural_network_implementation.py --help` for all options.

On Windows, keep the folder path short, because paths over 260 characters make figure saving fail.

## Team Members

| Team Member | Responsibility |
|---|---|
| Anupama Singh | Baseline models using Linear Regression and Decision Tree |
| Megha Gangal | Random Forest, XGBoost, Gaussian Naive Bayes, hyperparameter tuning, feature ablation, and feature-importance analysis |
| Supriya Kavitha Venkatesan | MLP/MaskMLP-inspired model and comparison analysis |

All team members will contribute to preprocessing, documentation, report writing, presentation preparation, and final review.

## Project Structure

```text
cmpe257-early-reading-prediction/
│
├── data/
├── notebooks/
├── src/
├── models/
├── results/
├── reports/
├── requirements.txt
└── README.md