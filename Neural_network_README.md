# Member 3: MLP / MaskMLP reproduction on ECRI

Reproduction of the neural-network part of Shangguan et al., *Scalable Early Childhood Reading Performance Prediction* (NeurIPS 2024 Datasets & Benchmarks). It implements the MLP and MaskMLP models, visualises their performance, and compares them with traditional ML. Results and discussion are in [REPORT.md](REPORT.md).

## Files

| File | Purpose |
|---|---|
| `Neural_network_implementation.py` | Experiments: models, cross-validation, analyses, figures |
| `Neural_network_report.py` | Paper comparison, sensitivity, robustness and collapse tables. It imports constants and plotting helpers from `Neural_network_implementation.py`. |
| `REPORT.md` | Results, review issues and conclusions |
| `requirements.txt` | Python dependencies (minimum versions) |
| `environment.yml`, `requirements-lock.txt` | Exact versions used for the reported experiments |
| `outputs/` | The experimental evidence behind REPORT.md (see below) |

## Experimental evidence (`outputs/`)

The repository keeps the outputs behind every number in REPORT.md. For each of the 22 runs it keeps:

| File | Contents |
|---|---|
| `config.json` | Every setting, the seed and the size of the pretraining pool. Data paths are stored as file names only. |
| `results.csv` | Per-fold and summary metrics for every model, split, target and subset |
| `significance.csv` | Paired t-tests over folds |
| `pretrain_log.csv` | MaskMLP pretraining loss and embedding spread, per epoch |
| `progress_breakdown.csv` | Accuracy by progress quintile |

**Also kept:**
- `outputs/main/importance.csv`, `outputs/main/loss_ablation.csv` and `outputs/main/raw_verification.json`
- `outputs/simsiam/ablation_s42.csv`, the extension run
- `outputs/main/figures/`
- `outputs/report/`, the consolidated tables and charts

**Left out of Git to keep the repository small (about 8 MB of outputs, 138 files, are tracked):**
- the per-student `predictions.csv` and `folds.csv` files (about 50 MB)
- the per-run figure copies in the seed and sensitivity folders (about 20 MB)

They are kept in `member3_deliverables.zip` and can be regenerated with the commands below.

## Reproducibility

The reported experiments were run with Python 3.9.12 on CPU, using numpy 1.21.5, pandas 1.4.2, scipy 1.7.3, scikit-learn 1.0.2, matplotlib 3.5.1, torch 1.13.0 and xgboost 1.7.1. To recreate that environment:

```bash
conda env create -f environment.yml
```

Alternatively, run `pip install -r requirements-lock.txt` in a Python 3.9 environment. `requirements.txt` only sets minimum versions; newer libraries should run the code, but may change results slightly.

## Data

The data is not included. Download `ECRIDataSet.csv` and `ECRIRawData.csv` from the authors' repository, [ecri-data/ECRI-Dataset](https://github.com/ecri-data/ECRI-Dataset), and place them next to the scripts.

## What is implemented

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

Logistic regression and XGBoost run on exactly the same folds and rows as a controlled reference. Members 1 and 2 own the traditional-model implementations. Use `--models neural` to run the neural models only.

## Design decisions

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
- **Evaluation rows.** The paper's text excludes incomplete students, but its Table 1 implies they were included.
  - Default `--eval-rows all` uses every labelled student.
  - `--eval-rows complete` follows the text.
- **Training details.**
  - Inputs are min-max scaled, so −1 stays the only negative value.
  - A grouped 20 % validation split is used for early stopping (patience 15).
  - Optimiser: Adam, learning rate 1e-3, batch size 64.
  - Pretraining and fine-tuning each run for up to 100 epochs.
- **Embedding collapse.** The cosine loss only pulls positive pairs together, so the encoder can map every student to the same embedding. In our implementation, cosine pretraining caused embedding collapse in every run. The script records embedding spread every epoch and warns when it falls below 10 % of its initial value.
- **Extension, not in the paper.** `--pretrain-loss simsiam` is Eq. 1 with a predictor and stop-gradient. It is used only to investigate the collapse: it is excluded from the Table 4 ablation and labelled `extension` in every report table.

## Running

```bash
pip install -r requirements-lock.txt   # or requirements.txt for minimum versions

# Complete experiment: primary reproduction + feature importance + loss ablation
python Neural_network_implementation.py --dataset ECRIDataSet.csv --raw-dataset ECRIRawData.csv --output-dir outputs/main --importance --loss-ablation

# Quick check (reduced epochs, one split and target)
python Neural_network_implementation.py --dataset ECRIDataSet.csv --raw-dataset ECRIRawData.csv --output-dir outputs/smoke --models neural --group school --target RMwidRS_target --epochs 5 --pretrain-epochs 5 --no-tsne

# Comparison tables (works on any subset of runs)
python Neural_network_report.py --outputs outputs
```

The commands for the seed, scope, evaluation-row and collapse-investigation runs are listed in REPORT.md, Section 8. Run `python Neural_network_implementation.py --help` for all options.

On Windows, keep the folder path short, because paths over 260 characters make figure saving fail.
