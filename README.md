# CMPE 257-01: Early Prediction of Reading Performance

## Project Overview

This machine learning project focuses on predicting future reading performance in young learners. The goal is to identify students who may experience reading difficulties early enough so that teachers and reading specialists can provide additional support.

The model will use students’ previous assessment results, demographic information, learning environment details, and intervention data. The system will serve as a decision-support tool and will not replace professional judgment or formal assessments.

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
- Compare traditional machine learning and neural network models.
- Analyze which features are most important for prediction.
- Evaluate model performance at the school and student levels.

## Data Preprocessing

Following the paper, the supervised experiments use 706 complete labeled students. The dataset contains no missing values for the selected 16 features and the two reading targets.

The 16 paper features include:

- Eight initial reading assessment features
- Gender
- Intervention group
- Age
- Number of at-risk students in the class
- Teacher/classroom assessment features

Students are grouped by school, and preprocessing is performed inside each cross-validation fold to prevent data leakage.

## Models

The project compares:

- Logistic Regression and Decision Tree baseline models
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)
- MaskMLP-inspired neural network

The Random Forest and XGBoost experiments are implemented using five-fold cross-validation grouped by school.

## Evaluation

The project predicts two binary reading-progress targets:

- Word Identification
- Word Attack

The main evaluation metrics are:

- Accuracy
- Area under the Precision–Recall curve (PR-AUC)
- Standard deviation across the five folds
- Feature-importance analysis

The paper-aligned experiments do not use a fixed 70/15/15 train-validation-test split. Unlabeled students may be used for MaskMLP pretraining, but supervised models use labeled students only.

## Paper-Aligned Results

| Target | Model | Mean Accuracy | Accuracy SD | Mean PR-AUC | PR-AUC SD |
|---|---|---:|---:|---:|---:|
| Word Identification | Random Forest | 0.6375 | 0.0614 | 0.6920 | 0.0929 |
| Word Identification | XGBoost | 0.6177 | 0.0385 | 0.6835 | 0.1030 |
| Word Attack | Random Forest | 0.5863 | 0.0551 | 0.6686 | 0.0992 |
| Word Attack | XGBoost | 0.6146 | 0.0699 | 0.6786 | 0.1024 |

Random Forest performed slightly better for Word Identification, while XGBoost performed slightly better for Word Attack. Feature-importance results are available in:

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

## Team Members

| Team Member | Responsibility |
|---|---|
| Anupama Singh | Baseline models using Linear Regression and Decision Tree |
| Megha Gangal | Random Forest, XGBoost, hyperparameter tuning, feature ablation, and feature-importance analysis |
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
```

