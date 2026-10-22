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

The preprocessing steps will include:

- Handling missing values.
- Encoding categorical variables.
- Standardizing assessment scores.
- Detecting and handling anomalies.
- Selecting relevant features.
- Dividing the data into training, validation, and testing sets.
- Checking for class imbalance and data leakage.

## Models

We plan to compare the following models:

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)
- MaskMLP-inspired neural network for incomplete data

## Evaluation

The models will be evaluated using appropriate performance measures, including:

- Accuracy
- Area Under the ROC Curve (AUC)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Classification reports
- Feature importance analysis

The final evaluation metrics will depend on whether the prediction task is treated as a regression or classification problem.

## Team Members

| Team Member | Responsibility |
|---|---|
| Anupama Singh | Baseline models using Linear Regression and Decision Tree |
| Megha Gangal | Random Forest, XGBoost, and feature-importance analysis |
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

