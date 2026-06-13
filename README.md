"# Fraud-classification-with-ANN-Deep-Learning" 

# Credit Card Default Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science project aimed at predicting the probability of credit card default using client demographics, historical bill statements, and repayment behaviors.

## 📌 Project Overview
Financial institutions require robust mechanisms to identify risky borrowers and minimize credit defaults. This project utilizes the **Default of Credit Card Clients Dataset** to build and compare machine learning classification models. The primary objective is to accurately classify whether a client will default on their credit card payment in the consecutive month.

## 📊 Dataset Details
The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
* **Instances:** 30,000 observations (representing distinct credit card clients in Taiwan from April to September 2005).
* **Features:** 23 explanatory variables + 1 binary target variable.
* **Target Variable:** `default.payment.next.month` (1 = Yes/Default; 0 = No/Non-default).

### Feature Categories:
* **Demographic Factors:** `LIMIT_BAL` (Credit Limit), `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`.
* **History of Past Payments (April - Sept 2005):** `PAY_0` to `PAY_6` (Tracking delays in months).
* **Amount of Bill Statement (April - Sept 2005):** `BILL_AMT1` to `BILL_AMT6`.
* **Amount of Previous Payment (April - Sept 2005):** `PAY_AMT1` to `PAY_AMT6`.

## ⚙️ Key Workflow & Methodology

### 1. Data Preprocessing & Cleaning
* Checked for missing data values (the dataset contains no explicit null entries).
* Handled undocumented/mislabeled structural categories in `EDUCATION` and `MARRIAGE` features.
* Renamed `PAY_0` to `PAY_1` to preserve structural indexing alignment with monthly bill tracking.

### 2. Exploratory Data Analysis (EDA)
* Identified severe target class imbalance (~78% non-defaulters vs. ~22% defaulters).
* Analysed default correlations across demographic parameters (e.g., higher default rates among younger groups or lower credit limits).
* Checked multicollinearity trends among `BILL_AMT` sequences using correlation matrices.

### 3. Feature Engineering & Scaling
* Implemented **One-Hot Encoding** on categorical features (`SEX`, `MARRIAGE`, `EDUCATION`).
* Applied **StandardScaler** to continuous numeric tracking variables to adjust scalar ranges for linear and metric-based models.
* *[Optional: Mention if you handled class imbalance here using SMOTE, Random Under Sampler, etc.]*

### 4. Model Training & Evaluation
We implemented and cross-evaluated the following classification algorithms:
* Logistic Regression
* Random Forest Classifier
* XGBoost / Gradient Boosting
* *[Add/Remove models based on what you actually used]*

---

## 📈 Experimental Results

Given the class imbalance, performance was evaluated primarily using **F1-Score**, **ROC-AUC Score**, and the **Confusion Matrix** alongside baseline accuracy.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| **Random Forest** | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| **XGBoost (Best)** | **0.XX** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |

### Key Takeaways:
* *[Example: Payment history in the immediate past month (PAY_1) proved to be the single most powerful feature metric predicting default propensity.]*
* *[Example: Ensembling methods substantially outperformed individual linear models in tracking default variances.]*

---
