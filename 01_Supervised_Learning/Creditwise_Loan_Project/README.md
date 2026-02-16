# CreditWise Loan Approval Prediction

## Overview

This project focuses on predicting whether a loan application will be approved or not
based on applicant financial and demographic information.

The objective was to perform end-to-end machine learning workflow including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and encoding
- Model training
- Model comparison
- Performance evaluation

Multiple classification algorithms were compared to determine the best performing model.

---

## Dataset

The dataset contains information about loan applicants, including:

- Applicant Income
- Coapplicant Income
- Credit Score
- Debt-to-Income Ratio
- Employment Status
- Education Level
- Property Area
- Loan Amount
- Loan Approval Status (Target)

Missing values were handled using imputation techniques.

---

## Data Preprocessing

Steps performed:

- Handling missing values
  - Numerical → Mean imputation
  - Categorical → Most frequent value
- Encoding categorical variables
  - Label Encoding
  - One-Hot Encoding
- Feature scaling using StandardScaler
- Train-test split (80/20)

---

## Exploratory Data Analysis

EDA was performed to understand relationships between features and loan approval:

- Distribution plots
- Boxplots
- Class balance visualization
- Credit score vs approval analysis
- Correlation heatmap

These insights helped identify important predictors.

---

## Models Used

The following supervised learning models were trained:

1. Logistic Regression
2. Gaussian Naive Bayes
3. K-Nearest Neighbors (KNN)

All models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Results

Based on evaluation metrics:

**Best Performing Model:** Gaussian Naive Bayes  
(Highest precision among the compared models)

This indicates that probabilistic assumptions worked well for this dataset.

---

## Key Learnings

- Importance of feature scaling for distance-based models like KNN
- Effect of feature distributions on Naive Bayes performance
- Differences between linear and probabilistic classifiers
- Practical understanding of evaluation metrics
- End-to-end ML workflow implementation

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature selection
- Deployment using Flask or FastAPI
