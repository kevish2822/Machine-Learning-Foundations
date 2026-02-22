# Employee Attrition Prediction — Machine Learning Study

## Overview

This project focuses on predicting employee attrition using multiple supervised
machine learning algorithms. The objective was to understand how different
classification models behave on the same real-world dataset and to compare their
performance using appropriate evaluation metrics.

The project emphasizes practical experimentation, model comparison, and
interpretability rather than relying on a single algorithm.

The following models were implemented:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest (with class balancing and threshold tuning)
- Decision Tree (pre-pruned)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

---

## Dataset Description

Dataset used: **IBM HR Analytics Employee Attrition Dataset**

The dataset contains employee-related information such as:

- Demographics
- Job roles and experience
- Salary and compensation
- Work environment attributes
- Attrition status (target variable)

Target variable:

Attrition → Yes / No

Converted to binary:

Yes → 1  
No → 0  

---

## Data Preprocessing

The following preprocessing steps were performed:

- One-Hot Encoding for categorical variables
- Feature scaling for distance-based and linear models
- Train-test split
- Outlier analysis and transformation where necessary
- Handling class imbalance using class weights
- Threshold tuning for improving minority class detection

---

## Model Implementation

Multiple classification algorithms were trained and evaluated to compare their
ability to detect employee attrition.

### Logistic Regression

- Linear decision boundary
- Provided strong baseline performance
- Achieved the highest F1 score among tested models

### Support Vector Machine (SVM)

- Margin-based classification approach
- Produced balanced precision and recall
- Competitive performance compared to Logistic Regression

### Random Forest

- Ensemble learning using multiple decision trees
- Class weighting and threshold tuning applied
- Achieved improved recall after probability adjustment

### Decision Tree (Post-Pruned)

- Controlled model complexity using pruning parameters
- Demonstrated underfitting compared to ensemble methods

### K-Nearest Neighbors (KNN)

- Distance-based classification
- Sensitive to feature scaling
- Lower recall compared to other models

### Gaussian Naive Bayes

- Probabilistic classifier with independence assumptions
- Higher recall but lower precision due to feature correlations

---

## Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Since the dataset is imbalanced, **F1 Score** was considered the primary
evaluation metric for model comparison.

---

## Observations

- Logistic Regression achieved the best balance between precision and recall.
- SVM produced competitive results with stable performance.
- Random Forest required threshold tuning to improve minority class detection.
- Class imbalance significantly affected recall in several models.
- Ensemble methods improved recall but sometimes increased false positives.
- Simpler linear models performed effectively on this dataset.

---

## Feature Importance Analysis

Random Forest feature importance was used to identify the most influential
factors affecting employee attrition.

Important features included:

- Overtime
- Monthly Income
- Years at Company

This analysis provides interpretable insights into factors contributing to
employee turnover.

---

## Key Learning Outcomes

- Understanding classification algorithms and their assumptions
- Handling imbalanced datasets in machine learning
- Threshold tuning for improving recall
- Ensemble learning concepts using Random Forest
- Model comparison across multiple algorithms
- Interpreting model behavior using feature importance
- Trade-offs between precision and recall in real-world problems

---

## Project Files

- `employee_attrition.ipynb` → Complete workflow and experiments
- `model_comparison.csv` → Model performance comparison
- `feature_importance.png` → Random Forest feature importance visualization
- `HR-Employee-Attrition.csv` → Dataset used for this project
---

## Dependencies

All required libraries are listed in the root `requirements.txt` file.

---

## Future Improvements

Possible extensions include:

- Hyperparameter tuning using GridSearchCV
- Advanced ensemble models such as Gradient Boosting or XGBoost
- Model deployment using a web framework
- Interactive dashboards for HR analytics