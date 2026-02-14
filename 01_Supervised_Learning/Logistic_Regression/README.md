# Logistic Regression – From Scratch

## Overview

This module contains my from-scratch implementation of Logistic Regression using gradient descent.  
The goal was to understand how binary classification works internally instead of directly using sklearn.

The implementation includes:

- Sigmoid activation
- Gradient descent optimization
- Weight and bias updates
- Probability prediction
- Binary classification using threshold

---

## Dataset Used

Breast Cancer Wisconsin Dataset  
(Source: sklearn.datasets.load_breast_cancer)

- 569 samples
- 30 numerical features
- Binary classification (Malignant / Benign)

Feature scaling was applied before training since gradient descent is sensitive to feature magnitude.

---

## Model Performance

Evaluation was performed on a held-out test set (20%).

Accuracy: **0.9824**  
Precision: **0.9859**  
Recall: **0.9859**  
F1 Score: **0.9859**

These results indicate strong class separation and stable convergence using gradient descent.

---

## Evaluation Metrics Used

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The confusion matrix confirms that the model makes very few false predictions.

---

## Observations

- Feature scaling significantly improves convergence.
- Logistic Regression performs very well when classes are linearly separable.
- Threshold (default = 0.5) can be adjusted to trade off between precision and recall.
- Gradient descent learning rate affects convergence stability.

---

## Key Learning Outcomes

- Understanding of sigmoid activation and decision boundary
- Implementation of gradient-based optimization for classification
- Difference between probability prediction and class prediction
- Practical understanding of classification evaluation metrics