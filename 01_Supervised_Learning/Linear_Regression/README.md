# Linear Regression – From Scratch & Analytical Study

## 📌 Overview
This module focuses on understanding Linear Regression from both a mathematical and implementation perspective.  
The objective is to move beyond library usage and deeply understand how optimization and closed-form solutions work.

The following implementations are included:
- Linear Regression using Gradient Descent (from scratch)
- Linear Regression using Ordinary Least Squares (Normal Equation)
- Comparison between both approaches

---

## 🧠 Theory Summary

Linear Regression models the relationship between independent variables (X) and a dependent variable (y) using a linear equation:

y = Xβ + ε

Where:
- β represents model parameters (weights)
- ε represents error

The goal is to minimize the Mean Squared Error (MSE):

J(β) = (1/m) Σ (y_pred - y_actual)²

Two approaches were implemented:

### 1️⃣ Gradient Descent (Iterative Optimization)
- Uses learning rate (α)
- Updates weights iteratively
- Tracks cost reduction over iterations

### 2️⃣ Ordinary Least Squares (Closed-Form Solution)
Uses the Normal Equation:

β = (XᵀX)⁻¹ Xᵀ y

- No learning rate
- No iterations
- Direct analytical solution

---

## 🛠 Implementation Details

### Files Included
- `linear_regression_gd.py` → Gradient Descent implementation
- `linear_regression_ols.py` → Normal Equation implementation
- `notebook.ipynb` → Data preprocessing, visualization, and comparison

### Key Components Implemented
- Cost function (Mean Squared Error)
- Weight initialization
- Bias handling
- Learning rate tuning
- Cost history tracking
- Model evaluation
- Visualization of predictions

---

## 📊 Model Evaluation

Evaluation metrics used:
- Mean Squared Error (MSE)
- R² Score
- Predicted vs Actual Plot
- Cost vs Iterations Plot (for GD)

---

## 🔍 Observations & Insights

- Gradient Descent converges gradually depending on learning rate.
- Very high learning rate causes divergence.
- OLS provides an exact solution but can be computationally expensive for large feature sets.
- Both methods produce similar coefficients when GD converges properly.

---

## 🚀 Key Learning Outcomes

- Deep understanding of optimization in ML
- Practical implementation of gradient-based learning
- Difference between iterative and analytical solutions
- Importance of feature scaling in convergence

---

## 📦 Dependencies

See `requirements.txt` in the root directory.
