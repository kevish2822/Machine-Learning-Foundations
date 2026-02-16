# Linear Regression – From Scratch & Analytical Study

## Overview

This module focuses on understanding Linear Regression from both a mathematical
and implementation perspective. The goal was to move beyond simply using libraries
and instead understand how optimization and closed-form solutions work internally.

The following implementations are included:

- Linear Regression using Gradient Descent (from scratch)
- Linear Regression using Ordinary Least Squares (Normal Equation)
- Comparison between both approaches

---

## Theory Summary

Linear Regression models the relationship between independent variables (X) and a
dependent variable (y) using a linear equation:

y = Xβ + ε

Where:

- β represents model parameters (weights)
- ε represents error

The objective is to minimize the Mean Squared Error (MSE):

J(β) = (1/m) Σ (y_pred − y_actual)²

Two approaches were implemented:

### Gradient Descent (Iterative Optimization)

- Uses a learning rate (α)
- Updates weights iteratively
- Tracks cost reduction over iterations

### Ordinary Least Squares (Closed-Form Solution)

Uses the Normal Equation:

β = (XᵀX)⁻¹ Xᵀ y

- No learning rate required
- No iterative updates
- Direct analytical solution

---

## Implementation Details

### Files Included

- `linear_regression_gd.py` → Gradient Descent implementation
- `linear_regression_ols.py` → Normal Equation implementation
- `linear_regression_comparison.ipynb` → Data preprocessing, visualization, and comparison

### Key Components Implemented

- Cost function (Mean Squared Error)
- Weight initialization
- Bias handling
- Learning rate tuning
- Cost history tracking
- Model evaluation
- Visualization of predictions

---

## Model Evaluation

The models were evaluated using:

- Mean Squared Error (MSE)
- R² Score
- Predicted vs Actual plots
- Cost vs Iterations plot (for Gradient Descent)

---

## Observations

- Gradient Descent converges gradually depending on the learning rate.
- A very high learning rate can cause divergence.
- OLS provides an exact analytical solution but can be computationally expensive
  for large feature sets.
- Both approaches produce similar coefficients when Gradient Descent converges properly.

---

## Key Learning Outcomes

- Understanding optimization techniques used in machine learning
- Practical implementation of gradient-based learning
- Differences between iterative and analytical solutions
- Importance of feature scaling for faster convergence

---

## Dependencies

All required libraries are listed in the root `requirements.txt` file.