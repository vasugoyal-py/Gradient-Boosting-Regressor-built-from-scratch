# Gradient Boosting From Scratch

A from-scratch implementation of the **Gradient Boosting Regression** algorithm built using Python and NumPy.

The purpose of this project is to understand the mechanics of gradient boosting by implementing the algorithm manually rather than relying on pre-built ensemble models.

Decision trees are used as weak learners, and each successive tree learns to predict the residual errors made by the current ensemble.

---

## Features

- Custom Gradient Boosting Regressor implementation
- Sequential residual learning
- Decision trees as weak learners
- Configurable:
  - Number of estimators
  - Learning rate
  - Maximum tree depth
- Prediction by summing weighted outputs from all trees
- RMSE evaluation after training

---

## Implementations Included

### 1. Object-Oriented Version

A reusable `CustomGradientBoostingRegressor` class supporting:

- `fit()`
- `predict()`

This version is designed similarly to scikit-learn estimators while keeping the implementation simple and educational.

---

### 2. Functional Version

A second implementation written using standalone functions:

- `fit()`
- `predict()`

This version demonstrates the same algorithm without object-oriented programming, making the boosting process easier to follow step by step.

---

## Dataset

The models are trained using the **House Prices - Advanced Regression Techniques** dataset.

```
data/
├── train.csv
└── test.csv
```

---

## Data Preprocessing

Before training, the following preprocessing steps are performed:

- Missing value imputation
  - Numerical → Median
  - Categorical → Most Frequent
- One-Hot Encoding of categorical variables
- Standardization (Z-score normalization) of numerical features
- Conversion to NumPy arrays

---

## Repository Structure

```
gradient_boosting_from_scratch/
│
├── CustomGradientBoosting.py
├── FunctionalGradientBoosting.py
├── data/
│   ├── train.csv
│   └── test.csv
└── README.md
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
  - DecisionTreeRegressor
  - OneHotEncoder
  - SimpleImputer

> Only the decision tree implementation is borrowed from scikit-learn. The gradient boosting algorithm itself—including residual computation, sequential training, and prediction aggregation—is implemented manually.

---

## How Gradient Boosting Works

The algorithm follows these steps:

1. Start with an initial prediction (the mean of the target values).
2. Compute residuals (actual value − current prediction).
3. Train a shallow decision tree on the residuals.
4. Update the predictions using:

```
Prediction = Prediction + Learning Rate × Tree Prediction
```

5. Repeat for the desired number of trees.
6. During inference, sum the predictions from every tree together with the initial prediction.

Each tree attempts to correct the mistakes made by the previous ensemble, gradually improving the overall model.

---

## Future Improvements

Planned additions include:

- Decision Trees built completely from scratch
- Early Stopping
- Validation Set Monitoring
- Feature Importance Calculation
- Subsampling (Stochastic Gradient Boosting)
- XGBoost-inspired optimizations
- Histogram-based Gradient Boosting
- LightGBM-inspired leaf-wise growth
- CatBoost-inspired categorical feature handling
- Gradient Boosting for Classification

---

## Purpose

This repository is intended for educational purposes.

The goal is to gain a deeper understanding of:

- Gradient Boosting
- Residual Learning
- Additive Ensemble Models
- Learning Rate Effects
- Weak Learners
- Decision Trees in Boosting
- Ensemble Learning Fundamentals

---

## Notes

This implementation prioritizes readability and understanding over production performance.

The objective is to demonstrate the core ideas behind Gradient Boosting in a clear, step-by-step implementation that can be easily studied, modified, and extended.
