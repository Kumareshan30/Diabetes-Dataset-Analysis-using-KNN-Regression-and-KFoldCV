"""
Streamlit Dashboard: KNN Regression Model Complexity & Selection
- Custom KNN regressor
- Cross-validation analysis (train vs. test error)
- Automatic k selection via internal CV
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KDTree
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold as SklearnKFold, train_test_split


class KnnRegressor:
    """
    Custom KNN regressor using KDTree for neighbor search.
    """
    def __init__(self, k: int = 5):
        self.k = k
        self.tree = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.y_train = y
        self.tree = KDTree(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, idx = self.tree.query(X, k=self.k)
        return self.y_train[idx].mean(axis=1)


class KnnRegressorCV(BaseEstimator, RegressorMixin):
    """
    KNN with internal CV to choose best k from candidates.
    """
    def __init__(self, ks=None, cv_splits=5, random_state=None):
        self.ks = ks or list(range(1, 21))
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.best_k_ = None
        self.model_ = None

    def fit(self, X, y):
        cv = SklearnKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        best_score = np.inf
        for k in self.ks:
            errs = []
            for train_idx, val_idx in cv.split(X):
                knn = KnnRegressor(k).fit(X[train_idx], y[train_idx])
                errs.append(mean_squared_error(y[val_idx], knn.predict(X[val_idx])))
            avg_err = np.mean(errs)
            if avg_err < best_score:
                best_score, self.best_k_ = avg_err, k
        self.model_ = KnnRegressor(self.best_k_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def evaluate_knn_range(X, y, k_values, cv_splits=5, random_state=None):
    """
    Returns DataFrame: columns=[k, mean_train, mean_test, std_train, std_test]
    """
    records = []
    cv = SklearnKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    for k in k_values:
        train_errs, test_errs = [], []
        for train_idx, test_idx in cv.split(X):
            knn = KnnRegressor(k).fit(X[train_idx], y[train_idx])
            train_errs.append(mean_squared_error(y[train_idx], knn.predict(X[train_idx])))
            test_errs.append(mean_squared_error(y[test_idx], knn.predict(X[test_idx])))
        records.append({
            'k': k,
            'mean_train': np.mean(train_errs),
            'mean_test': np.mean(test_errs),
            'std_train': np.std(train_errs),
            'std_test': np.std(test_errs),
        })
    return pd.DataFrame.from_records(records)


def plot_cv_results(df_cv):
    """
    Returns a Matplotlib Figure of errors vs k with 95% CI.
    """
    ks = df_cv['k']
    fig, ax = plt.subplots(figsize=(10, 5))
    ci_train = 1.96 * df_cv['std_train'] / np.sqrt(df_cv.shape[0])
    ci_test = 1.96 * df_cv['std_test'] / np.sqrt(df_cv.shape[0])
    ax.errorbar(ks, df_cv['mean_train'], yerr=ci_train, label='Train MSE', fmt='-o', capsize=3)
    ax.errorbar(ks, df_cv['mean_test'], yerr=ci_test, label='Test MSE', fmt='-s', capsize=3)
    ax.set_xlabel('k (neighbors)')
    ax.set_ylabel('MSE')
    ax.set_title('KNN CV: Train vs Test Error')
    ax.legend()
    ax.grid(True)
    return fig


def main():
    st.title("KNN Regression: Model Complexity & Selection")

    # Load and split data
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Display model description
    st.header("1. Model Implementation")
    st.markdown("We implemented a custom `KnnRegressor` using `KDTree` for efficient neighbor lookups.")
    st.code(
        "class KnnRegressor: ...", language='python'
    )

    # Cross-validation evaluation
    st.header("2. Cross-Validation Results")
    k_vals = list(range(1, 31))
    df_cv = evaluate_knn_range(X_train, y_train, k_vals, cv_splits=5, random_state=42)
    st.subheader("Error Summary Table")
    st.dataframe(df_cv.set_index('k'))

    # Plot errors
    st.subheader("Error vs. k Plot")
    fig = plot_cv_results(df_cv)
    st.pyplot(fig)

    # Automatic k selection
    st.header("3. Automatic k Selection")
    knn_cv = KnnRegressorCV(ks=k_vals, cv_splits=5, random_state=42).fit(X_train, y_train)
    st.write(f"**Best k selected:** {knn_cv.best_k_}")
    y_pred = knn_cv.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    st.write(f"**Test MSE with k={knn_cv.best_k_}:** {test_mse:.4f}")

    # Interpretation
    st.header("4. Interpretation")
    st.markdown(
        "- Small k → low training error but high test error (overfitting)."
        "  Medium k → balanced. Large k → underfitting."
    )

if __name__ == '__main__':
    main()
