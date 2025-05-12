"""
Refactored KNN Regression with Cross-Validation
- Implements a custom KNN regressor
- Provides K‑fold cross-validation evaluation
- Nested cross-validation for automatic K selection
- Clear function/module structure
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KDTree, KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold as SklearnKFold, train_test_split


class KnnRegressor:
    """
    Custom K-Nearest Neighbors Regressor using KDTree for efficient neighbor search.
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
        if self.tree is None:
            raise ValueError("Model has not been fitted yet.")
        _, idx = self.tree.query(X, k=self.k)
        neighbour_vals = self.y_train[idx]
        return neighbour_vals.mean(axis=1)


class KnnRegressorCV(BaseEstimator, RegressorMixin):
    """
    KNN Regressor with internal cross-validation to select optimal k.
    """
    def __init__(self, ks: list = None, cv_splits: int = 5, random_state: int = None):
        self.ks = ks or list(range(1, 21))
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.best_k_ = None
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        cv = SklearnKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        best_score = np.inf

        for k in self.ks:
            fold_errors = []
            for train_idx, val_idx in cv.split(X):
                knn = KnnRegressor(k)
                knn.fit(X[train_idx], y[train_idx])
                y_pred = knn.predict(X[val_idx])
                fold_errors.append(mean_squared_error(y[val_idx], y_pred))

            mean_error = np.mean(fold_errors)
            if mean_error < best_score:
                best_score, self.best_k_ = mean_error, k

        # Final model with best k
        self.model_ = KnnRegressor(self.best_k_).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)


def evaluate_knn_range(
    X: np.ndarray,
    y: np.ndarray,
    k_values: list,
    cv_splits: int = 5,
    random_state: int = None
) -> dict:
    """
    Evaluate KNN for a range of k using cross-validation.
    Returns a dict with mean/std of train/test MSE for each k.
    """
    results = {k: {'train_err': [], 'test_err': []} for k in k_values}
    cv = SklearnKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for k in k_values:
        for train_idx, test_idx in cv.split(X):
            knn = KnnRegressor(k).fit(X[train_idx], y[train_idx])
            y_train_pred = knn.predict(X[train_idx])
            y_test_pred = knn.predict(X[test_idx])
            results[k]['train_err'].append(mean_squared_error(y[train_idx], y_train_pred))
            results[k]['test_err'].append(mean_squared_error(y[test_idx], y_test_pred))

    summary = {}
    for k, errs in results.items():
        summary[k] = {
            'mean_train': np.mean(errs['train_err']),
            'mean_test': np.mean(errs['test_err']),
            'std_train': np.std(errs['train_err']),
            'std_test': np.std(errs['test_err']),
        }
    return summary


def plot_cv_results(summary: dict):
    """
    Plot mean training and test errors with 95% confidence intervals.
    """
    ks = sorted(summary)
    means_train = [summary[k]['mean_train'] for k in ks]
    means_test = [summary[k]['mean_test'] for k in ks]
    errs_train = [1.96 * summary[k]['std_train'] / np.sqrt(summary[k]['std_train'].size) for k in ks]
    errs_test = [1.96 * summary[k]['std_test'] / np.sqrt(summary[k]['std_test'].size) for k in ks]

    plt.figure(figsize=(10, 5))
    plt.errorbar(ks, means_train, yerr=errs_train, label='Train MSE', fmt='-o', capsize=3)
    plt.errorbar(ks, means_test, yerr=errs_test, label='Test MSE', fmt='-s', capsize=3)
    plt.xlabel('k (neighbors)')
    plt.ylabel('Mean Squared Error')
    plt.title('KNN CV: Train vs Test Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Load data
    data = load_diabetes()
    X, y = data.data, data.target

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Evaluate range of k
    k_vals = list(range(1, 31))
    summary = evaluate_knn_range(X_train, y_train, k_vals, cv_splits=5, random_state=42)
    plot_cv_results(summary)

    # Fit KNN with best k via internal CV
    knn_cv = KnnRegressorCV(ks=k_vals, cv_splits=5, random_state=42).fit(X_train, y_train)
    y_pred_test = knn_cv.predict(X_test)

    print(f"Best k selected: {knn_cv.best_k_}")
    print(f"Test MSE with k={knn_cv.best_k_}: {mean_squared_error(y_test, y_pred_test):.4f}")


if __name__ == '__main__':
    main()

