"""
Streamlit Dashboard: KNN Regression Model Complexity & Prediction
- Custom KNN regressor using KDTree
- Cross-validation analysis (train vs test error)
- Automatic k selection via internal CV
- Dataset Overview & Preprocessing with outlier filtering
- Feature Correlation Heatmap
- Residual Analysis & Bias-Variance Decomposition
- Alternative Metrics & Comparison
- Ranking of top k values
- Interactive Prediction & Interpretation
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KDTree
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold as SklearnKFold, train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree


class KnnRegressor:
    """Custom KNN regressor using KDTree for neighbor search."""

    def __init__(self, k: int = 5):
        self.k = k
        self.tree = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = KDTree(X)
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, idx = self.tree.query(X, k=self.k)
        return self.y_train[idx].mean(axis=1)


class KnnRegressorCV(BaseEstimator, RegressorMixin):
    """KNN regressor with internal CV to select best k."""

    def __init__(self, ks=None, cv_splits=5, random_state=None):
        self.ks = ks or list(range(1, 21))
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.best_k_ = None
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        best_score = np.inf
        splitter = SklearnKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        for k in self.ks:
            errors = []
            for train_idx, val_idx in splitter.split(X):
                pred = KnnRegressor(k).fit(X[train_idx], y[train_idx]).predict(X[val_idx])
                errors.append(mean_squared_error(y[val_idx], pred))
            mean_err = np.mean(errors)
            if mean_err < best_score:
                best_score, self.best_k_ = mean_err, k
        self.model_ = KnnRegressor(self.best_k_).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


def evaluate_metric_cv(X, y, k_values, metric, cv_splits=5, random_state=42):
    """Evaluate KNN over k_values for given metric using CV; return DataFrame."""
    records = []
    splitter = SklearnKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    for k in k_values:
        train_vals, test_vals = [], []
        for tr, te in splitter.split(X):
            model = KnnRegressor(k).fit(X[tr], y[tr])
            y_tr = model.predict(X[tr])
            y_te = model.predict(X[te])
            if metric == 'MSE':
                train_vals.append(mean_squared_error(y[tr], y_tr))
                test_vals.append(mean_squared_error(y[te], y_te))
            elif metric == 'MAE':
                train_vals.append(mean_absolute_error(y[tr], y_tr))
                test_vals.append(mean_absolute_error(y[te], y_te))
            else:  # R2
                train_vals.append(r2_score(y[tr], y_tr))
                test_vals.append(r2_score(y[te], y_te))
        records.append({
            'k': k,
            'mean_train': np.mean(train_vals),
            'mean_test': np.mean(test_vals),
            'std_train': np.std(train_vals),
            'std_test': np.std(test_vals)
        })
    return pd.DataFrame(records)


def plot_cv_results(df_cv, metric):
    """Plot CV results vs k with confidence intervals."""
    ks = df_cv['k']
    ci_train = 1.96 * df_cv['std_train'] / np.sqrt(len(df_cv))
    ci_test = 1.96 * df_cv['std_test'] / np.sqrt(len(df_cv))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(ks, df_cv['mean_train'], yerr=ci_train, fmt='-o', capsize=3, label=f'Train {metric}')
    ax.errorbar(ks, df_cv['mean_test'], yerr=ci_test, fmt='-s', capsize=3, label=f'Test {metric}')
    ax.set_xlabel('k')
    ax.set_ylabel(metric)
    ax.set_title(f'Cross-Validation: {metric} vs k')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred):
    """Plot residuals vs predicted and histogram."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(0, linestyle='--', color='red')
    ax1.set_title('Residuals vs Predicted')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Residual')
    ax2.hist(residuals, bins=20, edgecolor='k')
    ax2.set_title('Residuals Distribution')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Count')
    plt.tight_layout()
    return fig, residuals


def bias_variance_decomp(X, y, k, X_test, y_test, cv_splits=5, random_state=42):
    """Estimate average bias^2 and variance via CV ensembles."""
    splitter = SklearnKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    preds = np.zeros((cv_splits, len(y_test)))
    for i, (tr, te) in enumerate(splitter.split(X)):
        preds[i] = KnnRegressor(k).fit(X[tr], y[tr]).predict(X_test)
    mean_pred = preds.mean(axis=0)
    var_pred = preds.var(axis=0)
    bias2 = (mean_pred - y_test) ** 2
    return bias2.mean(), var_pred.mean()


def plot_heatmap(corr):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(corr, cmap='coolwarm', aspect='auto')
    fig.colorbar(cax, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig


def main():
    st.title('KNN Regression Dashboard')
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # 0. Dataset Overview & Preprocessing
    st.header('0. Data Overview & Preprocessing')
    st.subheader('Snapshot')
    st.dataframe(df.head())
    st.subheader('Statistics')
    st.write(df.describe())
    t_min, t_max = st.slider('Filter target range', float(df.target.min()), float(df.target.max()),
                             (float(df.target.min()), float(df.target.max())))
    df = df[(df.target >= t_min) & (df.target <= t_max)]
    st.write(f'Filtered record count: {len(df)}')

    # 1. Feature Correlation Heatmap
    st.header('1. Feature Correlation Heatmap')
    corr = df.corr()
    st.pyplot(plot_heatmap(corr))

    # Prepare train/test split
    X = df[data.feature_names].values
    y = df.target.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 2. Cross-Validation & Alternative Metrics
    st.header('2. Cross-Validation & Alternative Metrics')
    metric = st.selectbox('Choose metric to optimize', ['MSE', 'MAE', 'R2'])
    ks = list(range(1, 31))
    df_cv = evaluate_metric_cv(X_train, y_train, ks, metric)
    st.subheader('CV Results')
    st.dataframe(df_cv.set_index('k'))
    st.pyplot(plot_cv_results(df_cv, metric))
    # Ranking top 5 ks
    if metric in ['MSE', 'MAE']:
        top5 = df_cv.nsmallest(5, 'mean_test')[['k', 'mean_test']]
    else:
        top5 = df_cv.nlargest(5, 'mean_test')[['k', 'mean_test']]
    st.write('Top 5 k values by test', metric)
    st.dataframe(top5.set_index('k'))

    # 3. Automatic k Selection & Evaluation
    st.header('3. Automatic k Selection & Evaluation')
    knn_cv = KnnRegressorCV(ks=ks, cv_splits=5, random_state=42).fit(X_train, y_train)
    best_k = knn_cv.best_k_
    st.write(f'**Best k selected**: {best_k}')
    y_pred_test = knn_cv.predict(X_test)
    st.write(f'Test MSE: {mean_squared_error(y_test, y_pred_test):.2f}')
    st.write(f'Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}')
    st.write(f'Test R²: {r2_score(y_test, y_pred_test):.2f}')

    # 4. Residual Analysis & Bias-Variance Decomposition
    st.header('4. Residual Analysis & Bias-Variance')
    fig_res, _ = plot_residuals(y_test, y_pred_test)
    st.pyplot(fig_res)
    bias2, var = bias_variance_decomp(X_train, y_train, best_k, X_test, y_test)
    st.write(f'Average bias²: {bias2:.2f}, Average variance: {var:.2f}')

    # 5. Interactive Prediction
    st.header('5. Interactive Prediction')
    k_input = st.slider('Choose k for prediction', 1, 30, best_k)
    knn_custom = KnnRegressor(k_input).fit(X_train, y_train)
    y_pred_custom = knn_custom.predict(X_test)
    st.write(f'Predictions using k = {k_input}')
    st.pyplot(plot_residuals(y_test, y_pred_custom)[0])

    # 6. Interpretation & Takeaways
    st.header('6. Interpretation & Takeaways')
    st.markdown(
        '- **Overfitting** for small k: low training error, high test error.  \n'
        '- **Underfitting** for large k: both errors increase.  \n'
        '- **Optimal k** balances bias-variance for best generalization.'
    )

    # 8. Model Explainability
    st.header('8. Model Explainability')
    surrogate = DecisionTreeRegressor(max_depth=3)
    y_train_pred = knn_cv.predict(X_train)
    surrogate.fit(X_train, y_train_pred)
    rules = export_text(surrogate, feature_names=list(data.feature_names))
    st.subheader('Surrogate Decision Tree Rules')
    # st.text(rules)
    st.subheader('Surrogate Tree Visualization')
    fig_tree, ax_tree = plt.subplots(figsize=(12, 8))
    plot_tree(surrogate,
              feature_names=list(data.feature_names),
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax_tree)
    ax_tree.set_title('Surrogate Decision Tree')
    st.pyplot(fig_tree)
    st.markdown("""
    **Surrogate Tree Interpretation:**
    
    1. **Root split on BMI:**
       - If BMI is at or below average, follow the left branch.  
       - If BMI is above average, follow the right branch.
    
    2. **Left branch (lower-than-average BMI):**
       - Check lab measure **s4**:
         - If s4 is below or equal to average, check **s5**:
           - If s5 is below or equal to average, the predicted progression score is around **103**.  
           - Otherwise, it is around **133**.  
         - If s4 is above average, check **blood pressure (bp)**:
           - If bp is below or equal to average, the score is around **149**.  
           - Otherwise, around **182**.  
    
    3. **Right branch (higher-than-average BMI):**
       - Check **s5**:
         - If s5 is below or equal to average, check BMI again at a higher threshold:
           - If BMI is slightly above average, the score is around **158**.  
           - Otherwise, around **211**.  
         - If s5 is above average, check BMI at another threshold:
           - If BMI is near average, the score is around **191**.  
           - Otherwise, around **224**.  
    """)


if __name__ == '__main__':
    main()
