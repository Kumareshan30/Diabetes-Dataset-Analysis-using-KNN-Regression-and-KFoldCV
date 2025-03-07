# KNN Regression with K-Fold Cross Validation

## Project Overview
This project implements **K-Nearest Neighbors (KNN) Regression** with **K-Fold Cross Validation** to evaluate model performance. The goal is to systematically test different values of **K** and analyze their impact on prediction accuracy. The dataset is preprocessed, normalized, and used to compare different hyperparameter settings.

Additionally, the notebook includes:
- **Automatic Model Selection**: The best value of **K** is chosen based on the lowest test error.
- **Error Distribution Analysis**: Evaluating how different **K** values affect prediction variance.
- **Feature Scaling Techniques**: Ensuring the model performs optimally with different feature ranges.
- **Performance Visualization**: Using line plots to track model performance across K values.
- **Comparison with Baseline Models**: Assessing how KNN performs compared to simple mean predictions.

## Folder Structure
KNN Regression with K-Fold Cross Validation/
      
   ├── KNN_regression_KFoldCV.ipynb  # Jupyter Notebook implementing KNN regression with K-Fold CV
   
   ├── README.md  # Project documentation
   

## Installation
Ensure you have Python 3.8+ installed and install dependencies using:
```sh
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter Notebook:
   ```sh
   jupyter notebook KNN_regression_KFoldCV.ipynb
2. Execute the cells step by step to:
- Load and preprocess the dataset.
- Implement KNN regression with different values of K.
- Evaluate model performance using K-Fold Cross Validation.
- Visualize the effect of K on model performance.


## Methodology

### 1. Data Preprocessing
- Handling missing values (if any).
- Normalizing numerical features for better distance calculations.

### 2. KNN Regression Implementation
- Using **scikit-learn's KNeighborsRegressor**.
- Testing different values of **K** (1 to 30).

### 3. K-Fold Cross Validation
- Implementing **L-Fold CV** where **L** is chosen dynamically.
- Calculating the **mean squared error (MSE)** for each **K**.
- Selecting the best **K** based on lowest test error.

### 4. Results & Analysis
- **Performance Visualization**: Line plots showing the variation of MSE with **K**.
- **Optimal K Selection**: Identifying the best **K** for prediction accuracy.
- **Error Distribution Analysis**: Visualizing how errors change with different **K** values.
