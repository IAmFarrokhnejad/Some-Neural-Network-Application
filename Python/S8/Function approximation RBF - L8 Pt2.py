# Purpose of the work: Design and implementation of an RBF network with Python to estimate body fat percentage based on various body measurements provided in the bodyfat dataset
# About the dataset(bodyfat): Each input feature is a 252x1 vector; Stored attributes: age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist


# Credits: Morteza Farrokhnejad, Ali Farrokhnejad
# Based on the original work by Prof. Dr. Ahmet Rizaner

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# Vectorized RBF function
def rbf(x, centers, spread):
    return np.exp(-np.linalg.norm(x[:, np.newaxis] - centers, axis=2)**2 / (2 * spread**2))

class RBFRidgeEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, n_centers=10, spread=3.0, alpha=1.0, l1_ratio=0.5):
        self.n_centers = n_centers
        self.spread = spread
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scaler = StandardScaler()
        self.feature_selector = SelectPercentile(f_regression, percentile=80)
        self.model = None
        
    def get_params(self, deep=True):
        return {
            'n_centers': self.n_centers,
            'spread': self.spread,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio
        }
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
        
    def fit(self, X, y):
        # Data preprocessing
        X_scaled = self.scaler.fit_transform(X)
        
        # Find RBF centers using GMM
        gmm = GaussianMixture(n_components=self.n_centers, random_state=42)
        gmm.fit(X_scaled)
        self.centers_ = gmm.means_
        
        # Generate RBF features
        X_rbf = rbf(X_scaled, self.centers_, self.spread)
        
        # Feature selection
        X_selected = self.feature_selector.fit_transform(X_rbf, y)
        
        # Configure and train regression model
        self.model = ElasticNetCV(
            alphas=[self.alpha],
            l1_ratio=[self.l1_ratio],
            cv=5,
            max_iter=10000,
            random_state=42
        )
        self.model.fit(X_selected, y)
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_rbf = rbf(X_scaled, self.centers_, self.spread)
        X_selected = self.feature_selector.transform(X_rbf)
        return self.model.predict(X_selected)

# Load and prepare data
dataset = 'PATH TO DATASET GOES HERE'
data = np.genfromtxt(dataset, delimiter=',')
X = data[:-1, :].T  # Input features (13 dimensions)
y = data[-1, :]      # Target values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Configure grid search
param_grid = {
    'n_centers': [12, 15, 18],      # Finer center count
    'spread': [2.5, 3.0, 3.5],      # Intermediate spreads
    'alpha': [0.05, 0.1, 0.2],      # Weaker regularization
    'l1_ratio': [0.85, 0.9, 0.95]   # High L1 focus
}

grid_search = GridSearchCV(
    RBFRidgeEstimator(),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Execute grid search
grid_search.fit(X_train, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)
print(f"Train MSE: {mean_squared_error(y_train, best_model.predict(X_train)):.2f}")
print(f"Test MSE: {mean_squared_error(y_test, best_model.predict(X_test)):.2f}")

# Generate predictions
y_pred = best_model.predict(X)

# Regression plot
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.6, edgecolors='w')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Body Fat Percentage')
plt.xlabel('Actual Body Fat (%)')
plt.ylabel('Predicted Body Fat (%)')
plt.grid(True)
plt.show()

# Residual analysis
residuals = y_test - best_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, alpha=0.6)
plt.hlines(0, y.min(), y.max(), colors='r')
plt.title('Residual Analysis')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()


# TODo:
# 1. More hyperparameter tuning
# 2. Feature engineering
# 3. More advanced methods