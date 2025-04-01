# Purpose of the work: Introduction to Radial Basis Function Networks for function approximation.


# Credits: Morteza Farrokhnejad, Ali Farrokhnejad
# Based on the original work by Prof. Dr. Ahmet Rizaner

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Vectorized RBF function
def rbf(x, centers, spread):
    return np.exp(-(x[:, np.newaxis] - centers)**2 / (2 * spread**2))

# Generate training data
P = np.arange(-1, 1.1, 0.1)
T = np.array([-.9602, -.5770, -.0729, .3771, .6405, .6600, .4609, .1336, -.2013, 
              -.4344, -.5000, -.3930, -.1647, .0988, .3072, .3960, .3449, .1816, 
              -.0312, -.2189, -.3201])

# Normalize data
scaler = StandardScaler()
P_normalized = scaler.fit_transform(P.reshape(-1, 1)).flatten()

# Find centers in normalized space
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(P_normalized.reshape(-1, 1))
centers = kmeans.cluster_centers_.flatten()

# Training parameters
eg = 0.02  # Error goal (sum squared error)
lr = 0.01  # Learning rate

def train_and_plot(spread, max_epochs, case_name):
    # Generate RBF features
    P_rbf = rbf(P_normalized, centers, spread)
    
    # Initialize weights and bias
    weights = np.zeros(P_rbf.shape[1])
    bias = 0.0
    sse_history = []
    
    # Manual gradient descent
    for epoch in range(max_epochs):
        # Forward pass
        y_pred = P_rbf.dot(weights) + bias
        error = y_pred - T
        sse = np.sum(error**2)
        sse_history.append(sse)
        
        # Early stopping
        if sse <= eg:
            print(f"{case_name}: Converged at epoch {epoch}")
            break
            
        # Backward pass
        grad_weights = P_rbf.T.dot(error)
        grad_bias = error.sum()
        
        # Update parameters
        weights -= lr * grad_weights
        bias -= lr * grad_bias
        
    # Generate predictions
    X_plot = np.arange(-1, 1.01, 0.01)
    X_normalized = scaler.transform(X_plot.reshape(-1, 1)).flatten()
    X_rbf = rbf(X_normalized, centers, spread)
    Y_pred = X_rbf.dot(weights) + bias
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(P, T, '+', markersize=10, label='Training Data')
    plt.plot(X_plot, Y_pred, 'r-', linewidth=2, label='Network Output')
    plt.title(f'RBF Network: {case_name} (Spread={spread})')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    plt.show()

# Case 1: Too small spread
train_and_plot(spread=0.01, max_epochs=100, case_name='Underfitting')

# Case 2: Too large spread
train_and_plot(spread=50.0, max_epochs=100, case_name='Overfitting')

# Case 3: Optimal spread
train_and_plot(spread=1.0, max_epochs=10000, case_name='Good Fit')