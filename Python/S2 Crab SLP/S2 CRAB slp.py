#Purpose of the work: identification of the sex of crabs based on physical dimensions.(classification using an SLP Neural Network)
#Physical characteristics included in the dataset: species, frontallip, rearwidth, length, width, and depth

#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=1000, error_goal=0.02):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.error_goal = error_goal
        self.weights = None
        self.bias = 0.0
        self.training_errors = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.max_epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi.reshape(1, -1))
                if prediction != target:
                    update = self.lr * (target - prediction)
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
                    
            error_rate = errors / n_samples
            self.training_errors.append(error_rate)
            
            if error_rate <= self.error_goal:
                print(f"Early stopping at epoch {epoch+1} with error {error_rate:.4f}")
                break
                
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1} - Error: {error_rate:.4f}")
                
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0).flatten()

# Load and prepare data
data = np.genfromtxt('Path to dataset goes here', delimiter=',') # CHANGE THIS PATH AND TEST THE CODE LATER
X = data[:-1, :].T  # Samples as rows, features as columns
T = data[-1, :]     # 1D array of targets

# Split dataset
X_train, X_test, T_train, T_test = train_test_split(
    X, T, test_size=0.15, random_state=42
)

# Initialize and train perceptron
perceptron = Perceptron(learning_rate=0.01, max_epochs=1000, error_goal=0.02)
perceptron.fit(X_train, T_train)

# Training set evaluation
Y_train_pred = perceptron.predict(X_train)
cm_train = confusion_matrix(T_train, Y_train_pred)

# Testing set evaluation
Y_test_pred = perceptron.predict(X_test)
cm_test = confusion_matrix(T_test, Y_test_pred)

# Plot confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(cm_train, display_labels=['Male', 'Female']).plot(ax=ax1)
ax1.set_title('Training Set Confusion Matrix')

ConfusionMatrixDisplay(cm_test, display_labels=['Male', 'Female']).plot(ax=ax2)
ax2.set_title('Testing Set Confusion Matrix')

plt.tight_layout()
plt.show()

print("Test Set Confusion Matrix:")
print(cm_test)