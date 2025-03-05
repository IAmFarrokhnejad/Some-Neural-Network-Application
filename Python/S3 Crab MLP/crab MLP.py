#Purpose of the work: identification of the sex of crabs based on physical dimensions.(classification using an MLP Neural Network)
#Physical characteristics included in the dataset: species, frontallip, rearwidth, length, width, and depth


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = np.genfromtxt('path to the dataset goes here', delimiter=',') # Specify the dataset path 
X = data[:-1, :].T  # Features (6 input features)
T = data[-1, :]     # Targets (binary: 0/1)

# Common configuration for all models
def create_model(hidden_layers):
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='logistic',
        solver='sgd',
        learning_rate='constant',
        learning_rate_init=0.01,
        max_iter=500,
        tol=1e-10,
        random_state=42,
        verbose=False
    )

def evaluate_model(model, X_train, T_train, X_test, T_test, scaled=False):
    # Training set evaluation
    Y_train = model.predict(X_train)
    threshold = np.mean(Y_train)
    Y_train_bin = (Y_train >= threshold).astype(int)
    
    if scaled:
        T_train_bin = (T_train > 0.1).astype(int)
    else:
        T_train_bin = T_train
    
    cm_train = confusion_matrix(T_train_bin, Y_train_bin)
    
    # Testing set evaluation
    Y_test = model.predict(X_test)
    Y_test_bin = (Y_test >= np.mean(Y_test)).astype(int)
    
    if scaled:
        T_test_bin = (T_test > 0.1).astype(int)
    else:
        T_test_bin = T_test
    
    cm_test = confusion_matrix(T_test_bin, Y_test_bin)
    
    return cm_train, cm_test

def plot_confusion_matrices(cm_train, cm_test, title_suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ConfusionMatrixDisplay(cm_train, display_labels=['Male', 'Female']).plot(ax=ax1)
    ax1.set_title(f'Training Set {title_suffix}')
    
    ConfusionMatrixDisplay(cm_test, display_labels=['Male', 'Female']).plot(ax=ax2)
    ax2.set_title(f'Testing Set {title_suffix}')
    
    plt.tight_layout()
    plt.show()

# Split dataset (first 170 training, last 30 testing)
X_train, X_test = X[:170], X[-30:]
T_train, T_test = T[:170], T[-30:]

# 1 - Simple feedforward network (6-20-1)
print("\nPart 1: 6-20-1 Network")
model1 = create_model((20,))
model1.fit(X_train, T_train)
cm_train1, cm_test1 = evaluate_model(model1, X_train, T_train, X_test, T_test)
plot_confusion_matrices(cm_train1, cm_test1, "6-20-1 Network")
print("Test Confusion Matrix (6-20-1):\n", cm_test1)

# 2 - Deeper network (6-20-20-1)
print("\nPart 2: 6-20-20-1 Network")
model2 = create_model((20, 20))
model2.fit(X_train, T_train)
cm_train2, cm_test2 = evaluate_model(model2, X_train, T_train, X_test, T_test)
plot_confusion_matrices(cm_train2, cm_test2, "6-20-20-1 Network")
print("Test Confusion Matrix (6-20-20-1):\n", cm_test2)

# 3 - Scaled targets network (6-20-1)
print("\nPart 3: Scaled Targets Network")
# Scale targets to 0.1-0.9 range
T_train_scaled = (T_train * 0.8) + 0.1
T_test_scaled = (T_test * 0.8) + 0.1

model3 = create_model((20,))
model3.fit(X_train, T_train_scaled)
cm_train3, cm_test3 = evaluate_model(model3, X_train, T_train_scaled, 
                                   X_test, T_test_scaled, scaled=True)
plot_confusion_matrices(cm_train3, cm_test3, "Scaled Targets Network")
print("Test Confusion Matrix (Scaled Targets):\n", cm_test3)