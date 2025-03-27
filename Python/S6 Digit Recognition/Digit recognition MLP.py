#Purpose of the work: Implementation of a recognition system for handwritten digits and analysis of the impact of different hyperparameters on the performance of the model(classification using an MLP Neural Network)
#About the dataset(MNIST): The dataset consists of 60000 training images and 10000 testing images of handwritten digits, each labeled with its corresponding digit from 0 to 9. The images are 28x28 grayscale pixels, representing the pixel intensities of the handwritten digits.


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Utility functions
def mkvec(matrix):
    return matrix.flatten()

def mvcec(matrix):
    return np.fliplr(matrix.reshape(matrix.shape[0], -1)).flatten()

def pchar(matrix, rows, cols, inverted=False):
    if inverted:
        matrix = -matrix
    reshaped_matrix = matrix.reshape(rows, cols)
    plt.imshow(reshaped_matrix, cmap='gray')
    plt.show()

def mbplr(matrix):
    return np.fliplr(matrix)

def load_data(folder_path):
    images = []
    labels = []
    
    for digit in range(10):
        digit_path = os.path.join(folder_path, str(digit))
        for file_name in os.listdir(digit_path):
            img_path = os.path.join(digit_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).flatten()
            images.append(img)
            labels.append(digit)
    
    return np.array(images), np.array(labels)

# Load and prepare data
train_folder = 'PATH/TO/TRAIN_FOLDER'
test_folder = 'PATH/TO/TEST_FOLDER'

# Load training data
X_train, y_train = load_data(train_folder)
X_train = X_train / 255.0  # Normalize to [0, 1]

# Load testing data
X_test, y_test = load_data(test_folder)
X_test = X_test / 255.0  # Normalize to [0, 1]

# Create and train MLP classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
    activation='tanh',          # Hyperbolic tangent activation
    solver='adam',              # Adam optimizer
    alpha=0.0001,               # L2 regularization term
    batch_size=500,             # Mini-batch size
    learning_rate_init=0.001,
    max_iter=50,                # Number of epochs
    verbose=True,
    random_state=42
)

# Train the model
mlp.fit(X_train, y_train)

# Evaluate performance
def evaluate_model(model, X, y, title):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()
    
    accuracy = np.mean(y == y_pred)
    print(f"{title} Accuracy: {accuracy:.4f}")
    return accuracy

# Training set evaluation
train_acc = evaluate_model(mlp, X_train, y_train, "Training Set")

# Testing set evaluation
test_acc = evaluate_model(mlp, X_test, y_test, "Testing Set")

# Plot learning curve
plt.figure(figsize=(10, 5))
plt.plot(mlp.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()