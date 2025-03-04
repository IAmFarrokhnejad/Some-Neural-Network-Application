#Purpose of the work:classifying trucks given their masses and lengths.(simple classification using an SLP Neural Network)


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt

# Perceptron training function
def train_perceptron(samples, targets, activation_fn, max_epochs=1000):
    n_features = samples.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0
    learning_rate = 1.0
    errors = 1
    epochs = 0
    
    while errors > 0 and epochs < max_epochs:
        errors = 0
        for i in range(len(samples)):
            x = samples[i]
            target = targets[i]
            net_input = np.dot(weights, x) + bias
            output = activation_fn(net_input)
            
            if output != target:
                errors += 1
                update = learning_rate * (target - output)
                weights += update * x
                bias += update
        epochs += 1
    
    return weights, bias

# Prepare data
i = np.array([[8, 10, 6, 5, 15, 2, 20, 2],
             [5, 6, 2, 2, 6, 4, 4, 3]])
t_binary = np.array([1, 1, 0, 0, 1, 0, 1, 0])
t_bipolar = np.array([1, 1, -1, -1, 1, -1, 1, -1])

samples = i.T  # Transpose to get 8 samples with 2 features each

# First training with binary targets (0/1)
weights_bin, bias_bin = train_perceptron(
    samples, t_binary,
    activation_fn=lambda x: 1 if x >= 0 else 0
)

print("Binary Targets Training:")
print(f"Weights: {weights_bin}")
print(f"Bias: {bias_bin}\n")

# Second training with bipolar targets (-1/1)
weights_bip, bias_bip = train_perceptron(
    samples, t_bipolar,
    activation_fn=lambda x: 1 if x >= 0 else -1
)

print("Bipolar Targets Training:")
print(f"Weights: {weights_bip}")
print(f"Bias: {bias_bip}")

# Visualization
x_points = i[0]
y_points = i[1]
labels = t_bipolar

# Separate classes
positive = labels == 1
negative = labels == -1

plt.scatter(x_points[positive], y_points[positive], c='blue', marker='o', label='Class 1')
plt.scatter(x_points[negative], y_points[negative], c='red', marker='x', label='Class -1')

# Calculate decision boundary
x_decision = np.linspace(np.min(x_points), np.max(x_points), 100)
y_decision = (-weights_bip[0] * x_decision - bias_bip) / weights_bip[1]

plt.plot(x_decision, y_decision, label='Decision Boundary')

plt.xlabel('Feature 1 (Mass)')
plt.ylabel('Feature 2 (Length)')
plt.title('Truck Classification with Perceptron')
plt.legend()
plt.grid(True)
plt.show()