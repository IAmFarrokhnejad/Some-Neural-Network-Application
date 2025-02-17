#Purpose of the work:classifying trucks given their masses and lengths.(simple classification using an SLP Neural Network)


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

i = np.array([[8, 10, 6, 5, 15, 2, 20, 2],
             [5, 6, 2, 2, 6, 4, 4, 3]])
t = np.array([[1], [1], [0], [0], [1], [0], [1], [0]])

# Create a neural network
net = nl.net.newp([[np.min(i[0]), np.max(i[0])], [np.min(i[1]), np.max(i[1])]], 1)

# Adapt the network to the first set of inputs
net.train(i.T,t)

# Display the weights and bias of the network
print("Weights:", net.layers[0].np['w'])
print("Bias:", net.layers[0].np['b'])

# Update targets - use bipolar representation
i = np.array([[8, 10, 6, 5, 15, 2, 20, 2],
             [5, 6, 2, 2, 6, 4, 4, 3]])
t = np.array([[1], [1], [-1], [-1], [1], [-1], [1], [-1]])

# Create a neural network
net = nl.net.newp([[np.min(i[0]), np.max(i[0])], [np.min(i[1]), np.max(i[1])]], 1, transf=nl.net.trans.HardLims())

# Train the network
net.train(i.T, t)

# Organize plot data
x = i[0]
y = i[1]
labels = t.T[0]

# Separate data points based on labels
positive_points = np.column_stack((x[labels == 1], y[labels == 1]))
negative_points = np.column_stack((x[labels == -1], y[labels == -1]))

# Scatter plot for positive and negative points
plt.scatter(positive_points[:, 0], positive_points[:, 1], c='blue', marker='o', label='Class 1')
plt.scatter(negative_points[:, 0], negative_points[:, 1], c='red', marker='x', label='Class -1')

# For decision boundary
weights = net.layers[0].np['w']
bias = net.layers[0].np['b']

# # What is this formula?!?!?!
x_decision = np.linspace(np.min(x), np.max(x), 100)
y_decision = -(weights[0][0] * x_decision + bias) / weights[0][1]


# Plot decision boundary
plt.plot(x_decision, y_decision, label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()