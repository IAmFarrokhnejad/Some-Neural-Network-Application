#Purpose of the work: Implementation of a simple self-organizing map(SOM) algorithm to visualize how weights adapt over time through unsupervised learning. 
#The code demonstrates the process of training a SOM, plotting the positions of the neurons at different iterations, and adjusting the weights based on neighborhood functions and learning rates.
#Key steps: 1.Initialization        2.Training          3.Visualization


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt

def neigh2d(index, dimensions, neighborhood_area):
    x = index // dimensions[1]
    y = index % dimensions[1]
    neighbors = []

    for dx in range(-neighborhood_area, neighborhood_area + 1):
        for dy in range(-neighborhood_area, neighborhood_area + 1):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < dimensions[0] and 0 <= ny < dimensions[1]:
                neighbors.append(nx * dimensions[1] + ny)
    return neighbors

def plotmap2d(weights, dimensions, iteration):
    plt.figure()
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            index = i * dimensions[1] + j
            plt.plot(weights[index, 0], weights[index, 1], 'bo')
    plt.title(f"Iteration {iteration}")
    plt.show()

# Initialize parameters
no_iter = 50000  # Number of iterations needed
pausetime = 0.0
Disp_Rate = 1000  # Display rate
N_I = 2  # Number of inputs
XX = 10  # X dimension
YY = 10  # Y dimension
N_N = XX * YY  # Number of neurons
N_A = 1  # Neighborhood area
ETHA = 0.9  # Learning rate
figcnt = 0  # Figure counter

W = np.random.rand(N_N, N_I)  # Initializing the weights
A0 = W.copy()

# The program logic
for iter in range(1, no_iter + 1):
    x1 = np.random.rand()
    x2 = np.random.rand()
    P = np.array([x1, x2])

    # The learning rate ETHA
    ETHA = 1 / (0.009 * iter + 1.1)
    if iter > 6000:
        ETHA = 0.01

    # The Neighborhood functions width
    if iter > 9000:
        N_A = 0
    elif iter > 4000:
        N_A = 1
    elif iter > 1000:
        N_A = 2
    else:
        N_A = 3

    # Finding the minimum distance
    distances = np.sum((P - W) ** 2, axis=1)
    n = np.argmin(distances)

    neigh = neigh2d(n, [XX, YY], N_A)

    # Adjusting the weights of the winning neuron and its neighborhoods
    for k in neigh:
        W[k, :] += ETHA * (P - W[k, :])

    # Plotting decision boundaries
    if iter % Disp_Rate == 0:
        figcnt += 1
        plt.pause(pausetime)
        plotmap2d(W, [XX, YY], iter)

    # Store some examples
    if iter == 50:
        A1 = W.copy()
    elif iter == 1000:
        A2 = W.copy()
    elif iter == 5000:
        A3 = W.copy()
    elif iter == 10000:
        A4 = W.copy()
    elif iter == 20000:
        A5 = W.copy()
    elif iter == 50000:
        A6 = W.copy()