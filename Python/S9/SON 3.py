#Purpose of the work: Implementation of a Self-Organizing Feature Map (SOFM) for color separation.

#Key components: 
    # 1.Initialization: The parameters for the SOFM, such as the number of iterations, display rates, learning rate, and neighborhood radius, are initialized. The weights of the neurons are also randomly initialized.
    # 2.Data: A set of color vectors (RGB values) is defined as the input data.
    # 3.Training Loop: The script iterates through a specified number of iterations to train the SOFM. In each iteration an input vector is selected and normalized, the learning rate is adjusted, the neighborhood radius is updated based on the current iteration, the winning neuron is identified based on the minimum distance to the input vector, the neighbors of the winning neuron are calculated, the weights of the winning neuron and its neighbors are updated.
    # 4.Plotting: During the training process, the feature map is plotted at specified intervals to visualize the evolution of the weight vectors.
    # 5.Classification: After training, the script classifies each input vector by determining the closest neuron and displays the class assignment.



#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt

def calclas(W, x, N_N):
    P = x / np.linalg.norm(x)
    ii = np.sum((P - W.T) ** 2, axis=1)
    n = np.argmin(ii) + 1
    print(f'Class of the entered vector: {n}')

def neigh1d(indx, tot, rds):
    all_nb = np.arange(indx - rds, indx + rds + 1)
    nb = all_nb[(all_nb > 0) & (all_nb <= tot)]
    return nb

def plotmap1d(W, iter):
    if W.shape[1] < 2:
        raise ValueError('W must have at least two columns.')
    plt.clf()
    plt.plot(W[0, :], W[1, :], 'bs')
    plt.xlabel('W(1,i)')
    plt.ylabel('W(2,i)')
    plt.title(f'Iteration = {iter}')
    plt.show()

# Parameters
no_iter = 5000
Disp_Rate_1 = 100
Disp_Rate_2 = 1000
N_I = 3
N_N = 3
N_A = 1
ETHA = 0.9

# Initializing the weights
W = 2 * np.random.rand(N_I, N_N) - 1
A0 = W

# Datas
xi = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [139, 123, 191],
    [44, 50, 170],
    [191, 95, 23],
    [202, 121, 12],
    [20, 194, 148],
    [84, 144, 70]
])

figcnt = 1

for iter in range(1, no_iter + 1):
    a = np.random.randint(0, 9)
    P = xi[a, :] / np.linalg.norm(xi[a, :])
    
    ETHA = 1 / (0.01 * iter + 1)
    if iter > 100:
        ETHA = 0.01

    if iter > 500:
        N_A = 0
    elif iter > 100:
        N_A = 1
    elif iter > 50:
        N_A = 2
    elif iter > 10:
        N_A = 3
    else:
        N_A = 4

    ii = np.sum((P - W.T) ** 2, axis=1)
    n = np.argmin(ii)
    
    neigh = neigh1d(n + 1, N_N, N_A) - 1

    for k in neigh:
        W[:, k] += ETHA * (P - W[:, k])

    if iter < 5 * Disp_Rate_1:
        if iter % Disp_Rate_1 == 0 or iter == 1:
            plotmap1d(W, iter)
            figcnt += 1
    else:
        if iter % Disp_Rate_2 == 0:
            plotmap1d(W, iter)
            figcnt += 1

for i in range(9):
    P = xi[i, :] / np.linalg.norm(xi[i, :])
    ii = np.sum((P - W.T) ** 2, axis=1)
    n = np.argmin(ii) + 1
    print(f'Input Vector: {i + 1} Class: {n}')