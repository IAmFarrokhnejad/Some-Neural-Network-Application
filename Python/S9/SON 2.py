#Purpose of the work: Demonstration and visualization of the training process of a self-organizing map (SOM) using a simple 1-dimensional neighborhood function.
#The code generates a set of weights that adaptively learn the structure of input data points, which are randomly sampled from a triangular distribution. By iterating through a set number of iterations and adjusting the weights based on the input samples and a dynamically changing learning rate and neighborhood size, the code showcases the convergence of the weights and their spatial organization over time.


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt

def neigh1d(indx, tot, rds): #Computes the 1-dimensional neighborhood indices for a given index "indx" within a total of "tot" units and a neighborhood radius "rds".
    all_nb = np.arange(indx - rds, indx + rds + 1)
    inx_nb = (all_nb >= 0) & (all_nb < tot)
    nb = all_nb[inx_nb]
    return nb

def plotmap1d(W, iter): #Plots the current state of the weight vectors "W" at a given iteration "iter".
    plt.clf()
    plt.plot(W[0, :], W[1, :], 'ob')
    plt.xlabel('W(0,i)')
    plt.ylabel('W(1,i)')
    plt.title(f'Iteration = {iter}')
    plt.show()

def trig_ex(): #Main function that initializes the weights, defines the training parameters, and iteratively updates the weights based on random input samples; Adapts the learning rate and neighborhood radius over time to facilitate convergence; Plots the weight vectors at specified intervals to visualize the training process.
    h = 2  # Height of the triangle
    w = 1  # Half width of the triangle
    no_iter = 50000  # Number of iterations needed
    Disp_Rate_1 = 250  # Display rate 1
    Disp_Rate_2 = 5000  # Display rate 2
    N_I = 2  # Number of input
    N_N = 65  # Number of neurons
    N_A = 1  # Neighbourhood area, radius
    ETHA = 0.9  # Learning rate

    # Initializing the weights
    W = 2 * np.random.rand(N_I, N_N) - 1

    figcnt = 1

    for iter in range(1, no_iter + 1):
        # Input vector
        x2 = 2 * np.random.rand()
        x1 = (w - (w / h) * x2) * np.random.rand() * np.sign(np.random.rand() - np.random.rand())
        P = np.array([x1, x2])

        # The learning rate ETHA
        ETHA = 1 / (0.005 * iter + 1)
        if iter > 10000:
            ETHA = 0.01

        # The Neighbourhood functions width, radius
        if iter > 22000:
            N_A = 0
        elif iter > 10000:
            N_A = 1
        elif iter > 5000:
            N_A = 2
        elif iter > 1000:
            N_A = 3
        else:
            N_A = 4

        # Finding the minimum distance
        ii = np.sum((P.reshape(-1, 1) - W) ** 2, axis=0)
        n = np.argmin(ii)

        # Calculate the neighbors of the winning unit
        neigh = neigh1d(n, N_N, N_A)

        # Adjusting the weights of the winning neuron and its neighbourhoods
        for k in neigh:
            W[:, k] += ETHA * (P - W[:, k])

        # Plotting decision boundaries
        if iter < 5 * Disp_Rate_1:
            if iter % Disp_Rate_1 == 0 or iter == 1:
                plt.figure(figcnt)
                plotmap1d(W, iter)
                figcnt += 1
        else:
            if iter % Disp_Rate_2 == 0:
                plt.figure(figcnt)
                plotmap1d(W, iter)
                figcnt += 1

        # Store some examples (optional, for future use)
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

if __name__ == "__main__":
    trig_ex()