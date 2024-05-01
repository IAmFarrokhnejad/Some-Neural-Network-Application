#Purpose of the work: Introduction to Radial Basis Function Networks for function approximation


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import neurolab as nl

# Function to calculate radial basis values for a set of inputs
def rbf(x, centers, spread):
    res = np.zeros((x.shape[0], len(centers)))
    try:
        for j in range(x.shape[1]):
            for i, c in enumerate(centers):
                res[:, i] += np.exp(-(x[:,j].T - c)**2 / (2 * (spread**2)))
    except IndexError:    
        for i, c in enumerate(centers):
            res[:, i] += np.exp(-(x - c)**2 / (2 * (spread**2)))
    finally:
        return res

# Basic Radial Basis Transfer Function
x = np.arange(-3, 3.1, 0.1)
a = np.exp(-x**2)
plt.plot(x, a)
plt.title('Radial Basis Transfer Function')
plt.xlabel('Input p')
plt.ylabel('Output a')
plt.show()

# Radial Basis Transfer Function with different centers
a2 = np.exp(-(x - 1.5)**2)
a3 = np.exp(-(x + 2)**2)
plt.plot(x, a, 'r', label='Center 0')
plt.plot(x, a2, 'b', label='Center 1.5')
plt.plot(x, a3, 'g', label='Center -2')
plt.grid()
plt.legend()
plt.show()

# Generate training data
P = np.arange(-1, 1.1, 0.1)
T = np.array([-.9602, -.5770, -.0729, .3771, .6405, .6600, .4609, .1336, -.2013, -.4344, -.5000, -.3930, -.1647, .0988, .3072, .3960, .3449, .1816, -.0312, -.2189, -.3201])
scaler = StandardScaler()
P_normalized = scaler.fit_transform(P.reshape(-1, 1)).flatten().T

# Perform k-means clustering to choose centers
P_centers = 10
kmeans = KMeans(n_clusters=P_centers, random_state=42).fit(P.reshape(-1, 1))
centers = kmeans.cluster_centers_.flatten()

# Define training parameters 
eg = 0.02  # sum-squared error goal
lr = 0.01 # network learning rate

# # When the spread of the radial basis neurons is too low
sc = 0.01  # spread constant

# Pass inputs through hidden layer
P_rbf = rbf(P_normalized, centers, sc)

# Create SLP for the rest of the network
net = nl.net.newp([[0, 1] for _ in range(P_centers)], 1, transf= nl.net.trans.PureLin())

# Train network
net.train(P_rbf, T.reshape(-1, 1), epochs=100, show=10, goal=eg, lr=lr)

# Plot original data and network output
plt.plot(P, T, '+', label='Training Data')
X = np.arange(-1, 1.01, 0.01)
Y = net.sim(rbf(X, centers, sc))
plt.plot(X, Y, label='Network Output')
plt.legend()
plt.show()

# When the spread of the radial basis neurons is too high
sc = 50.0  # spread constant

# Pass inputs through hidden layer
P_rbf = rbf(P_normalized, centers, sc)

# Create SLP for the rest of the network
net = nl.net.newp([[0, 1] for _ in range(P_centers)], 1, transf= nl.net.trans.PureLin())

# Train network
net.train(P_rbf, T.reshape(-1, 1), epochs=100, show=10, goal=eg, lr=lr)

# Plot original data and network output
plt.plot(P, T, '+', label='Training Data')
X = np.arange(-1, 1.01, 0.01)
Y = net.sim(rbf(X, centers, sc))
plt.plot(X, Y, label='Network Output')
plt.legend()
plt.show()

# When the spread of the radial basis neurons is moderate
sc = 1.0  # spread constant

# Pass inputs through hidden layer
P_rbf = rbf(P_normalized, centers, sc)

# Create SLP for the rest of the network
net = nl.net.newp([[0, 1] for _ in range(P_centers)], 1, transf= nl.net.trans.PureLin())

# Train network
net.train(P_rbf, T.reshape(-1, 1), epochs=10000, show=10, goal=eg, lr=lr)

# Plot original data and network output
plt.plot(P, T, '+', label='Training Data')
X = np.arange(-1, 1.01, 0.01)
Y = net.sim(rbf(X, centers, sc))
plt.plot(X, Y, label='Network Output')
plt.legend()
plt.show()