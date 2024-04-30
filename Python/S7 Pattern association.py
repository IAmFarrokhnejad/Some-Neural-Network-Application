#Purpose of the work: development of an auto-associative memory neural network capable of storing and retrieving the five characters: S, P, E, U, and T. Testing and training the patterns is done using Hebb rule.
#Technical information: The network should accept a 5x3 input array, convert it into a 15-element vector, and apply the Hebb rule to update the weight matrix. The trained network should then be used in a test mode, where the input array is presented without updating the weights. The network's output should be displayed as a 5x3 array for comparison with the original input.


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt

def pchar(pattern, rows, cols):
    pattern_matrix = np.array(pattern).reshape(rows, cols)
    plt.imshow(-pattern_matrix, cmap="gray", interpolation="nearest")
    
def verpat(inpat, weight):
    # Plot the input pattern and network output together
    plt.subplot(1, 2, 1)
    pchar(inpat, 5, 3)
    plt.title('Input')
    plt.subplot(1, 2, 2)
    pchar(np.sign(inpat @ weight), 5, 3)
    plt.title('Output')
    plt.show()
    plt.close()

# Define patterns (1 for black pixels, -1 for white pixels)
S = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
P = np.array([1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1])
E = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1])
U = np.array([1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1])
T = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1])

# Display patterns
for i, pattern in enumerate([S, P, E, U, T]):
    plt.subplot(1, 5, i+1)
    pchar(pattern, 5, 3)
plt.show()
plt.close()

# Calculate weight matrices

Ws = np.outer(S, S)
Wp = np.outer(P, P)
We = np.outer(E, E)
Wu = np.outer(U, U)
Wt = np.outer(T, T)
W = Ws + Wp + We + Wu + Wt

# Display network outputs for S and P patterns
for i, pattern in enumerate([S, P]):
    plt.subplot(1,2, i+1)
    pchar(np.sign(pattern @ W), 5, 3)
plt.show()
plt.close()

# Display network outputs for S pattern using Ws, Wp, We, and Wu
W = Ws + Wp + We + Wu
pchar(np.sign(S @ W), 5, 3)
plt.show()
plt.close()

# Define function to corrupt a pattern
def corrupt(pattern, percent):
    corrupted_pattern = np.copy(pattern)
    indices_to_flip = np.random.choice(len(pattern), int(np.ceil(percent*len(pattern)/100)))
    corrupted_pattern[indices_to_flip] *= -1
    return corrupted_pattern

# Display network outputs for T pattern before and after corruption
verpat(T, W)
verpat(corrupt(T, 10), W)