#Purpose of the work: identification of the sex of crabs based on physical dimensions.(classification using an SLP Neural Network)
#Physical characteristics included in the dataset: species, frontallip, rearwidth, length, width, and depth


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore') # Ignore warnings for float overflow in exponent when training

# Import data from file; SPECIFY PATH TO DATASET
data = np.genfromtxt('PATH\\TO\\DATA', delimiter=',')
X = data[:-1, :].T # Each row contains one data sample
T = data[-1, :].reshape(-1, 1) # Column vector of classes (each row corresponds to respective row in X) 

# Randomly split the data into training and testing sets
Xtr, Xts, Ttr, Tts = train_test_split(X, T, test_size=0.15, random_state=42)

# Create a perceptron neural network
net = nl.net.newp(np.column_stack((Xtr.min(axis=0), Xtr.max(axis=0))), 1)

# Specify training parameters and train the network
error = net.train(Xtr, Ttr, epochs=1000, show=10, goal=2e-2, lr=0.01)

# Evaluate the trained network on the training set
Ytr = net.sim(Xtr)

# Plot confusion matrix for the training set
cmtr = confusion_matrix(Ttr, Ytr)
cmtrDisp = ConfusionMatrixDisplay(confusion_matrix=cmtr)
cmtrDisp.plot()
plt.title('Confusion Matrix - Training Set')

# Evaluate the trained network on the testing set   
Yts = net.sim(Xts)

# Plot confusion matrix for the testing set
cmts = confusion_matrix(Tts, Yts)
cmtsDisp = ConfusionMatrixDisplay(confusion_matrix=cmts)
cmtsDisp.plot()
plt.title('Confusion Matrix - Testing Set')

print(f'Confusion matrix:\n{cmts}')

# Show plots
plt.show()