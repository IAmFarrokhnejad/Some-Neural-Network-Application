#Purpose of the work: identification of the sex of crabs based on physical dimensions.(classification using an MLP Neural Network)
#Physical characteristics included in the dataset: species, frontallip, rearwidth, length, width, and depth


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner

import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore') # Ignore warnings for float overflow in exponent when training

# Import data from file; SPECIFY PATH TO DATASET
data = np.genfromtxt('PATH\\TO\\DATASET', delimiter=',')
X = data[:-1, :].T # Each row contains one data sample
T = data[-1, :].reshape(-1, 1) # Column vector of classes (each row corresponds to respective row in X)

# 1 - Simple feedforward network
# Create 6-20-1 feedforward neural network
net = nl.net.newff(np.column_stack((X.min(axis=0), X.max(axis=0))), [20, 1], [nl.net.trans.LogSig(), nl.net.trans.PureLin()])

# Separate training and testing data
Xtr = X[:170, :]
Ttr = T[:170, :]
Xts = X[-30:, :]
Tts = T[-30:, :]

# Specify training parameters and train network 
net.train(Xtr, Ttr, epochs=500, show=10, goal=1e-10)

# Evaluate trained network on the training set
Ytr = net.sim(Xtr)
Ytr_bin = Ytr>=np.mean(Ytr) # Map outputs to binary values

# Plot confusion matrix for the training set
cmtr = confusion_matrix(Ttr, Ytr_bin)

cmtrDisp = ConfusionMatrixDisplay(confusion_matrix=cmtr)
cmtrDisp.plot()
plt.title('Confusion Matrix - Training Set')

# Evaluate the trained network on the testing set   
Yts = net.sim(Xts)
Yts_bin = Yts>=np.mean(Yts) # Map outputs to binary values

# Plot confusion matrix for the testing set
cmts = confusion_matrix(Tts, Yts_bin)
cmtsDisp = ConfusionMatrixDisplay(confusion_matrix=cmts)
cmtsDisp.plot()
plt.title('Confusion Matrix - Testing Set')

print(f'Confusion matrix:\n{cmts}')

# Show plots
plt.show()

# 2 = More hidden layers
# Create 6-20-20-1 feedforward neural network
net = nl.net.newff(np.column_stack((X.min(axis=0), X.max(axis=0))), [20, 20, 1], [nl.net.trans.LogSig(),
                                                                              nl.net.trans.LogSig(),
                                                                              nl.net.trans.PureLin()])
# Separate training and testing data
Xtr = X[:170, :]
Ttr = T[:170, :]
Xts = X[-30:, :]
Tts = T[-30:, :]

# Specify training parameters and train network 
net.train(Xtr, Ttr, epochs=500, show=10, goal=1e-10)

# Evaluate trained network on the training set
Ytr = net.sim(Xtr)
Ytr_bin = Ytr>=np.mean(Ytr) # Map outputs to binary values

# Plot confusion matrix for the training set
cmtr = confusion_matrix(Ttr, Ytr_bin)

cmtrDisp = ConfusionMatrixDisplay(confusion_matrix=cmtr)
cmtrDisp.plot()
plt.title('Confusion Matrix - Training Set')

# Evaluate the trained network on the testing set   
Yts = net.sim(Xts)
Yts_bin = Yts>=np.mean(Yts) # Map outputs to binary values

# Plot confusion matrix for the testing set
cmts = confusion_matrix(Tts, Yts_bin)
cmtsDisp = ConfusionMatrixDisplay(confusion_matrix=cmts)
cmtsDisp.plot()
plt.title('Confusion Matrix - Testing Set')

print(f'Confusion matrix:\n{cmts}')

# Show plots
plt.show()

# 3 = 
# Create 6-20-1 feedforward neural network
net = nl.net.newff(np.column_stack((X.min(axis=0), X.max(axis=0))), [20, 1], [nl.net.trans.LogSig(),
                                                                              nl.net.trans.PureLin()])
# Separate training and testing data
Xtr = X[:170, :]
Ttr = (T[:170, :]*0.8)+0.1
Xts = X[-30:, :]
Tts = (T[-30:, :]*0.8)+0.1

# Specify training parameters and train network 
net.train(Xtr, Ttr, epochs=500, show=10, goal=1e-10)

# Evaluate trained network on the training set
Ytr = net.sim(Xtr)
Ytr_bin = Ytr>=np.mean(Ytr) # Map outputs to binary values
Ttr_bin = Ttr>0.1
# Plot confusion matrix for the training set
cmtr = confusion_matrix(Ttr_bin, Ytr_bin)

cmtrDisp = ConfusionMatrixDisplay(confusion_matrix=cmtr)
cmtrDisp.plot()
plt.title('Confusion Matrix - Training Set')

# Evaluate the trained network on the testing set   
Yts = net.sim(Xts)
Yts_bin = Yts>=np.mean(Yts) # Map outputs to binary values
Tts_bin = Tts>0.1 #Map targets to binary values

# Plot confusion matrix for the testing set
cmts = confusion_matrix(Tts_bin, Yts_bin)
cmtsDisp = ConfusionMatrixDisplay(confusion_matrix=cmts)
cmtsDisp.plot()
plt.title('Confusion Matrix - Testing Set')

print(f'Confusion matrix:\n{cmts}')

# Show plots
plt.show()