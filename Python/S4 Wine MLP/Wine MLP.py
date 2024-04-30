#Purpose of the work: Classification of wines(with generalization to ensure the accuracy of classification).(classification using an MLP Neural Network)
#Dataset includes a selection of Italian wines from the same region, made from three different grape varieties.
#Attributes included:Alcohol, Malic acid, Ash, Alkalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanidins, Color intensity, Hue, OD280/OD315 of diluted wines, Proline

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
X = data[:-3, :].T # Each row contains one data sample
T = data[-3:, :].reshape(3,178).T # Target classes (each row corresponds to respective row in X)

# Adjust target values to 0.1 and 0.9
T = 0.8 * T + 0.1

# First part - 13-20-3 feedforward neural network

# Shuffle data
rinx = np.random.permutation(178)
xt = X[rinx, :]
tt = T[rinx, :]

# Create neural network
net = nl.net.newff(np.column_stack((X.min(axis=0), X.max(axis=0))), [20, 3], [nl.net.trans.LogSig(), nl.net.trans.PureLin()])

# Split data into training and testing sets
Xtr = xt[:150, :]
Ttr = tt[:150, :]
Xts = xt[150:, :]
Tts = tt[150:, :]

# Configure neural network parameters
net.train(Xtr, Ttr, epochs=1000, show=10, goal=1e-7)

# Evaluate trained network on the training set
Ytr = net.sim(Xtr)

# Plot confusion matrix for the training set
cmtr = confusion_matrix(Ttr.argmax(axis=1), Ytr.argmax(axis=1))
cmtrDisp = ConfusionMatrixDisplay(confusion_matrix=cmtr)
cmtrDisp.plot()
plt.title('Confusion Matrix - Training Set')

# Print results
print(f'Training confusion matrix:\n{cmtr}')

c = (sum(sum(cmtr))-sum(cmtr.diagonal()))/sum(sum(cmtr)) # Wrong classifications
print(f'Percentage Correct Classification: {100 * (1-c):.2f}%')

# Evaluate the trained network on the testing set   
Yts = net.sim(Xts) 

# Plot confusion matrix for the testing set
cmts = confusion_matrix(Tts.argmax(axis=1), Yts.argmax(axis=1))
cmtsDisp = ConfusionMatrixDisplay(confusion_matrix=cmts)
cmtsDisp.plot()
plt.title('Confusion Matrix - Testing Set')

# Print results
print(f'Testing confusion matrix:\n{cmts}')

c = (sum(sum(cmts))-sum(cmts.diagonal()))/sum(sum(cmts)) #Wrong classifications
print(f'Percentage Correct Classification: {100 * (1-c):.2f}%')

# Show plots
plt.show()