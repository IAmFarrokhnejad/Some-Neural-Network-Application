#Purpose of the work: Implementation of a recognition system for handwritten digits and analysis of the impact of different hyperparameters on the performance of the model(classification using an MLP Neural Network)
#About the dataset(MNIST): The dataset consists of 60000 training images and 10000 testing images of handwritten digits, each labeled with its corresponding digit from 0 to 9. The images are 28x28 grayscale pixels, representing the pixel intensities of the handwritten digits.


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner

import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# # Predefined functions:
# # mkvec, mvcec, pchar, mbplr
def mkvec(matrix):
    return matrix.flatten()

def mvcec(matrix):
    return np.fliplr(matrix.reshape(matrix.shape[0], -1)).flatten()

def pchar(matrix, rows, cols, inverted=False):
    if inverted:
        matrix = -matrix
    reshaped_matrix = matrix.reshape(rows, cols)
    plt.imshow(reshaped_matrix, cmap='gray')
    plt.show()

def mbplr(matrix):
    return np.fliplr(matrix)

# # train_data_import
# Specify dataset folder
folder = 'PATH/TO/FOLDER'
images_tr = []
labels_tr = []

# Import training images
for dgt in range(10):
    filelist = os.listdir(os.path.join(os.getcwd(), folder, str(dgt)))
    for smp in range(len(filelist)):
        fullFileName = os.path.join(folder, str(dgt), filelist[smp])
        img = cv2.imread(fullFileName, cv2.IMREAD_GRAYSCALE)
        images_tr.append(img.flatten())
        dgt_mat = np.zeros((10, 1))
        dgt_mat[dgt] = 1
        labels_tr.append(dgt_mat.flatten())

images_tr = np.array(images_tr).T / 255 # Normalize inputs to be in range [0,1]
labels_tr = np.array(labels_tr).T

# # test_data_import
# Specify dataset folder
folder = 'PATH/TO/FOLDER'
images_tst = []
labels_tst = []

# Import testing images
for dgt in range(10):
    filelist = os.listdir(os.path.join(os.getcwd(), folder, str(dgt)))
    for smp in range(len(filelist)):
        fullFileName = os.path.join(folder, str(dgt), filelist[smp])
        img = cv2.imread(fullFileName, cv2.IMREAD_GRAYSCALE)
        images_tst.append(img.flatten())
        dgt_mat = np.zeros((10, 1))
        dgt_mat[dgt] = 1
        labels_tst.append(dgt_mat.flatten())

images_tst = np.array(images_tst).T / 255 # Normalize inputs to be in range [0,1]
labels_tst = np.array(labels_tst).T

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(images_tr.shape[0],)),
    tf.keras.layers.Dense(100, activation='tanh'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define training parameters
numEpochs = 50
batchSize = 500

# Training loop
for epoch in range(numEpochs):
    # Shuffle data
    indices = np.arange(images_tr.shape[1])
    np.random.shuffle(indices)
    images_tr_shuffled = images_tr[:, indices]
    labels_tr_shuffled = labels_tr[:, indices]

    # Train the model on batches
    for batchNum in range(0, images_tr.shape[1], batchSize):
        endIdx = min(batchNum + batchSize, images_tr.shape[1])
        images_batch = images_tr_shuffled[:, batchNum:endIdx]
        labels_batch = labels_tr_shuffled[:, batchNum:endIdx]
        model.train_on_batch(images_batch.T, labels_batch.T)

# Evaluate the trained model on the training set
results_tr = model.predict(images_tr.T).argmax(axis=1)

# Plot confusion matrix for the training set
cmtr = confusion_matrix(labels_tr.T.argmax(axis=1), results_tr)
cmtrDisp = ConfusionMatrixDisplay(confusion_matrix=cmtr)
cmtrDisp.plot()
plt.title('Confusion Matrix - Training Set')

# Evaluate the trained model on the testing set
results_tst = model.predict(images_tst.T).argmax(axis=1)

# Plot confusion matrix for the testing set
cmtst = confusion_matrix(labels_tst.T.argmax(axis=1), results_tst)
cmtstDisp = ConfusionMatrixDisplay(confusion_matrix=cmtst)
cmtstDisp.plot()
plt.title('Confusion Matrix - Testing Set')

# Calculate accuracies
c_tr = np.sum(cmtr.diagonal()) / np.sum(cmtr)
c_tst = np.sum(cmtst.diagonal()) / np.sum(cmtst)

print(f"Training Accuracy: {c_tr}")
print(f"Testing Accuracy: {c_tst}")

# Show plots
plt.show()