#Purpose of the work: Introduction to image processign with Python and demonstration of some useful functions(pchar, mbplr, mkvec, mvcec, corrupt)
#Supported image fromats: BMP(Windows Bitmap), GIF(Graphics Interchange Format), JPEG(Joint Photographic Experts Group),  PNG(Portable Network Graphics), TIFF(Tagged Image File Format)

#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
import glob

# Demonstration of the predefined functions:

def pchar(matrix, rows, cols):
    reshaped_matrix = matrix.reshape(rows, cols)
    plt.imshow(reshaped_matrix, cmap='gray')
    plt.show()      

def mbplr(matrix):
    return (2*matrix)-1

def mkvec(matrix):
    return matrix.flatten()

def mvcec(matrix):
    return np.fliplr(matrix.reshape(matrix.shape[0], -1)).flatten()

def corrupt(matrix, percentage):
    num_corrupted = int((percentage / 100) * matrix.size)
    indices = np.random.choice(matrix.size, num_corrupted, replace=False)
    matrix_flat = matrix.flatten()
    matrix_flat[indices] = (matrix_flat[indices]*-1).astype("int16")
   
    return matrix_flat.reshape(matrix.shape)

# Show PT.bmp 
pt = imageio.imread('PT.bmp')
plt.imshow(pt)
plt.show()

# Show pattern defined by a matrix
x = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=np.int8)
pchar(x, 4, 3)

# 
x1 = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]], dtype=np.int8)
x2 = mbplr(x1)
x3 = mkvec(x2)
xp = mvcec(x2)
pchar(xp, 5, 5)

plt.imshow(x1, cmap='gray')
plt.show()
plt.imshow(mbplr(x1), cmap='gray')
plt.show()
plt.imshow(-mbplr(x1), cmap='gray')
plt.show()

x4 = corrupt(x3, 10)
pchar(x4, 5, 5)

c2 = imageio.imread('sym2.bmp')
plt.imshow(c2)
plt.show()

c1 = imageio.imread('sym1.bmp')
c1v = mkvec(c1)
c1b = -mbplr(c1)
pchar(c1b, 80, 80)

c3 = imageio.imread('sym3.bmp')
c3v = mkvec(c3)
c3b = -mbplr(c3)
c3c = corrupt(c3b, 10)
pchar(c3c, 80, 80)
plt.show()

image = imageio.imread('monalisa.png')
plt.imshow(image)
plt.show()
binimage = np.array(image > 127, dtype=np.int8)
plt.imshow(binimage, cmap='gray')
plt.show()

image2 = imageio.imread('lion.png')
igray = np.array(Image.fromarray(image2).convert('L'))
ibw = np.array(igray > 127, dtype=np.uint8)
imagen = image2 + 2 * np.random.randn(*image2.shape)
imagen = np.array(imagen, dtype=np.uint8)
plt.imshow(imagen)
plt.show()

folder = 'mnist_ar\\testing\\'
images_tst = []
labels_tst = []

for dgt in range(10):
    filelist = glob.glob(os.path.join(folder, str(dgt), '*.png'))
    for fullFileName in filelist:
        img = np.array(Image.open(fullFileName))
        images_tst.append(img.flatten())
        dgt_mat = np.zeros(10)
        dgt_mat[dgt] = 1
        labels_tst.append(dgt_mat)

images_tst = np.array(images_tst)
labels_tst = np.array(labels_tst)