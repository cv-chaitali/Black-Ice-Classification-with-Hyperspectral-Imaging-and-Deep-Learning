"""This module is for dataset size or shape change ;
 plotting the channel's images ; visualising ground truth.
In our case we needed
 to rotate the images so, one can 
 remove the "this one" line if your
 images are in the right direction"""

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
# import modifiedwithacc
from operator import truediv

from plotly.offline import init_notebook_mode

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral


## GLOBAL VARIABLES
dataset = 'BI'
test_ratio = 0.7
windowSize = 25

def loadData(name):
    if name == 'BI':
        data = np.load(r'C:\Users\10meter_original.npy')
        labels = np.load(r'C:\Users\sourceGT_corrfoprtria;.npy')
    return data,labels

X,y = loadData(dataset)

X_new  = np.rot90(X)

# data visualisation of the original dataset

import numpy as np
import matplotlib.pyplot as plt

def plot_band(X_new):
    plt.figure(figsize=(10,10))
    band_no = np.random.randint(X_new.shape[2])
    plt.imshow(X_new[:,:, band_no], cmap='viridis')
    plt.title(f'Band-{band_no}', fontsize=14)
    plt.axis('off')
    plt.colorbar()


    plt.show()
plot_band(X_new)

#data visualisation of the ground truth 

plt.figure(figsize=(8, 6))
plt.imshow(y, cmap='viridis')
plt.axis('off')
plt.colorbar(ticks= range(0,16))
plt.show()
