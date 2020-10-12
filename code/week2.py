##################################

# Data visualisation

##################################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# use sklearn for PCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

## %matplotlib inline
import matplotlib.pyplot as plt

### Use the dataset for previous logistic regression first
# assume covariance is an identity matrix
sigma1 = np.identity(5)
# mean is an array of 0
mean1 = np.zeros(5)
# number of samples are 200
n1 = 200
# generate n gaussian distributed data points
x = np.random.multivariate_normal(mean1, sigma1, n1)
# split the training test data by half
# simulate y by p = 0.5
y = np.random.choice([0, 1], size=n1, p=[.5, .5])
# split data
train_x = x[0:100,:]
train_y = y[0:100]
test_x = x[100:200,:]
test_y = y[100:200]

# Plot the simulated data
# Convert mean = 0 and sd = 1 for variable before PCA
from sklearn.preprocessing import StandardScaler
standard_x = StandardScaler().fit_transform(train_x)
print(standard_x.shape)
# compute covariance matrix
covar_matrix = np.matmul(standard_x.T , standard_x)
print('The shape of variance matrix =', covar_matrix.shape)
# Compute eigenvalues and eigenvectors
from scipy.linalg import eigh
values, vectors = eigh(covar_matrix, eigvals=(3,4))
print('Shape of eigenvectors =',vectors.shape)
# converting the eigenvectors into (2,d) shape for easyness of further computations
vectors = vectors.T
print('Updated shape of eigenvectors =',vectors.shape)
# project the data by eigenvectors
new_coordinates = np.matmul(vectors, standard_x.T)
# appending y results
new_coordinates = np.vstack((new_coordinates, train_y)).T
dataframe = pd.DataFrame(data=new_coordinates, columns=('1st_principal', '2nd_principal', 'y'))
print(dataframe.head())
# plotting
import seaborn as sn
sn.FacetGrid(dataframe, hue='y', size=6).map(plt.scatter,'1st_principal', '2nd_principal').add_legend()
plt.show()

# the input of my previous logistic regression is a 5-dimensional input, so PCA can be applied
pca = PCA()
# set dimension wanted
pca.n_components = 2
# PCA for dimension reduction
pca_result = pca.fit_transform(train_x)
# test percentage
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
cum_var_explained = np.cumsum(percentage_var_explained)




##################################

# Data simulation

##################################



##################################

# Example data and benchmark

##################################



##################################

# Interpret the output

##################################