##################################

# Data simulation

##################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from scipy.linalg import eigh
import seaborn as sn

##################################
# Linear separable dataset
##################################

# use sklearn to generate separable data
separable = False
while not separable:
    samples = make_classification(n_samples=200, n_features=5, n_informative=2, n_redundant=0)
    red = samples[0][samples[1] == 0]
    blue = samples[0][samples[1] == 1]
    separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(5)])

# split data
train_x = samples[0][0:100,:]
train_y = samples[1][0:100]
test_x = samples[0][100:200,:]
test_y = samples[1][100:200]

# Plot the simulated data
# Convert mean = 0 and sd = 1 for variable before PCA
standard_x = StandardScaler().fit_transform(train_x)
print(standard_x.shape)
# compute covariance matrix
covar_matrix = np.matmul(standard_x.T , standard_x)
print('The shape of variance matrix =', covar_matrix.shape)
# Compute eigenvalues and eigenvectors
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
sn.FacetGrid(dataframe, hue='y', size=6).map(plt.scatter,'1st_principal', '2nd_principal').add_legend()
plt.show()




##################################
# Linear non separable dataset
##################################

# the linear non separable dataset is used the one for logistic regression

# simulate a data set for a logistic regression model with 5 dimension:
# assume covariance is an identity matrix
sigma1 = np.identity(5)
# mean is an array of 0
mean1 = np.zeros(5)
# number of samples are 200
n1 = 200
# generate n gaussian distributed data points
x = np.random.multivariate_normal(mean1, sigma1, n1)

# calculate p, set regression coefficients beta as an array of 1s
p = 1/(1 + np.exp(-np.sum(x,axis = 1)))

# split the training test data by half
# simulate y by p
y = np.random.binomial(1,p,size= n1)

# split data
train_x = x[0:100,:]
train_y = y[0:100]
test_x = x[100:200,:]
test_y = y[100:200]


##################################
# Non-linear separable dataset
##################################

# use sklearn to generate separable data
separable = False
while not separable:
    samples = make_classification(n_samples=200, n_features=5, n_informative=2, n_redundant=0)
    red = samples[0][samples[1] == 0]
    blue = samples[0][samples[1] == 1]
    sign = red.copy()
    sign[sign < 0] = -1
    sign[sign >= 0] = 1
    red2 = np.square(red) * sign
    separable = any([red2[:, k].max() < blue[:, k].min() or red2[:, k].min() > blue[:, k].max() for k in range(5)])

# split data and calculate a non-linear dataset
train_x = samples[0][0:100,:]
train_y = samples[1][0:100]
test_x = samples[0][100:200,:]
test_y = samples[1][100:200]

sign1 = train_x.copy()
sign1[sign1 < 0] = -1
sign1[sign1 >= 0] = 1
train_x2 = np.square(train_x)*sign1

sign2 = test_x.copy()
sign2[sign2 < 0] = -1
sign2[sign2 >= 0] = 1
test_x2 = np.square(test_x)*sign2

# Plot the simulated data
# Convert mean = 0 and sd = 1 for variable before PCA
standard_x = StandardScaler().fit_transform(train_x)
print(standard_x.shape)
# compute covariance matrix
covar_matrix = np.matmul(standard_x.T , standard_x)
print('The shape of variance matrix =', covar_matrix.shape)
# Compute eigenvalues and eigenvectors
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
sn.FacetGrid(dataframe, hue='y', size=6).map(plt.scatter,'1st_principal', '2nd_principal').add_legend()
plt.show()

##################################
# Non-linear non separable dataset
##################################

# the non-linear non separable dataset is an expansion of the one for logistic regression

# simulate a data set for a logistic regression model with 5 dimension:
# assume covariance is an identity matrix
sigma1 = np.identity(5)
# mean is an array of 0
mean1 = np.zeros(5)
# number of samples are 200
n1 = 200
# generate n gaussian distributed data points
x = np.random.multivariate_normal(mean1, sigma1, n1)

# calculate p, set regression coefficients beta as an array of 1s and p to be non-linear by beta*x^2
sign = x.copy()
sign[sign < 0] = -1
sign[sign >= 0] = 1
x2 = np.square(x)*sign

p = 1/(1 + np.exp(-np.sum(x2,axis = 1)))

# split the training test data by half
# simulate y by p
y = np.random.binomial(1,p,size= n1)

# split data
train_x = x[0:100,:]
train_y = y[0:100]
test_x = x[100:200,:]
test_y = y[100:200]