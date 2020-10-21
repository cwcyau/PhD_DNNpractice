######################################

# 2. Bayesian NNs on logistic regression

######################################

# A simple example for regression

# Importing the necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.autograd import Variable

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


# Creating our Logistic regression class with bayesian layers NN

@variational_estimator
class BayesianLogistic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 100)
        self.blinear2 = BayesianLinear(100, 20)
        self.blinear3 = BayesianLinear(20, output_dim)

    def forward(self, x):
        x1 = self.blinear1(x)
        x2 = self.blinear2(x1)
        return self.blinear3(x2)

# Defining a confidence interval evaluating function

# def evaluate_regression(regressor,
#                         X,
#                         y,
#                         samples = 100,
#                         std_multiplier = 2):
#     preds = [regressor(X) for i in range(samples)]
#     preds = torch.stack(preds)
#     means = preds.mean(axis=0)
#     stds = preds.std(axis=0)
#     ci_upper = means + (std_multiplier * stds)
#     ci_lower = means - (std_multiplier * stds)
#     ic_acc = (ci_lower <= y) * (ci_upper >= y)
#     ic_acc = ic_acc.float().mean()
#     return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

# Creating regressor and loading data

regressor = BayesianLogistic(5, 1)
optimizer = optim.SGD(regressor.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()



######################################

# 2.1 Performance test on Linear separable dataset

######################################

print('2.1')

# use sklearn to generate separable data
from sklearn.datasets import make_classification
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

# make dataset iterable
train_loader_x = Variable(torch.Tensor(train_x))
train_loader_y = Variable(torch.Tensor(train_y))
test_loader_x = Variable(torch.Tensor(test_x))
test_loader_y = Variable(torch.Tensor(test_y))

# train the data

iteration = 0
for epoch in range(5):
    for i, (datapoints, labels) in enumerate(zip(train_loader_x,train_loader_y)):

        # reshape train data y
        labels = labels.view(1)

        optimizer.zero_grad()

        loss = criterion(regressor(datapoints), labels)

        loss.backward()
        optimizer.step()

        # iteration += 1
        # if iteration % 100 == 0:
        #     ic_acc, under_ci_upper, over_ci_lower = evaluate_regression(regressor,
        #                                                                 test_loader_x,
        #                                                                 test_loader_y,
        #                                                                 samples=100,
        #                                                                 std_multiplier=3)
        #
        #     print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper,
        #                                                                               over_ci_lower))
        #     print("Loss: {:.4f}".format(loss))
#         if epoch == 0 and i == 0:
#             print('initial loss: ', loss.item())
#
# print('final loss: ', loss.item())

print('Finished training bayesian NN:')

# make predictions
test = []
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = regressor(inputs)
        predicted = torch.round(outputs)
        test.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of linear separable dataset (Bayesian): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = regressor(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of linear separable dataset (Bayesian): %d %%' % (
    100 * correct / total))








######################################

# 2.2 Performance test on Linear non-separable dataset

######################################

print('2.2')

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
# simulate y by p = 0.5
y = np.random.binomial(1,p,size= n1)

# split data
train_x = x[0:100,:]
train_y = y[0:100]
test_x = x[100:200,:]
test_y = y[100:200]

# make dataset iterable
train_loader_x = Variable(torch.Tensor(train_x))
train_loader_y = Variable(torch.Tensor(train_y))
test_loader_x = Variable(torch.Tensor(test_x))
test_loader_y = Variable(torch.Tensor(test_y))

# train the data

iteration = 0
for epoch in range(5):
    for i, (datapoints, labels) in enumerate(zip(train_loader_x,train_loader_y)):

        # reshape train data y
        labels = labels.view(1)

        optimizer.zero_grad()

        loss = criterion(regressor(datapoints), labels)

        loss.backward()
        optimizer.step()

print('Finished training bayesian NN:')

# make predictions
test = []
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = regressor(inputs)
        predicted = torch.round(outputs)
        test.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of linear non-separable dataset (Bayesian): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = regressor(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of linear non-separable dataset (Bayesian): %d %%' % (
    100 * correct / total))


######################################

# 2.3 Performance test on Non-linear separable dataset

######################################

print('2.3')

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

# make dataset iterable
train_loader_x = Variable(torch.Tensor(train_x2))
train_loader_y = Variable(torch.Tensor(train_y))
test_loader_x = Variable(torch.Tensor(test_x2))
test_loader_y = Variable(torch.Tensor(test_y))

# train the data

iteration = 0
for epoch in range(5):
    for i, (datapoints, labels) in enumerate(zip(train_loader_x,train_loader_y)):

        # reshape train data y
        labels = labels.view(1)

        optimizer.zero_grad()

        loss = criterion(regressor(datapoints), labels)

        loss.backward()
        optimizer.step()

print('Finished training bayesian NN:')

# make predictions
test = []
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = regressor(inputs)
        predicted = torch.round(outputs)
        test.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of non-linear separable dataset (Bayesian): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = regressor(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of non-linear separable dataset (Bayesian): %d %%' % (
    100 * correct / total))


######################################

# 2.4 Performance test on Non-linear non-separable dataset

######################################

print('2.4')

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
# simulate y by p = 0.5
y = np.random.binomial(1,p,size= n1)

# split data
train_x = x[0:100,:]
train_y = y[0:100]
test_x = x[100:200,:]
test_y = y[100:200]

# make dataset iterable
train_loader_x = Variable(torch.Tensor(train_x))
train_loader_y = Variable(torch.Tensor(train_y))
test_loader_x = Variable(torch.Tensor(test_x))
test_loader_y = Variable(torch.Tensor(test_y))

# train the data

iteration = 0
for epoch in range(5):
    for i, (datapoints, labels) in enumerate(zip(train_loader_x,train_loader_y)):

        # reshape train data y
        labels = labels.view(1)

        optimizer.zero_grad()

        loss = criterion(regressor(datapoints), labels)

        loss.backward()
        optimizer.step()

print('Finished training bayesian NN:')

# make predictions
test = []
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = regressor(inputs)
        predicted = torch.round(outputs)
        test.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of non-linear non-separable dataset (Bayesian): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = regressor(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of non-linear non-separable dataset (Bayesian): %d %%' % (
    100 * correct / total))
