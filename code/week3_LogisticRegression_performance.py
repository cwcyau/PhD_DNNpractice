######################################

# 1. Logistic regression

######################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression

# Logistic regression
# use multiple layers NN
class LogisticDNN(nn.Module):

    def __init__(self):
        super(LogisticDNN, self).__init__()
        self.linear1 = nn.Linear(5, 60)
        self.linear2 = nn.Linear(60, 20)
        self.linear3 = nn.Linear(20, 10)
        self.linear4 = nn.Linear(10, 5)
        self.linear5 = nn.Linear(5, 1)


    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x

NNmodel = LogisticDNN()

DEmodel = LogisticDNN()

# define a loss function and optimizers
criterion = nn.MSELoss()
NN_optimizer = optim.SGD(NNmodel.parameters(), lr=0.001)


######################################

# 1.1 Performance test on Linear separable dataset

######################################

print('1.1')

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


##############################
# test on logistic regression
##############################
LR = LogisticRegression(solver='liblinear', random_state=0).fit(train_x, train_y)
# accuracy score for train data
print('Accuracy of the 100 train numbers of linear separable dataset (LR): %d %%' %
      (100 * LR.score(train_x, train_y)))
# predicted accuracy score for test data
print('Accuracy of the 100 test numbers of linear separable dataset (LR): %d %%' %
      (100 * LR.score(test_x, test_y)))

##############################
# test on basic neural network
##############################

# make dataset iterable
train_loader_x = Variable(torch.Tensor(train_x))
train_loader_y = Variable(torch.Tensor(train_y))
test_loader_x = Variable(torch.Tensor(test_x))
test_loader_y = Variable(torch.Tensor(test_y))

# model training
iter = 0
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(zip(train_loader_x,train_loader_y)):
        inputs, labels = data
        # reshape train data y
        labels = labels.view(1)

        # zero the parameter gradients
        NN_optimizer.zero_grad()
        # Forward pass
        outputs = NNmodel(inputs)
        # Compute Loss
        NN_loss = criterion(outputs, labels)
        # Backward pass
        NN_loss.backward()
        # optimize
        NN_optimizer.step()

        #print statistics
        # running_loss += loss.item()
        # if i % 20 == 19:  # print every 20 mini-batches
        #      print('[%d, %5d] loss: %.3f' %
        #            (epoch + 1, i + 1, running_loss / 2000))
        #      running_loss = 0.0

print('Finished training basic NN:')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = NNmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of linear separable dataset (NN): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = NNmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of linear separable dataset (NN): %d %%' % (
    100 * correct / total))

##############################
# Training an ensemble of 5 MLPs with MSE
##############################
de = []
de_optimizers = []
for _ in range(5):
    de.append(DEmodel)
    de_optimizers.append(torch.optim.SGD(params=DEmodel.parameters(), lr=0.001))
# train
for i, net in enumerate(de):
    # print('Training network ',i+1)
    for epoch in range(5):
        for j, data in enumerate(zip(train_loader_x, train_loader_y)):
            inputs, labels = data
            # reshape train data y
            labels = labels.view(1)

            de_optimizers[i].zero_grad()
            # reshape train data y
            DE_loss = criterion(net(inputs), labels)

            DE_loss.backward()

            de_optimizers[i].step()

    #         if epoch == 0 and j == 0:
    #             print('initial loss: ', DE_loss.item())
    # print('final loss: ', DE_loss.item())

print('Finished training Deep ensembles:')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DEmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of linear separable dataset (DE): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DEmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of linear separable dataset (DE): %d %%' % (
    100 * correct / total))



######################################

# 1.2 Performance test on Linear non-separable dataset

######################################

print('1.2')

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

##############################
# test on logistic regression
##############################
LR = LogisticRegression(solver='liblinear', random_state=0).fit(train_x, train_y)
# accuracy score for train data
print('Accuracy of the 100 train numbers of linear non-separable dataset (LR): %d %%' %
      (100 * LR.score(train_x, train_y)))
# predicted accuracy score for test data
print('Accuracy of the 100 test numbers of linear non-separable dataset (LR): %d %%' %
      (100 * LR.score(test_x, test_y)))

##############################
# test on basic neural network
##############################

# make dataset iterable
train_loader_x = Variable(torch.Tensor(train_x))
train_loader_y = Variable(torch.Tensor(train_y))
test_loader_x = Variable(torch.Tensor(test_x))
test_loader_y = Variable(torch.Tensor(test_y))

# model training
iter = 0
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(zip(train_loader_x,train_loader_y)):
        inputs, labels = data
        # reshape train data y
        labels = labels.view(1)

        # zero the parameter gradients
        NN_optimizer.zero_grad()
        # Forward pass
        outputs = NNmodel(inputs)
        # Compute Loss
        NN_loss = criterion(outputs, labels)
        # Backward pass
        NN_loss.backward()
        # optimize
        NN_optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 20 == 19:  # print every 20 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

print('Finished training basic NN:')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = NNmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of linear non-separable dataset (NN): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = NNmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of linear non-separable dataset (NN): %d %%' % (
    100 * correct / total))

##############################
# Training an ensemble of 5 MLPs with MSE
##############################
de = []
de_optimizers = []
for _ in range(5):
    de.append(DEmodel)
    de_optimizers.append(torch.optim.SGD(params=DEmodel.parameters(), lr=0.001))
# train
for i, net in enumerate(de):
    # print('Training network ',i+1)
    for epoch in range(5):
        for j, data in enumerate(zip(train_loader_x, train_loader_y)):
            inputs, labels = data
            # reshape train data y
            labels = labels.view(1)

            de_optimizers[i].zero_grad()
            # reshape train data y
            DE_loss = criterion(net(inputs), labels)

            DE_loss.backward()

            de_optimizers[i].step()

    #         if epoch == 0 and j == 0:
    #             print('initial loss: ', DE_loss.item())
    # print('final loss: ', DE_loss.item())

print('Finished training Deep ensembles:')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DEmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of linear non-separable dataset (DE): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DEmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of linear non-separable dataset (DE): %d %%' % (
    100 * correct / total))

######################################

# 1.3 Performance test on Non-linear separable dataset

######################################

print('1.3')

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

##############################
# test on logistic regression
##############################
LR = LogisticRegression(solver='liblinear', random_state=0).fit(train_x2, train_y)
# accuracy score for train data
print('Accuracy of the 100 train numbers of non-linear separable dataset (LR): %d %%' %
      (100 * LR.score(train_x2, train_y)))
# predicted accuracy score for test data
print('Accuracy of the 100 test numbers of non-linear separable dataset (LR): %d %%' %
      (100 * LR.score(test_x2, test_y)))

##############################
# test on basic neural network
##############################

# make dataset iterable
train_loader_x = Variable(torch.Tensor(train_x2))
train_loader_y = Variable(torch.Tensor(train_y))
test_loader_x = Variable(torch.Tensor(test_x2))
test_loader_y = Variable(torch.Tensor(test_y))

# model training
iter = 0
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(zip(train_loader_x,train_loader_y)):
        inputs, labels = data
        # reshape train data y
        labels = labels.view(1)

        # zero the parameter gradients
        NN_optimizer.zero_grad()
        # Forward pass
        outputs = NNmodel(inputs)
        # Compute Loss
        NN_loss = criterion(outputs, labels)
        # Backward pass
        NN_loss.backward()
        # optimize
        NN_optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 20 == 19:  # print every 20 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

print('Finished training basic NN:')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = NNmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of non-linear separable dataset (NN): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = NNmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of non-linear separable dataset (NN): %d %%' % (
    100 * correct / total))

##############################
# Training an ensemble of 5 MLPs with MSE
##############################
de = []
de_optimizers = []
for _ in range(5):
    de.append(DEmodel)
    de_optimizers.append(torch.optim.SGD(params=DEmodel.parameters(), lr=0.001))
# train
for i, net in enumerate(de):
    # print('Training network ',i+1)
    for epoch in range(5):
        for j, data in enumerate(zip(train_loader_x, train_loader_y)):
            inputs, labels = data
            # reshape train data y
            labels = labels.view(1)

            de_optimizers[i].zero_grad()
            # reshape train data y
            DE_loss = criterion(net(inputs), labels)

            DE_loss.backward()

            de_optimizers[i].step()

    #         if epoch == 0 and j == 0:
    #             print('initial loss: ', DE_loss.item())
    # print('final loss: ', DE_loss.item())

print('Finished training Deep ensembles:')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DEmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of non-linear separable dataset (DE): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DEmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of non-linear separable dataset (DE): %d %%' % (
    100 * correct / total))

######################################

# 1.4 Performance test on Non-linear non-separable dataset

######################################

print('1.4')

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

##############################
# test on logistic regression
##############################
LR = LogisticRegression(solver='liblinear', random_state=0).fit(train_x, train_y)
# accuracy score for train data
print('Accuracy of the 100 train numbers of non-linear non-separable dataset (LR): %d %%' %
      (100 * LR.score(train_x, train_y)))
# predicted accuracy score for test data
print('Accuracy of the 100 test numbers of non-linear non-separable dataset (LR): %d %%' %
      (100 * LR.score(test_x, test_y)))

##############################
# test on basic neural network
##############################

# make dataset iterable
train_loader_x = Variable(torch.Tensor(train_x))
train_loader_y = Variable(torch.Tensor(train_y))
test_loader_x = Variable(torch.Tensor(test_x))
test_loader_y = Variable(torch.Tensor(test_y))

# model training
iter = 0
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(zip(train_loader_x,train_loader_y)):
        inputs, labels = data
        # reshape train data y
        labels = labels.view(1)

        # zero the parameter gradients
        NN_optimizer.zero_grad()
        # Forward pass
        outputs = NNmodel(inputs)
        # Compute Loss
        NN_loss = criterion(outputs, labels)
        # Backward pass
        NN_loss.backward()
        # optimize
        NN_optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 20 == 19:  # print every 20 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

print('Finished training basic NN:')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = NNmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of non-linear non-separable dataset (NN): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = NNmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of non-linear non-separable dataset (NN): %d %%' % (
    100 * correct / total))

##############################
# Training an ensemble of 5 MLPs with MSE
##############################
de = []
de_optimizers = []
for _ in range(5):
    de.append(DEmodel)
    de_optimizers.append(torch.optim.SGD(params=DEmodel.parameters(), lr=0.001))
# train
for i, net in enumerate(de):
    # print('Training network ',i+1)
    for epoch in range(5):
        for j, data in enumerate(zip(train_loader_x, train_loader_y)):
            inputs, labels = data
            # reshape train data y
            labels = labels.view(1)

            de_optimizers[i].zero_grad()
            # reshape train data y
            DE_loss = criterion(net(inputs), labels)

            DE_loss.backward()

            de_optimizers[i].step()

    #         if epoch == 0 and j == 0:
    #             print('initial loss: ', DE_loss.item())
    # print('final loss: ', DE_loss.item())

print('Finished training Deep ensembles:')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DEmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 train numbers of non-linear non-separable dataset (DE): %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DEmodel(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 100 test numbers of non-linear non-separable dataset (DE): %d %%' % (
    100 * correct / total))

