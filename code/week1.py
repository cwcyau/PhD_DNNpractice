#########################################

# Implement a logistic regression

#########################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

# simulate a data set for a logistic regression model with 5 dimension:
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


#############################################
# run standard logistic regression on simulated data
#############################################
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', random_state=0).fit(train_x, train_y)
# intercept
model.intercept_
# coefficient
model.coef_
# predicted probability
model.predict_proba(test_x)
# predicted value
model.predict(test_x)
# predicted accuracy score for test data
model.score(test_x, test_y)

#############################################
# apply neural network
#############################################
import torch.nn.functional as F

# use a trial single layer NN first
class Logistictest(nn.Module):

    def __init__(self):
        super(Logistictest, self).__init__()
        self.linear = nn.Linear(5,1)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

model = Logistictest()

# define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# make dataset iterable
train_loader_x = Variable(torch.Tensor(train_x))
train_loader_y = Variable(torch.Tensor(train_y))
test_loader_x = Variable(torch.Tensor(test_x))
test_loader_y = Variable(torch.Tensor(test_y))
# train_loader_x = torch.utils.data.DataLoader(dataset=train_x, batch_size=4, shuffle=True)
# train_loader_y = torch.utils.data.DataLoader(dataset=train_y, batch_size=4, shuffle=True)
# test_loader_x = torch.utils.data.DataLoader(dataset=test_x, batch_size=4, shuffle=False)
# test_loader_y = torch.utils.data.DataLoader(dataset=test_y, batch_size=4, shuffle=False)

# model training
iter = 0
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(zip(train_loader_x,train_loader_y)):
        inputs, labels = data
        # reshape train data y
        labels = labels.view(1)

        # zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute Loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = model(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network 100 train numbers: %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = model(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network 100 test numbers: %d %%' % (
    100 * correct / total))


#############################################
# apply it on a multiple layers DNN
#############################################

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

model = LogisticDNN()

# define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


# model training
iter = 0
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(zip(train_loader_x,train_loader_y)):
        inputs, labels = data
        # reshape train data y
        labels = labels.view(1)

        # zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute Loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# make predictions
correct = 0
total = 0
with torch.no_grad():
    for data in zip(train_loader_x,train_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = model(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network 100 train numbers: %d %%' % (
    100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test_loader_x,test_loader_y):
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = model(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network 100 test numbers: %d %%' % (
    100 * correct / total))
