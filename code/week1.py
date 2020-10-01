#########################################

# Implement a logistic regression

#########################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# make dataset iterable
train_loader_x = torch.utils.data.DataLoader(dataset=train_x, batch_size=4, shuffle=True)
train_loader_y = torch.utils.data.DataLoader(dataset=train_y, batch_size=4, shuffle=True)
test_loader_x = torch.utils.data.DataLoader(dataset=test_x, batch_size=4, shuffle=False)
test_loader_y = torch.utils.data.DataLoader(dataset=test_y, batch_size=4, shuffle=False)

# model training
iter = 0
for epoch in range(5):  # loop over the dataset multiple times

    for i, data in enumerate(zip(train_loader_x,train_loader_y)):
        inputs, labels = data
        inputs = inputs.type(torch.LongTensor)

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

        iter+=1
        if iter%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for inputs, labels in zip(test_loader_x,test_loader_y):
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))



#############################################
# apply it on a multiple layers DNN
#############################################
