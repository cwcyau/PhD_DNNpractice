################################

#  Pytorch lightning

################################

### import required packages

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from blitz.modules import BayesianLinear


################################

# Lightning basic DNN

################################

class LitLogisticDNN(pl.LightningModule):

    def __init__(self):
        super().__init__()
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

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs, labels = batch
        # reshape train data y
        labels = labels.view(1,-1)

        outputs = self.forward(inputs)

        criterion = nn.MSELoss()

        loss = criterion(outputs, labels)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    # def test_step(self, batch, batch_idx):
    #     inputs, labels = batch
    #
    #     outputs = self.forward(inputs)
    #     criterion = nn.MSELoss()
    #     loss = criterion(outputs, labels)
    #     return {'val_loss': loss}


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

    def prepare_data(self):
        # simulate a data set for a logistic regression model with 5 dimension:
        # assume covariance is an identity matrix
        sigma1 = np.identity(5)
        # mean is an array of 0
        mean1 = np.zeros(5)
        # number of samples are 200
        n1 = 5000
        # generate n gaussian distributed data points
        x = np.random.multivariate_normal(mean1, sigma1, n1)

        # calculate p, set regression coefficients beta as an array of 1s
        p = 1 / (1 + np.exp(-np.sum(x, axis=1)))

        # split the training test data by half
        # simulate y by p = 0.5
        y = np.random.binomial(1, p, size=n1)

        # split data
        train_x = x[0:np.int(n1/2), :]
        train_y = y[0:np.int(n1/2)]
        test_x = x[np.int(n1/2):n1, :]
        test_y = y[np.int(n1/2):n1]

        # make dataset iterable
        train_loader_x = torch.tensor(train_x).float()
        train_loader_y = torch.tensor(train_y).float()
        # test_loader_x = torch.tensor(test_x).float()
        # test_loader_y = torch.tensor(test_y).float()

        self.train_data = TensorDataset(train_loader_x, train_loader_y)
        # self.test_data = TensorDataset(test_loader_x, test_loader_y)


    def train_dataloader(self):
        return DataLoader(self.train_data)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data)

    # def test_step(self):

BasicNN = LitLogisticDNN()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(BasicNN)

# test performance
sigma1 = np.identity(5)
mean1 = np.zeros(5)
n1 = 1000
x = np.random.multivariate_normal(mean1, sigma1, n1)
p = 1 / (1 + np.exp(-np.sum(x, axis=1)))
y = np.random.binomial(1, p, size=n1)
test_loader_x = torch.tensor(x).float()
test_loader_y = torch.tensor(y).float()
test_data = DataLoader(TensorDataset(test_loader_x,test_loader_y))

correct = 0
total = 0
with torch.no_grad():
    for data in test_data:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = BasicNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 1000 test numbers of linear separable dataset (LightningNN): %d %%' % (
    100 * correct / total))


################################

# Lightning Deep ensembles

################################

DeepEnsemble1 = LitLogisticDNN()
trainer1 = pl.Trainer(max_epochs=5)
trainer1.fit(DeepEnsemble1)
DeepEnsemble2 = LitLogisticDNN()
trainer2 = pl.Trainer(max_epochs=5)
trainer2.fit(DeepEnsemble2)
DeepEnsemble3 = LitLogisticDNN()
trainer3 = pl.Trainer(max_epochs=5)
trainer3.fit(DeepEnsemble3)

correct = 0
total = 0
with torch.no_grad():
    for data in test_data:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DeepEnsemble1(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

tot1 = total
cor1 = correct
print('Accuracy of the first ensemble: %d %%' % (
    100 * correct / total))

with torch.no_grad():
    for data in test_data:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DeepEnsemble2(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

tot2 = total
cor2 = correct
print('Accuracy of the second ensemble: %d %%' % (
    100 * (correct - cor1) / (total-tot1)))

with torch.no_grad():
    for data in test_data:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DeepEnsemble3(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the third ensemble: %d %%' % (
    100 * (correct - cor2) / (total - tot2)))

print('Average accuracy of three ensembles: %d %%' % (
    100 * correct / total))

################################

# Lightning Bayesian NN

################################

class LitBayesian(pl.LightningModule):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 60)
        self.blinear2 = BayesianLinear(60, 20)
        self.blinear3 = BayesianLinear(20, 10)
        self.blinear4 = BayesianLinear(10,5)
        self.blinear5 = BayesianLinear(5, output_dim)

    def forward(self, x):
        x1 = self.blinear1(x)
        x2 = self.blinear2(x1)
        x3 = self.blinear3(x2)
        x4 = self.blinear4(x3)
        x5 = self.blinear5(x4)
        return x5

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        inputs, labels = batch
        # reshape train data y
        labels = labels.view(1, -1)

        outputs = self.forward(inputs)

        criterion = nn.MSELoss()

        loss = criterion(outputs, labels)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def prepare_data(self):
        # simulate a data set for a logistic regression model with 5 dimension:
        # assume covariance is an identity matrix
        sigma1 = np.identity(5)
        # mean is an array of 0
        mean1 = np.zeros(5)
        # number of samples are 200
        n1 = 1000
        # generate n gaussian distributed data points
        x = np.random.multivariate_normal(mean1, sigma1, n1)

        # calculate p, set regression coefficients beta as an array of 1s
        p = 1 / (1 + np.exp(-np.sum(x, axis=1)))

        # split the training test data by half
        # simulate y by p = 0.5
        y = np.random.binomial(1, p, size=n1)

        # make dataset iterable
        train_loader_x = torch.tensor(x).float()
        train_loader_y = torch.tensor(y).float()

        self.train_data = TensorDataset(train_loader_x, train_loader_y)

    def train_dataloader(self):
        return DataLoader(self.train_data)

BayesianNN = LitBayesian(5,1)
trainer = pl.Trainer(max_epochs=5)
trainer.fit(BayesianNN)

correct = 0
total = 0
with torch.no_grad():
    for data in test_data:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = BayesianNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the 1000 test numbers of linear separable dataset (BayesianNN): %d %%' % (
    100 * correct / total))