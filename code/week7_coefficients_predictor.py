################################

#  Fit coefficients and predict NN by the simulated new dataset

################################

# 1. import data and packages
import numpy as np
import pandas as pd

# Read CSV train data file into DataFrame
train_df = pd.read_csv("../data/kaggle_titanic/train.csv")

# Read CSV test data file into DataFrame
test_df = pd.read_csv("../data/kaggle_titanic/test.csv")

# 2. data quality check
# check missing values in train data
train_df.isnull().sum()
# data adjustment
train_data = train_df.copy()
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
# double check missing values in adjusted train data
train_data.isnull().sum()

## Create categorical variable for traveling alone
train_data['TravelAlone'] = np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)

#create categorical variables and drop some variables

training = pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training

# apply change to test data
test_df.isnull().sum()

test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)
test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing

final_test.head()

# 3. data analysis

# add 16 boundary for age
final_train['IsMinor'] = np.where(final_train['Age'] <= 16, 1, 0)
final_test['IsMinor'] = np.where(final_test['Age'] <= 16, 1, 0)

# 4. logistic regression
from sklearn.linear_model import LogisticRegression


# model evaluation procedures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

# create X (features) and y (response)
Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C',
                     'Embarked_S', 'Sex_male', 'IsMinor']
X = final_train[Selected_features]
y = final_train['Survived']

# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

# get logistic regression coefficients and intercept
coef = logreg.coef_
inter = logreg.intercept_

# calculate mean and std from the input data and simulate x
train_mean = X_train.mean()
train_std = X_train.std()
X_simulated = np.random.normal(train_mean, train_std, size=(len(X_train), len(train_mean)))

# calculate the output using simulated inputs and fitted coefficients
y_simulated = 1/(1 + np.exp(- (np.matmul(X_simulated,np.transpose(coef)) + inter)))
y_round = np.round(y_simulated)

# logfit = LogisticRegression()
# logreg.fit(X_train, y_train)



#####################################

# basic neural network predictor

#####################################

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# change train test data for neural network
train_loader_x = torch.tensor(X_simulated).float()
train_loader_y = torch.tensor(y_round).float()
test_loader_x = torch.tensor(X_test.values).float()
test_loader_y = torch.tensor(y_test.values).float()

train_loader = DataLoader(TensorDataset(train_loader_x, train_loader_y))
test_loader = DataLoader(TensorDataset(test_loader_x, test_loader_y))

class BasicNN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8,1)

    def forward(self, x):
        x = self.linear1(x)
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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0001)
        return optimizer

NN = BasicNN()
trainer = pl.Trainer(max_epochs=250)
trainer.fit(NN, train_loader)

# test performance
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = NN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of basic neural network using the simulated input: %d %%' % (
    100 * correct / total))

#####################################

# deep neural network predictor

#####################################

class DNN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 100)
        self.linear2 = nn.Linear(100, 20)
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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer

DeepNN = DNN()
trainer = pl.Trainer(max_epochs=100)
trainer.fit(DeepNN, train_loader)

# count number of parameters
# from prettytable import PrettyTable
# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in DeepNN.named_parameters():
#         if not parameter.requires_grad: continue
#         param = parameter.numel()
#         table.add_row([name, param])
#         total_params += param
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params

# get regression coefficients
w1 = list(DeepNN.parameters())[0].data.numpy()
b1 = list(DeepNN.parameters())[1].data.numpy()
w2 = list(DeepNN.parameters())[2].data.numpy()
b2 = list(DeepNN.parameters())[3].data.numpy()
w3 = list(DeepNN.parameters())[4].data.numpy()
b3 = list(DeepNN.parameters())[5].data.numpy()
w4 = list(DeepNN.parameters())[6].data.numpy()
b4 = list(DeepNN.parameters())[7].data.numpy()
w5 = list(DeepNN.parameters())[8].data.numpy()
b5 = list(DeepNN.parameters())[9].data.numpy()

beta = np.matmul(np.matmul(np.matmul(np.transpose(w1),np.transpose(w2)),np.matmul(np.transpose(w3),np.transpose(w4))),np.transpose(w5))
bias = b5 + np.matmul(w5,b4) + np.matmul(w5,np.matmul(w4,b3)) + np.matmul(np.matmul(w5,w4),np.matmul(w3,b2)) + np.matmul(np.matmul(np.matmul(w5,w4),np.matmul(w3,w2)),b1)

# # simulate data
#
# # get new survived status from the input coeficients
# y_new = np.matmul(X.values,beta) + bias
# y_df = pd.DataFrame(np.round(y_new))
# y_df.columns = ['survived']
#
# # split test and train data
# X_newtrain, X_newtest, y_newtrain, y_newtest = train_test_split(X, y_df, test_size=0.2, random_state=2)
#
# # change train test data for neural network
# train_loader_x = torch.tensor(X_newtrain.values).float()
# train_loader_y = torch.tensor(y_newtrain.values).float()
# test_loader_x = torch.tensor(X_newtest.values).float()
# test_loader_y = torch.tensor(y_newtest.values).float()
#
# train_loader = DataLoader(TensorDataset(train_loader_x, train_loader_y))
# test_loader = DataLoader(TensorDataset(test_loader_x, test_loader_y))
#
# #LinearNN = Linear()
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(LinearNN, train_loader)
#
# test performance
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = DeepNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of deep neural network using the simulated input: %d %%' % (
    100 * correct / total))


################################

# Lightning Bayesian NN

################################

from blitz.modules import BayesianLinear

class LitBayesian(pl.LightningModule):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 100)
        self.blinear2 = BayesianLinear(100, 20)
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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
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

BayesianNN = LitBayesian(8,1)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(BayesianNN, train_loader)

# test performance
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = BayesianNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of bayesian neural network using the simulated input: %d %%' % (
    100 * correct / total))



