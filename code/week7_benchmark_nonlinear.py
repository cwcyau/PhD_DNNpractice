####################################

# benchmark

####################################

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

# # recursive feature elimination
from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import RFE
#
# cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"]
# X = final_train[cols]
# y = final_train['Survived']
# # Build a logreg and compute the feature importances
# model = LogisticRegression()
# # create the RFE model and select 8 attributes
# rfe = RFE(model, 8)
# # rfe = rfe.fit(X, y)
# # summarize the selection of the attributes
# # print('Selected features: %s' % list(X.columns[rfe.support_]))
#
# # feature ranking and cross-validation
# from sklearn.feature_selection import RFECV
# # Create the RFE object and compute a cross-validated score.
# # The "accuracy" scoring is proportional to the number of correct classifications
# rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
# rfecv.fit(X, y)


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

#####################################

# performance of basic lightning NN

#####################################

### import required packages

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class Linear(pl.LightningModule):

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

# change train test data for neural network
train_loader_x = torch.tensor(X_train.values).float()
train_loader_y = torch.tensor(y_train.values).float()
test_loader_x = torch.tensor(X_test.values).float()
test_loader_y = torch.tensor(y_test.values).float()

train_loader = DataLoader(TensorDataset(train_loader_x, train_loader_y))
test_loader = DataLoader(TensorDataset(test_loader_x, test_loader_y))

LinearNN = Linear()
trainer = pl.Trainer(max_epochs=100)
trainer.fit(LinearNN, train_loader)

# test performance
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = LinearNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the linear neural network: %d %%' % (
    100 * correct / total))

######################################################

# performance of lightning NN with a non-linear layer

######################################################

### import required packages

import torch.nn.functional as F

class OneRelu(pl.LightningModule):

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
        x = F.relu(self.linear3(x))
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

ReluNN = OneRelu()
trainer = pl.Trainer(max_epochs=100)
trainer.fit(ReluNN, train_loader)

# test performance
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = ReluNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the basic neural network with a non-linear layer: %d %%' % (
    100 * correct / total))

######################################################

# performance of lightning NN with a non-linear sigmoid layer

######################################################

### import required packages

class OneSigmoid(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 100)
        self.linear2 = nn.Linear(100, 20)
        self.linear3 = nn.Linear(20, 10)
        self.linear4 = nn.Linear(10, 5)
        self.linear5 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(self.linear3(x))
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

SigmoidNN = OneSigmoid()
trainer = pl.Trainer(max_epochs=100)
trainer.fit(SigmoidNN, train_loader)

# test performance
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = SigmoidNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the basic neural network with a non-linear sigmoid layer: %d %%' % (
    100 * correct / total))

######################################################

# performance of lightning NN with three non-linear layers

######################################################

### import required packages

import torch.nn.functional as F

class ThreeRelu(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 100)
        self.linear2 = nn.Linear(100, 20)
        self.linear3 = nn.Linear(20, 10)
        self.linear4 = nn.Linear(10, 5)
        self.linear5 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
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

TReluNN = ThreeRelu()
trainer = pl.Trainer(max_epochs=100)
trainer.fit(TReluNN, train_loader)

# test performance
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = TReluNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the basic neural network with three non-linear layers: %d %%' % (
    100 * correct / total))

######################################################

# performance of lightning NN with three non-linear sigmoid layers

######################################################


class ThreeSigmoid(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 100)
        self.linear2 = nn.Linear(100, 20)
        self.linear3 = nn.Linear(20, 10)
        self.linear4 = nn.Linear(10, 5)
        self.linear5 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
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

TSigmoidNN = ThreeSigmoid()
trainer = pl.Trainer(max_epochs=100)
trainer.fit(TSigmoidNN, train_loader)

# test performance
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = TSigmoidNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the basic neural network with three non-linear sigmoid layers: %d %%' % (
    100 * correct / total))



