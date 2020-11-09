####################################

# benchmark

####################################

# 1. import data and packages
import numpy as np
import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.simplefilter(action='ignore')


# Read CSV train data file into DataFrame
train_df = pd.read_csv("../data/kaggle_titanic/train.csv")

# Read CSV test data file into DataFrame
test_df = pd.read_csv("../data/kaggle_titanic/test.csv")

# preview train data
train_df.head()

print('The number of samples into the train data is {}.'.format(train_df.shape[0]))

# preview test data
test_df.head()

print('The number of samples into the test data is {}.'.format(test_df.shape[0]))

# 2. data quality check
# check missing values in train data
train_df.isnull().sum()

# percent of missing "Age"
print('Percent of missing "Age" records is %.2f%%' %((train_df['Age'].isnull().sum()/train_df.shape[0])*100))

ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

# mean age
print('The mean of "Age" is %.2f' %(train_df["Age"].mean(skipna=True)))
# median age
print('The median of "Age" is %.2f' %(train_df["Age"].median(skipna=True)))

# percent of missing "Cabin"
print('Percent of missing "Cabin" records is %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0])*100))

# percent of missing "Embarked"
print('Percent of missing "Embarked" records is %.2f%%' %((train_df['Embarked'].isnull().sum()/train_df.shape[0])*100))

print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(train_df['Embarked'].value_counts())
sns.countplot(x='Embarked', data=train_df, palette='Set2')
plt.show()

# data adjustment
train_data = train_df.copy()
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)

# double check missing values in adjusted train data
train_data.isnull().sum()

plt.figure(figsize=(15,8))
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
train_data["Age"].plot(kind='density', color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

## Create categorical variable for traveling alone
train_data['TravelAlone'] = np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)

#create categorical variables and drop some variables
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()

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

# recursive feature elimination
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"]
X = final_train[cols]
y = final_train['Survived']
# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE model and select 8 attributes
rfe = RFE(model, 8)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))

# feature ranking and cross-validation
from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

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

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +
      "and a specificity of %.3f" % (1-fpr[idx]) +
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))

#####################################

# performance of basic lightning NN

#####################################

### import required packages

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class LitLogisticDNN(pl.LightningModule):

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

BasicNN = LitLogisticDNN()
trainer = pl.Trainer(max_epochs=100)
trainer.fit(BasicNN, train_loader)

# test performance
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # reshape test data y
        labels = labels.view(1)

        outputs = BasicNN(inputs)

        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the basic neural network: %d %%' % (
    100 * correct / total))

#####################################

# performance of basic lightning deep ensembles

#####################################

DeepEnsemble1 = LitLogisticDNN()
trainer1 = pl.Trainer(max_epochs=100)
trainer1.fit(DeepEnsemble1,train_loader)
DeepEnsemble2 = LitLogisticDNN()
trainer2 = pl.Trainer(max_epochs=100)
trainer2.fit(DeepEnsemble2,train_loader)
DeepEnsemble3 = LitLogisticDNN()
trainer3 = pl.Trainer(max_epochs=100)
trainer3.fit(DeepEnsemble3,train_loader)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
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
    for data in test_loader:
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
    for data in test_loader:
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
print('Accuracy of the Bayesian NN: %d %%' % (
    100 * correct / total))

