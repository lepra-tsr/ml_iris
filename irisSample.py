# https://www.kaggle.com/ash316/ml-from-scratch-with-iris

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

iris = pd.read_csv("./iris.csv", delimiter=",", header=0)

train, test = train_test_split(iris, test_size=0.3)
print(train.shape)
print(test.shape)
train_X = train[['SepalLengthCm', 'SepalWidthCm',
                 'PetalLengthCm', 'PetalWidthCm', ]]
train_y = train.Species
test_X = test[['SepalLengthCm', 'SepalWidthCm',
               'PetalLengthCm', 'PetalWidthCm', ]]
test_y = test.Species

# print(train_X.head(2))
# print(test_X.head(2))
# print(train_y.head())

print(' === use sepal and petal: ')
model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('accuracy of SVM: ', metrics.accuracy_score(prediction, test_y))

model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('accuracy of LR: ', metrics.accuracy_score(prediction, test_y))

model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('accuracy of DT: ', metrics.accuracy_score(prediction, test_y))

petal = iris[['PetalLengthCm', 'PetalWidthCm', 'Species']]
train_p, test_p = train_test_split(petal, test_size=0.3, random_state=0)
train_x_p = train_p[['PetalWidthCm', 'PetalLengthCm']]
train_y_p = train_p.Species
test_x_p = test_p[['PetalWidthCm', 'PetalLengthCm']]
test_y_p = test_p.Species

sepal = iris[['SepalLengthCm', 'SepalWidthCm', 'Species']]
train_s, test_s = train_test_split(sepal, test_size=0.3, random_state=0)
train_x_s = train_s[['SepalWidthCm', 'SepalLengthCm']]
train_y_s = train_s.Species
test_x_s = test_s[['SepalWidthCm', 'SepalLengthCm']]
test_y_s = test_s.Species

print(' === use sepal or petal: ')
model = svm.SVC()
model.fit(train_x_p, train_y_p)
prediction = model.predict(test_x_p)
print('accuracy of SVM(petal): ', metrics.accuracy_score(prediction, test_y_p))

model = svm.SVC()
model.fit(train_x_s, train_y_s)
prediction = model.predict(test_x_s)
print('accuracy of SVM(sepal): ', metrics.accuracy_score(prediction, test_y_s))

model = LogisticRegression()
model.fit(train_x_p, train_y_p)
prediction = model.predict(test_x_p)
print('accuracy of LR(petal): ', metrics.accuracy_score(prediction, test_y_p))

model = LogisticRegression()
model.fit(train_x_s, train_y_s)
prediction = model.predict(test_x_s)
print('accuracy of LR(sepal): ', metrics.accuracy_score(prediction, test_y_s))
