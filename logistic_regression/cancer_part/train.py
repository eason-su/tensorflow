# -*- encoding:utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from numpy import shape
from sklearn import metrics
import numpy as np


def read_data(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [eval(line.strip()) for line in lines]
    lines = [[x, y] for [x, y] in lines if y != None]
    X, y = zip(*lines)
    X = np.array(X)
    y = np.array(y)
    return X, y


X_train, y_train = read_data("train_data")
X_test, y_test = read_data("test_data")

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
y_pred = model.predict(X_train)
print("训练集", "MSE:", metrics.mean_squared_error(y_train, y_pred))

y_pred = model.predict(X_test)
print("测试集", "MSE:", metrics.mean_squared_error(y_test, y_pred))
