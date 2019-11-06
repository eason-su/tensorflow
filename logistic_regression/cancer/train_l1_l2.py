# -*- encoding:utf-8 -*-
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from numpy import shape
from sklearn import metrics
import numpy as np


def read_data(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [eval(line.strip()) for line in lines]
    X, y = zip(*lines)
    X = np.array(X)
    y = np.array(y)
    return X, y


X_train, y_train = read_data("train_data")
X_test, y_test = read_data("test_data")


def train_model(reg):
    print(reg)
    model = LogisticRegression(penalty=reg)
    model.fit(X_train, y_train)
    print("w", model.coef_)
    # print (model.intercept_)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    e_train = metrics.mean_squared_error(y_train, y_pred_train)
    e_test = metrics.mean_squared_error(y_test, y_pred_test)

    print("训练集MSE:{}".format(e_train))
    print("测试集MSE:{}".format(e_test))
    print("训练测试差异{}".format(e_test - e_train))
    print("\n")


train_model(reg="None")
train_model(reg="l1")
train_model(reg="l2")
