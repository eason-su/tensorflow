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
    data = [[x, y] for [x, y] in lines if y != None]
    X_unknow = [x for [x, y] in lines if y == None]
    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)
    X_unknow = np.array(X_unknow)
    return X, y, X_unknow


def get_data(X_unknow, y_proba):
    X_new = []
    Y_new = []
    X_unknow = X_unknow.tolist()
    y_proba = y_proba.tolist()
    threshold = 0.95
    for x, y in zip(X_unknow, y_proba):
        if y[0] > threshold:
            X_new.append(x)
            Y_new.append(0)
        if y[1] > threshold:
            X_new.append(x)
            Y_new.append(1)
    return X_new, Y_new


X_train, y_train, X_unknow = read_data("train_data")
X_test, y_test, _ = read_data("test_data")

for i in range(0, 100):
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_unknow)
    X_new, y_new = get_data(X_unknow, y_proba)
    print("第{}轮训练 新增{}个样本".format(i, len(y_new)))
    X_train = np.array(X_train.tolist() + X_new)
    y_train = np.array(y_train.tolist() + y_new)
    y_pred = model.predict(X_test)
    print("测试集", "MSE:", metrics.mean_squared_error(y_test, y_pred))
