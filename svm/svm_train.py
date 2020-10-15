# -*- encoding:utf-8 -*-
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV


def read_data(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [eval(line.strip()) for line in lines]
    X, y = list(zip(*lines))
    X = np.array(X)
    y = np.array(y)
    return X, y


X_train, y_train = read_data("train_data")
X_test, y_test = read_data("test_data")

# C对样本错误的容忍

model= svm.SVC()
#非常慢
'''
model.fit(X_train,y_train)
print(model.support_vectors_)
print(model.support_)
print(len(model.support_))
score = model.score(X_test,y_test)
print(score)
'''
# 网格搜索
search_space = {'C': np.logspace(-3, 3, 7)}
print(search_space['C'])
model = svm.SVC()
gridsearch = GridSearchCV(model, param_grid=search_space)
gridsearch.fit(X_train, y_train)
cv_performance = gridsearch.best_score_
test_performance = gridsearch.score(X_test, y_test)
print("C", gridsearch.best_params_['C'])
