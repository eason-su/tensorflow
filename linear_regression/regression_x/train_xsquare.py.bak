#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
reload(sys) 
sys.setdefaultencoding('utf-8') 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
def extend_feature(x):
	#return [x[0]]
	#x=[2]
	#x=[2,4]
	return [x[0],x[0]*x[0]]

def read_data(path):
	with open(path) as f :
		lines=f.readlines()
	lines=[eval(line.strip()) for line in lines]
	X,y=zip(*lines)
	X=[ extend_feature(x) for x in X]
	X=np.array(X)
	y=np.array(y)
	return X,y
X_train,y_train=read_data("train_paracurve_data")
X_test,y_test=read_data("test_paracurve_data")
model = LinearRegression()
model.fit(X_train, y_train)
print model.coef_
print model.intercept_


'''
y_pred_train = model.predict(X_train)
train_mse=metrics.mean_squared_error(y_train, y_pred_train)
print "特征+平方非线性"
print "MSE:", train_mse
y_pred_test = model.predict(X_test)
test_mse=metrics.mean_squared_error(y_test, y_pred_test)
print "MSE:",test_mse
print "推广mse差", test_mse-train_mse
'''
