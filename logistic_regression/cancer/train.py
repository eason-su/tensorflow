# -*- encoding:utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from numpy import shape
from sklearn import metrics
from sklearn.metrics import log_loss
import numpy as np
def read_data(path):
	with open(path) as f :
		lines=f.readlines()
	lines=[eval(line.strip()) for line in lines]
	X,y=zip(*lines)
	X=np.array(X)
	y=np.array(y)
	return X,y
X_train,y_train=read_data("train_data")
X_test,y_test=read_data("test_data")
 
 
model = LogisticRegression()
model.fit(X_train, y_train)
 
print (model.coef_)
print (model.intercept_)
 
y_pred = model.predict(X_test)
y_pred=model.predict_proba(X_test)
print (y_pred)
loss=log_loss(y_test,y_pred)
print ("KL_loss:",loss)
 
 
