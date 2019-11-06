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
def curve(x_train,w,w0):
	results=x_train.tolist()
	for i in range(0,100):
		x1=1.0*i/10
		x2=-1*(w[0]*x1+w0)/w[1]
		results.append([x1,x2])
	results=["{},{}".format(x1,x2) for [x1,x2] in results]
	return results	
	

X_train,y_train=read_data("train_data")
X_test,y_test=read_data("test_data")
 
 
model = LogisticRegression()
model.fit(X_train, y_train)
 
print "w",model.coef_
print "w0",model.intercept_

#y_pred = model.predict(X_test)
#print y_pred
y_pred=model.predict_proba(X_test)
print y_pred
#loss=log_loss(y_test,y_pred)
#print "KL_loss:",loss
#loss=log_loss(y_pred,y_test)
#print "KL_loss:",loss
'''
curve_results=curve(X_train,model.coef_.tolist()[0],model.intercept_.tolist()[0])
with open("train_with_splitline","w") as f :
	f.writelines("\n".join(curve_results))
'''


 
 
