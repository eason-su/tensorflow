# -*- encoding:utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from numpy import shape
from sklearn import metrics
import numpy as np
import random
def curve(x_train,w,w0):
	results=x_train.tolist()
	results=[x[0:2] for x in results]
	step=0.0001
	for i in np.arange(-0.2,1.2,step):
		x1=i+step
		x2=-1*(w[0]*x1+w0)/(w[1]+w[2]*x1)
		if abs(x2)>5.0:
			continue
		results.append([x1,x2])
	results=["{},{}".format(x1,x2) for [x1,x2] in results]
	return results	
def get_data(center_label,num=100):
	X_train=[]
	y_train=[]
	sigma=0.01
	for point,label in center_label:
		c1,c2=point
		for _ in range(0,num):
			x1=c1+random.uniform(-sigma,sigma)
			x2=c2+random.uniform(-sigma,sigma)
			X_train.append([x1,x2])
			y_train.append([label])
	return X_train,y_train
		
center_label=[[[0,0],1],[[1,1],1],[[0,1],0],[[1,0],0]]
X_train,y_train=get_data(center_label)
#X_train=10*[[0,0],[1,1],[1,0],[0,1]]
X_train=[ x+[x[0]*x[1]] for x in X_train]
X_train=np.array(X_train)
 
#model = LogisticRegression(penalty="l2")
model = LogisticRegression()
model.fit(X_train, y_train)
 
print (model.coef_)
print (model.intercept_)
curve_results=curve(X_train,model.coef_.tolist()[0],model.intercept_.tolist()[0])

with open("no_separa_traindata.csv","w") as f :
	f.writelines("\n".join(curve_results[0:400]))
with open("no_separa_train_with_splitline.csv","w") as f :
	f.writelines("\n".join(curve_results))



 
 
