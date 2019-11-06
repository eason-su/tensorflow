# -*- encoding:utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from numpy import shape
from sklearn import metrics
def array_to_list(data):
	data=data.tolist()
	return data 
loaded_data= datasets.load_breast_cancer()
#loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)
X_train=array_to_list(X_train)
X_test=array_to_list(X_test)
y_train=array_to_list(y_train)
y_test=array_to_list(y_test)
train_data=zip(X_train,y_train)
train_data=[str(list(s)) for s in train_data]
test_data=zip(X_test,y_test)
test_data=[str(list(s)) for s in test_data]
with open("train_data","w") as f :
	f.writelines("\n".join(train_data))
with open("test_data","w") as f :
	f.writelines("\n".join(test_data))
 
