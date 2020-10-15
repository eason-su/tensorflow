#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import importlib
importlib.reload(sys) 
#sys.setdefaultencoding('utf-8')
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import json
import random
from scipy.sparse import csr_matrix
import pickle
def read_data(path,num):
	with open(path) as f:
		lines=f.readlines()#[0:100]
	#lines=random.sample(lines,5000)
	lines=[line.strip().split("\t") for line in lines]
	y_data=[]
	data=[]
	data2=[]
	col=[]
	row=[]
	f=-1
	for i,line in enumerate(lines):
		y_data.append(int(line[0]))
		data2.append("\t".join(line[1:]))
		for ss in line[1:]:
			c,score=ss.split(":")
			data.append(float(score))
			row.append(i)
			col.append(int(c))
	x_data=csr_matrix((data, (row,col)), shape=(len(lines), num+1))
	y_data=np.array(y_data)
	return x_data,y_data,data2
def compress(x_data,normal_features,ys,enc):
	step=500
	results1=[]
	results2=[]
	start=0
	while start<len(x_data):
		print(start,len(x_data))
		end=start+step
		sub_x=enc.transform(x_data[start:end]).toarray()
		compress_features=[ "\t".join([ "{}:{}".format(i,s) for i,s in enumerate(ss) if s>0]) for ss in sub_x]
		combine_features=[ "\t".join([ "{}:{}".format(i+num,s) for i,s in enumerate(ss) if s>0]) for ss in sub_x]
		results1.extend([ "{}\t{}".format(y,f) for y,f in zip(ys[start:end],compress_features)])
		results2.extend(["{}\t{}\t{}".format(y,f1,f2) for y,f1,f2 in zip(ys[start:end],normal_features,combine_features)])
		start=end
	return results1,results2

with open("feature/feature_index_normal") as f:
	num=len(json.load(f))+1
X_train,Y_train,normal_features_train=read_data("feature/train/all_normal",num)
X_test,Y_test,normal_features_test=read_data("feature/test/all_normal",num)
n_estimators=100
gbm1 = GradientBoostingClassifier(n_estimators=n_estimators, subsample=1, max_depth=11,min_samples_split=10,verbose=2)
gbm1.fit(X_train, Y_train)
x_train = gbm1.apply(X_train).reshape(-1, n_estimators)
x_test = gbm1.apply(X_test).reshape(-1, n_estimators)
enc = OneHotEncoder()
enc.fit(np.concatenate([x_train,x_test]))
results1,results2=compress(x_train,normal_features_train,Y_train.tolist(),enc)
with open("feature/train/all_compress","w") as f:
	f.writelines("\n".join(results1))
with open("feature/train/all_combine","w") as f:
	f.writelines("\n".join(results2))

results1,results2=compress(x_test,normal_features_test,Y_test.tolist(),enc)
with open("feature/test/all_compress","w") as f:
	f.writelines("\n".join(results1))
with open("feature/test/all_combine","w") as f:
	f.writelines("\n".join(results2))
model=(enc,gbm1)
with open("model/compress_model","w") as f:
	pickle.dump(model,f)


