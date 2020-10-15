import json
import math
import xlearn as xl
import os
import FM
from sklearn.metrics import roc_auc_score
import numpy as np
from keras.models import load_model
def process_feature(ss):
	feature=[s.split(":") for s in ss]
	feature=[ [int(index),float(score)]for [index,score] in feature]
	return feature	
def read_data(path):
	with open(path) as f:
		lines=set(f.readlines())#[0:5]
	lines=[line.strip().split("\t") for line in lines]
	return [ [int(line[0]) ,process_feature(line[1:]) ] for line in lines]
def read_profile(path):
	with open(path) as f:
		lines=f.readlines()#[0:5]
	lines=[eval(line.strip()) for line in lines]
	x1,x2=list(zip(*[[int(user_cluster[user]),int(jd_cluster[jd])]for [user,jd] in lines]))
	return [np.array(x1),np.array(x2)]

	
ts=["normal","compress","combine"]
for t in ts:
	model=FM.FM()
	data=read_data("feature/test/all_{}".format(t))
	label,feature=list(zip(*data))	
	model.load("model/all_{}".format(t))
	y=[model.predict(f) for f in feature]
	y=np.array(y)
	label=np.array(label)
	auc_score = roc_auc_score(label,y)
	print(t,auc_score)




