#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import random
import math
def logistic(x,w):
	d=sum([ s1*s2 for [s1,s2] in zip(x,w)])
	r=1.0/(1+math.exp(-1*d))
	return r

def cal_entropy(s1,s2):
	s2=min(0.999,max(s2,0.001))
	if s1==0:
		return math.log(1.0/(1-s2))
	else:
		return math.log(1.0/s2)
def cal_error(data,w):
	y2=[ logistic(x,w) for [x,_] in data]
	y=[s[1][0] for s in data]
	mse=sum([   (s1-s2)*(s1-s2) for [s1,s2] in zip(y,y2)])/len(data)
	entropy=sum([   cal_entropy(s1,s2) for [s1,s2] in zip(y,y2)])/len(data)
	return  mse,entropy
def read_data(path):
	with open(path) as f :
		lines=f.readlines()
	lines=[eval(line.strip()) for line in lines]
	return lines

data=read_data("train_data")
results=[]
num=100
step=0.2
for i in range(-num,num):
	w1=1.87+step*i
	#for j in range(-num,num):
	w2=-1.87#+step*j
	e1,e2=cal_error(data,[w1,w2])
	results.append("{},{},{}".format(w1,e1,e2))

with open("mse_entropy.csv","w") as f  :
	f.writelines("\n".join(results))









