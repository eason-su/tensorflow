#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
reload(sys) 
sys.setdefaultencoding('utf-8') 
import random
import math

def read_data(path):
	with open(path) as f :
		lines=f.readlines()
	lines=[eval(line.strip()) for line in lines]
	
def cal_mse(data,w,b=3):
	y2=[ w*x[0]+b for [x,_] in data]
	y=[s[1][0] for s in data]
	mse=sum([   (s1-s2)*(s1-s2) for [s1,s2] in zip(y,y2)])/len(data)
	return  mse

with open("train_data") as f :
	lines=f.readlines()
data=[eval(line.strip()) for line in lines]
sub_data1=random.sample(data,10)
sub_data2=random.sample(data,10)
sub_data3=random.sample(data,50)
sub_data4=random.sample(data,100)
results=["w,all_sample,one_sample1,one_sample2,50_sample,100_sampe"]
for i in range(-2000,2000):
	w=5+1.0*i/500
	mse=cal_mse(data,w)
	sub_mse1=cal_mse(sub_data1,w)
	sub_mse2=cal_mse(sub_data2,w)
	sub_mse3=cal_mse(sub_data3,w)
	sub_mse4=cal_mse(sub_data4,w)
	results.append("{},{},{},{},{},{}".format(w,mse,sub_mse1,sub_mse2,sub_mse3,sub_mse4))

with open("mse_curve.csv","w") as f  :
	f.writelines("\n".join(results))









