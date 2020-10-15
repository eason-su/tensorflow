#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
reload(sys) 
sys.setdefaultencoding('utf-8') 
import random
def get_data(w,num):
	x=[ random.uniform(0,5) for  i in range(0,num)]
	y=[ w*s for s in x]
	return zip(x,y)
def train_step_pow(data,w,rate=0.03):
	g=sum([ (w*x-y)*x  for [x,y] in data])/len(data)
	w=w-rate*g
	return w
def train_step_abs(data,w,rate=0.03):
	g=sum([ x if  (w*x-y)>0 else -1*x  for [x,y] in data])/len(data)
	w=w-rate*g
	return w
	
def cal_data_error(data,w):
	error=[(w*x-y)*(w*x-y)  for [x,y] in data ]
	return error


#第一个参数是w 第二个参数是数量
data=get_data(10,10) +get_data(6,2)
w1=w2=7
#pre_errors=cal_data_error(data,w)
for i in range(0,5000):
	w1=train_step_pow(data,w1)#正规mse训练
	w2=train_step_abs(data,w2)#绝对值mse训练
	if i%50==0:
		#errors=cal_data_error(data,w)
		#mse_delta=[ "%.3f"%(e2-e1) for [e1,e2] in zip(errors,pre_errors)]
		#pre_errors=errors
		print "{},{}".format(w1,w2)
		#print " ".join(mse_delta)


