# -*- coding:utf-8 -*-
import sys
import importlib
importlib.reload(sys)
sys.setdefaultencoding('utf-8')
import re
import json

input_path=sys.argv[1]	 
output_path=sys.argv[2]	 
with open(input_path) as f :
	lines=f.readlines()
lines=[line.decode("utf-8").strip().split("\t") for line in lines]
schema=lines[0][1:]
data=lines[1:]
results=[]
for info in data:
	user=info[0]
	feature=info[1:]
	feature_result=[]
	for i,s in enumerate(feature):
		ss=re.split('/|,|\|',s) 
		feature_result.extend([ "user:{}:{}|1.0".format(schema[i].replace("|",""),f)  for f in ss if len(f)>0 and f!="-"])
	results.append("{}\t{}".format(user," ".join(feature_result)))	
with open(output_path,"w") as f:
	f.writelines("\n".join(results))






	
