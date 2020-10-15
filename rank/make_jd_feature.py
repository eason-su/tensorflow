# -*- coding:utf-8 -*-
import sys
import importlib
importlib.reload(sys)
sys.setdefaultencoding('utf-8')
import re
import json
import jieba.analyse

def extract_keywords(sentence):
	keywords=jieba.analyse.extract_tags(sentence,topK=3)
	return keywords
	
input_path=sys.argv[1]	 
output_path=sys.argv[2]	 
with open(input_path) as f :
	lines=f.readlines()#[0:5000]
lines=[line.decode("utf-8").strip().split("\t") for line in lines]
schema=lines[0][1:]
data=lines[1:]
results=[]
for info in data:
	try:
		job=info[0]
		feature=info[1:]
		feature_result=[]
		if len(schema)!=len(feature):
			continue
		for i,s in enumerate(feature):
			if schema[i]!="job_description" and schema[i]!="jd_title":
				ss=re.split('/|,|\|',s) 
			else:
				ss=extract_keywords(s)
			if schema[i]=="max_salary" or schema[i]=="min_salary":
				ss=[ str(int(s)/1000)]
			if schema[i]=="key" and s=="null":
				continue
			ss=[s.replace(" ","") for s in ss]
			feature_result.extend([ "job:{}:{}|1.0".format(schema[i].replace("|",""),f)  for f in ss if len(f)>0 and f!="-" and f!="\N"])
		results.append("{}\t{}".format(job," ".join(feature_result)))	
	except:
		pass
with open(output_path,"w") as f:
	f.writelines("\n".join(results))






	




	
