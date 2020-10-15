#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import importlib
importlib.reload(sys) 
sys.setdefaultencoding('utf-8') 
import random
import json
def read_feature(path,profile={}):
	with open(path) as f:
		lines=f.readlines()
	lines=[line.decode('utf-8').strip().split("\t") for line in lines]
	lines=[[key,feature.split(" ")] for [key,feature] in lines]
	lines=[[key,[ s.split("|") for s in feature]] for [key,feature] in lines]
	lines=[[key,[ [ss[0],float(ss[1])] for ss in feature if len(ss)==2 and len(ss[0])>0]+profile.get(key,[]) ] for [key,feature] in lines]
	return dict(lines)

def balance(data):
	results=[]
	p_list=data["p"]
	n_list=data["n"]
	n_list=random.sample(n_list,len(p_list))
	random.shuffle(p_list)
	results.extend(p_list)
	results.extend(n_list)
	return results

def read_data(path):
	with open(path) as f:
		lines=f.readlines()[1:]
	lines=[line.strip().split("\t") for line in lines ]
	results={"p":[],"n":[]}
	count=0
	for user_id,jd,v1,v2,v3 in lines:
		count+=1
		if user_id not in user_feature:
			continue
		if jd not in jd_feature:
			continue
		feature=user_feature[user_id]+"\t"+jd_feature[jd]
		score=int(v1)+int(v2)+int(v3)
		if score>0:
			results["p"].extend(score*["1"+"\t"+feature])
		else:
			results["n"].extend(["0"+"\t"+feature])
	return results


def filter(features):
	results={}
	for ss in list(features.values()):
		for s,_ in ss:
			results[s]=results.get(s,0)+1
	results=[ s  for [s,num] in list(results.items()) if num>10 and num<0.7*len(features)]
	return results
def filter_low_frequence_feature(user_feature,jd_feature):
	f1=list(filter(user_feature))
	f2=list(filter(jd_feature))
	return f1+f2

def cal_feature_index(user_feature,jd_feature):
	features=filter_low_frequence_feature(user_feature,jd_feature)
	feature_index=dict(list(zip(features,list(range(1,len(features)+1)))))
	jd_feature=dict([ [key,"\t".join(["{}:{}".format(feature_index[s],score) for [s,score] in l if s in feature_index]) ]for key,l in list(jd_feature.items())])
	user_feature=dict([ [key,"\t".join(["{}:{}".format(feature_index[s],score) for [s,score] in l if s in feature_index]) ]for key,l in list(user_feature.items())])
	return feature_index,jd_feature,user_feature

user_feature_path=sys.argv[1]	
user_profile_path=sys.argv[2]
jd_feature_path=sys.argv[3]
with open(user_profile_path) as f:
	user_profile=json.load(f)
#user_profile={}
user_feature=read_feature(user_feature_path,profile=user_profile)
jd_feature=read_feature(jd_feature_path)
feature_index,jd_feature,user_feature=cal_feature_index(user_feature,jd_feature)
ts=["train","test"]
for t in ts:
	results=read_data("feature/{}_data".format(t))
	results=balance(results)
	print(len(results))
	print()
	with open("feature/{}/all_normal".format(t),"w") as f:
		f.writelines("\n".join(results))
with open("feature/feature_index_normal","w") as f:
	print("特征维度",len(feature_index))
	json.dump(feature_index,f,ensure_ascii=False)
with open("feature/jd_feature_index","w") as f:
	json.dump(jd_feature,f)
with open("feature/user_feature_index","w") as f:
	json.dump(user_feature,f)



