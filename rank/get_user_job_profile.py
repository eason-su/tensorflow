#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import importlib
importlib.reload(sys) 
sys.setdefaultencoding('utf-8') 
import random
import json
import math

def read_feature(path):
	with open(path) as f:
		lines=f.readlines()
	lines=[line.strip().split("\t") for line in lines]
	lines=[[key,feature.split(" ")] for [key,feature] in lines]
	lines=[[key,[ s.split("_") for s in feature]] for [key,feature] in lines]
	lines=[[key,[ ss[0] for ss in feature if len(ss)==2 and len(ss[0])>0]] for [key,feature] in lines]
	return dict(lines)


def cal_profile(path,user_feature,jd_feature):
	with open(path) as f:
		lines=f.readlines()[1:]
	lines=[line.strip().split("\t") for line in lines ]
	count=0
	user_profile={}
	job_profile={}
	user_word_count={}
	job_word_count={}
	doc_count=0
	for user,jd,v1,v2,v3 in lines:
		if count%1000==0:
			print(count,len(lines))
		count+=1
		if user not in user_feature:
			continue
		if jd not in jd_feature:
			continue
		score=int(v1)+int(v2)+int(v3)
		score=score if score>0 else -1
		doc_count+=1
		if user not in user_profile:
			user_profile[user]={}
		if jd not in job_profile:
			job_profile[jd]={}
		for s in set(user_feature[user]):
			job_profile[jd][s]=job_profile[jd].get(s,0)+score
			user_word_count[s]=user_word_count.get(s,0.0)+1.0
		for s in set(jd_feature[jd]):
			user_profile[user][s]=user_profile[user].get(s,0)+score
			job_word_count[s]=job_word_count.get(s,0.0)+1.0
	job_idf=cal_idf(job_word_count,doc_count)
	user_idf=cal_idf(user_word_count,doc_count)
	user_profile=get_top(user_profile,job_idf,"userProfile",k=20)
	job_profile=get_top(job_profile,user_idf,"jobProfile",k=20)
	return user_profile,[]#job_profile

def cal_idf(word_count,count):
	result={}
	for word,num in list(word_count.items()):
		result[word]=math.log(1.0*count/num)
	return result
		
def get_top(profile,idf,prefix,k=20):
	results={}
	for key,l in list(profile.items()):
		l=[[f,score*idf[f]]for f,score in list(l.items())]
		l=sorted(l,key=lambda s:s[1],reverse=True)[0:k]
		l=[["{}:{}".format(prefix,f),score] for [f,score] in l if score >0]
		if len(l)>0:
			results[key]=l
	return results

user_feature_path=sys.argv[1]	
jd_feature_path=sys.argv[2]	

user_feature=read_feature(user_feature_path)
jd_feature=read_feature(jd_feature_path)
user_profile,job_profile=cal_profile("feature/train_data",user_feature,jd_feature)
with open("feature/user_profile","w") as f:
	json.dump(user_profile,f,ensure_ascii=False)
#with open("feature/job_profile","w") as f:
#	f.writelines("\n".join(job_profile))



