#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import importlib
importlib.reload(sys) 
sys.setdefaultencoding('utf-8') 
from annoy import AnnoyIndex
import random
import pickle
import json
length = 100
u = AnnoyIndex(length,metric="angular")
u.load('article.ann') 
with open("index_url") as f:
	index_url=pickle.load(f)
with open("article_topics") as f:
	lines=f.readlines()
lines=[line.strip() for line in lines]
results=[]
for line in lines:
	data=json.loads(line)
	url=data["__url"]
	aid=data["article_id"]
	index,distance=u.get_nns_by_item(aid,6,include_distances=True)
	urls=[ "{}\t{}".format(index_url[i],d) for [i,d] in zip(index,distance)]
	results.append("原始url\t{}".format(url))
	results.append("相关推荐")
	results.append("\n".join(urls)+"\n")
	
with open("relation_article","w") as f :
	f.writelines("\n".join(results))
	



