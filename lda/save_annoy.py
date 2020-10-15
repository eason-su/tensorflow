from annoy import AnnoyIndex
import random
import pickle
import json
length = 100
t = AnnoyIndex(length,metric="angular")
index_url={}
with open("article_topics") as f:
	lines=f.readlines()
lines=[line.strip() for line in lines]
for line in lines:
	data=json.loads(line)
	url=data["__url"]
	aid=data["article_id"]
	feature=[0 for _ in range(0,length)]
	index_url[aid]=url
	for index,score in data["topic_lda"]:
		feature[index]=score
    	t.add_item(aid, feature)
t.build(10) 
t.save('article.ann')
with open("index_url","w") as f:
	pickle.dump(index_url,f)


