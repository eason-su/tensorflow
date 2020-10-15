#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import importlib

importlib.reload(sys)
#sys.setdefaultencoding('utf-8')
import json
from gensim import corpora, models, similarities
import pickle
import random
import math


def cal_entropy(topic):
    p = [score for _, score in topic]
    s = sum(p)
    p = [score / s for score in p]
    entropy = sum([-1 * score * math.log(score) for score in p])
    return entropy


print("读取文件")
with open("forward_index","rb") as f:
    lines = f.readlines()
print("读取结束")
lines = [line.strip() for line in lines]
train = []
article_info = []
print("解析数据")
for line in lines:
    data = json.loads(line)
    if len(data["article_title_words"]) < 5:
        continue
    # 标题词，摘要词，正文词
    # 真实场景中，加权，去停用词
    train.append(data["article_abstract_words"])
    article_info.append({"article_id": data['article_id'], "__url": data['__url']})
lines = []
print("解析结束")
dictionary = corpora.Dictionary(train)
corpus = [dictionary.doc2bow(text) for text in train]
print("开始训练lda")
num_topics = 100
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
print("lda训练结束")

# 计算词的分布
all_word_num = len(list(dictionary.keys()))
word_topic_distribute = {}
for topic_num in range(0, num_topics):
    topic_list = lda.show_topic(topic_num, topn=all_word_num)
    for [word, score] in topic_list:
        if word not in word_topic_distribute:
            word_topic_distribute[word] = []
        word_topic_distribute[word].append([topic_num, score])
word_entropy = [[word, cal_entropy(topic_distribute)] for word, topic_distribute in list(word_topic_distribute.items())]
word_entropy = sorted(word_entropy, key=lambda s: s[1])
word_entropy = ["{}\t{}".format(s[0], s[1]) for s in word_entropy]
with open("word_entropy", "w") as f:
    f.writelines("\n".join(word_entropy))
# 计算文章的分布
article_result = []
for feature, article_info in zip(corpus, article_info):
    topic = lda.get_document_topics(feature)
    topic = [(s[0], float(s[1])) for s in topic]
    article_info['topic_lda'] = topic
    article_info['entropy'] = cal_entropy(topic)
    article_result.append(article_info)
article_result = sorted(article_result, key=lambda s: s['entropy'])
article_result = [json.dumps(s, ensure_ascii=False) for s in article_result]
with open("article_topics", "w") as f:
    f.writelines("\n".join(article_result))
with open("lda.model", "wb") as f:
    pickle.dump(lda, f)
with open("dictionary", "wb") as f:
    pickle.dump(dictionary, f)