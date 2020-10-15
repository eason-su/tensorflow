# -*- encoding:utf-8 -*-
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import math
def cal_distance(v1,v2):
	return sum([ math.pow(s1-s2,2) for [s1,s2] in  zip(v1,v2)])
def cal_cluster_distance(label_X,centers):
	center_distance_num=[[0,0] for _ in centers]
	for label,x in label_X:
		center=centers[label]
		d=cal_distance(center,x)
		center_distance_num[label][0]+=d
		center_distance_num[label][1]+=1
	d=sum([ sum_distance/num for [sum_distance,num] in center_distance_num])
	return d
def find_clusters_num(X,max_num,min_num):
	k_inner={}
	#从k=2遍历到k=10
	for k in range(min_num,max_num+1):
		model=KMeans(n_clusters=k)
		model.fit(X)
		y_pred = model.predict(X)
		label_X=list(zip(y_pred,X))
		centers=model.cluster_centers_.tolist()
		inner=cal_cluster_distance(label_X,centers)
		k_inner[k]=inner
	max_delta_distance=-1
	max_delta_k=-1
	for k in range(min_num,max_num):
		d=k_inner[k]-k_inner[k+1]
		if d>max_delta_distance:
			max_delta_distance=d
			max_delta_k=k+1
	return max_delta_k
	
	

# X为样本特征，Y为样本簇类别，共1000个样本，每个样本2个特征，对应x和y轴，共4个簇，
# 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],cluster_std=[0.4, 0.2, 0.2, 0.2])
#X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1]],cluster_std=[0.4, 0.2, 0.2])
k=find_clusters_num(X,10,2)
print(k)
'''
model=KMeans(n_clusters=10)
model.fit(X)
y_pred = model.predict(X)
print y_pred
'''
