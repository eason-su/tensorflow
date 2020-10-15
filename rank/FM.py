import json
import math
import xlearn as xl
import os
def sigmoid(x):
	return 1.0/(1+math.exp(-1*x))
class FM:
	def __init__(self):
		self.item={}
		self.bias={}
		self.vector={}
	def train(self,data_path,model_path,lr=0.2,regular_lambda=0.002,epoch=50):
		fm_model = xl.create_fm()
		fm_model.setTrain(data_path)
		os.system("mkdir {}".format(model_path))
		fm_model.setTXTModel("{}/model.txt".format(model_path))
		param = {'task':'binary', 'lr':lr, 'lambda':regular_lambda,'epoch':epoch,'k':4}
		fm_model.fit(param, "{}/model.out".format(model_path))


	def load(self,model_path):
		with open("{}/model.txt".format(model_path)) as f:
			lines=f.readlines()
		lines=[line.strip().split(": ") for line in lines]
		for key,value in lines:
			if key=="bias":
				self.bias=float(value)
			else:
				t,index=key.split("_")
				index=int(index)
				if t=="i":
					self.item[index]=float(value)
				if t=="v":
					self.vector[index]=[float(s) for s in value.split(" ")]
	def cal_cross(self,feature):
		vector_score_list=[[self.vector[index],score] for [index,score] in feature ]
		k=len(vector_score_list[0][0])
		result=0
		for i in range(0,k):
			s1=0
			s2=0
			for vector,score in vector_score_list:
				s1+=vector[i]*score
				s2+=vector[i]*vector[i]*score*score
			result+=s1*s1-s2
		return 0.5*result
	def predict(self,feature):
		result=0
		score1=sum([self.item[index]*score for [index,score] in feature])
		score2=self.cal_cross(feature)
		result=score1+score2+self.bias
		result=sigmoid(result)
		return result
if __name__ == '__main__':
	model=FM()
	#model.train("train_data","model")
	model.load("model")
	feature="16895:1.0	6270:1.0	33476:1.0"
	feature=[s.split(":") for s in feature.split("\t")]
	feature=[ [int(index),float(score)]for [index,score] in feature]
	print(model.predict(feature))






