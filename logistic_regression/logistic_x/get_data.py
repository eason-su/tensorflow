import random
def generate_data(num):
	results=[]
	count=0
	while True:
		x1=random.uniform(0,10)
		x2=random.uniform(0,10)
		if abs(x1-x2)<2:
			continue
		
		y=1 if x1>x2 else 0
		results.append(str([[x1,x2],[y]]))
		count+=1
		if count>=num:
			break
	return results
train_data=generate_data(500)
test_data=generate_data(100)
with open("train_data","w") as f :
	f.writelines("\n".join(train_data))
with open("test_data","w") as f :
	f.writelines("\n".join(test_data))
