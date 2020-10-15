import random
def generate_data(num):
	results=[]
	for i in range(0,num):
		x=random.uniform(-10,10)
		y=x*x+x+3.0+2.0*random.gauss(0,10)+0.01*random.gauss(0,50)
		results.append(str([[x],[y]]))
	return results
train_data=generate_data(500)
test_data=generate_data(100)
with open("train_paracurve_data","w") as f :
	f.writelines("\n".join(train_data))
with open("test_paracurve_data","w") as f :
	f.writelines("\n".join(test_data))
