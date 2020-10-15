import random
with open("/data/zhilian_job/table3_action.txt") as f:
	lines=f.readlines()[1:]#[0:1000]
lines=[line.strip() for line in lines ]
num=len(lines)/10
random.shuffle(lines)
test_data=lines[0:num]
train_data=lines[num:]
with open("feature/train_data","w") as f:
	f.writelines("\n".join(train_data))
with open("feature/test_data","w") as f:
	f.writelines("\n".join(test_data))








