from annoy import AnnoyIndex
import random
f = 40
t = AnnoyIndex(f,metric="angular")
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)
t.build(10) 
t.save('test.ann')
