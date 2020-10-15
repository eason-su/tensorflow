from annoy import AnnoyIndex
import random

f = 40
u = AnnoyIndex(f)
u.load('test.ann') 
v = [random.gauss(0, 1) for z in range(f)]
index,distance=u.get_nns_by_vector(v, 10,include_distances=True)
print(index)
print(distance)
