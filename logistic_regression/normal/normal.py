from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
scaler = preprocessing.StandardScaler().fit(X)
print ("mean",scaler.mean_)
print ("scale",scaler.scale_)


print (scaler.transform(X))
