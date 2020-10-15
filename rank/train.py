import json
import math
import xlearn as xl
import os
import FM
from sklearn.metrics import roc_auc_score
import numpy as np

ts=["normal","compress","combine"]
#status=["delivered","satisfied","browsed"]
for t in ts:
	model=FM.FM()
	model.train("feature/train/all_{}".format(t),"model/all_{}".format(t),epoch=500)





