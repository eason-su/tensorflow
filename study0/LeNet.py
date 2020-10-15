'''



from keras import backend as K # 兼容不同后端的代码
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

# 图片格式问题
# K.image_data_format() == 'channels_last'
# 默认是last是通道  K.set_image_dim_ordering("tf")
# K.image_data_format() == 'channels_first' #  K.set_image_dim_ordering("th")

class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv2D(20,kernel_size=5,padding='same',
                         input_shape=input_shape,activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(50,kernel_size=5,padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))

        model.add(Dense(classes,activation='softmax'))
        return model


train = pd.read_csv('train.csv')
y_train_full = train['label']
X_train_full = train.drop(['label'], axis=1)
X_test_full = pd.read_csv('test.csv')

'''