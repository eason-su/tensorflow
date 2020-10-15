#coding：utf-8

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

BATCH_SIZE = 8  #一次喂入神经网络多少组数据

seed = 23455  #

#基于seed产生随机数

rng = np.random.RandomState(seed)

#随机数返回32行2列的矩阵 表示32组 体积和体重 作为输入数据集

X = rng.rand(32,2)

#从X这个32行2列的矩阵中 取出一行 判断如果和小于1 就给Y赋值1  如果和不小于1  则给Y赋值0

#Y作为输入数据集的标签（正确答案）  人为给出零件的合格与否 1合格  0不合格

Y = [[int(x0 + x1 < 1)] for (x0,x1) in X]

#上面一行代码类似于下面

#for (x0,x1) in X:
#    if x0+x1<1:
#        Y=1
#    else:
#        Y=0

print("X:\n",X)
print("Y:\n",Y)

#1定义神经网络的输入、参数、输出，定义前向传播过程

x = tf.placeholder(tf.float32, shape=(None, 2))  #输入有 体积 重量 两个特征  数据组数不定
y_ = tf.placeholder(tf.float32, shape=(None, 1))  #标准答案  每个标签一个元素
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))  #[2,3]中 2对应x  3表示隐藏层用3个神经元
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))  #[3,1]中 1对应y  3表示隐藏层用3个神经元

#前向传播过程描述
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#2定义损失函数及反向传播方法。
loss = tf.reduce_mean(tf.square(y-y_)) #均方误差
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)  #梯度下降  学习率0.001
#train_step = tf.train.MomentumOptimize(0.001，0.9).minimize(loss) #Momentum优化器
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss) #Adam优化器

#3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer() #变量初始化
    sess.run(init_op)

    #输出目前（未经训练）的参数取值
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))
    print("\n")

    #训练模型
    STEPS = 3000 #训练3000轮
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training step(s),loss on all data is %g" % (i, total_loss))

    #输出训练后的参数取值
    print("\n")
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))

