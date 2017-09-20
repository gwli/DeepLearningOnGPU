# -*- coding:utf-8 -*-
import numpy as np
def sample_data(size, length=100):
    """ 随机mean=4 std=1.5的数据 :param size: :param length: :return: """
    data = []
    for _ in range(size):
    data.append(sorted(np.random.normal(4, 1.5, length)))

def random_data(size, length=100):
    """ 随机生成数据 :param size: :param length: :return: """
    x = np.random.random(length)
    data.append(x)

def preprocess_data(x):
    """ 计算每一组数据平均值和方差 :param x: :return: """
    return [[np.mean(data), np.std(data)] for data in x]



#G 和 D 的连接之间也需要做出处理。
# 先求出G_output3的各行平均值和方差
MEAN = tf.reduce_mean(G_output3, 1) # 平均值，但是是1D向量
MEAN_T = tf.transpose(tf.expand_dims(MEAN, 0)) # 转置
STD = tf.sqrt(tf.reduce_mean(tf.square(G_output3 - MEAN_T), 1))
DATA = tf.concat(1, [MEAN_T,
tf.transpose(tf.expand_dims(STD, 0))] # 拼接起来
