# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

"""
Created by gaoyw on 2018/5/22
"""

# mnist = tf.contrib.learn.datasets.load_dataset("mnist")
mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

def load_data():
    train_data = mnist.train.images  # Returns np.array
    print(type(train_data))
    print("train_data.shape:", train_data.shape)  # (55000, 784)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)  # 将uint8转成int32
    print("train_labels.shape;", train_labels.shape)  # (55000,)
    eval_data = mnist.test.images  # Returns np.array
    print("eval_data.shape:", eval_data.shape)  # (10000, 784)
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    print("eval_labels.shape:", eval_labels.shape)  # (10000,)
    return mnist


# data是用于进行迭代输入的数据，有特征和标签，batch_size是每一批的数目
def batch_iter(data, batch_size):
    input_x, input_y = data
    input_x = np.array(input_x)
    input_y = np.array(input_y)
    data_size = input_x.shape[0]
    num_batches_per_epoch = int((data_size-1)/batch_size)  # 一轮迭代有多少个batch，也就是遍历一次所有数据，需要分几批次送入网络
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index*batch_size
        end_index = min((batch_index+1)*batch_size, data_size)
        return_x = input_x[start_index:end_index]
        return_y = input_y[start_index:end_index]
        yield (return_x, return_y)


if __name__ == '__main__':
    load_data()
