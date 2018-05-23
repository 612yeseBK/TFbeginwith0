# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import cnn_mnist.my_load_data as my_load
"""
Created by gaoyw on 2018/5/22
"""
class my_model(object):
    def __init__(self):
        self.train_x = tf.placeholder(tf.float32, [None, 784])
        self.train_y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(self.train_x, [-1, 28, 28, 1])  # -1指任意的图片个数，28*28大小，1指单通道
        ## conv1 layer ##
        with tf.name_scope("conv2"):
            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))  # patch 5x5, in size 1, out size 32 一个截断正态分布，即在（μ-2σ，μ+2σ）内，标准差定义为0.1
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)  # output size 28x28x32  卷积层，padding='SAME'指填充0使图片大小不变
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # output size 14x14x32

        ## conv2 layer ##
        with tf.name_scope("conv1"):
            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)) # patch 5x5, in size 32, out size 64
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)  # output size 14x14x64
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # output size 7x7x64

        ## func1 layer ##
        with tf.name_scope("func1"):
            W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
            # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 激活函数
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)  # dropout防止过拟合

        ## func2 layer ##
        with tf.name_scope("output"):
            W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
            self.prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 用softmax进行分类

        # the error between prediction and real data
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(self.train_y * tf.log(self.prediction), reduction_indices=[1]))  # 用交叉熵来定义损失函数




