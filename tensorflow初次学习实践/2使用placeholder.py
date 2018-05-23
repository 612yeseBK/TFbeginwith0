# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf

"""
Created by gaoyw on 2018/5/14
"""
def user_placeholder():
    graph1 = tf.Graph()
    with graph1.as_default():
        value1 = tf.placeholder(dtype=tf.float64)
        value2 = tf.constant([2,3],dtype=tf.float64)
        mul = value1*value2
    with tf.Session(graph=graph1) as mysess:
        # tf.global_variables_initializer().run()  #这是用来初始化变量用的，我们这里没有使用变量，所以不进行初始化也可以使用
        value = load_from_remote()
        print(mysess.run(mul,feed_dict={value1:value}))


def load_from_remote():
    return [2,3]

if __name__ == "__main__":
    user_placeholder()