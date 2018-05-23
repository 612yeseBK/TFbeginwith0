# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf

"""
Created by gaoyw on 2018/5/14
"""
'''基本的运算'''
def basic_operation():
    v1 = tf.Variable(2)
    v2 = tf.Variable(3)
    addv = v1 + v2
    print(v1)  # <tf.Variable 'Variable:0' shape=() dtype=int32_ref>
    print(addv)  # Tensor("add:0", shape=(), dtype=int32)
    print(type(v1))  # <class 'tensorflow.python.ops.variables.Variable'> variable本身是一个类别，不是tensor
    print(type(addv))  # <class 'tensorflow.python.framework.ops.Tensor'>

    c1 = tf.constant(10)
    c2 = tf.constant(5)
    addc = c1 + c2
    print(addc)  # Tensor("add_1:0", shape=(), dtype=int32)
    print(type(addc))  # <class 'tensorflow.python.framework.ops.Tensor'>
    print(type(c1))  # <class 'tensorflow.python.framework.ops.Tensor'> constant本身是一个tensor

    # 创建一个session
    sess = tf.Session()
    # 初始化变量
    tf.global_variables_initializer().run(session=sess)

    print(addv.eval(session=sess))  # 5
    print(sess.run(addv))  # 5
    print(addc.eval(session=sess))  # 15
    #  如果这里不传入session的话 No default session is registered. Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`

    sess.close()

    # 建立一个新的空的图
    graph = tf.Graph()
    with graph.as_default():
        value1 = tf.constant([1,2])
        value2 = tf.Variable([2,4])
        mul = value2/value1

    # 以下写法会报错，很奇怪
    # mysess = tf.Session(graph=graph)
    # tf.initialize_all_variables().run(session=mysess)
    # print(mysess.run(mul))
    # print(mul.eval())


    with tf.Session(graph = graph) as mysess:
        tf.global_variables_initializer().run()
        print(mysess.run(mul))  # [2. 2.]
        print(mul.eval())  # [2. 2.]  这时候我们就不想要传入session了，因为使用了

if __name__ == "__main__":
    basic_operation()