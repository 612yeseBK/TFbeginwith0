# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import os

"""
Created by gaoyw on 2018/5/17
"""
def test_name_scope():
    c_0 = tf.constant(0, name="c")  # 得到名称为c的操作
    print(c_0)  #Tensor("c:0", shape=(), dtype=int32)
    c_1 = tf.constant(2, name="c")  # 重名！得到名称为c_1的操作
    print(c_1) # Tensor("c_1:0", shape=(), dtype=int32)


    # 外层的命名空间.
    with tf.name_scope("outer"):
        c_2 = tf.constant(2, name="c")  # outer/c
        print(c_2)  # Tensor("outer/c:0", shape=(), dtype=int32)
        with tf.name_scope("inner"):  # 嵌套的命名空间
            c_3 = tf.constant(3, name="c")  # outer/inner/c
            print(c_3)  # Tensor("outer/inner/c:0", shape=(), dtype=int32)

        c_4 = tf.constant(4, name="c")  # 重名！变为outer/c_1
        print(c_4)  # Tensor("outer/c_1:0", shape=(), dtype=int32)

        with tf.name_scope("inner"):
            c_5 = tf.constant(5, name="c")  # 重名！变为outer/inner_1/c
            print(c_5)  # Tensor("outer/inner_1/c:0", shape=(), dtype=int32)

def test_device():
    # 这里创建的操作自动选择，优先GPU
    weights = tf.random_normal([3, 2])
    with tf.device("/device:CPU:0"):
        # 这里创建的操作将使用CPU
        img = tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)
    with tf.device("/device:GPU:0"):
        # 这里创建的操作将使用GPU.
        result = tf.matmul(weights, img)
    sess = tf.Session()
    tf.global_variables_initializer().run(session = sess)
    print(sess.run(result))
    '''[[ 2.0981257  2.9940946  3.8900633]
         [ 8.1600275 11.460288  14.76055  ]
         [-2.6751652 -3.7684445 -4.861724 ]]'''

    train_batch = tf.constant([2.3])
    with tf.device("/job:ps/task:0"):  # 一些参数用来分配运算任务给哪个机器的
        weights_1 = tf.Variable(tf.truncated_normal([784, 100]))
        biases_1 = tf.Variable(tf.zeroes([100]))

    with tf.device("/job:ps/task:1"):
        weights_2 = tf.Variable(tf.truncated_normal([100, 10]))
        biases_2 = tf.Variable(tf.zeroes([10]))

    with tf.device("/job:worker"):
        layer_1 = tf.matmul(train_batch, weights_1) + biases_1
        layer_2 = tf.matmul(train_batch, weights_2) + biases_2

    # 使用tf.train.replica_device_setter自动分配
    with tf.device(tf.train.replica_device_setter(ps_tasks=3)):
        # tf.Variable objects对象会被自动循环分配任务
        w_0 = tf.Variable(...)  # placed on "/job:ps/task:0"
        b_0 = tf.Variable(...)  # placed on "/job:ps/task:1"
        w_1 = tf.Variable(...)  # placed on "/job:ps/task:2"
        b_1 = tf.Variable(...)  # placed on "/job:ps/task:0"

        input_data = tf.placeholder(tf.float32)  # placed on "/job:worker"
        layer_0 = tf.matmul(input_data, w_0) + b_0  # placed on "/job:worker"
        layer_1 = tf.matmul(layer_0, w_1) + b_1  # placed on "/job:worker"


def test_tensor_like():
    '''tf.Tensor
        tf.Variable
        numpy.ndarray
        list（以及类似于张量的对象的列表）
        标量 Python 类型：bool、float、int、str'''

    '''每次您使用同一个类似于张量的对象时，TensorFlow 将创建新的 tf.Tensor。
    如果类似于张量的对象很大（例如包含一组训练示例的 numpy.ndarray），且您多次使用该对象，您可能会用光内存。
    要避免出现此问题，请在类似于张量的对象上手动调用 tf.convert_to_tensor 一次'''

def test_options():
    y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

    with tf.Session() as sess:
        # 定义选项
        options = tf.RunOptions()
        options.output_partition_graphs = True
        options.trace_level = tf.RunOptions.FULL_TRACE

        # 定义元数据容器
        metadata = tf.RunMetadata()

        sess.run(y, options=options, run_metadata=metadata)

        # 打印每个设备上执行的子图结构.
        print(metadata.partition_graphs)

        # 打印每个操作执行时候的时间.
        print(metadata.step_stats)

def test_tensorboard():
    x = tf.placeholder(tf.float32, shape=[3])
    y = tf.square(x)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sum_path = os.path.join(dir_path, 'temp')  # 不要使用斜杠

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(sum_path, sess.graph)
        print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
        writer.close()

def test_multi_graph():
    g_1 = tf.Graph()
    with g_1.as_default():
        # Operations created in this scope will be added to `g_1`.
        c = tf.constant("Node in g_1")

        # Sessions created in this scope will run operations from `g_1`.
        sess_1 = tf.Session()

    g_2 = tf.Graph()
    with g_2.as_default():
        # Operations created in this scope will be added to `g_2`.
        d = tf.constant("Node in g_2")

    # Alternatively, you can pass a graph when constructing a `tf.Session`:
    # `sess_2` will run operations from `g_2`.
    sess_2 = tf.Session(graph=g_2)

    assert c.graph is g_1
    assert sess_1.graph is g_1

    assert d.graph is g_2
    assert sess_2.graph is g_2

    # Print all of the operations in the default graph.
    g = tf.get_default_graph()
    print(g.get_operations())  # []
    print(g_1.get_operations())  # [<tf.Operation 'Const' type=Const>]
    print(g_2.get_operations())  # [<tf.Operation 'Const' type=Const>]

if __name__ == "__main__":
    # test_name_scope()
    # test_device()
    # test_options()  # 有bug
    test_tensorboard()
    # test_multi_graph()
