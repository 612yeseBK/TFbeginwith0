# -*- coding: UTF-8 -*-
import numpy as np
import os
import tensorflow as tf
"""
Created by gaoyw on 2018/5/16
"""
def testgraph():
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0, name="b")  # also tf.float32 implicitly
    total = a + b
    print(a)  # Tensor("Const:0", shape=(), dtype=float32)
    print(b)  # Tensor("b:0", shape=(), dtype=float32)
    print(total)  # Tensor("add:0", shape=(), dtype=float32)
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    sess = tf.Session()
    print(sess.run(total))  # 7.0
    print(sess.run({'ab': (a, b), 'total': total}))  # {'ab': (3.0, 4.0), 'total': 7.0}
    sess.close()

'''使用占位符去进行运算操作'''
def testpalceholder():
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y
    sess = tf.Session()
    print(sess.run(z, feed_dict={x: 3, y: 4.5}))
    print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))


'''从数据集里面获取数据'''
def testdataset():
    my_data = [
        [0, 1, ],
        [2, 3, ],
        [4, 5, ],
        [6, 7, ],
    ]
    slices = tf.data.Dataset.from_tensor_slices(my_data)
    next_item = slices.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(next_item))
            except tf.errors.OutOfRangeError:
                break

def testlayer():
    # 创建一个节点的密集层，传入n个三元数组，units=1输出二维数组
    # shape=[None,3]应该是指多个三元数组
    # linear_model是定义了一个层，层接收输入，并产生输出，units参数表示输出的节点
    x = tf.placeholder(tf.float32, shape=[None, 3])
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # x是一个shape=[2,3]的数组，对其中每一个三元数组都会经过层的全连接，并产生一个输出,这里全连接层的权值因当时由tf自己随机初始化的
        print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
        '''[[-0.01844335]
            [-1.0990629 ]]'''
    x1 = tf.placeholder(tf.float32, shape=[None, 3])
    y1 = tf.layers.dense(x1, units=2)  # 这是另一个简化写法，可以将层的输入直接作为参数传入进去，这里我们设置对每一组输入，其输出的节点是两个
    # 这种方式没有办法访问layer对象，可能会对调试有问题，后面再来重新测试
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(y1, {x1: [[1, 2, 3], [4, 5, 6]]}))
        '''[[ 2.3299496 -2.353556 ]
            [ 4.2549276 -5.1509485]]'''

'''weindows不要使用，我的有bug'''
def testtensorboard():
    # 存储event文件夹
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sum_path = os.path.join(dir_path, 'models')  # 不要使用斜杠
    # 构造计算图
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0)
    total = a + b
    # 写入tensorboard该要
    writer = tf.summary.FileWriter(sum_path)
    writer.add_graph(tf.get_default_graph())
    # 运行图
    sess = tf.Session()
    sess.run(total)

'''这里的内容可以在7特征列的一些使用了里面看到'''
def testfeatureColumn():
    features = {
        'sales': [[5], [10], [8], [9]],
        'department': ['sports', 'sports', 'gardening', 'gardening']
    }
    department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
    department_column = tf.feature_column.indicator_column(department_column)

    columns = [
        tf.feature_column.numeric_column('sales'),
        department_column
    ]
    inputs = tf.feature_column.input_layer(features, columns)
    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    with tf.Session() as sess:
        print(sess.run((var_init, table_init)))  # (None, None)
        print(sess.run(inputs))
        # 第一和二列的1,0表示sports和gardening的onehot编码，第三列表示销售额
        '''[[ 1.  0.  5.]
             [ 1.  0. 10.]
             [ 0.  1.  8.]
             [ 0.  1.  9.]]'''

def testtrain():
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
    # 构建从输入到输出的训练的连接层
    linear_model = tf.layers.Dense(units=1)
    # 输出的预测权值
    y_pred = linear_model(x)
    # 定义损失函数
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    # 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    # 使用优化器使得训练函数最小化
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # 执行训练过程100次
    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)
    print(sess.run(y_pred))


if __name__ == "__main__":
    # testgraph()
    # testpalceholder()
    # testdataset()
    # testlayer()
    # testtensorboard()
    # testfeatureColumn()
    testtrain()