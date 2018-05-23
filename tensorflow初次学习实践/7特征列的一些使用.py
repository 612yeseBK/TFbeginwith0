# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.feature_column.feature_column import _LazyBuilder


"""
Created by gaoyw on 2018/5/16
"""

'''numeric_column是将特征以数值方式读取进入'''
def testNumeric():
    price = {'price': [[1.], [2.], [3.], [4.]]}  # 4行样本
    column = tf.feature_column.numeric_column('price', normalizer_fn=lambda x: x + 2)
    tensor = tf.feature_column.input_layer(price, [column])
    with tf.Session() as sess:
        print(sess.run(tensor))
        '''[[3.]
             [4.]
             [5.]
             [6.]]'''

    '''使用多个特征的情况下'''
    features = {'price': [[1.], [2.], [3.], [4.]] , 'num':[[11],[22],[33],[44]]}
    columns = []
    columns.append(tf.feature_column.numeric_column('price'))
    columns.append(tf.feature_column.numeric_column('num'))
    tensors = tf.feature_column.input_layer(features, columns)
    with tf.Session() as sess:
        print(sess.run(tensors))
        '''[[11.  1.]
             [22.  2.]
             [33.  3.]
             [44.  4.]]'''


def testbucketized():
    years = {'years': [1999, 2013, 1987, 2005]}
    years_fc = tf.feature_column.numeric_column('years')
    column = tf.feature_column.bucketized_column(years_fc, [1990, 2000, 2010])
    # [1990, 2000, 2010]这三个数将时间分成了4段，对于每一段时间，我们使用onehot编码，因此下面的输出也是onehot形式
    tensor = tf.feature_column.input_layer(years, [column])
    with tf.Session() as session:
        print(session.run(tensor))
        '''[[0. 1. 0. 0.] 1999在第二段
             [0. 0. 0. 1.] 2013在第四段
             [1. 0. 0. 0.] 1987在第一段
             [0. 0. 1. 0.]] 2005在第三段'''


'''这是先将类别手工转成数字，然后再由数字表示独立的类别，也是使用了onehot'''
def testcategorical():
    pets = {'pets': [2, 3, 0, 1]}  # 猫0，狗1，兔子2，猪3
    column = tf.feature_column.categorical_column_with_identity(
        key='pets',
        num_buckets=4)
    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])
    with tf.Session() as session:
        print(session.run(tensor))
        '''[[0. 0. 1. 0.]
             [0. 0. 0. 1.]
             [1. 0. 0. 0.]
             [0. 1. 0. 0.]]'''

def testvocabulary():
    pets = {'pets': ['rabbit', 'pig', 'dog', 'mouse', 'cat']}

    column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='pets',
        vocabulary_list=['cat', 'dog', 'rabbit', 'pig', 'mouse'],
        dtype=tf.string,
        default_value=-1,
        num_oov_buckets=0  # 表示除了vocabulary_list指定的那些以外新增加的列数
    )
    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run(tensor))
        '''[[0. 0. 1. 0. 0.]
             [0. 0. 0. 1. 0.]
             [0. 1. 0. 0. 0.]
             [0. 0. 0. 0. 1.]
             [1. 0. 0. 0. 0.]]'''
    # 也可以从文件里读取特征列词汇表
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fc_path = os.path.join(dir_path, 'pets_fc.txt')
    column1 = tf.feature_column.categorical_column_with_vocabulary_list(
        key='pets',
        vocabulary_list=fc_path
        # dtype=tf.string,
        # default_value=-1,
        # num_oov_buckets=0  # 表示除了vocabulary_list指定的那些以外新增加的列数
    )
    indicator1 = tf.feature_column.indicator_column(column1)
    tensor1 = tf.feature_column.input_layer(pets, [indicator1])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run(tensor1))
        '''[[0. 0. 1. 0. 0.]
             [0. 0. 0. 1. 0.]
             [0. 1. 0. 0. 0.]
             [0. 0. 0. 0. 1.]
             [1. 0. 0. 0. 0.]]'''

'''先求出每个字符串的hash值，然后再根据hash值做分桶'''
def testhash():
    colors = {'colors': ['green', 'red', 'blue', 'yellow', 'pink', 'blue', 'red', 'indigo']}

    column = tf.feature_column.categorical_column_with_hash_bucket(
        key='colors',
        hash_bucket_size=5,
    )

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(colors, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run(tensor))
        # 以下的分类结果很明显将不一样的颜色分到了一起，可以调整分桶的数目求得更精确的分类
        '''[[0., 0., 0., 0., 1.],#green
       [1., 0., 0., 0., 0.],#red
       [1., 0., 0., 0., 0.],#blue
       [0., 1., 0., 0., 0.],#yellow
       [0., 1., 0., 0., 0.],#pink
       [1., 0., 0., 0., 0.],#blue
       [1., 0., 0., 0., 0.],#red
       [0., 1., 0., 0., 0.]]'''

'''交叉特征列'''
def testcross():
    featrues = {
        'longtitude': [19, 61, 30, 9, 45],
        'latitude': [45, 40, 72, 81, 24]
    }

    longtitude = tf.feature_column.numeric_column('longtitude')
    latitude = tf.feature_column.numeric_column('latitude')
    # 为经度和维度各自建立特征列，分别是3个分桶
    longtitude_b_c = tf.feature_column.bucketized_column(longtitude, [33, 66])
    latitude_b_c = tf.feature_column.bucketized_column(latitude, [33, 66])
    # 建立交叉特征列 ，分桶数量是12
    column = tf.feature_column.crossed_column([longtitude_b_c, latitude_b_c], 12)

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(featrues, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run(tensor))
        '''[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
             [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
             [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
             [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]]'''

'''指示列Indicator Columns和嵌入列Embeding Columns
指示列并不直接操作数据，但它可以把各种分类特征列转化成为input_layer()方法接受的特征列。

当我们遇到成千上万个类别的时候，独热列表就会变的特别长[0,1,0,0,0,....0,0,0]。
嵌入列可以解决这个问题，它不再限定每个元素必须是0或1，而可以是任何数字，从而使用更少的元素数表现数据。
指示列会将那些很多0的数据，存储为这里有多少个0，而不是存储这么多0

'''

def testembedding():
    features = {'pets': ['dog', 'cat', 'rabbit', 'pig', 'mouse']}

    pets_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
        'pets',
        ['cat', 'dog', 'rabbit', 'pig'],
        dtype=tf.string,
        default_value=-1)
    # 3表示进行嵌入之后，输出的数据，这里和词嵌入是很相似的
    column = tf.feature_column.embedding_column(pets_f_c, 3)
    tensor = tf.feature_column.input_layer(features, [column])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run(tensor))
        '''[[-0.15245248 -0.150342   -0.6848194 ]
         [-0.05933502  0.3347138  -0.533208  ]
         [-0.27823684 -0.1790009  -0.3491666 ]
         [ 0.03789547  0.5379434   0.38569418]
         [ 0.          0.          0.        ]]'''

'''上面介绍了Tensorflow用于生成特征列的9个方法（tf.feature_column...），每个方法最终都会得到分类列或者密集列，可以从tensorflow的文档里看到这几种的区别
'''

'''默认的CategoricalColumn所有分类的权重都是一样的，没有轻重主次。而权重分类特征列则可以为每个分类设置权重。'''
def testweight():
    features = {'color': [['R'], ['A'], ['G'], ['B'], ['R']],
                'weight': [[1.0], [5.0], [4.0], [8.0], [3.0]]}

    color_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B', 'A'], dtype=tf.string, default_value=-1
    )

    # 给分类结果设置权重
    column = tf.feature_column.weighted_categorical_column(color_f_c, 'weight')

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(features, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run(tensor))
        '''[[1. 0. 0. 0.]
             [0. 0. 0. 5.]
             [0. 4. 0. 0.]
             [0. 0. 8. 0.]
             [3. 0. 0. 0.]]'''


if __name__ == "__main__":
    # testNumeric()
    # testbucketized()
    # testcategorical()
    # testvocabulary()
    # testhash()
    # testcross()
    # testembedding()
    testweight()