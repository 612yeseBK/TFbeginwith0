# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf

"""
Created by gaoyw on 2018/5/16
"""
def testcreat():
    '''0 rank 0阶 标量'''
    mammal = tf.Variable("Elephant", tf.string)
    ignition = tf.Variable(451, tf.int16)
    floating = tf.Variable(3.14159265359, tf.float64)
    its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)
    '''一阶张量 数组'''
    mystr = tf.Variable(["Hello"], tf.string)
    cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
    first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
    its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

    '''2阶张量 矩阵'''
    mymat = tf.Variable([[7], [11]], tf.int16)
    myxor = tf.Variable([[False, True], [True, False]], tf.bool)
    linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
    squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
    rank_of_squares = tf.rank(squarish_squares)
    mymatC = tf.Variable([[7], [11]], tf.int32)

    '''高阶张量 这是表示图片的长宽高以及颜色，是一个四阶张量'''
    my_image = tf.zeros([10, 299, 299, 3])

def testgetrank():
    t = tf.Variable([[4, 9], [16, 25]], tf.int32)
    img = tf.zeros([10, 299, 299, 3])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        v1 = sess.run(tf.rank(t))
        v2 = sess.run(tf.rank(img))
        print(v1, v2)  # 2,4

def testslice():
    vertor = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    martrix = tf.constant([[2, 5], [3, 4]])
    tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    vertorslice = vertor[1:3]
    martrixslice = martrix[:, 1]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(vertorslice))  # [2 3]
        print(sess.run(martrixslice))  # [5 4]
        print(sess.run(tensor[:, :, 1]))  # [[2 4] [6 8]]

def testgetshape():
    sess = tf.Session()
    t = tf.Variable([[4, 9], [16, 25]], tf.int32)
    img = tf.zeros([10, 299, 299, 3])
    my_matrix = tf.Variable([[1, 2], [2, 4], [4, 3]])
    zeros = tf.zeros(my_matrix.shape[1])
    print(t.shape)  # (2, 2)
    print(img.shape)  # (10, 299, 299, 3)
    print(my_matrix.shape)  # (3, 2)
    print(zeros)  # Tensor("zeros_1:0", shape=(2,), dtype=float32) 行数为2，列数未知
    sess.close()

def testreshape():
    rank_three_tensor = tf.ones([3, 4, 5])  # 3x4x5=60共60的数字立方体
    matrix = tf.reshape(rank_three_tensor, [6, 10])  # 变为6x10 matrix
    matrixB = tf.reshape(matrix, [3, -1])  # 3x20matrix，-1 表示自动计算这维度元素个数.
    matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # 4x3x5数字立方

    # notgood = tf.reshape(matrixB, [6, 2])  # 代码报错，因为6x2不等于60，无法决定取舍
    # notgood2 = tf.reshape(matrixAlt, [13, 2, -1])  # 代码报错，因为13x2无法被60整除!

    # 测试发现，reshape会将原来的元素，顺序排列，然后先依据目标shape最里层的数字去分割，然后在分割之后按照外层数字分割
    # 不一定正确
    v1 = tf.constant([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])  # 2*3*2
    v2 = tf.reshape(v1, [4, 3, 1])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(matrix))
        print(sess.run(matrixB))
        print(sess.run(v2))

'''cast对数据类型进行转换'''
def testcast():
    float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
    with tf.Session() as sess:
        print(sess.run(float_tensor))

def testeval():
    sess = tf.Session()

    constant = tf.constant([1, 2, 3])
    tensor = constant * constant
    print(tensor.eval(session=sess))  # [1 4 9]
    sess.close()


'''tf.print相当于定义了一个打印操作的计算，只有实际计算到它的时候，这个计算才会触发'''
def testprint():
    sess = tf.Session()

    t = tf.Variable(5, tf.int32)
    t = tf.Print(t, [t,t.shape])  # 这里只是对t定义了了一次打印计算，但是没有实际进行，只有在计算到它的时候才会实际运行
    result = t + 1

    init = tf.global_variables_initializer()
    sess.run(init)

    print("===========")
    print(t)  # 此时t只是一个tensor，还没有计算
    print("=====")
    sess.run(result)  # 评估result时，会用到t，t的计算又会用到tf.Print，从而打印它的输入，也就是原来的t的值


if __name__ == "__main__":
    # testgetrank()
    # testslice()
    # testgetshape()
    # testreshape()
    # testcast()
    # testeval()
    testprint()