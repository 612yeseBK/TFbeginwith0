# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf

"""
Created by gaoyw on 2018/5/15
"""

'''from_tensor_slices函数接受一个数组并返回表示该数组切片的 tf.data.Dataset'''
def testfrom_tensor_slices():
    arr = [[1.0,1.2], [2.0,2.1],[3.0,3.1], [4.0,4.1], [5.0,5.1]]
    nparr = np.array(arr)
    dataset = tf.data.Dataset.from_tensor_slices(nparr)
    print(dataset)  # <TensorSliceDataset shapes: (2,), types: tf.float64>
    print(type(dataset))  # <class 'tensorflow.python.data.ops.dataset_ops.TensorSliceDataset'>
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))
            '''[1.  1.2]
                [2.  2.1]
                [3.  3.1]
                [4.  4.1]
                [5.  5.1]'''
    '''当我们传入的是dict时，这时候返回给我们的是字典dataset,切片数目根据字典对应value的shape来处理，处理结果如下'''
    input = {"n1": nparr, "n2": nparr}
    dataset = tf.data.Dataset.from_tensor_slices(input)
    print(dataset)  # <TensorSliceDataset shapes: {n1: (2,), n2: (2,)}, types: {n1: tf.float64, n2: tf.float64}>
    print(type(dataset))  # <class 'tensorflow.python.data.ops.dataset_ops.TensorSliceDataset'>
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(one_element))
            except tf.errors.OutOfRangeError:
                break
            '''{'n1': array([1. , 1.2]), 'n2': array([1. , 1.2])}
                {'n1': array([2. , 2.1]), 'n2': array([2. , 2.1])}
                {'n1': array([3. , 3.1]), 'n2': array([3. , 3.1])}
                {'n1': array([4. , 4.1]), 'n2': array([4. , 4.1])}
                {'n1': array([5. , 5.1]), 'n2': array([5. , 5.1])}'''

    '''当我们传入的是组合对象时，它会按照组合的方式返回'''
    input = {"n1": nparr, "n2": nparr}
    dataset = tf.data.Dataset.from_tensor_slices((input,nparr))
    print(dataset)  # <TensorSliceDataset shapes: ({n2: (2,), n1: (2,)}, (2,)), types: ({n2: tf.float64, n1: tf.float64}, tf.float64)>
    print(type(dataset))  # <class 'tensorflow.python.data.ops.dataset_ops.TensorSliceDataset'>
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(one_element))
            except tf.errors.OutOfRangeError:
                break
            '''({'n2': array([1. , 1.2]), 'n1': array([1. , 1.2])}, array([1. , 1.2]))
                ({'n2': array([2. , 2.1]), 'n1': array([2. , 2.1])}, array([2. , 2.1]))
                ({'n2': array([3. , 3.1]), 'n1': array([3. , 3.1])}, array([3. , 3.1]))
                ({'n2': array([4. , 4.1]), 'n1': array([4. , 4.1])}, array([4. , 4.1]))
                ({'n2': array([5. , 5.1]), 'n1': array([5. , 5.1])}, array([5. , 5.1]))'''

'''batch方法将之前完全分割的按传入的参数组合起来'''
def testbatch():
    arr = [[1.0,1.2], [2.0,2.1],[3.0,3.1], [4.0,4.1], [5.0,5.1],[6.0,6.1]]
    nparr = np.array(arr)
    dataset = tf.data.Dataset.from_tensor_slices(nparr)
    dataset = dataset.batch(2)
    print(dataset)  # <BatchDataset shapes: (?, 2), types: tf.float64>
    print(type(dataset))  # <class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(one_element))
            except tf.errors.OutOfRangeError:
                break
            '''[[1.  1.2]
                 [2.  2.1]]
                [[3.  3.1]
                 [4.  4.1]]
                [[5.  5.1]
                 [6.  6.1]]'''
    input = {"n1": nparr, "n2": nparr}
    dataset = tf.data.Dataset.from_tensor_slices(input)
    dataset = dataset.batch(2)
    print(dataset)  # <BatchDataset shapes: {n1: (?, 2), n2: (?, 2)}, types: {n1: tf.float64, n2: tf.float64}>
    print(type(dataset))  # <class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(one_element))
            except tf.errors.OutOfRangeError:
                break
            '''{'n1': array([[1. , 1.2],
                       [2. , 2.1]]), 'n2': array([[1. , 1.2],
                       [2. , 2.1]])}
                {'n1': array([[3. , 3.1],
                       [4. , 4.1]]), 'n2': array([[3. , 3.1],
                       [4. , 4.1]])}
                {'n2': array([[5. , 5.1],
                       [6. , 6.1]]), 'n1': array([[5. , 5.1],
                       [6. , 6.1]])}'''

'''tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，
如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。'''
def testargmax():
    A = [[1, 3, 4, 5, 6]]
    B = [[1, 3, 4], [2, 4, 1]]
    a1 = tf.constant([2, 3, 5], name="a1")
    maxA = tf.argmax(A, 1)
    print(type(A))
    maxB = tf.argmax(B, 1)
    maxA1 = tf.argmax(a1, 0) #此处的axis需要设定为0
    with tf.Session() as sess:
        print(sess.run(maxA))  # [4]
        print(sess.run(maxB))  # [2 1]
        print(sess.run(maxA1))  # 2

'''这个tf.newaxis是为了帮助数组创建一个新的维度，其中引号:表示原来的数组
可以从shape上面考虑这个构造，tf.newaxis在引号的哪一边就可以在原来的shape的哪一边加个1
下面a1_new1就是shape（2，）加了一个1，变成了（1,2）
'''
def testnewaxis():
    a1 = tf.constant([2, 2], name="a1")
    print(a1)
    a1_new1 = a1[tf.newaxis, :]
    a1_new2 = a1[:,tf.newaxis]
    with tf.Session() as sess:
        print(sess.run(a1))  # [2 2]
        print(sess.run(a1_new1))  # [[2 2]]
        print(sess.run(a1_new2))  # [[2] [2]]

def testprint():
    sess = tf.Session()

    t = tf.constant([2,3,4,5])
    t = tf.Print(t, [t,t.shape])  # 这里只是对t定义了了一次打印计算，但是没有实际进行，只有在计算到它的时候才会实际运行
    # ,t作为输入，然后打印出[t,t.shape],然后将t的值赋值给新的t
    result = t + 1

    init = tf.global_variables_initializer()
    sess.run(init)

    print("===========")
    print(t.eval(session=sess))  # 此时t只是一个tensor，还没有计算
    print("=====")
    sess.run(result)  # 评估result时，会用到t，t的计算又会用到tf.Print，从而打印它的输入，也就是原来的t的值

if __name__=="__main__":
    # testfrom_tensor_slices()
    # testbatch()
    # testargmax()
    # testnewaxis()
    testprint()