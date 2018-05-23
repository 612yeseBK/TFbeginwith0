# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import operator
import math

"""
Created by gaoyw on 2018/4/16
"""

'''
dataSet是一个list
dataSet = [[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
featList = [example[i] for example in dataSet]
这时候返回的featList是第i列组成的list
'''
def test1():
    dataSet = [ [1, 2, 3, 4, 'no'],
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    featList = [example[2] for example in dataSet]
    print("返回了数组的第3列，下标为2")
    print(featList)


'''
x=np.array([3, 1, 2])
np.argsort(x)
argsort()是numpy里的函数,用于返回一个numpy里面里面从小到大的index数组,这里返回的是数组的下标，数组本身没有变
可通过在数组前添加负号的方式改变升降序的规则
如果x是一个多维数组,那么可以在argsort里面添加参数axis=0/1
axis取0表示按列排序，某一列的数据顺序被返回了，并写入该列
axis取1表示按行排序，某一行的数据顺序被返回了，并写入该行
一维数组的axis一般不取值，在函数里面默认是-1，取0也可以，但取1不行，暂时没看源码，有时间补上
'''
def test_argsort():
    distances1 = np.array([1,3,2,5,34,3,4,8])
    print("这是一维数组，axis=-1的情况")
    print(np.argsort(distances1, axis=-1))
    print("这是一维数组，axis=0的情况，降序")
    print(np.argsort(-distances1, axis=0))
    distances = np.array([[0, 3], [1, 2]])
    print("这是二维数组，axis=1的情况")
    print(np.argsort(distances, axis=1))
    print("这是二维数组，axis=0的情况")
    print(np.argsort(distances, axis=0))


'''
s.strip(rm)，当rm空时,默认删除末尾空白符(包括'\n','\r','\t',' ')
'''
def test_strip():
    line = '1212 2 aew  '
    print("将字符串的末尾空白符删除")
    print(line.strip())

'''
 str - - 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等
 num - - 分割次数，如果为1，表示只找到第一个位置，分割完之后就不再处理了
'''
def test_split():
    str = "Line1-abcdef \nLine2-abc\tLine4- abcd"
    print("将字符串按默认所有空字符分割")
    print(str.split())
    print("将字符串按空格分割，并且只分隔第一个空格")
    print(str.split(' ', 1))

def testzwf():
    attr=[1,2,3,4,5]
    print("测试占位符：%s" % attr)
    attr = 1
    print("测试占位符：%s" % attr)
    attr = "你好"
    print("测试占位符：%s" % attr)
    attr = 0.111
    print("测试占位符：%s" % attr)

'''
set可以将一个list转换为不存在重复元素的set集合
'''
def testset():
    attr=[1,2,2,3,3,4]
    print("测试set：%s" % set(attr))

'''
测试切片函数的功能
a.extend(b)可以将b拼接到a上面
'''
def testsplit():
    attr = [1,2,3,4,5,6]
    # [:3]表示去[0,3)不包括3，前3个
    print("测试取前三个的切片功能：%s" % attr[:3])
    print("测试取4个之后的切片功能：%s" % attr[4:])
    un = attr[:3]
    un.extend(attr[4:])
    print("测试取去除第4个之后：%s" % un)

'''
list.append(object) 向列表中添加一个对象object
list.extend(sequence) 把一个序列seq的内容添加到列表中
'''
def testappendandextend():
    music_media = ['compact disc', '8-track tape', 'long playing record']
    new_media = ['DVD Audio disc', 'Super Audio CD']
    music_media2 = ['compact disc', '8-track tape', 'long playing record']
    new_media2 = ['DVD Audio disc', 'Super Audio CD']
    music_media.append(new_media)
    music_media2.extend(new_media2)
    print('music_media.append(new_media) = %s' % music_media)
    print('music_media2.extend(new_media2) = %s' % music_media2)
'''
sorted函数传入iterator数组
 key的值应该是一个函数，这个函数接收一个参数并且返回一个用于比较的关键字
 operator module中有itemgetter,attrgetter两个函数，可以更方便地获取元组中的元素和自定义对象中的属性
 reverse=true表示降序
'''
def testsorted():
    classCount = {"排第2": 2,"排第3":3,"排第1":1}
    sortedclass = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print("排序后的字典：%s" % sortedclass)

"""
 L.count(value) -> integer -- return number of occurrences of value
用来计算list里面某个元素出现的次数
"""
def testcount():
    classco = ['a','a','a',2,2,3,3,4,4]
    print("classco里面的元素%s在里面出现了%s次" % (classco[0],classco.count(classco[0])))

'''
只有set可以计算交并集
'''
def testbingji():
    a=set([1,1,2,3,4,5])
    b=set([3,4,5,6,7,8])
    c = a|b
    d = a&b
    # 只有set可以去并集交集之类的
    print("a和b取并集之后：%s" % c)
    print("a和b取交集之后：%s" % d)

'''
list乘以数字是将矩阵里面的元素复制并extend到里面
'''
def test0time():
    a = [0,1]*100
    print("[0]乘以100是多少：")
    print(a)

'''
测试相加
numpy的数组相加是矩阵加法
list相加是extend方法
'''
def testpluse():
    ps = np.zeros(10)
    a = [1,2,3,4,5,5,6,7,6,4]
    b = [1,2,3,2,2,2,2,3,3,4]
    ps = ps + a
    ps = ps + b
    c = a + b
    print('测试矩阵相加：%s' % ps)
    print('测试数组相加：%s' % c)

'''np.tile(a,shape)里面的shape参数表示的是一种延展的方式，shape这个tuple里面最又边的表示a矩阵的自底层基本元素，它的数字表示里复制的次数
shape的第二层如果有，并且a数组没有第二层，就将原来的整层看成第二层一个元素，并按照shape里的值进行拓展，
如果本身a有第二层，就在原来的第二层里面把原有的元素拓展几倍，其余的层数参照这个规则
'''
def testtile():
    # a = np.array([0, 1, 2])
    a = np.array([[0, 1], [2, 3]])
    print('np.tile(a, 2)=', np.tile(a, 2))
    print('np.tile(a, (2, 2))=', np.tile(a, (2, 2)))
    print('np.tile(a, (2, 1, 2))=', np.tile(a, (2, 2, 2)))

'''multiply和*表示两个矩阵对应元素相乘'''
def testMultiply():
    two_dim_matrix_one = np.array([[1, 2, 3], [4, 5, 6]])
    another_two_dim_matrix_one = np.array([[7, 8, 9], [4, 7, 1]])

    # 对应元素相乘 element-wise product
    element_wise = two_dim_matrix_one * another_two_dim_matrix_one
    print('element wise product: %s' % (element_wise))

    # 对应元素相乘 element-wise product
    element_wise_2 = np.multiply(two_dim_matrix_one, another_two_dim_matrix_one)
    print('element wise product: %s' % (element_wise_2))

'''数组的转置必须是符合矩阵的shape，也就是shape的长度为2的，否则不行'''
def testT():
    two_dim_matrix_one = np.array([[1, 2, 3], [4, 5, 6]])
    another_two_dim_matrix_one = np.array([[7, 8, 9], [4, 7, 1]])
    print("转置T：%s" % two_dim_matrix_one.T)
    print("转置transpose：%s" % another_two_dim_matrix_one.transpose())

'''高维数组转置
(1, 0, 2)是指下标的位置的改变，将原来下标为1的放到了第一位，下标为0的放到了第二位，下标为2的不变，我感觉直接思考这个转置过程是很困难的
元素11在a中的位置是a[0][2][1]，经过b = a.transpose(1, 0, 2)之后，11在b中的位置就变成b[2][0][1]。再比如元素28，在a中的位置a[1][2][3]，在b中为：a[2][1][3].
'''
def testHt():
    a = np.array(range(30)).reshape(2, 3, 5)
    print(a)
    print(a.transpose(1, 0, 2))


'''测试np里面数组的切片函数'''
def testqiepian():
    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20]])
    print("X[:,0]=%s" % X[:,0])
    #X[:,0]=[ 0  3  6  9 12 15 18]
    print("X[:,1]=%s" % X[:,1])
    #X[:,1]=[ 1  4  7 10 13 16 19]
    print("X[1,:]=%s" % X[1,:])
    #X[1,:]=[3 4 5]
    '''列坐标下标为[1,3)'''
    print("X[:,1:3]=\n%s" % X[:,1:3])
    '''X[:,1:3]=
[[ 1  2]
 [ 4  5]
 [ 7  8]
 [10 11]
 [13 14]
 [16 17]
 [19 20]]'''
    '''行下标为[1,3)'''
    print("X[1:3,:]=\n%s" % X[1:3, :])


'''mat是矩阵创建，它的shape只能是长度为2比如（3,2）表示一个3*2的矩阵'''
'''array是创建列表，它的shape的长度可以大于2，比如（2,3,2）'''
def testmatandarray():
    b2 = np.mat([[1, 22], [0, 23], [0, 23], [0, 23], [1, 23], [0, 23], [1, 23], [0, 23]])
    b3 = np.array([[1, 22], [0, 23], [0, 23], [0, 23], [1, 23], [0, 23], [1, 23], [0, 23]])
    print(b2)
    print(b3)

'''A或者getA()表示将一个矩阵matrix转换为数组array'''
def testgetA():
    w = [[1, 22], [0,23],[0,23],[0,23],[1,23],[0,23], [1,23], [0,23]] # w是一个list
    print(w)
    w1 = np.array(w)
    print(w1)
    w2 = np.mat(w)
    print(w2)
    w3 = w2.A
    print(w3)

'''
nonzeros(a)返回数组a中值不为零的元素的下标，
它的返回值是一个长度为a.ndim(数组a的轴数)的元组，
元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值
以下的；例子里面，nps里面两个数组对应数字可以组合成一个坐标，这个坐标就是b2里面不为0元素的坐标
'''
def testnonzero():
    b2 = np.array([[1, 22], [0,23],[0,23],[0,23],[1,23],[0,23], [1,23], [0,23]])
    nps = np.nonzero(b2)
    print(nps)
    b1 = np.array([1,2,0,4,1,0,2,0])
    nps1 = np.nonzero(b1)
    print(nps1[0])

'''
这里的数组和数值比较会返回一个和数组性状一致的bool数组，
这个bool数组作为参数会将调用者对应的true的元素返回并可以进行操作，
不过这里的调用者和bool数组的形状必须是一样的
'''
def testcompare():
    mat = np.array([1.,2.,3.,4.,5.])
    print(np.shape(mat))
    retArray = np.ones(np.shape(mat)[0])
    print(retArray)
    print(mat<=2.)
    retArray[mat<=2.] = -1
    print(retArray)
    dataMatrix = np.matrix([[1., 2.1],
                            [1.5, 1.6],
                            [1.3, 1.],
                            [1., 1.],
                            [2., 1.]])
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    print(retArray)
    print(dataMatrix[:, 0] <= 1.2)
    retArray[dataMatrix[:, 0] <= 1.2] = -1.0
    print(retArray)


'''
choice(a, size=None, replace=True, p=None)
a：一维数组或者int型变量，如果是数组，就按照里面的范围来进行采样，如果是单个变量，则采用np.arange(a)的形式

size : int 或者 tuple of ints, 可选参数
决定了输出的shape. 如果给定的是, (m, n, k), 那么 m * n * k 个采样点将会被采样. 默认为零，也就是只有一个采样点会被采样回来。

replace : 布尔参数，可选参数
决定采样中是否有重复值

p :一维数组参数，可选参数
对应着a中每个采样点的概率分布，如果没有标出，则使用标准分布
'''
def testchoice():
    a = np.random.choice(5, 3)
    print(a)
    a=np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
    print(a)
    a=np.random.choice(5, 3, replace=False)
    print(a)
    a=np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
    print(a)
    aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
    a=np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
    print(a)
    X = np.array(np.random.choice(2, size=(100,)))#(100,)在这里等于100，区别于（100,1），这里生成一行
    print(X)

'''将label里面的每个数字扩展成编码形式'''
def testone_hot():
    SIZE = 6
    CLASS = 8
    label1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])
    sess1 = tf.Session()
    print('label1:', sess1.run(label1))
    b = tf.one_hot(label1, CLASS, 1, 0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(b)
        print('after one_hot', sess.run(b))


'''tf.stack（）这是一个矩阵拼接的函数，tf.unstack（）则是一个矩阵分解的函数
axis表示是从哪一个轴进行划分
'''
def teststack():
    a = tf.constant([[1, 2, 3], [11, 22, 33]])
    b = tf.constant([[4, 5, 6], [44, 55, 66]])
    hh = tf.constant(
        [[[1,2],[3,4]],
         [[5,6],[7,8]],
         [[9,10],[11,12]]
         ]
    )
    hh1 = tf.unstack(hh,axis=1)
    c = tf.stack([a, b], axis=0)
    d = tf.unstack(c, axis=0)
    e = tf.unstack(c, axis=1)
    f = tf.unstack(c, axis=2)
    with tf.Session() as sess:
        print(sess.run(c))
        '''[[[ 1  2  3]
  [11 22 33]]
 [[ 4  5  6]
  [44 55 66]]]'''
        print(sess.run(d))
        '''[array([[ 1,  2,  3],
       [11, 22, 33]]), array([[ 4,  5,  6],
       [44, 55, 66]])]'''
        print(sess.run(e))
        '''[array([[1, 2, 3],
       [4, 5, 6]]), array([[11, 22, 33],
       [44, 55, 66]])]'''
        print(sess.run(f))
        '''[array([[ 1, 11],
       [ 4, 44]]), array([[ 2, 22],
       [ 5, 55]]), array([[ 3, 33],
       [ 6, 66]])]'''
        print(sess.run(hh1))
        '''
        [array([[ 1,  2],
       [ 5,  6],
       [ 9, 10]]), array([[ 3,  4],
       [ 7,  8],
       [11, 12]])]
        '''


'''用于矩阵的连接，这里的后一个参数是指axis'''
def testconcat():
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    t3 = tf.concat([t1, t2],1)
    with tf.Session() as sess:
        print(sess.run(t3))


def testOneHot():
    x = np.array(np.random.choice(2, size=(10,)))
    x_one_hot = tf.one_hot(x, 4)
    rnn_inputs = tf.unstack(x_one_hot, axis=1)
    with tf.Session() as sess:
        print(x)
        print(sess.run(x_one_hot))
        print(sess.run(rnn_inputs))

def testargmax():
    a=np.zeros((3,2))
    print(a)

'''python里面的log是以自然常数e作为底数的'''
def testlog():
    p=0.625
    ex = -(p*math.log(p)+(1-p)*math.log(1-p))
    ex = math.log(math.e)
    print(ex)

if __name__ == '__main__':
    # test1()
    # test_argsort()
    # test_strip()
    # test_split()
    # testzwf()
    # testset()
    # testsplit()
    # testappendandextend()
    # testsorted()
    # testcount()
    # testbingji()
    # test0time()
    # testpluse()
    # testtile()
    # testMultiply()
    # testT()
    # testHt()
    # testqiepian()
    # testmatandarray()
    # testgetA()
    # testnonzero()
    # testcompare()
    # testargmax()
    # testchoice()
    # testone_hot()
    # teststack()
    # testconcat()
    # testwatch()
    test()