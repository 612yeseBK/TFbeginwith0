# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf

"""
Created by gaoyw on 2018/5/16
"""
def testcreat():
    sess = tf.Session()

    v1 = tf.get_variable("v1", [1, 2, 3])  # 随机0～1之间
    v2 = tf.get_variable("v2", [1, 2, 3], initializer=tf.zeros_initializer)  # 全0
    v3 = tf.get_variable("v3", dtype=tf.int32, initializer=tf.constant([23, 42]))  # 不能同时设定shape！

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(v1))
    print(sess.run(v2))
    print(sess.run(v3))
    '''[[[ 0.6321472   0.26857984  0.02652645]
          [-0.34536093  0.11227393 -0.79649436]]]
        [[[0. 0. 0.]
          [0. 0. 0.]]]
        [23 42]
        '''
def testcollection():
    sess = tf.Session()
    v1 = tf.get_variable("v1", [1, 2, 3])
    tf.add_to_collection("mycollection", v1)  # 将v1放入名称为mycollection的集合
    mycollection = tf.get_collection("mycollection")
    print(mycollection)  # [<tf.Variable 'v1:0' shape=(1, 2, 3) dtype=float32_ref>]
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(mycollection[0]))  # 获取v1
    '''[[[ 0.15184581  0.27375472  0.39042556]
        [ 0.23136508 -0.21778977  0.02947795]]]'''
    '''若果希望变量不被训练，可以由两种办法'''
    # 将变量初始化的时候，放入到集合tf.GraphKeys.LOCAL_VARIABLES中去
    my_local = tf.get_variable(name="my_local",
                               shape=(),
                               collections=[tf.GraphKeys.LOCAL_VARIABLES])
    # 或者设置trainable为false
    my_non_trainable = tf.get_variable("my_non_trainable",
                                       shape=(),
                                       trainable=False)

def testdevice():
    # 可以把变量限定在特定设备，例如下面代码把v变量放置在GPU1下面：
    with tf.device("/device:GPU:1"):
        v = tf.get_variable("v", [1])
    # 分布式计算中变量的设备非常重要，放置不当可能导致运算缓慢甚至出错。tf.train.replica_device_setter方法可以自动化处理。
    cluster_spec = {
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
    with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
        v = tf.get_variable("v", shape=[20, 20])

def testinitializer():
    v1 = tf.get_variable("v1", [1, 2, 3])
    sess = tf.Session()
    tf.global_variables_initializer().run() # 一次性初始化所有的变量
    sess.run(v1.initializer) # 只初始化某一个变量
    sess.run(tf.report_uninitialized_variables()) # 报告那些变量没有初始化

    # tf.global_variables_initializer()初始化时候并没有固定顺序，
    # 如果某个变量依赖于其他变量的初始值进行初始化，那么需要使用initialized_value来获取某个变量初始化的值
    # 比如下面就是 w 依赖于 v 的初始化值
    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    w = tf.get_variable("w", initializer=v.initialized_value() + 1)


def test_use():
    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    w = v + 1   # 变量在运算的过程中，会自动转变成tensor
    w = tf.Print(w, [w, w.shape], message="这里打印的是w的值")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(w)
        print(sess.run(w))


'''使用assign_add可以给改变变量的值
和加号不同的地方在于，这里改变值，会将原来的变量改变，加号只是参与了运算，并没有改变变量的值
'''
def test_assign():
    sess = tf.Session()
    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    print(v)  # <tf.Variable 'v:0' shape=() dtype=float32_ref>
    assignment = v.assign_add(1)
    # 使用了assign_add之后，变量依然变成了tensor
    print(assignment)  # Tensor("AssignAdd:0", shape=(), dtype=float32_ref)
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(assignment))  # 或者assignment.op.run(), 或assignment.eval()  # 输出：1.0

def test_read():
    sess = tf.Session()
    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    w1 = v + 1
    assignment = v.assign_add(1)
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(v.read_value()))  # 0.0
    with tf.control_dependencies([w1]):  # 这里表示，进行加法计算之后，原变量的值没有改变
        print(sess.run(v.read_value()))  # 0.0
    with tf.control_dependencies([assignment]): # 这里表示，assignment进行计算之后，去查看v变量的值，这时候，打印出来的v变量已经变成了1
        w = v.read_value()
        print(sess.run(w))  # 1.0


''' 关于变量共享问题'''
# 我们在下面定义了一个卷积层的连接
def conv_relu(input, kernel_shape, bias_shape):
    # 创建变量 "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # 创建变量"biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def wrong_use():
    input1 = tf.random_normal([1, 10, 10, 32])
    x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
    x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape=[32])  # 失败！ 这种使用的错误之处在于，变量weight和biases是不是要在第二次使用时重复使用
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(x))
    '''ValueError: Variable weights already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:'''


# 这里表示在不同的作用域调用conv_relu，表明我们要创建两个不同的变量,也就是说，两层的卷积层的变量会是不同的，这代表了一个两层网络的过滤器
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # 这里创建的变量被重新命名为 "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # 这里创建的变量被重新命名为 "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])


def re_use1():
    input1 = tf.random_normal([1, 10, 10, 32])
    input2 = tf.random_normal([1, 20, 20, 32])
    with tf.variable_scope("model"):
        output1 = my_image_filter(input1)
    with tf.variable_scope("model", reuse=True):  # 注意这里reuse=True ，表明，第一个过滤器的参数会被用到这一个里面，每层的给子对应
        output2 = my_image_filter(input2)


def re_use2():
    input1 = tf.random_normal([1, 10, 10, 32])
    input2 = tf.random_normal([1, 20, 20, 32])
    with tf.variable_scope("model") as scope:
        output1 = my_image_filter(input1)
        scope.reuse_variables()  # 使用了另一种用法就行设置
        output2 = my_image_filter(input2)


def re_use3():
    input1 = tf.random_normal([1, 10, 10, 32])
    input2 = tf.random_normal([1, 20, 20, 32])
    with tf.variable_scope("model") as scope:
        output1 = my_image_filter(input1)
    with tf.variable_scope(scope, reuse=True):  # 这种方法更加推荐，相当于给某一个scope命了名，然后另一个scope初始化时选择哪一个来使用
        output2 = my_image_filter(input2)



if __name__ == '__main__':
    # testcreat()
    # testcollection()
    # testdevice()
    # test_use()
    # test_assign()
    # test_read()
    # re_use1()
    # re_use2()
    re_use3()