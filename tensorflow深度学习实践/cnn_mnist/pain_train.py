

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST-data', one_hot=True)


# 以下为自定义函数，为了方便构造网络
def compute_accuracy(v_xs, v_ys):
    global prediction  # 全局变量
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})  # 预测值  keep_prob=1表示不抛弃
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # 是否预测正确，equal返回一个bool值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求准确率
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 一个截断正态分布，即在（μ-2σ，μ+2σ）内，标准差定义为0.1
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 卷积层，padding='SAME'指填充0使图片大小不变


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 池化层


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  # dropout参数
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # -1指任意的图片个数，28*28大小，1指单通道
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

## func1 layer ##
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 激活函数
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout防止过拟合

## func2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 用softmax进行分类

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # 用交叉熵来定义损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 使得误差最小

sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())  # 初始化

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 用SGD训练，一次训练100个数据
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    # 每过100批，测试一下准确率
    if i % 100 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))