# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from cnn_mnist.my_nn import my_model
import cnn_mnist.my_load_data as my_load

"""
Created by gaoyw on 2018/5/22
"""

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


model = my_model()
mnist = my_load.load_data()
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.cross_entropy)  # 使得误差最小
sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())  # 初始化

def compute_accuracy(v_xs, v_ys):
    y_pre = sess.run(model.prediction, feed_dict={model.train_x: v_xs, model.keep_prob: 1})  # 预测值  keep_prob=1表示不抛弃
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # 是否预测正确，equal返回一个bool值
    correct_prediction = tf.Print(correct_prediction, [correct_prediction,correct_prediction.shape])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求准确率
    result = sess.run(accuracy, feed_dict={model.train_x: v_xs, model.train_y: v_ys, model.keep_prob: 1})
    return result

def train_proc():
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 用SGD训练，一次训练100个数据
        sess.run(train_step, feed_dict={model.train_x: batch_xs, model.train_y: batch_ys, model.keep_prob: 0.5})
        # 每过100批，测试一下准确率
        if i % 100 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))

if __name__ == "__main__":
    train_proc()