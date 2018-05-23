# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.tools import inspect_checkpoint as chkp

"""
Created by gaoyw on 2018/5/17
"""


def test_save():
    v1 = tf.get_variable("v1", shape=[5], initializer=tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape=[5], initializer=tf.ones_initializer)

    # 定义一些运算操作
    inc_v1 = v1.assign(v1 + 3)
    dec_v2 = v2.assign(v2 - 3)

    # 初始化变量
    init_op = tf.global_variables_initializer()

    # 自动添加默认图的所有变量，保存和恢复
    saver = tf.train.Saver()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sum_path = os.path.join(dir_path, 'temp/test1.ckpt')  # 不要使用斜杠

    with tf.Session() as sess:
        sess.run(init_op)
        inc_v1.op.run()
        dec_v2.op.run()

        # 执行保存操作.
        save_path = saver.save(sess, sum_path)
        print("Model saved in path: %s" % save_path)


def test_restore():
    tf.reset_default_graph()

    v1 = tf.get_variable("v1", shape=[3])
    v2 = tf.get_variable("v2", shape=[5])

    saver = tf.train.Saver()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sum_path = os.path.join(dir_path, 'temp/test1.ckpt')  # 不要使用斜杠
    with tf.Session() as sess:
        # 恢复的时候是根据变量名字来匹配的，并且变量的shape应当与保存的是一致的
        # 对于那些已经保存的变量，可以不用初始化了
        saver.restore(sess, sum_path)
        print("Model restored.")
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())


# 这里是指存储部分变量的用法
def test_save_part():
    tf.reset_default_graph()

    # 在第二遍运行时候可以修改v1的[3]为2，但不可以修改v2的[5]，会导致与存储的变量形状不同而失败
    v1 = tf.get_variable("v1", [3], initializer=tf.zeros_initializer)
    v2 = tf.get_variable("v2", [5], initializer=tf.ones_initializer)

    # 传入参数，存储一个v2
    saver = tf.train.Saver({"v2": v2})

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sum_path = os.path.join(dir_path, 'temp/test3.ckpt')  # 不要使用斜杠

    with tf.Session() as sess:
        v1.initializer.run()  # 因为v1没有被保存，所以需要初始化
        v2.initializer.run()  # 第二遍运行时候注释此行
        saver.save(sess, sum_path)  # 第二遍运行时候注释此行
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())


def test_read_part():
    tf.reset_default_graph()

    # 在第二遍运行时候可以修改v1的[3]，但不可以修改v2的[5]，会导致与存储的变量形状不同而失败
    v1 = tf.get_variable("v1", [2], initializer=tf.zeros_initializer)
    v2 = tf.get_variable("v2", [5], initializer=tf.ones_initializer)

    saver = tf.train.Saver({"v2": v2})  # 这里的字典是指我们将要从保存的变量里面恢复的值，这里

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sum_path = os.path.join(dir_path, 'temp/test3.ckpt')  # 不要使用斜杠

    with tf.Session() as sess:
        v1.initializer.run()  # 因为v1没有被保存，所以需要初始化
        saver.restore(sess, sum_path)  # 读取v2

        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())


def test_read_part2():
    tf.reset_default_graph()

    # 在第二遍运行时候可以修改v1的[3]，但不可以修改v2的[5]，会导致与存储的变量形状不同而失败
    v1 = tf.get_variable("v1", [5], initializer=tf.zeros_initializer)
    v2 = tf.get_variable("v2", [5], initializer=tf.ones_initializer)

    saver = tf.train.Saver({"v1": v2})  # 这里的字典是指我们将要从保存的变量里面恢复的值，这里的key是指从保存的变量名称，value是指实际存入的值
    # 当我们从test1的文件里面获取相应的值，这时候我们就是从那里面将名字为v1的变量值取出来，并将它赋给了我们这里的v2,
    # saver = tf.train.Saver()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sum_path = os.path.join(dir_path, 'temp/test1.ckpt')  # 不要使用斜杠

    with tf.Session() as sess:
        v1.initializer.run()  # 因为v1没有被保存，所以需要初始化
        saver.restore(sess, sum_path)  # 读取v2

        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())
        '''v1 : [0. 0. 0. 0. 0.]
            v2 : [3. 3. 3. 3. 3.]'''


def test_inspect_check():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sum_path = os.path.join(dir_path, 'temp/test3.ckpt')

    # 打印全部
    chkp.print_tensors_in_checkpoint_file(sum_path, tensor_name='', all_tensors=True, all_tensor_names=True)
    '''tensor_name:  v2
        [1. 1. 1. 1. 1.]'''

    # 只打印v1，由于上一个案例中没有存储v1，所以会失败
    chkp.print_tensors_in_checkpoint_file(sum_path, tensor_name='v1', all_tensors=False, all_tensor_names=False)
    # 如果all_tensor_names是true，则会打印所有的名字，all_tensors为true，则会打印出所有的变量
    '''tensor_name:  v1
        Key v1 not found in checkpoint'''

    # 只打印v2
    chkp.print_tensors_in_checkpoint_file(sum_path, tensor_name='v2', all_tensors=False, all_tensor_names=False)
    '''tensor_name:  v2
        [1. 1. 1. 1. 1.]'''


def test_save_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_path = os.path.join(dir_path, 'model')
    v1 = tf.get_variable("v1", [5], initializer=tf.zeros_initializer)
    v2 = tf.get_variable("v2", [5], initializer=tf.ones_initializer)
    sum = v1 + v2
    builder = tf.saved_model.builder.SavedModelBuilder(exp_path)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 这个方法假设变量都已经初始化好了
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.TRAINING]  # 这里是标签，可以使用自己自定义的，也可以使用系统定义好的参数，
                                             # 如tf.saved_model.tag_constants.SERVING与tf.saved_model.tag_constants.TRAINING
                                             )
        builder.save()


def test_load_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_path = os.path.join(dir_path, 'model')
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], exp_path)
        x = sess.graph.get_tensor_by_name('v1:0')
        y = sess.graph.get_tensor_by_name('v2:0')
        print(sess.run(x))
        '''SavedModel file does not exist at:
        E:\machinelearning\tensorflow文档\我的tensorflow学习记录\tensorflow初次学习实践\model/{saved_model.pbtxt|saved_model.pb}
        这里由于是运行在windows系统里面，所以tensorflow默认添加的是linux里面的路径连接符反斜杠，无法在windows里面找到对应的文件'''
        # 通常情况下，我们用于使用的时候，需要获取输入的变量，一般是一个placeholder，但是只有知道这个tensor的名字才能获取到



# 我们可以使用signature来用以约定如何获得变量
def test_save_model_signature():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_path = os.path.join(dir_path, 'model_2')
    v1 = tf.get_variable("v1", [5], initializer=tf.zeros_initializer)
    v2 = tf.get_variable("v2", [5], initializer=tf.ones_initializer)
    v3 = tf.get_variable("v3", [5], initializer=tf.ones_initializer)
    sum = v1 + v2
    # 将输入变量v1,v2设置字典key为转变成input_x，pro_x,输出变量设置key为，pro_x
    inputs = {'input_x': tf.saved_model.utils.build_tensor_info(v1),
              'pro_x': tf.saved_model.utils.build_tensor_info(v2)}
    outputs = {'output': tf.saved_model.utils.build_tensor_info(v3)}
    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')
    # 建立一个signature对象，test_sig_name是将对象转变成string的方法名
    builder = tf.saved_model.builder.SavedModelBuilder(exp_path)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 这个方法假设变量都已经初始化好了
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.TRAINING],  # 这里是标签，可以使用自己自定义的，也可以使用系统定义好的参数，
                                             # 如tf.saved_model.tag_constants.SERVING与tf.saved_model.tag_constants.TRAINING
                                             {'test_signature': signature}  # 将所有的signature对象注册到graph里面，很明显，这里可以注册很多个
                                             )
        builder.save()


def test_load_signature():
    signature_key = 'test_signature'
    input_key = 'input_x'
    output_key = 'output'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_path = os.path.join(dir_path, 'model')
    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], exp_path)
        signature = meta_graph_def.signature_def  # 获取所有的签名文件
        x_tensor_name = signature[signature_key].inputs[input_key].name  # 获取我们需要的那份签名里面的变量名称
        y_tensor_name = signature[signature_key].outputs[output_key].name
        x = sess.graph.get_tensor_by_name(x_tensor_name)  # 根据变量名获取变量本身
        y = sess.graph.get_tensor_by_name(y_tensor_name)
        print(sess.run(x))
        '''SavedModel file does not exist at:
        E:\machinelearning\tensorflow文档\我的tensorflow学习记录\tensorflow初次学习实践\model/{saved_model.pbtxt|saved_model.pb}
        这里由于是运行在windows系统里面，所以tensorflow默认添加的是linux里面的路径连接符反斜杠，无法在windows里面找到对应的文件'''
        # 通常情况下，我们用于使用的时候，需要获取输入的变量，一般是一个placeholder，但是只有知道这个tensor的名字才能获取到


if __name__ == "__main__":
    # test_save()
    # test_restore()
    # test_save_part()
    # test_read_part()
    # test_read_part2()
    # test_inspect_check()
    # test_save_model()
    # test_load_model()
    test_save_model_signature()