# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    # 这里定义的是模型的输入层，将输入与特征字典进行对应处理
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    # print("======")
    # print(params['feature_columns'])
    '''[_NumericColumn(key='SepalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
        _NumericColumn(key='SepalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
         _NumericColumn(key='PetalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
         _NumericColumn(key='PetalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]'''
    # print(features)
    '''{'PetalLength': <tf.Tensor 'IteratorGetNext:0' shape=(?,) dtype=float32>,
        'SepalWidth': <tf.Tensor 'IteratorGetNext:3' shape=(?,) dtype=float32>,
        'PetalWidth': <tf.Tensor 'IteratorGetNext:1' shape=(?,) dtype=float32>,
        'SepalLength': <tf.Tensor 'IteratorGetNext:2' shape=(?,) dtype=float32>}'''

    # 下面是用来定义隐藏层的形状，参数hidden_units是我们实例化时输入的数组，表示表示共几层，每层的多少个节点，下面还有激活函数的方式，这里是设定好的激活函数
    # 这里的变量 net 表示网络的当前顶层。在第一次迭代中，net 表示输入层。在每次循环迭代时，tf.layers.dense 使用变量 net 创建一个新层，该层将前一层的输出作为其输入。
    # 第一个隐藏层使用的net是我们之前定义的输入层
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    # 这里定义的是输出层，不使用激活函数，参数n_classes由用户定义，表示输出的类别，模型会为每个类别输出一个值，用来打分
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    # print(logits)  # Tensor("dense_2/BiasAdd:0", shape=(?, 3), dtype=float32)

    # Compute predictions.
    # argmax返回的是logit里面的最大值的索引号
    predicted_classes = tf.argmax(logits, 1)
    # print(predicted_classes)  # Tensor("ArgMax:0", shape=(?,), dtype=int64)

    # model参数会在我们用户使用某个经过实例化的模型调用predict函数时，框架自动传入一个tf.estimator.ModeKeys.PREDICT
    # train对应的是tf.estimator.ModeKeys.TRAIN
    # evaluate 对应的是tf.estimator.ModeKeys.EVAL
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],  # Tensor("strided_slice_1:0", shape=(?, 1), dtype=int64) ，相当于把[2]，变成了[[2]]
            'probabilities': tf.nn.softmax(logits),  # 归一化为概率
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    # 定义损失函数 稀疏柔性最大交叉熵sparse_softmax_cross_entropy 对于分类问题很有效
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    # 对比传入的标签和预测的标签，计算预测的精确度
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    # 如果是训练的调用
    assert mode == tf.estimator.ModeKeys.TRAIN
    # 定义优化器
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    # 使用优化器使损失函数最小化
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    # 返回训练好的实例
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
