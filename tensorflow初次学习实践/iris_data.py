import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv
    # 从远程下载了一个数据，下载到默认文件夹，并且需要传入文件的名称，那么train_path就是下载到本地的文件的地址
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.
    # train变量是读取了本地文件的一个数据，csv文件是一种类似于表格的数据
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    # 这里是先使用了pop（y_name）把Species这一列表格返回给train_y，然后剩下的是特征数据返回给train_x，
    train_x, train_y = train, train.pop(y_name)
    # print(train_x)
    '''
             SepalLength  SepalWidth  PetalLength  PetalWidth
    0            6.4         2.8          5.6         2.2
    1            5.0         2.3          3.3         1.0
    2            4.9         2.5          4.5         1.7

    ..           ...         ...          ...         ...

    118          4.8         3.0          1.4         0.1
    119          5.5         2.4          3.7         1.0

    [120 rows x 4 columns]
    '''
    # print(train_y)
    '''
    0      2
    1      1
    2      2
    3      0
    4      0
    5      0
          ..

    117    0
    118    0
    119    1
    Name: Species, Length: 120, dtype: int64
    '''

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    # features就是上述处理的train_x
    # labels就是train_y
    """An input function for training"""
    # Convert the inputs to a Dataset.
    # print(dict(features))
    # dict 会将features转换为字典，表格列标题是key，列的内容是value
    #tf.data.Dataset.from_tensor_slices
    '''
    函数接受一个数组并返回表示该数组切片的 tf.data.Dataset。例如，一个包含 mnist 训练数据的数组的形状为 (60000, 28, 28)。
    将该数组传递给 from_tensor_slices，会返回一个包含 60000 个切片的 Dataset 对象，其中每个切片都是一个 28x28 的图像。
    该函数的用法可以参考tensorflow里面的一些用法
    '''
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # print(dataset)
    '''<TensorSliceDataset shapes: ({PetalLength: (), PetalWidth: (), SepalWidth: (), SepalLength: ()}, ()),
     types: ({PetalLength: tf.float64, PetalWidth: tf.float64, SepalWidth: tf.float64, SepalLength: tf.float64}, tf.int64)>'''

    # Shuffle, repeat, and batch the examples.
    # shuffle会对数据进行打乱，repeat方法在结束时重启dataset，batch是对切片的dataset进行批量组合
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # print(dataset)
    '''<BatchDataset shapes: ({PetalLength: (?,), PetalWidth: (?,), SepalWidth: (?,), SepalLength: (?,)}, (?,)),
    types: ({PetalLength: tf.float64, PetalWidth: tf.float64, SepalWidth: tf.float64, SepalLength: tf.float64}, tf.int64)>'''

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
