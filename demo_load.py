# -*- coding: utf-8 -*-
import tempfile
import tensorflow as tf
import matplotlib.pyplot as plt

train_files = tf.train.match_filenames_once("skin_train.tfrecords")


# 定义一个解析TFRecord文件的方法
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    ## 解析从读取到的内容
    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    # 将string转换回来
    retyped_imaged = tf.cast(decoded_images, tf.float32)
    # reshape 成图片矩阵
    images = tf.reshape(retyped_imaged, [225, 225, 3])

    labels = tf.cast(features['label'], tf.int64)
    return images, labels


image_size = 299  # 定义神经网络输入层图片的大小。
batch_size = 100  # 定义组合数据batch的大小。
shuffle_buffer = 10000  # 定义随机打乱数据时buffer的大小。

# 从TFRecord文件创建数据集。这里可以提供多个文件。

dataset = tf.data.TFRecordDataset(train_files)

# map()函数表示对数据集中的每一条数据进行调用解析方法。
dataset = dataset.map(parser)

# 对数据进行shuffle和batching操作。这里省略了对图像做随机调整的预处理步骤。
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

# 重复NUM_EPOCHS个epoch。
NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)



# 定义遍历数据集的迭代器。
iterator = dataset.make_initializable_iterator()

# 读取数据，可用于进一步计算
image, label = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    for i in range(10):
        x, y = sess.run([image, label])
        print(y)



