# -*- coding: utf-8 -*-
# 从 TFRecord 中读取并保存图片
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load():
    SAVE_PATH = 'skin_train.tfrecords'
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([SAVE_PATH])

    # 从 TFRecord 读取内容并保存到 serialized_example 中
    _, serialized_example = reader.read(filename_queue)
    # 读取 serialized_example 的格式
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # 解析从 serialized_example 读取到的内容
    images = tf.decode_raw(features['image_raw'], tf.uint8)
    labels = tf.cast(features['label'], tf.int64)

    with tf.Session() as sess:
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 因为我这里只有 2 张图片，所以下面循环 2 次
        for i in range(10000):
            # 获取一张图片和其对应的类型
            label, image = sess.run([labels, images])
            # 将string转换回来
            image = np.fromstring(image, dtype=np.float32)
            # reshape 成图片矩阵
            image = tf.reshape(image, [225, 225, 3])
            print(image)
            plt.imshow(image.eval())
            plt.show()
            # 因为要保存图片，所以将其转换成 uint8
            image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
            # 按照 jpeg 格式编码
            image = tf.image.encode_jpeg(image)
            # 保存图片
            # with tf.gfile.GFile('pic_%d.jpg' % i, 'wb') as f:
               # f.write(sess.run(image))


if __name__ == '__main__':
    load()