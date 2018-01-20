# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import threadpool
# 图片文件路径
cwd = 'images/'

# 疾病目前设置为10种
classes = {'多形性红斑', '感染性胼胝', '呼吸道念珠菌感染', '化脓性汗腺炎', '进行性色素性皮肤病', '肉芽肿', '湿疹', '荨麻疹', '银屑病'}

# 要生成的文件
writer = tf.python_io.TFRecordWriter("skin_train.tfrecords")

j = 0


# 随机调整颜色
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    else:
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)


# 随机调整大小方向旋转
def preprocess_for_train(image, height, width, bbox):

    # 查看是否存在标注框，如果没有标注框，则认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机的截取图片中一个块，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.7)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.7)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。大小调整的算法是随机选择的
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像的色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


# 定义函数转变变量类型
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成训练数据
def make_example(label, image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


# 循环处理数据存储到tfr
def dispose():
    j = 0
    for index, name in enumerate(classes):
        class_path = cwd + name + '/'
        for img_name in os.listdir(class_path):
            try:
                img_path = class_path + img_name  # 每一个图片的地址
                image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
                with tf.Session() as sess:
                    img_data = tf.image.decode_jpeg(image_raw_data)
                    for i in range(10):
                        result = preprocess_for_train(img_data, 225, 225, None)
                        print(result)
                       # img_dispose = tf.image.convert_image_dtype(result.eval(),dtype=tf.uint8)
                        example = make_example(index, result.eval())
                        writer.write(example.SerializeToString())
            except:
                pass

            j = j + 1
            print("已经处理好了" + str(j) + "张" + str(name) + "图片")




#print("已经处理好了所有图片，并且储存成功，敬请使用吧")

pool = threadpool.ThreadPool(10)
requests = threadpool.makeRequests(dispose())
[pool.putRequest(req) for req in requests]
pool.wait()
