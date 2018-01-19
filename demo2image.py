# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np



#图片文件路径
cwd='dataset/'

#疾病目前设置为10种
classes={'多形性红斑', '感染性胼胝', '呼吸道念珠菌感染', '化脓性汗腺炎', '进行性色素性皮肤病', '肉芽肿', '湿疹', '荨麻疹', '遗传性淋巴水肿', '银屑病'}


#要生成的文件
writer= tf.python_io.TFRecordWriter("skin_train.tfrecords")



#随机调整颜色
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)

#随机调整大小方向旋转
def preprocess_for_train(image, height, width, bbox):
    # 查看是否存在标注框，如果没有标注框，则认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    #转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机的截取图片中一个块，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。大小调整的算法是随机选择的
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #使用一种随机的顺序调整图像的色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image

#定义函数转变变量类型
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#生成训练数据
def make_example(label,image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example

#循环处理数据存储到tfr
for index,name in enumerate(classes):
    class_path=cwd+name+'/'
    for img_name in os.listdir(class_path):
        img_path=class_path+img_name #每一个图片的地址

        with tf.Session() as sess:
            image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
            img_data = tf.image.decode_jpeg(image_raw_data)
            # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
            # 运行多次获得多张图片
            for i in range(10):
                result = preprocess_for_train(img_data, 225, 225, None)
                example = make_example(index,result.eval())
                # example = tf.train.Example(features=tf.train.Features(feature={
                #     "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                #     'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[result]))
                # }))  # example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  #





