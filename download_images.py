# coding=utf-8

import os
import pymysql
import re
import requests

# 连接数据库
db = pymysql.connect("172.29.28.18", "tensorflow", "tensorflow", "tensorflow", charset="utf8")
cursor = db.cursor()
sql = "select disease_classify,case_imgs from t_case"
try:
    # 执行SQL语句
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    # 计数

    j = 0
    # 循环读取数据
    for row in results:
        disease_name = row[0]
        case_imgs = row[1]
        # 清洗以及修改url
        if disease_name is not None and case_imgs is not None:
            url = case_imgs.split(",", )
            urls = ["https://dzj-prod-1.oss-cn-shanghai.aliyuncs.com/" + x for x in url]
            name = re.split('[,;；，、 ]', disease_name, )
            most_name = name[0]
            path = "images/"
            imagepath = path + most_name

            #判断是否存在，创建文件夹
            if not os.path.exists(imagepath):
                os.makedirs(imagepath)
            #下载图片
            for pic_urls in urls:
                try:
                    pic = requests.get(pic_urls, timeout=5)  # 超时异常判断 5秒超时
                except requests.exceptions.ConnectionError:
                    print('当前图片无法下载')
                    continue
                file_name = imagepath + "/" + most_name + str(j) + ".jpg"  # 拼接图片名
                print(file_name + "         已下载完毕")
                # 将图片存入本地
                fp = open(file_name, 'wb')
                fp.write(pic.content)  # 写入图片
                fp.close()
                j += 1




except:
    print("Error: unable to fetch data")  # 关闭数据库连接
db.close()
