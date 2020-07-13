# 数据集处理部分
# 将数据集分为3部分，训练集（Training）、公共测试集（PublicTest）和私有测试集（PrivateTest）
# 每一部分又分为2部分，数据和标签
# 标签: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv
import os
import numpy as np
import h5py

file = 'data/fer2013.csv'

# 创建list以存储数据和标签信息
Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []
PrivateTest_x = []
PrivateTest_y = []

# 创建数据文件data.h5
datapath = os.path.join('data', 'data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

# 循环处理每次读1行，进行相应归类处理。
with open(file, 'r') as csvin:
    data = csv.reader(csvin)
    for row in data:
        if row[-1] == 'Training':
            temp_list = []
            for pixel in row[1].split():
                temp_list.append(int(pixel))
            I = np.asarray(temp_list)
            Training_y.append(int(row[0]))
            Training_x.append(I.tolist())

        if row[-1] == "PublicTest":
            temp_list = []
            for pixel in row[1].split():
                temp_list.append(int(pixel))
            I = np.asarray(temp_list)
            PublicTest_y.append(int(row[0]))
            PublicTest_x.append(I.tolist())

        if row[-1] == 'PrivateTest':
            temp_list = []
            for pixel in row[1].split():
                temp_list.append(int(pixel))
            I = np.asarray(temp_list)

            PrivateTest_y.append(int(row[0]))
            PrivateTest_x.append(I.tolist())
print('training data shape')
print(np.shape(Training_x))
print('public test data shape')
print(np.shape(PublicTest_x))
print('private test data shape')
print(np.shape(PrivateTest_x))

# 在data.h5中保存每一个数据集部分
datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_pixel", dtype='uint8', data=Training_x)
datafile.create_dataset("Training_label", dtype='int64', data=Training_y)
datafile.create_dataset("PublicTest_pixel", dtype='uint8', data=PublicTest_x)
datafile.create_dataset("PublicTest_label", dtype='int64', data=PublicTest_y)
datafile.create_dataset("PrivateTest_pixel", dtype='uint8', data=PrivateTest_x)
datafile.create_dataset("PrivateTest_label", dtype='int64', data=PrivateTest_y)
datafile.close()

print("Save data finish")
