# Face Detection and Facial Expression Recognition

***

实现对于图像的人脸检测并进行表情识别。

代码参考结合了[ Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)和 [Facial-Expression-Recognition.Pytorch](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)。

## 代码简介：

***

- [final.py](./final.py)为最终将人脸检测和人脸表情识别结合的文件。
- [fer.py](./fer.py) 是pytorch对于fer2013的数据集文件。
- [preprocess_fer2013.py](preprocess_fer2013.py)是对于fer2013数据集的处理文件。
- [mainpro_FER.py](mainpro_FER.py)是对于人脸表情识别模型的训练文件。
- [utils.py](utils.py)是人脸表情识别模型用到的工具文件。
- [visualize.py](visualize.py)为对于人脸表情识别的预测文件。
- [box_utils_numpy.py](box_utils_numpy.py)是MNN的RFB模型的工具文件，有一些用到的函数。
- [ultraface_py_mnn.py](ultraface_py_mnn.py)为MNN的RFB模型的预测文件。

## 使用说明：

***

- 在imgs文件夹下，放入相关原始图片。
- 执行[final.py](final.py)文件。
- 在results文件夹下，会产生相关的结果图片。人脸画框+表情识别。