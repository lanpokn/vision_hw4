from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
import joblib
import os
import time
import pickle


def mkdir_or_exist(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 数据读取
def getData(filePath):
    TrainData = []
    for childDir in os.listdir(filePath):
        if 'data_batch' in childDir:
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            train = np.reshape(data[b'data'], (10000, 3, 32 * 32))
            labels = np.reshape(data[b'labels'], (10000, 1))
            fileNames = np.reshape(data[b'filenames'], (10000, 1))
            datalebels = zip(train, labels, fileNames)
            TrainData.extend(datalebels)
        if childDir == "test_batch":
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            test = np.reshape(data[b'data'], (10000, 3, 32 * 32))
            labels = np.reshape(data[b'labels'], (10000, 1))
            fileNames = np.reshape(data[b'filenames'], (10000, 1))
            TestData = zip(test, labels, fileNames)
    return TrainData, TestData


orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
# train_path = './data/features/train/'
# test_path = './data/features/test/'
# train_path = './data/fog_features/train/'
# test_path = './data/fog_features/test/'
# train_path = './data/fog_features2/train/'
# test_path = './data/fog_features2/test/'
train_path = './data/fog_features3/train/'
test_path = './data/fog_features3/test/'



# 特征提取
def getFeat(TrainData, TestData):
    # 测试数据
    for data in TestData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image) / 255.0
        #TODO use hog instead formal
        # fd1 ,hog_image= hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True,feature_vector=True)
        # fd1 ,hog_image= hog(gray, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualize=True,feature_vector=True)
        fd = hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))
        # fd = np.resize(hog_image, (16, 16)).flatten()
        # fd = np.resize(gray, (16, 16)).flatten()
        # import ipdb; ipdb.set_trace()
        # 添加标签和保存图像
        fd = np.concatenate((fd, data[1]))
        # print(fd)
        filename = list(data[2])
        fd_name = str(filename[0], encoding="utf-8").split('.')[0] + '.feat'
        mkdir_or_exist(train_path)
        fd_path = os.path.join(train_path, fd_name)
        joblib.dump(fd, fd_path)
    print("Test features are extracted and saved.")
    # 训练数据
    for data in TrainData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image) / 255.0
        # fd = np.resize(gray, (16, 16)).flatten()
        #TODO use hog instead formal
        # fd1 ,hog_image= hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True,feature_vector=True)
        # fd1 ,hog_image= hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True,feature_vector=True)
        fd = hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))
        # fd = np.resize(hog_image, (16, 16)).flatten()
        #

        # 添加标签和保存图像
        fd = np.concatenate((fd, data[1]))
        # print(fd[0])
        filename = list(data[2])
        fd_name = str(filename[0], encoding="utf-8").split('.')[0] + '.feat'
        mkdir_or_exist(test_path)
        fd_path = os.path.join(test_path, fd_name)
        joblib.dump(fd, fd_path)
    print("Train features are extracted and saved.")


# 图像灰度化
def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


if __name__ == '__main__':
    t0 = time.time()
    filePath = 'cifar-10-batches-py'
    # filePath = "data/hog"
    print('extracting feats......')
    TrainData, TestData = getData(filePath)
    getFeat(TrainData, TestData)
    t1 = time.time()
    print("Features are extracted and saved.")
    print('total time is:%f' % (t1 - t0))
