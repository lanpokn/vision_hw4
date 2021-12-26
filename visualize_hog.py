from skimage.feature import hog
from skimage import exposure
from skimage import data, color, exposure


import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

#注：本地电脑的pyqt5上有难以解决的不过（不是缺少链接库导致的），难以可视化，所以转到了ipynb上，代码没有改动

if __name__ == '__main__':
    img = cv2.imread('jinx.jpg')
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO: 提取hog特征
    # cv2.imshow('jinx',gray_image)
    # cv2.waitKey(0)
    plt.switch_backend('eglfs')
    # fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
    fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1),visualize=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    #hog_image_rescaled = np.repeat(hog_image_rescaled[:,:,np.newaxis], 3, axis=2)
    hog_vis = gray_image * hog_image_rescaled
    print(type(hog_image_rescaled))

    # 可视化特征
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('input image')
    ax[1].imshow(hog_vis, cmap='bone')
    ax[1].set_title('visualization of HOG features')
    plt.show()
