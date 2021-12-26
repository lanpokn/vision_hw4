from skimage.feature import hog
from skimage import exposure
from skimage import data, color, exposure


import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


def read_path(file_pathname):
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        ####change to hog
        image = color.rgb2gray(img)
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
        #hog_image_rescaled = np.repeat(hog_image_rescaled[:,:,np.newaxis], 3, axis=2)
        final_img = image * hog_image_rescaled
        print(type(hog_image_rescaled))
        cv2.imshow('final_img',final_img)
        cv2.waitKey(0)
        #cv2.imwrite('/tool/HOGCNN/xixi'+"/"+filename,h)       


if __name__ == '__main__':
    img = cv2.imread('jinx.jpg')
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO: 提取hog特征


    # 可视化特征
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('input image')
    ax[1].imshow(hog_vis, cmap='bone')
    ax[1].set_title('visualization of HOG features')
    plt.show()
