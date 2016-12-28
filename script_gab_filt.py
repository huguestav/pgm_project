# -*- coding: utf-8 -*-
"""
Convolution of an image with oriented Gabor filters
"""
import cv2
from gabor_pre import build_filters
from matplotlib import pyplot as plt
import numpy as np

image_name = 'corel_55.jpg'
img = cv2.imread(image_name,0)
kerne = build_filters()
kernels = np.asarray(kerne)
dst = []

for filt in kernels:
    dst_temp = cv2.filter2D(img, -1, filt)
    dst.append(dst_temp)    

plt.imshow(dst[2])