# -*- coding: utf-8 -*-
"""
Computes the various Gabor filters for one image
"""
import cv2
from gabor_pre import build_filters
from matplotlib import pyplot as plt
import numpy as np

image_name = 'corel_55.jpg'
img = cv2.imread(image_name,0)

theta_min = 0
theta_max = np.pi
theta_step = np.pi/4
scale_min = 3
scale_max = 9
scale_step = 2


[kernel_even, kernel_odd] = build_filters(theta_min, theta_max, theta_step, scale_min, scale_max, scale_step)

kern_even = np.asarray(kernel_even)
kern_odd= np.asarray(kernel_odd)

dst_even = []
dst_odd = []

for filt in kern_even:
    dst_temp = cv2.filter2D(img, -1, filt)
    dst_even.append(dst_temp)  

print('odd')
for filte in kern_odd:
    dst_tempe = cv2.filter2D(img, -1, filte)
    dst_odd.append(dst_tempe)

