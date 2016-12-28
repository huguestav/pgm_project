# -*- coding: utf-8 -*-
"""
Builds Gabor filters at various scales and orientations
"""

import cv2
import numpy as np

def build_filters():
    filters = []
    ksize = 5
    for theta in np.arange(0, np.pi, np.pi/4):
        for lambdap in np.arange(10,80, 30):
            kern_even = cv2.getGaborKernel((ksize, ksize), 20, theta, lambdap, 0.5, 0, ktype=cv2.CV_32F)
            kern_odd = cv2.getGaborKernel((ksize, ksize), 20, np.pi/2-theta, lambdap, 0.5, 0, ktype=cv2.CV_32F)
            kern_even /= 1.5*kern_even.sum()
            kern_odd /= 1.5*kern_odd.sum()
            filters.append(kern_even)
            filters.append(kern_odd)
    return filters
