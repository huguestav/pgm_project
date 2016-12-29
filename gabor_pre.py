# -*- coding: utf-8 -*-
import cv2
import numpy as np

#Builds the Gabor filters for the required parameters

def build_filters(theta_min, theta_max, theta_step, scale_min, scale_max, scale_step, lambdap = 1., sigma = 20, gamma = 0.5, psi = 0):
    filters_even = []
    filters_odd = []

    for theta in np.arange(theta_min, theta_max, theta_step):
        for ksize in np.arange(scale_min, scale_max, scale_step):
          kern_even = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambdap, gamma, psi, ktype=cv2.CV_32F)
          kern_odd = cv2.getGaborKernel((ksize, ksize), sigma, theta+np.pi, lambdap, gamma, np.pi/2-psi, ktype=cv2.CV_32F)
          kern_even /= 1.5*kern_even.sum()
          kern_odd /= 1.5*kern_odd.sum()
          filters_even.append(kern_even)
          filters_odd.append(kern_odd)
    return [filters_even, filters_odd]
