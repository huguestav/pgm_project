# -*- coding: utf-8 -*-
"""
Builds Gabor filters
"""
import math
import numpy as np

def GaborFilter(ksize, theta, lambdap, sigma, gamma, psi):
    xmin = -math.floor(ksize/2)
    xmax = math.floor(ksize/2)
    ymin = -math.floor(ksize/2)
    ymax = math.floor(ksize/2)

    (x,y) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))
    
    x_theta = x*np.cos(theta) + y*np.sin(theta)
    y_theta = -x * np.sin(theta) + y*np.cos(theta)
    
    filt_even = np.exp(-(x_theta**2 + (gamma*y_theta)**2)/(2*sigma**2))*np.cos(2*np.pi/lambdap * x_theta + psi)
    filt_odd = np.exp(-(x_theta**2 + (gamma*y_theta)**2)/(2*sigma**2))*np.sin(2*np.pi/lambdap * x_theta + psi)
    
    return [filt_even, filt_odd]