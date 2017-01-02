# -*- coding: utf-8 -*-
import numpy as np
from math import floor

#Builds the Gabor filters for the required parameters

def GaborFilter(ksize, theta, lambdap, sigma, gamma, psi):
    xmin = -floor(ksize/2)
    xmax = floor(ksize/2)
    ymin = -floor(ksize/2)
    ymax = floor(ksize/2)

    (x,y) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))

    x_theta = x*np.cos(theta) + y*np.sin(theta)
    y_theta = -x * np.sin(theta) + y*np.cos(theta)

    filt_even = np.exp(-(x_theta**2 + (gamma*y_theta)**2)/(2*sigma**2))*np.cos(2*np.pi/lambdap * x_theta + psi)
    filt_odd = np.exp(-(x_theta**2 + (gamma*y_theta)**2)/(2*sigma**2))*np.sin(2*np.pi/lambdap * x_theta + psi)

    return [filt_even, filt_odd]


def build_filters(theta_min, theta_max, theta_step, scale_min, scale_max, scale_step, lambdap = 0.5, sigma = 20, gamma = 0.5, psi = 0):
    filters_even = []
    filters_odd = []

    for theta in np.arange(theta_min, theta_max, theta_step):
        for ksize in np.arange(scale_min, scale_max, scale_step):
          [kern_even, kern_odd] = GaborFilter(ksize, theta, lambdap, sigma, gamma, psi)
          kern_even /= 1.5*kern_even.sum()
          zk = np.absolute(kern_odd)
          Z = np.sum(zk)
          kern_odd = kern_odd/(1.5*Z)
          filters_even.append(kern_even)
          filters_odd.append(kern_odd)
    return [filters_even, filters_odd]
