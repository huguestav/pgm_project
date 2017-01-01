# -*- coding: utf-8 -*-
import numpy as np
from gabor_build import GaborFilter

#Builds the Gabor filters for the required parameters

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
