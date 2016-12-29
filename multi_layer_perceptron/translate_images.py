#!/usr/bin/env python2

import numpy as np


def translate_images(images):
    (n_samples, height, width) = images.shape

    t_images = np.zeros((8,n_samples, height, width))

    # Build the four translation matrices
    left_1 = np.diag(np.ones(height - 1),1)
    left_1[-1,-1] = 1

    left_2 = np.diag(np.ones(height - 1),-1)
    left_2[0,0] = 1

    right_1 = np.diag(np.ones(width - 1),1)
    right_1[0,0] = 1

    right_2 = np.diag(np.ones(width - 1),-1)
    right_2[-1,-1] = 1

    for i in range(n_samples):
        image = images[i,:,:]
        t_images[0,i,:,:] = left_1.dot(image)
        t_images[1,i,:,:] = left_2.dot(image)

        t_images[2,i,:,:] = image.dot(right_1)
        t_images[3,i,:,:] = image.dot(right_2)

        t_images[4,i,:,:] = left_1.dot(image).dot(right_1)
        t_images[5,i,:,:] = left_1.dot(image).dot(right_2)

        t_images[6,i,:,:] = left_2.dot(image).dot(right_1)
        t_images[7,i,:,:] = left_2.dot(image).dot(right_2)

    return t_images
