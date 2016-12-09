#!/usr/bin/env python2

import numpy as np
from skimage import color

nb_images = 100
width = 120
height = 180

images_path = "images_rgb"
labels_path = "labels_raw"

# Initialize the arrays
images  = np.zeros((nb_images, width, height, 3))
images_lab = np.zeros((nb_images, width, height, 3))
labels = np.zeros((nb_images, width, height))

# Build the arrays
for i in range(nb_images):
    # Load the data from files
    image = np.loadtxt(open('{path}/corel_{i}'.format(path=images_path, i=i+1)))
    label = np.loadtxt(open('{path}/corel_{i}'.format(path=labels_path, i=i+1)))

    # Reshape the data using "order='F'" trick
    image = np.reshape(image, (width, height, 3), order='F')
    label = np.reshape(label, (width, height), order='F')

    images[i] = image
    labels[i] = label

    # Convert image to CIE-LAB (image has to takes values between 0 and 1)
    images_lab[i] = color.rgb2lab(image / 255.)


# Save the arrays into files
np.save('images_rgb', images)
np.save('images_lab', images_lab)
np.save('labels', labels)
