#!/usr/bin/env python2

import numpy as np
from skimage import color
import os

images_path_name = "images_rgb"
labels_path_name = "labels_raw"

dataset_folders = [name for name in os.listdir('.') if "Dataset" in name]

# Open each dataset
for f in dataset_folders:
	print(f + "...")
	
	images_path = f + "/" + images_path_name
	labels_path = f + "/" + labels_path_name

	files = [name for name in os.listdir(labels_path) if os.path.isfile(images_path+"/"+name)]
	nb_images = len(files)
	width, height = np.loadtxt(open(images_path+"/"+name)).shape
	height = height / 3

	# Initialize the arrays
	images  = np.zeros((nb_images, width, height, 3))
	images_lab = np.zeros((nb_images, width, height, 3))
	labels = np.zeros((nb_images, width, height))

	# Build the arrays
	for i,name in enumerate(files):
		# Load the data from files
		image = np.loadtxt(open(images_path+"/"+name))
		label = np.loadtxt(open(labels_path+"/"+name))

		# Reshape the data using "order='F'" trick
		images[i] = np.reshape(image, (width, height, 3), order='F')
		labels[i] = np.reshape(label, (width, height), order='F')

		# Convert image to CIE-LAB (image has to takes values between 0 and 1)
		images_lab[i] = color.rgb2lab(images[i] / 255.)


	# Save the arrays into files
	np.save(f + '/images_rgb', images)
	np.save(f + '/images_lab', images_lab)
	np.save(f + '/labels', labels)

print("Done.")
