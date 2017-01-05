#!/usr/bin/env python2

import numpy as np
from skimage import color
from skimage.io import imsave
import os

def colorize(arr):
        (h,w) = arr.shape
        new_arr = np.empty((h,w,3))
        cdict = {0 : [0,0,0], 1 : [150,0,0], 2: [0,150,0], 3: [0,0,150],
                4 : [241, 196, 15], 5: [125, 60, 152], 6: [243, 156, 18], 7: [255,255,255]}
        for y in range(h):
                for x in range(w):
                        new_arr[y][x] = cdict[arr[y][x]]
        return new_arr

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
	width, height = np.loadtxt(open(images_path+"/"+files[0])).shape
	height = height / 3
	
	new_order = [int(u.split("_")[1])-1 for u in files]
	f_o = zip(new_order,files)
	f_o.sort()
	files = [u[1] for u in f_o]

	# Initialize the arrays
	images  = np.zeros((nb_images, width, height, 3))
	images_lab = np.zeros((nb_images, width, height, 3))
	labels = np.zeros((nb_images, width, height))
	
	if not os.path.exists(f+"/Images"):
		os.makedirs(f+"/Images")
	if not os.path.exists(f+"/Labels"):
                os.makedirs(f+"/Labels")

	# Build the arrays
	for i,name in enumerate(files):
		# Load the data from files
		image = np.loadtxt(open(images_path+"/"+name))
		label = np.loadtxt(open(labels_path+"/"+name))

		# Reshape the data using "order='F'" trick
		images[i] = image.reshape((width, height, 3), order='F')
		labels[i] = np.reshape(label, (width, height), order='F')
		
		imsave(f+"/Images/"+name+'.jpg', images[i].astype(np.uint8))
		imsave(f+"/Labels/"+name+'.jpg', colorize(labels[i]).astype(np.uint8))
		# Convert image to CIE-LAB (image has to takes values between 0 and 1)
		images[i] = images[i] / 255.
		images_lab[i] = color.rgb2lab(images[i])

	# Save the arrays into files
	np.save(f + '/images_rgb', images)
	np.save(f + '/images_lab', images_lab)
	np.save(f + '/labels', labels)

print("Done.")
