import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

from sklearn.neural_network import BernoulliRBM

images = np.load('Corel_Dataset/images_rgb.npy')
labels = np.load('Corel_Dataset/labels.npy')
nb_hidden = 30
nb_labels = 7
reg_w = 8
reg_h = 8
reg_incr = 4
nb_iterations = 10
temperature = 500.
(n_samples, width, height, p) = images.shape
trainingset_idx = range(n_samples / 3)

img = images.reshape(n_samples, width,height,3) / 255.
X = images.reshape(n_samples, width * height * 3)
Y = labels.reshape(n_samples, width, height, 1)

def convert_into_regions(img, reg_w, reg_h, reg_incr, nb_labels):
	global num_figure

	assert((width - reg_w) % reg_incr == 0)
	assert((height - reg_h) % reg_incr == 0)

	nb_reg_w = (width - reg_w) / reg_incr + 1
	nb_reg_h = (height - reg_h) / reg_incr + 1
	nb_reg = nb_reg_w * nb_reg_h
	L_r = np.empty((nb_reg, reg_w * reg_h * nb_labels))

	for i in range(nb_reg_w):
		p_wt = i * reg_incr
		p_wb = p_wt + reg_w
		for j in range(nb_reg_h):
			p_hl = j * reg_incr
			p_hr = p_hl + reg_h
			L_r[i * nb_reg_h + j] = img[p_wt:p_wb,p_hl:p_hr].reshape(reg_w * reg_h * nb_labels)

	return L_r

# Vectorize Y
Y = Y.reshape(n_samples * width * height,1)
Y = (Y == np.arange(nb_labels)) * 1
Y = Y.reshape(n_samples, width, height, nb_labels)

# Regions
print("Transform regions...")
L_r = np.array([]).reshape(0, reg_w * reg_h * nb_labels)
for i in trainingset_idx:
	L_r = np.append(L_r, convert_into_regions(Y[i], reg_w, reg_h, reg_incr, nb_labels), axis=0)
	print "{}/{}\r".format(str(i + 1), str(n_samples/3))

L_r_test = convert_into_regions(Y[1], reg_w, reg_h, reg_incr, nb_labels)

# Model
model = BernoulliRBM(n_components=nb_hidden, n_iter = nb_iterations, random_state=0, verbose=True)

print("Training...")
model.fit(L_r)
print("Done.")

w = model.components_

energy_r = np.dot(L_r_test, w.T)
proba_r = np.log(1 + np.exp(energy_r/temperature))
print(np.sum(proba_r))
