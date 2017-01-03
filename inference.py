#!/usr/bin/env python2

import numpy as np
import sys
import pickle

from sklearn.externals import joblib
from skimage import color
from matplotlib import pyplot as plt
from time import time
from build_data import build_data


def colorize(arr):
	(h,w) = arr.shape
	new_arr = np.empty((h,w,3))
	cdict = {0 : [0,0,0], 1 : [150,0,0], 2: [0,150,0], 3: [0,0,150],
		4 : [241, 196, 15], 5: [125, 60, 152], 6: [243, 156, 18], 7: [255,255,255]}
	for y in range(h):
		for x in range(w):
			new_arr[y][x] = [u / 255. for u in cdict[arr[y][x]]]
	return new_arr

def final_plot(arr, title, i):
	plt.subplot(2,2,i)
	plt.axis('off')
	plt.title(title)
	plt.imshow(arr)

tic = time()

if len(sys.argv) < 3:
        exit("Usage : python inference.py <Corel|Sowerby> <image_num>")

image_num = int(sys.argv[2]) - 1
folder = sys.argv[1] + '_Dataset/'
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')

dataname = sys.argv[1].lower()



# mlp_model = joblib.load("models/" + 'mlp_corel_1.pkl')
# mlp_moments = pickle.load(open("models/mlp_moments_sowerby.pkl", "rb" ))
# regional_rbm = joblib.load("models/" + 'regional_rbm_corel.pkl')



mlp_file = "models/mlp_{dataname}_1.pkl".format(dataname=dataname)
moments_file = "models/mlp_moments_{dataname}.pkl".format(dataname=dataname)
rbm_r_file = "models/regional_rbm_{dataname}.pkl".format(dataname=dataname)

mlp_model = joblib.load(mlp_file)
mlp_moments = pickle.load(open(moments_file, "rb" ))
regional_rbm = joblib.load(rbm_r_file)




w_regional = regional_rbm.components_
mean = mlp_moments["mean"]
std = mlp_moments["std"]
nb_labels = len(np.unique(labels))

# rmb parameters
# reg_w = 8
# reg_h = 8
# reg_incr_w = 4
# reg_incr_h = 4
reg_w = 6
reg_h = 4
reg_incr_w = 4
reg_incr_h = 1




precision_acc = 3
np.random.seed(3)
(n_samples, height, width, p) = images.shape
n_steps = width * height

Y = labels.reshape(n_samples, height * width)
X = images.reshape(n_samples, height, width, 3)
X = X[image_num].reshape(1, height, width, 3)
image = color.lab2rgb(X.reshape(height,width,3))
final_plot(image, 'original', 1)

# Build test data (we are testing on one image)
X = build_data(X)
(_, _, _, size_input) = X.shape

Y_test = Y[image_num].reshape(width * height)
X_test = X[0].reshape(width * height, size_input)
final_plot(colorize(Y_test.reshape(height, width)), 'ground truth', 2)

# Normalize the data
X_test = X_test - mean
X_test = X_test / std

###############################################################################
###############################################################################
###############################################################################
def convert_into_regions(reg_w, reg_h, reg_incr_w, reg_incr_h, width, height):

    nb_reg_w = (width - reg_w) / reg_incr_w + 1
    nb_reg_h = (height - reg_h) / reg_incr_h + 1
    pixel_to_reg = [[[] for j in range(width)] for i in range(height)]
    for i in range(nb_reg_w):
        p_wt = i * reg_incr_w
        p_wb = p_wt + reg_w
        for j in range(nb_reg_h):
            p_hl = j * reg_incr_h
            p_hr = p_hl + reg_h
            for wp in range(p_wt,p_wb):
                for hp in range(p_hl,p_hr):
                    pixel_to_reg[hp][wp].append([p_hl,p_hr,p_wt,p_wb])
    return pixel_to_reg

###############################################################################
###############################################################################
###############################################################################

def draw_finite(p):
    """
    Draw a sample from a distribution on a finite set
    that takes value k with probability p(k)
    """
    q = np.cumsum(p)
    u = np.random.random()
    i = 0
    while u > q[i]:
        i += 1
    return i

###############################################################################
###############################################################################
###############################################################################

# Initialize the gibbs sampling using the mlp clasifier
Y_proba = mlp_model.predict_proba(X_test)
Y_init = np.argmax(Y_proba, axis=1)
initial_accuracy = round(np.sum(Y_init == Y_test) / float(width * height), precision_acc)
final_plot(colorize(Y_init.reshape(height, width)), 'initial (MLP) : ' + str(initial_accuracy), 3)

# Make Y_guess into a vector
Y_guess = (Y_init.copy().reshape(width*height, 1) == np.arange(nb_labels)) * 1

# Do the gibbs sampling
rand_order = np.arange(width*height)
np.random.shuffle(rand_order)

fixed_labels = np.eye(7)

pixel_to_reg = convert_into_regions(reg_w,reg_h, reg_incr_w, reg_incr_h, width, height)

Y_guess = Y_guess.reshape((height, width, nb_labels))
size_region = reg_w * reg_h * nb_labels

for i in range(n_steps):
    if (i * 10) / n_steps != ((i-1) * 10) / n_steps:
        print "Sampling :", (i * 10) / n_steps * 10, "%"
    num_pixel = rand_order[i]
    p = num_pixel / width
    r = num_pixel % width
    pixel = (p,r)
    #
    probas_rbm = np.empty(nb_labels)
    for k in range(nb_labels):
        Y_guess[pixel] = fixed_labels[k]
        reg_list = pixel_to_reg[pixel[0]][pixel[1]]
        reg_list_size = len(reg_list)
        idxs = range(reg_list_size)
        L_r_pixel = np.empty((reg_list_size,size_region))
        for idx in idxs:
            r = reg_list[idx]
            [p_hl,p_hr,p_wt,p_wb] = r
            L_r_pixel[idx] = Y_guess[p_hl:p_hr,p_wt:p_wb].reshape(size_region)
        energy_r = np.dot(L_r_pixel, w_regional.T)
        proba_r = np.log(1 + np.exp(energy_r))
        probas_rbm[k] = np.sum(proba_r)
    probas_rbm = np.exp(probas_rbm - np.max(probas_rbm))
    # probas_rbm = probas_rbm / np.sum(probas_rbm)
    #
    probas = probas_rbm * Y_proba[num_pixel]
    probas = probas / np.sum(probas)
    # Sample from distribution
    Y_guess[pixel] = fixed_labels[draw_finite(p=probas)]


Y_guess_ = Y_guess.reshape((height*width, nb_labels))
Y_guess_ = np.argmax(Y_guess_, axis=1)
new_accuracy = round(np.sum(Y_guess_ == Y_test) / float(width * height), precision_acc)
n_diff = abs(np.sum(Y_guess_ == Y_test) - np.sum(Y_init == Y_test))
print "new_accuracy :", new_accuracy
print "initial_accuracy :", initial_accuracy
print "Number of pixels that have changed :", n_diff
print "Time : ", round(time()-tic,1), "s"
final_plot(colorize(Y_guess_.reshape(height, width)), 'final : ' + str(new_accuracy), 4)
plt.show()
