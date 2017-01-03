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

def normalize(X, mean, std):
    return (X-mean) / std

tic = time()

if len(sys.argv) < 3:
        exit("Usage : python inference.py <Corel|Sowerby> <image_num>")

image_num = int(sys.argv[2]) - 1
folder = sys.argv[1] + '_Dataset/'
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')

dataname = sys.argv[1].lower()


mlp_file = "models/mlp_{dataname}_1.pkl".format(dataname=dataname)
moments_file = "models/mlp_moments_{dataname}.pkl".format(dataname=dataname)
rbm_r_file = "models/regional_rbm_{dataname}.pkl".format(dataname=dataname)
rbm_g_file = "models/global_rbm_{dataname}.pkl".format(dataname=dataname)

mlp_model = joblib.load(mlp_file)
mlp_moments = pickle.load(open(moments_file, "rb" ))
regional_rbm = joblib.load(rbm_r_file)
global_rbm = joblib.load(rbm_g_file)

w_rbm_g = global_rbm.components_
w_rbm_r = regional_rbm.components_

mean = mlp_moments["mean"]
std = mlp_moments["std"]
nb_labels = len(np.unique(labels))

# rmb parameters
if dataname == "corel":
    reg_w_g = 18
    reg_h_g = 12
    reg_incr_h_g = 12
    reg_incr_w_g = 18

    reg_w_r = 8
    reg_h_r = 8
    reg_incr_w_r = 4
    reg_incr_h_r = 4
if dataname == "sowerby":
    reg_w_g = 8
    reg_h_g = 8
    reg_incr_w_g = 8
    reg_incr_h_g = 8

    reg_w_r = 6
    reg_h_r = 4
    reg_incr_w_r = 4
    reg_incr_h_r = 1


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
X_test = normalize(X_test, mean, std)
final_plot(colorize(Y_test.reshape(height, width)), 'ground truth', 2)


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

pixel_to_reg_g = convert_into_regions(reg_w_g,reg_h_g, reg_incr_w_g, reg_incr_h_g,
                                    width, height)

pixel_to_reg_r = convert_into_regions(reg_w_r,reg_h_r, reg_incr_w_r, reg_incr_h_r,
                                    width, height)



# Build the two information dicts
info_r = {
    "reg_w_r": reg_w_r,
    "reg_h_r": reg_h_r,
    "reg_incr_w_r": reg_incr_w_r,
    "reg_incr_h_r": reg_incr_h_r,
    "pixel_to_reg_r": pixel_to_reg_r,
    "w_rbm_r": w_rbm_r,
}

info_g = {
    "reg_w_g": reg_w_g,
    "reg_h_g": reg_h_g,
    "reg_incr_w_g": reg_incr_w_g,
    "reg_incr_h_g": reg_incr_h_g,
    "pixel_to_reg_g": pixel_to_reg_g,
    "w_rbm_g": w_rbm_g,
}



def proba_from_energy(energy):
    """
    The purpose of this funtion is to compute the probability from the energy
    trying to avoid overflow.
    """
    big_idx = energy > 700
    small_idx = np.logical_not(big_idx)
    big_values = energy[big_idx]
    small_values = energy[small_idx]
    #
    result = np.zeros(energy.shape)
    result[big_idx] = big_values
    result[small_idx] = np.log(1 + np.exp(small_values))
    return result


def compute_proba_rbm(Y_guess, pixel_to_reg, pixel, nb_labels, fixed_labels,
                      size_region, w_rbm, scale="g"):
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
        energy_r = np.dot(L_r_pixel, w_rbm.T)
        if scale == "g":
            proba_r = proba_from_energy(energy_r)
        else:
            proba_r = np.log(1 + np.exp(energy_r))
        probas_rbm[k] = np.sum(proba_r)
    probas_rbm = probas_rbm - np.max(probas_rbm)

    return probas_rbm


def gibbs_sampling(Y_guess, n_steps, rand_order, info_r, info_g, distrib_mlp):
    reg_w_r = info_r["reg_w_r"]
    reg_h_r = info_r["reg_h_r"]
    reg_incr_w_r = info_r["reg_incr_w_r"]
    reg_incr_h_r = info_r["reg_incr_h_r"]
    pixel_to_reg_r = info_r["pixel_to_reg_r"]
    w_rbm_r = info_r["w_rbm_r"]

    reg_w_g = info_g["reg_w_g"]
    reg_h_g = info_g["reg_h_g"]
    reg_incr_w_g = info_g["reg_incr_w_g"]
    reg_incr_h_g = info_g["reg_incr_h_g"]
    pixel_to_reg_g = info_g["pixel_to_reg_g"]
    w_rbm_g = info_g["w_rbm_g"]

    (height, width, nb_labels) = Y_guess.shape
    fixed_labels = np.eye(nb_labels)
    n_tot = width * height

    size_region_r = reg_w_r * reg_h_r * nb_labels
    size_region_g = reg_w_g * reg_h_g * nb_labels

    for i in rand_order[range(n_steps)]:
        p = i / width
        r = i - p * width
        pixel = (p,r)
        # Compute probas rbm regional
        probas_rbm_r = compute_proba_rbm(Y_guess, pixel_to_reg_r, pixel, nb_labels,
                                fixed_labels, size_region_r, w_rbm_r, scale="r")
        # Compute probas rbm global
        probas_rbm_g = compute_proba_rbm(Y_guess, pixel_to_reg_g, pixel, nb_labels,
                                       fixed_labels, size_region_g, w_rbm_g)
        # Compute general proba
        probas = probas_rbm_g + probas_rbm_r + np.log(distrib_mlp[i])
        probas = probas - np.max(probas)
        probas = np.exp(probas)
        probas = probas / np.sum(probas)

        # Sample from distribution
        Y_guess[pixel] = fixed_labels[draw_finite(p=probas)]

    return Y_guess


Y_guess = Y_guess.reshape((height, width, nb_labels))

if len(sys.argv) == 4:
    n_runs = int(sys.argv[3])
else:
    n_runs = 1

for i in range(n_runs):
    Y_guess = gibbs_sampling(Y_guess, n_steps, rand_order, info_r, info_g, Y_proba)

    print "iteration :", i+1
    # Prints
    Y_guess_ = Y_guess.reshape((height*width, nb_labels))
    Y_guess_ = np.argmax(Y_guess_, axis=1)
    new_accuracy = np.sum(Y_guess_ == Y_test) / float(width * height)
    print "new_accuracy :", new_accuracy

    n_better = (new_accuracy - initial_accuracy) * width * height
    print "Number of pixels that have changed :", n_better
    print ""



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
