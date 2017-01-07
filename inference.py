#!/usr/bin/env python2

import numpy as np
import sys
import pickle
import parameters
from sklearn.externals import joblib

from skimage import color
from matplotlib import pyplot as plt
from time import time
from build_data import build_data

class info:

	def make_regions(self):
	    nb_reg_w = (self.width - self.reg_w) / self.reg_incr_w + 1
	    nb_reg_h = (self.height - self.reg_h) / self.reg_incr_h + 1
	    self.pixel_to_reg = [[[] for j in range(self.width)] for i in range(self.height)]
	    for i in range(nb_reg_w):
		p_wt = i * self.reg_incr_w
		p_wb = p_wt + self.reg_w
		for j in range(nb_reg_h):
		    p_hl = j * self.reg_incr_h
		    p_hr = p_hl + self.reg_h
		    for wp in range(p_wt,p_wb):
			for hp in range(p_hl,p_hr):
			    self.pixel_to_reg[hp][wp].append([p_hl,p_hr,p_wt,p_wb])

	def __init__(self, database, scale, w, width, height):
		if database == "Corel" and scale == "region":
			params = parameters.param_rbm_corel.reg
		if database == "Corel" and scale == "global":
			params = parameters.param_rbm_corel.glob
		if database == "Sowerby" and scale == "region":
			params = parameters.param_rbm_corel.reg
		if database == "Sowerby" and scale == "global":
			params = parameters.param_rbm_corel.glob
		self.reg_w = params.reg_w
    		self.reg_h = params.reg_h
    		self.reg_incr_w = params.reg_incr_w
    		self.reg_incr_h = params.reg_incr_h
    		self.w_rbm = w
		self.width = width
		self.height = height
		self.make_regions()

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

def get_args():
	if len(sys.argv) < 3:
		exit("Usage : python " + sys.argv[0] + " <Corel|Sowerby> <image_num> [<nb_iter>]")
	if len(sys.argv) == 4:
		n_runs = int(sys.argv[3])
	else:
    		n_runs = 1
	image_num = int(sys.argv[2]) - 1
	database = sys.argv[1]
	return database, image_num, n_runs, time()

def get_models(dataname):

	mlp_file = "models/mlp_{dataname}_1.pkl".format(dataname=dataname)
	moments_file = "models/mlp_moments_{dataname}.pkl".format(dataname=dataname)
	rbm_r_file = "models/regional_rbm_{dataname}.pkl".format(dataname=dataname)
	rbm_g_file = "models/global_rbm_{dataname}.pkl".format(dataname=dataname)

	mlp_model = joblib.load(mlp_file)
	mlp_moments = pickle.load(open(moments_file, "rb" ))
	regional_rbm = pickle.load(open(rbm_r_file, "rb"))
	global_rbm = pickle.load(open(rbm_g_file, "rb"))

	return mlp_model, mlp_moments, regional_rbm, global_rbm

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
    reg_w_r = info_r.reg_w
    reg_h_r = info_r.reg_h
    reg_incr_w_r = info_r.reg_incr_w
    reg_incr_h_r = info_r.reg_incr_h
    pixel_to_reg_r = info_r.pixel_to_reg
    w_rbm_r = info_r.w_rbm

    reg_w_g = info_g.reg_w
    reg_h_g = info_g.reg_h
    reg_incr_w_g = info_g.reg_incr_w
    reg_incr_h_g = info_g.reg_incr_h
    pixel_to_reg_g = info_g.pixel_to_reg
    w_rbm_g = info_g.w_rbm

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


# Load Parameters ########################
database, image_num, n_runs, tic = get_args()
folder = database + '_Dataset/'
dataname = database.lower()
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')

# saved models
mlp_model, mlp_moments, regional_rbm, global_rbm = get_models(dataname)

w_rbm_g = global_rbm
w_rbm_r = regional_rbm

mean = mlp_moments["mean"]
std = mlp_moments["std"]
nb_labels = len(np.unique(labels))
precision_acc = 3
np.random.seed(3)

(n_samples, height, width, p) = images.shape
n_steps = width * height

# Raw image ###############################
X = images.reshape(n_samples, height, width, 3)
X = X[image_num].reshape(1, height, width, 3)
image = color.lab2rgb(X.reshape(height,width,3))
final_plot(image, 'original', 1)

# Feautre image ###########################
print "Compute Features..."
X = build_data(X)
(_, _, _, size_input) = X.shape
X_test = X[0].reshape(width * height, size_input)
X_test = normalize(X_test, mean, std)

# Ground truth labels #####################
Y = labels.reshape(n_samples, height * width)
Y_test = Y[image_num].reshape(width * height)
final_plot(colorize(Y_test.reshape(height, width)), 'ground truth', 2)

# Initial labeling (MLP) ##################
Y_proba = mlp_model.predict_proba(X_test)
Y_init = np.argmax(Y_proba, axis=1)
initial_accuracy = round(np.sum(Y_init == Y_test) / float(width * height), precision_acc)
final_plot(colorize(Y_init.reshape(height, width)), 'initial (MLP) : ' + str(initial_accuracy), 3)

# Make Y_guess into a vector
Y_guess = (Y_init.copy().reshape(width*height, 1) == np.arange(nb_labels)) * 1
Y_guess = Y_guess.reshape((height, width, nb_labels))

# Sampling parameters
rand_order = np.arange(width*height)
np.random.shuffle(rand_order)

# Build the two information class
info_r = info(database, "region", w_rbm_r, width, height)
info_g = info(database, "global", w_rbm_g, width, height)

# Sampling ##################################
print "Sample..."
for i in range(n_runs):

    Y_guess = gibbs_sampling(Y_guess, n_steps, rand_order, info_r, info_g, Y_proba)
    Y_guess_ = Y_guess.reshape((height*width, nb_labels))
    Y_guess_ = np.argmax(Y_guess_, axis=1)
    new_accuracy = round(np.sum(Y_guess_ == Y_test) / float(width * height), precision_acc)
    n_better = np.sum(Y_guess_ == Y_test) - np.sum(Y_init == Y_test)

    print "iteration :", i+1
    print "new_accuracy :", new_accuracy
    print "Number of pixels that have changed :", n_better, "\n"

# Results ##################################
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
