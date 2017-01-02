#!/usr/bin/env python2

import numpy as np

from sklearn.externals import joblib
from skimage import color
from matplotlib import pyplot as plt
from time import time

tic = time()


folder = 'Corel_Dataset/'
images = np.load(folder + 'images_lab_old.npy')
labels = np.load(folder + 'labels_old.npy')

(n_samples, height, width, p) = images.shape
Y = labels.reshape(n_samples, width * height)

# Load the distribution on image 70 given by the mlp classifier
distrib_mlp = np.load('mlp_distrib_corel_70_old.npy')
distrib_mlp = np.power(distrib_mlp, 0.90)
idx_test = 70


# Load the regional rbm learned on the first 60 images
# regional_rbm = joblib.load('regional_rbm_1.pkl')
global_rbm = joblib.load('global_rbm_corel_old.pkl')
w_rbm_g = global_rbm.components_

regional_rbm = joblib.load('regional_rbm_corel_old.pkl')
w_rbm_r = regional_rbm.components_

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

nb_labels = 7

reg_w_g = 18
reg_h_g = 12
reg_incr_h_g = 12
reg_incr_w_g = 18

reg_w_r = 8
reg_h_r = 8
reg_incr_w_r = 4
reg_incr_h_r = 4

# Build test data (we are testing on one image)
# idx_test = 70
Y_test = Y[idx_test]
Y_test = Y_test.reshape(width * height)


# Initialize the gibbs sampling using the mlp clasifier
Y_guess = np.argmax(distrib_mlp, axis=1)

initial_accuracy = np.sum(Y_guess == Y_test) / float(width * height)
print "initial_accuracy :", initial_accuracy


# Make Y_guess into a vector
Y_guess = (Y_guess.reshape(width*height, 1) == np.arange(nb_labels)) * 1

# Do the gibbs sampling
n_steps = width * height
np.random.seed(3)
rand_order = np.arange(width*height)
np.random.shuffle(rand_order)

Y_guess = Y_guess.reshape((height, width, nb_labels))
pixel_to_reg_g = convert_into_regions(reg_w_g,reg_h_g, reg_incr_w_g, reg_incr_h_g,
                                    width, height)

pixel_to_reg_r = convert_into_regions(reg_w_r,reg_h_r, reg_incr_w_r, reg_incr_h_r,
                                    width, height)

def trick(energy):
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
            proba_r = trick(energy_r)
        else:
            proba_r = np.log(1 + np.exp(energy_r))
        probas_rbm[k] = np.sum(proba_r)
    # probas_rbm = np.exp(probas_rbm - np.max(probas_rbm))
    probas_rbm = probas_rbm - np.max(probas_rbm)

    return probas_rbm


# def gibbs_sampling(Y_guess, n_steps, rand_order, reg_w,reg_h,
#                    reg_incr_w, reg_incr_h, pixel_to_reg, distrib_mlp, w_rbm):
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
        #
        probas = probas_rbm_g + probas_rbm_r + np.log(distrib_mlp[i])
        probas = probas - np.max(probas)
        probas = np.exp(probas)
        probas = probas / np.sum(probas)

        # probas = probas_rbm * distrib_mlp[i]
        # probas = probas / np.sum(probas)
        # Sample from distribution
        Y_guess[pixel] = fixed_labels[draw_finite(p=probas)]

    return Y_guess

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


n_runs = 5
for i in range(n_runs):
    Y_guess = gibbs_sampling(Y_guess, n_steps, rand_order, info_r, info_g, distrib_mlp)

    print "iteration :", i+1
    # Post prod
    Y_guess_ = Y_guess.reshape((height*width, nb_labels))
    Y_guess_ = np.argmax(Y_guess_, axis=1)
    new_accuracy = np.sum(Y_guess_ == Y_test) / float(width * height)
    print "new_accuracy :", new_accuracy

    n_better = (new_accuracy - initial_accuracy) * width * height
    print "Number of pixels that have changed :", n_better
    print ""




plt.imshow(Y_guess_.reshape((height, width)) / 7.)
plt.show()

print("Time : "+str(time()-tic)+"s")
