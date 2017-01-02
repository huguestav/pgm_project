#!/usr/bin/env python2

import numpy as np

from sklearn.externals import joblib
from skimage import color
from matplotlib import pyplot as plt

folder = '../Corel_Dataset/'
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')

(n_samples, height, width, p) = images.shape
Y = labels.reshape(n_samples, width * height)

# Load the distribution on image 70 given by the mlp classifier
distrib_mlp = np.load('mlp_distrib_image_70.npy')

# Load the regional rbm learned on the first 60 images
regional_rbm = joblib.load('regional_rbm_1.pkl')
w_regional = regional_rbm.components_

###############################################################################
###############################################################################
###############################################################################
def convert_into_regions(img, reg_w, reg_h, reg_incr, nb_labels, height, width):
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
            cste = reg_w * reg_h * nb_labels
            L_r[i * nb_reg_h + j] = img[p_wt:p_wb,p_hl:p_hr].reshape(cste)
    return L_r
###############################################################################
###############################################################################
###############################################################################

nb_labels = 7
reg_w = 8
reg_h = 8
reg_incr = 4

# Build test data (we are testing on one image)
idx_test = 70
Y_test = Y[idx_test]
Y_test = Y_test.reshape(width * height)


# Initialize the gibbs sampling using the mlp clasifier
Y_guess = np.argmax(distrib_mlp, axis=1)

initial_accuracy = np.sum(Y_guess == Y_test) / float(width * height)
print "initial_accuracy :", initial_accuracy


# Make Y_guess into a vector
Y_guess = (Y_guess.reshape(width*height, 1) == np.arange(nb_labels)) * 1

# Do the gibbs sampling
n_steps = 10
np.random.seed(3)
rand_order = np.arange(width*height)
np.random.shuffle(rand_order)

fixed_labels = np.eye(7)

n_tot = width * height
for i in rand_order[range(n_steps)]:
    label_field_test = np.copy(Y_guess)
    label_field_test = label_field_test.reshape((height, width, nb_labels))
    #
    p = i / width
    r = i - p * width
    pixel = (p,r)
    #
    probas_rbm = np.zeros(7)
    for k in range(nb_labels):
        label_field_test[pixel] = fixed_labels[k]
        L_r_test = convert_into_regions(label_field_test, reg_w,
                                    reg_h, reg_incr, nb_labels, width, height)
        energy_r = np.dot(L_r_test, w_regional.T)
        proba_r = np.log(1 + np.exp(energy_r))
        probas_rbm[k] = np.sum(proba_r)
    probas_rbm = np.exp(probas_rbm - np.max(probas_rbm))
    probas_rbm = probas_rbm / np.sum(probas_rbm)
    #
    probas = probas_rbm * distrib_mlp[i]
    probas = probas / np.sum(probas)
    # Sample from distribution
    value = np.random.choice(nb_labels,1,p=probas)[0]
    Y_guess[i] = fixed_labels[value]


Y_guess_ = np.argmax(Y_guess, axis=1)
new_accuracy = np.sum(Y_guess_ == Y_test) / float(width * height)
print "new_accuracy :", new_accuracy
print "initial_accuracy :", initial_accuracy

n_better = (new_accuracy - initial_accuracy) * width * height
print "Number of pixels that have changed :", n_better


# plt.imshow(Y_guess_.reshape((height, width)) / 7.)
# plt.show()


