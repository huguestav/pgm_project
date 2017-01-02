#!/usr/bin/env python2

import numpy as np
import sys

from sklearn.externals import joblib
from skimage import color
from matplotlib import pyplot as plt
from time import time
from build_data import build_data

tic = time()

if len(sys.argv) < 2:
        exit("Usage : python inference.py <image_num>")
image_num = int(sys.argv[1]) - 1

folder = 'Corel_Dataset/'
images = np.load(folder + 'images_lab_old.npy')
labels = np.load(folder + 'labels_old.npy')
nb_labels = 7

(n_samples, height, width, p) = images.shape
Y = labels.reshape(n_samples, height * width)
X = images.reshape(n_samples, height, width, 3)
X = X[image_num].reshape(1, height, width, 3)

image = color.lab2rgb(X.reshape(height,width,3))
plt.subplot(2,2,1)
plt.axis('off')
plt.title('original')
plt.imshow(image)

# Build test data (we are testing on one image)
X = build_data(X)
(_, _, _, size_input) = X.shape

Y_test = Y[image_num].reshape(width * height)
plt.subplot(2,2,2)
plt.axis('off')
plt.title('ground truth')
plt.imshow(Y_test.reshape(height, width))

X_test = X.reshape(width * height, size_input)

# Load the regional rbm learned on the first 60 images
# regional_rbm = joblib.load("models/" + 'regional_rbm_1.pkl')
regional_rbm = joblib.load("models/" + 'regional_rbm_corel.pkl')
w_regional = regional_rbm.components_
mlp_model = joblib.load("models/" + 'mlp_corel_1.pkl')

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
reg_w = 8
reg_h = 8
reg_incr_w = 4
reg_incr_h = 4

# Initialize the gibbs sampling using the mlp clasifier
#Y_proba = mlp_model.predict_proba(X_test)
distrib_mlp = np.load("models/" + 'mlp_distrib_corel_70.npy')
Y_proba = distrib_mlp
Y_guess = np.argmax(Y_proba, axis=1)
initial_accuracy = np.sum(Y_guess == Y_test) / float(width * height)
print "initial_accuracy :", initial_accuracy

plt.subplot(2,2,3)
plt.axis('off')
plt.title('initial (MLP) : ' + str(initial_accuracy))
plt.imshow(Y_guess.reshape(height, width))


# Make Y_guess into a vector
Y_guess = (Y_guess.reshape(width*height, 1) == np.arange(nb_labels)) * 1

# Do the gibbs sampling
n_steps = width * height/2
np.random.seed(3)
rand_order = np.arange(width*height)
np.random.shuffle(rand_order)

fixed_labels = np.eye(7)

n_tot = width * height
pixel_to_reg = convert_into_regions(reg_w,reg_h, reg_incr_w, reg_incr_h, width, height)

Y_guess = Y_guess.reshape((height, width, nb_labels))
size_region = reg_w * reg_h * nb_labels

for i in rand_order[range(n_steps)]:
    p = i / width
    r = i - p * width
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
    probas = probas_rbm * Y_proba[i]
    probas = probas / np.sum(probas)
    # Sample from distribution
    Y_guess[pixel] = fixed_labels[draw_finite(p=probas)]


Y_guess_ = Y_guess.reshape((height*width, nb_labels))
Y_guess_ = np.argmax(Y_guess_, axis=1)
new_accuracy = np.sum(Y_guess_ == Y_test) / float(width * height)
n_better = (new_accuracy - initial_accuracy) * width * height
print "new_accuracy :", new_accuracy
print "initial_accuracy :", initial_accuracy
print "Number of pixels that have changed :", n_better
print "Time : ", time()-tic, "s"

plt.subplot(2,2,4)
plt.axis('off')
plt.title('final : ' + str(new_accuracy))
plt.imshow(Y_guess_.reshape(height, width))
plt.show()
