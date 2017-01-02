#!/usr/bin/env python2
from build_data import build_data
# import sys
# sys.path.append('..')


import numpy as np
from sklearn.neural_network import MLPClassifier

from skimage import color
from matplotlib import pyplot as plt

folder = '../Corel_Dataset/'
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')

(n_samples, height, width, p) = images.shape
Y = labels.reshape(n_samples, width * height)

X = build_data(images, labels)
(n_samples, height, width, size_input) = np.shape(X)


# Shuffle the images
np.random.seed(3)
order = np.arange(n_samples)
np.random.shuffle(order)
# X = X[order]
# Y = Y[order]



# Build training data
nb_labels = len(np.unique(Y))
train_size = 60
# X_train = X[:train_size]
# X_train = X_train.reshape(train_size * width * height, size_input)

# Y_train = Y[:train_size]
# Y_train = Y_train.reshape(train_size * width * height,1)
# Y_train = (Y_train == np.arange(nb_labels)) * 1


# # Build test data
# test_size = n_samples - train_size
# X_test = X[train_size:]
# X_test = X_test.reshape(test_size * width * height, size_input)

# Y_test = Y[train_size:]
# Y_test = Y_test.reshape(test_size * width * height)


# Get the saved mlp
from sklearn.externals import joblib
mlp = joblib.load('mlp_corel_1.pkl')


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
from sklearn.neural_network import BernoulliRBM

nb_hidden = 30
nb_labels = 7
reg_w = 8
reg_h = 8
reg_incr = 4
nb_iterations = 20
temperature = 500.
trainingset_idx = range(60)

img = images.reshape(n_samples, height,width,3) / 255.
# X = images.reshape(n_samples, width * height * 3)
Y_rbm = labels.reshape(n_samples, height, width, 1)
# Y_rbm = Y_rbm[order]

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
            L_r[i * nb_reg_h + j] = img[p_wt:p_wb,p_hl:p_hr].reshape(reg_w * reg_h * nb_labels)
    return L_r

# Vectorize Y
Y_rbm = Y_rbm.reshape(n_samples * width * height,1)
Y_rbm = (Y_rbm == np.arange(nb_labels)) * 1
Y_rbm = Y_rbm.reshape(n_samples, height, width, nb_labels)

# Regions
print("Transform regions...")
L_r = np.array([]).reshape(0, reg_w * reg_h * nb_labels)
for i in trainingset_idx:
    L_r = np.append(L_r, convert_into_regions(Y_rbm[i], reg_w, reg_h, reg_incr,
                    nb_labels, width, height), axis=0)
    print "{}/{}\r".format(str(i + 1), str(n_samples/3))

L_r_test = convert_into_regions(Y_rbm[1], reg_w, reg_h, reg_incr, nb_labels)

# Model
model = BernoulliRBM(n_components=nb_hidden, learning_rate=0.01,
                    n_iter = nb_iterations, random_state=0, verbose=True)

print("Training...")
model.fit(L_r)
print("Done.")

w = model.components_

# energy_r = np.dot(L_r_test, w.T)
# proba_r = np.log(1 + np.exp(energy_r))
# print(np.sum(proba_r))
idx_test = 70


X_test = X[idx_test]
X_test = X_test.reshape(width * height, size_input)

Y_test = Y[idx_test]
Y_test = Y_test.reshape(width * height)



# Initialize the gibbs sampling using the mlp clasifier
distrib_mlp = mlp.predict_proba(X_test)
Y_guess = np.argmax(distrib_mlp, axis=1)

initial_accuracy = np.sum(Y_guess == Y_test) / float(width * height)
print "initial_accuracy :", initial_accuracy



# compare = labels[idx_test].reshape(width*height)
# weird_accuracy = np.sum(Y_guess == compare) / float(width * height)
# print "weird_accuracy :", weird_accuracy



# plt.imshow(Y_test.reshape((height, width)) / 7.)
# plt.imshow(color.lab2rgb(images[idx_test]))
# plt.imshow(Y_guess.reshape((height, width)) / 7.)
# plt.show()




# Make Y_guess into a vector
Y_guess = (Y_guess.reshape(width*height, 1) == np.arange(nb_labels)) * 1


# Do the gibbs sampling
n_steps = 1000
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
        energy_r = np.dot(L_r_test, w.T)
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
print "n better :", n_better


# plt.imshow(Y_guess_.reshape((height, width)) / 7.)
# plt.show()


