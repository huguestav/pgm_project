#!/usr/bin/env python2

from build_data import build_data

import numpy as np
import sys
from parameters import param_mlp as params
from sklearn.neural_network import MLPClassifier
from skimage import color
from matplotlib import pyplot as plt
from sklearn.externals import joblib

def run_test(mlp, X, Y):
	Y_proba = mlp.predict_proba(X)
	Y_guess = np.argmax(Y_proba, axis=1)
	n, w, h = Y.shape
	return np.sum(Y_guess == Y) / float(n * w * h)

def save_mlp(mlp, filename):
    pickle.dump(rbm, open("models/" + filename, "wb" ))
    return 0

def get_args():
        if len(sys.argv) < 2:
                exit("Usage : python " + sys.argv[0] + " <Corel|Sowerby>")
        database = sys.argv[1].lower()
        return database

database = get_args()
folder = database.title() + "_Dataset/"
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')
filename = 'mlp_' + database + '_1test.pkl'

(n_samples, height, width, p) = images.shape
Y = labels.reshape(n_samples, width * height)
Y = Y[:10]
X = images[:10]
X = build_data(images, save_moments=database)

size_input = X.shape[4]
train_size = 5
test_size = n_samples - train_size
nb_labels = len(np.unique(Y))

# Shuffle the images
np.random.seed(3)
order = np.arange(n_samples)
np.random.shuffle(order)
# X = X[order]
# Y = Y[order]

# Build training data
X_train = X[:train_size]
X_train = X_train.reshape(train_size * width * height, size_input)

Y_train = Y[:train_size].reshape(train_size * width * height,1)

# Build test data
X_test = X[train_size:].reshape(test_size * width * height, size_input)

Y_test  = Y[train_size:].reshape(test_size * width * height)

# Train
Y_train_vector = (Y_train == np.arange(nb_labels)) * 1
mlp = MLPClassifier(	hidden_layer_sizes = params.hidden_layer_sizes,
			max_iter = params.max_iter,
			alpha = params.alpha,
                    	solver = params.solver,
			activation = params.activation,
			verbose = params.verbose,
			tol = params.tol,
			random_state = params.random_state,
			learning_rate_init = params.learning_rate_init)
mlp.fit(X_train, Y_train_vector)

# Test the classifier on train data
accuracy_train = run_test(mlp, X_train, Y_train)
print "training accuracy :", accuracy_train

# Test the classifier on test data
accuracy_test = run_test(mlp, X_test, Y_test)
print "test accuracy :", accuracy_test

# Save mlp classifier to file
save_mlp(mlp, "models/" + filename)
