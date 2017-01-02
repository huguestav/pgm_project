#!/usr/bin/env python2

import matplotlib.image as mpimg
import numpy as np
from sklearn.neural_network import MLPClassifier

# images = np.load('images_rgb.npy')
images = np.load('images_lab.npy')
labels = np.load('labels.npy')

(n_samples, width, height, p) = images.shape

X = images.reshape(n_samples, width * height, 3)
Y = labels.reshape(n_samples, width * height)


# Learn on pixel i
nb_labels = 7
i = 10

X_i = X[:, i, :]
labels_i = Y[:, i].reshape((n_samples, 1))
Y_i = (labels_i == np.arange(nb_labels)) * 1


# Build training data
train_size = 80
X_train_i = X_i[:train_size]
Y_train_i = Y_i[:train_size]


# Train
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='adam',activation='logistic', verbose=10, tol=1e-4,
                    random_state=1, learning_rate_init=.1)
mlp.fit(X_train_i, Y_train_i)


# Build test data
X_test_i = X_i[train_size:]
Y_test_i = Y_i[train_size:]
n_test = Y_test_i.shape[0]

Y_proba = mlp.predict_proba(X_test_i)
# Y_predicted_ = mlp.predict(X_test_i)


Y_proba_max = np.max(Y_proba, axis=1).reshape(n_test,1)
Y_predicted = (Y_proba_max == Y_proba) * 1

score = 0
for k in range(n_test):
    if np.array_equal(Y_test_i[k], Y_predicted[k]):
        score = score + 1. / n_test

accuracy = mlp.score(X_test_i, Y_test_i)

print "accuracy", accuracy
print "score", score


