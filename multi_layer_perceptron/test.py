#!/usr/bin/env python2

import matplotlib.image as mpimg
import numpy as np
from sklearn.neural_network import MLPClassifier


folder = '../Corel_Dataset/'
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')

(n_samples, width, height, p) = images.shape

X = images.reshape(n_samples, width * height, 3)
Y = labels.reshape(n_samples, width * height)


# Shuffle the images
np.random.seed(1)
order = np.arange(n_samples)
np.random.shuffle(order)
X = X[order]
Y = Y[order]

# Learn
nb_labels = 7


# Build training data
train_size = 60
X_train = X[:train_size]
X_train = X_train.reshape(train_size * width * height, 3)

Y_train = Y[:train_size]
Y_train = Y_train.reshape(train_size * width * height,1)
Y_train = (Y_train == np.arange(nb_labels)) * 1


# Build test data
test_size = n_samples - train_size
X_test = X[train_size:]
X_test = X_test.reshape(test_size * width * height, 3)

Y_test = Y[train_size:]
Y_test = Y_test.reshape(test_size * width * height)


# Train
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    # solver='adam',activation='logistic', verbose=10, tol=1e-4,
                    solver='adam',activation='logistic', verbose=10, tol=0.1,
                    random_state=1, learning_rate_init=.1)
mlp.fit(X_train, Y_train)



# Test the classifier
Y_proba = mlp.predict_proba(X_test)

Y_proba_max = np.argmax(Y_proba, axis=1)

score = np.sum(Y_proba_max == Y_test)
accuracy = score / float(test_size * width * height)


print "accuracy", accuracy


