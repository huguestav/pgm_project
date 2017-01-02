#!/usr/bin/env python2

from build_data import build_data

import numpy as np
from sklearn.neural_network import MLPClassifier
from skimage import color
from matplotlib import pyplot as plt
from sklearn.externals import joblib

folder = 'Corel_Dataset/'
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')

(n_samples, height, width, p) = images.shape
Y = labels.reshape(n_samples, width * height)

X = build_data(images)
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
X_train = X[:train_size]
X_train = X_train.reshape(train_size * width * height, size_input)

Y_train = Y[:train_size]
Y_train = Y_train.reshape(train_size * width * height,1)
Y_train = (Y_train == np.arange(nb_labels)) * 1


# Build test data
test_size = n_samples - train_size
X_test = X[train_size:]
X_test = X_test.reshape(test_size * width * height, size_input)

Y_test = Y[train_size:]
Y_test = Y_test.reshape(test_size * width * height)


# Train
mlp = MLPClassifier(hidden_layer_sizes=(80,), max_iter=100, alpha=1e-4,
                    # solver='adam',activation='logistic', verbose=10, tol=1e-4,
                    solver='adam',activation='logistic', verbose=10, tol=0.01,
                    random_state=1, learning_rate_init=.001)
mlp.fit(X_train, Y_train)



# Test the classifier
Y_proba = mlp.predict_proba(X_test)

Y_proba_max = np.argmax(Y_proba, axis=1)

score = np.sum(Y_proba_max == Y_test)
accuracy = score / float(test_size * width * height)

print "test accuracy :", accuracy

# Save mlp classifier to file
filename = 'mlp_corel_1.pkl'
joblib.dump(mlp, "models/" + filename)


# Test the classifier on training data
Y_proba = mlp.predict_proba(X_train)
Y_proba_max = np.argmax(Y_proba, axis=1)

Y_train = Y[:train_size]
Y_train = Y_train.reshape(train_size * width * height)

score = np.sum(Y_proba_max == Y_train)
accuracy = score / float(train_size * width * height)

print "training accuracy :", accuracy



# Save the distribution for one image
idx_save = 70
X_save = X[idx_save]
X_save = X_save.reshape(width * height, size_input)

mlp_distrib_image_save = mlp.predict_proba(X_save)
np.save('mlp_distrib_corel_70', mlp_distrib_image_save)


################################################################################

# idx = 1
# image = images[order[idx]]
# image = color.lab2rgb(image)
# # plt.imshow(image)
# # plt.show()




# X_test = X[idx]
# X_test = X_test.reshape(width * height, size_input)

# Y_test = Y[idx]
# Y_test = Y_test.reshape(width * height)


# Y_proba = mlp.predict_proba(X_test)
# Y_proba_max = np.argmax(Y_proba, axis=1)

# print np.sum(Y_proba_max == Y_test) / float((width * height))

# Y_guess = Y_proba_max.reshape((height, width))


# Y_test = Y_test.reshape((height, width))



# plt.figure(1)
# plt.imshow(image)
# plt.figure(2)
# plt.imshow(Y_guess / 7.)
# plt.show()

