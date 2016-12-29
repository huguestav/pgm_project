#!/usr/bin/env python2

import matplotlib.image as mpimg
import numpy as np
from sklearn.neural_network import MLPClassifier
import cv2


folder = '../Corel_Dataset/'
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')

(n_samples, width, height, p) = images.shape

X = images.reshape(n_samples, width * height, 3)
Y = labels.reshape(n_samples, width * height)


# images = np.load(folder + 'images_rgb.npy')


filtered_images = np.zeros((n_samples, width, height))
for i in range(n_samples):
    big = (7,7)
    medium = (5,5)
    img = images[i,:,:,0]
    f_image_big = cv2.GaussianBlur(img, big, 1)
    f_image_medium = cv2.GaussianBlur(img, medium, 1)
    f_image = f_image_big - f_image_medium
    filtered_images[i] = f_image

# from matplotlib import pyplot as plt
# from skimage import color
# gray_image = np.zeros((width, height, 3))
# gray_image[:,:,0] = f_image

# conv_img = color.lab2rgb(gray_image)
# conv_img = color.lab2rgb(f_image)
# # conv_img = color.lab2rgb(img)
# # plt.imshow(conv_img)

# plt.imshow(filtered_images[9])
# plt.show()


# img = images[10]


# plt.imshow(img * 255)
# plt.show()


# from skimage import transform as tf
# tform = tf.SimilarityTransform(scale=1, translation=(0, 1))

# image = images[0,:,:,0]
# translated = tf.warp(image, tform)


# def add_neighbors(image):
#   """
#   Image has only one dimension.
#   """
#   (width, height) = image.shape



# Shuffle the images
np.random.seed(2)
order = np.arange(n_samples)
np.random.shuffle(order)
X = X[order]
Y = Y[order]

X_2 = filtered_images
X_2 = X_2[order]

# Learn
nb_labels = 7


# Build training data
train_size = 60
X_train = X[:train_size]
X_train = X_train.reshape(train_size * width * height, 3)
X_2_train = X_2[:train_size]
X_2_train = X_2_train.reshape(train_size * width * height, 1)

X_train = np.append(X_train, X_2_train, axis=1)

# np.append(X_train, X_2_train, axis=0)

Y_train = Y[:train_size]
Y_train = Y_train.reshape(train_size * width * height,1)
Y_train = (Y_train == np.arange(nb_labels)) * 1


# Build test data
test_size = n_samples - train_size
X_test = X[train_size:]
X_test = X_test.reshape(test_size * width * height, 3)
X_2_test = X_2[train_size:]
X_2_test = X_2_test.reshape(test_size * width * height,1)

X_test = np.append(X_test, X_2_test, axis=1)

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


