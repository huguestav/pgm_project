#!/usr/bin/env python2

import matplotlib.image as mpimg
import numpy as np
from sklearn.neural_network import MLPClassifier
from translate_images import translate_images
import cv2
from skimage import color
from matplotlib import pyplot as plt


folder = '../Corel_Dataset/'
images = np.load(folder + 'images_lab.npy')
labels = np.load(folder + 'labels.npy')

(n_samples, height, width, p) = images.shape
Y = labels.reshape(n_samples, width * height)

# Pre-filter the images
# size = 3
# f = (size,size)
# filtered_images = np.zeros(images.shape)
# for i in range(n_samples):
#     filtered_images[i] = cv2.GaussianBlur(images[i],f,1)
# images = filtered_images

# Filter the images
size = 7
f = (size,size)
filtered_images_l = np.zeros(images.shape)
for i in range(n_samples):
    filtered_images_l[i] = cv2.GaussianBlur(images[i],f,1)

size = 5
f = (size,size)
filtered_images_m = np.zeros(images.shape)
for i in range(n_samples):
    filtered_images_m[i] = cv2.GaussianBlur(images[i],f,1)

size = 3
f = (size,size)
filtered_images_s = np.zeros(images.shape)
for i in range(n_samples):
    filtered_images_s[i] = cv2.GaussianBlur(images[i],f,1)


difference_of_g_1 = filtered_images_m - filtered_images_s
difference_of_g_1 = difference_of_g_1[:,:,:,0]

difference_of_g_2 = filtered_images_l - filtered_images_s
difference_of_g_2 = difference_of_g_2[:,:,:,0]

difference_of_g_3 = filtered_images_l - filtered_images_m
difference_of_g_3 = difference_of_g_3[:,:,:,0]

# Translate the difference of gaussians
diff_of_g_translated_1 = translate_images(difference_of_g_1)
diff_of_g_translated_2 = translate_images(difference_of_g_2)
diff_of_g_translated_3 = translate_images(difference_of_g_3)


# Normalize the images
# image = images[0]
# image = cv2.equalizeHist(image)
# image = color.lab2rgb(image)

# plt.imshow(image)
# plt.show()


# Build 3x3 data
l_images = images[:,:,:,0]
a_images = images[:,:,:,1]
b_images = images[:,:,:,2]


# Translate the images
l_translated = translate_images(l_images)
a_translated = translate_images(a_images)
b_translated = translate_images(b_images)


# Build the input data
# size_input = 3 + 8 * 2
size_input = 3 + 8 * 0 + 3
X = np.zeros((n_samples, height, width, size_input))
X[:,:,:,0] = l_images
X[:,:,:,1] = a_images
X[:,:,:,2] = b_images
X[:,:,:,3] = difference_of_g_1
X[:,:,:,4] = difference_of_g_2
X[:,:,:,5] = difference_of_g_3

for i in range(8):
    print ''
    # X[:,:,:,i+6]  = a_translated[i]
    # X[:,:,:,i+14] = b_translated[i]
    # X[:,:,:,i+19] = l_translated[i]
    # X[:,:,:,i+21] = diff_of_g_translated_1[i]
    # X[:,:,:,i+29] = diff_of_g_translated_1[i]


# Shuffle the images
np.random.seed(3)
order = np.arange(n_samples)
np.random.shuffle(order)
X = X[order]
Y = Y[order]

# Learn
nb_labels = 7


# Build training data
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
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    # solver='adam',activation='logistic', verbose=10, tol=1e-4,
                    solver='adam',activation='logistic', verbose=10, tol=0.01,
                    random_state=1, learning_rate_init=.01)
mlp.fit(X_train, Y_train)



# Test the classifier
Y_proba = mlp.predict_proba(X_test)

Y_proba_max = np.argmax(Y_proba, axis=1)

score = np.sum(Y_proba_max == Y_test)
accuracy = score / float(test_size * width * height)

print "accuracy", accuracy



image = images[order[70]]
image = color.lab2rgb(image)
plt.imshow(image)
plt.show()




X_test = X[70]
X_test = X_test.reshape(width * height, size_input)

Y_test = Y[70]
Y_test = Y_test.reshape(width * height)


Y_proba = mlp.predict_proba(X_test)
Y_proba_max = np.argmax(Y_proba, axis=1)

print np.sum(Y_proba_max == Y_test) / float((width * height))

Y_guess = Y_proba_max.reshape((height, width))


Y_test = Y_test.reshape((height, width))



plt.figure(1)
plt.imshow(image)
plt.figure(2)
plt.imshow(Y_guess / 7.)
plt.show()

