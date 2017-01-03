import numpy as np
from sklearn.neural_network import MLPClassifier
from translate_images import translate_images
from cv2 import GaussianBlur, filter2D
from gabor_pre import build_filters


def build_data(images, save_moments=False):
    (n_samples, height, width, p) = images.shape

    # Filter the images
    size = 7
    f = (size,size)
    filtered_images_l = np.zeros(images.shape)
    for i in range(n_samples):
        filtered_images_l[i] = GaussianBlur(images[i],f,1)

    size = 5
    f = (size,size)
    filtered_images_m = np.zeros(images.shape)
    for i in range(n_samples):
        filtered_images_m[i] = GaussianBlur(images[i],f,1)

    size = 3
    f = (size,size)
    filtered_images_s = np.zeros(images.shape)
    for i in range(n_samples):
        filtered_images_s[i] = GaussianBlur(images[i],f,1)


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

    theta_min = 0
    theta_max = np.pi
    theta_step = np.pi/4
    scale_min = 3
    scale_max = 15
    scale_step = 5
    [kernel_even, kernel_odd] = build_filters(theta_min, theta_max, theta_step,
                                                scale_min, scale_max, scale_step)

    # gabor filter 1
    n = 8
    images_gab = np.zeros((n, n_samples, height, width))
    images_gab_translated = np.zeros((n, 8, n_samples, height, width))

    for k in range(n/2):
        # Even
        gab = np.zeros(images.shape)
        for i in range(n_samples):
            gab[i] = filter2D(images[i], -1, kernel_even[k])
        images_gab[2*k] = gab[:,:,:,0]
        images_gab_translated[2*k] = translate_images(images_gab[2*k])
        # Odd
        gab = np.zeros(images.shape)
        for i in range(n_samples):
            gab[i] = filter2D(images[i], -1, kernel_odd[k])
        images_gab[2*k+1] = gab[:,:,:,0]
        images_gab_translated[2*k+1] = translate_images(images_gab[k])


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
    n_gab = 8
    n_input = 3 + 3 + n_gab
    size_input = n_input + 8 * (5+n_gab)
    X = np.zeros((n_samples, height, width, size_input))
    X[:,:,:,0] = l_images
    X[:,:,:,1] = a_images
    X[:,:,:,2] = b_images

    X[:,:,:,3] = difference_of_g_1
    X[:,:,:,4] = difference_of_g_2
    X[:,:,:,5] = difference_of_g_3

    for i in range(n_gab):
        X[:,:,:,6+i] = images_gab[i]


    for i in range(8):
        # X[:,:,:,i+n_input+8*2] = l_translated[i]
        X[:,:,:,i+n_input+8*0] = diff_of_g_translated_1[i]
        X[:,:,:,i+n_input+8*1] = diff_of_g_translated_2[i]
        X[:,:,:,i+n_input+8*2] = diff_of_g_translated_3[i]
        X[:,:,:,i+n_input+8*3] = a_translated[i]
        X[:,:,:,i+n_input+8*4] = b_translated[i]
        for j in range(n_gab):
            X[:,:,:,i+n_input+8*(j+5)] = images_gab_translated[j,i]


    if save_moments:
        # Normalize X
        mean = np.mean(X, axis=(0,1,2))
        X = X - mean

        std = np.std(X, axis=(0,1,2))
        X = X / std

        import pickle
        moments = {"mean": mean, "std": std}
        pickle.dump(moments, open( "models/mlp_moments.pkl", "wb" ))

    return X

