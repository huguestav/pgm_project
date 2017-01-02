import matplotlib.pyplot as plt
import numpy as np

from sklearn.externals import joblib
from sklearn.neural_network import BernoulliRBM


def convert_into_regions(img, reg_w, reg_h, reg_incr_w, reg_incr_h, nb_labels, width, height):

    nb_reg_w = (width - reg_w) / reg_incr_w + 1
    nb_reg_h = (height - reg_h) / reg_incr_h + 1
    nb_reg = nb_reg_w * nb_reg_h
    L_r = np.empty((nb_reg, reg_w * reg_h * nb_labels))

    for i in range(nb_reg_w):
        p_wt = i * reg_incr_w
        p_wb = p_wt + reg_w
        for j in range(nb_reg_h):
            p_hl = j * reg_incr_h
            p_hr = p_hl + reg_h
            size = reg_w * reg_h * nb_labels
            L_r[i * nb_reg_h + j] = img[p_hl:p_hr,p_wt:p_wb].reshape(size)
    return L_r


def train_rbm(labels, nb_hidden, reg_w, reg_h, reg_incr_w, reg_incr_h, nb_iterations):
    nb_labels = len(np.unique(labels))
    (n_samples, height, width) = labels.shape
    trainingset_idx = range(60)
    Y = labels.reshape(n_samples, height, width, 1)

    # Vectorize Y
    Y = Y.reshape(n_samples * width * height,1)
    Y = (Y == np.arange(nb_labels)) * 1
    Y = Y.reshape(n_samples, height, width, nb_labels)

    # Regions
    print("Transform regions...")
    L_r = np.array([]).reshape(0, reg_w * reg_h * nb_labels)
    for i in trainingset_idx:
        L_r = np.append(L_r, convert_into_regions(Y[i], reg_w, reg_h, reg_incr_w, reg_incr_h,
                                                nb_labels, width, height), axis=0)
        print "Image {}/{}\r".format(str(i + 1), len(trainingset_idx))

    # Model
    rbm = BernoulliRBM(n_components=nb_hidden, learning_rate=0.01,
                        n_iter = nb_iterations, random_state=0, verbose=True)

    print("Training...")
    rbm.fit(L_r)
    print("Done.")

    return rbm


def save_rbm(rbm, filename):
    joblib.dump(rbm, "models/" + filename)
    return 0


labels = np.load('Corel_Dataset/labels_old.npy')
filename = 'regional_rbm_corel.pkl'
nb_hidden = 30
reg_w = 8
reg_h = 8
reg_incr_w = 4
reg_incr_h = 4
nb_iterations = 20

regional_rbm = train_rbm(labels, nb_hidden, reg_w, reg_h, reg_incr_w, reg_incr_h, nb_iterations)
save_rbm(regional_rbm, filename)


# labels = np.load('Corel_Dataset/labels_old.npy')
# filename = 'global_rbm_corel.pkl'
# nb_hidden = 15
# reg_w = 18
# reg_h = 12
# reg_incr = 12
# nb_iterations = 5
# height = 120
# width = 180


# global_rbm = train_rbm(labels, nb_hidden, reg_w, reg_h, reg_incr, nb_iterations)
# save_rbm(global_rbm, filename)



