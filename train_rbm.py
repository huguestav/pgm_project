import matplotlib.pyplot as plt
import numpy as np

from sklearn.externals import joblib
from sklearn.neural_network import BernoulliRBM

# Attention, ceci est la vieille fonction qui confond width et height
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
            size = reg_w * reg_h * nb_labels
            L_r[i * nb_reg_h + j] = img[p_wt:p_wb,p_hl:p_hr].reshape(size)
    return L_r


def train_rbm(labels, nb_hidden, reg_w, reg_h, reg_incr, nb_iterations):
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
        L_r = np.append(L_r, convert_into_regions(Y[i], reg_w, reg_h, reg_incr,
                                                nb_labels, width, height), axis=0)
        print "{}/{}\r".format(str(i + 1), str(n_samples/3))

    # Model
    rbm = BernoulliRBM(n_components=nb_hidden, learning_rate=0.01,
                        n_iter = nb_iterations, random_state=0, verbose=True)

    print("Training...")
    rbm.fit(L_r)
    print("Done.")

    return rbm


def save_rbm(rbm, filename):
    joblib.dump(rbm, filename)
    return 0


labels = np.load('Corel_Dataset/labels_old.npy')
filename = 'regional_rbm_corel.pkl'
nb_hidden = 30
reg_w = 8
reg_h = 8
reg_incr = 4
nb_iterations = 20

# !!!! Attention, j'utilise toujours la vieille fonction qui renvoie
# directement l'image transformée en régions mais qui confond width et height
regional_rbm = train_rbm(labels, nb_hidden, reg_w, reg_h, reg_incr, nb_iterations)
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



