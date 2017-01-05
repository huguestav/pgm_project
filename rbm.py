import numpy as np
import pickle

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

def logistic(x):
	return (1. + np.tanh(x / 2.)) / 2.


def fit(X, nb_hidden, batch_size, n_iter, learning_rate):
	np.random.seed(0)
	n_samples, nb_visible = X.shape
	w = np.random.normal(0, 0.01, (nb_hidden, nb_visible))
        intercept_h = np.zeros(nb_hidden)
        intercept_v = np.zeros(nb_visible)
        h_samples = np.zeros((batch_size, nb_hidden))
        n_batches = int(np.ceil(float(n_samples) / batch_size))
        slices = [ range(batch_size * i, batch_size * (i + 1)) for i in range(n_batches)]

        for iteration in range(n_iter):
		print("Iteration " + str(iteration + 1) + "/" + str(n_iter))
		for batch_slice in slices:
                	v_pos = X[batch_slice]
			h_pos = logistic(np.dot(v_pos, w.T) + intercept_h)
        		v_neg = logistic(np.dot(h_samples, w) + intercept_v)
			v_neg = np.random.random(size=v_neg.shape) < v_neg
        		h_neg = logistic(np.dot(v_neg, w.T) + intercept_h)

        		lr = float(learning_rate) / batch_size
        		update = np.dot(v_pos.T, h_pos).T - np.dot(v_neg.T, h_neg).T
        		w += lr * update
        		intercept_h += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        		intercept_v += lr * (v_pos.sum(axis=0) - v_neg.sum(axis=0))
			h_samples = np.random.uniform(size=h_neg.shape) < h_neg
	
	return w

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

    # Model Training
    print("Training...")
    w = fit(L_r, nb_hidden = nb_hidden, batch_size=10, n_iter = nb_iterations, learning_rate=0.01)
    print("Done.")

    return w


def save_rbm(rbm, filename):
    pickle.dump(rbm, open("models/" + filename, "wb" ))
    return 0

"""
labels = np.load('Corel_Dataset/labels.npy')
filename = 'regional_rbm_corel.pkl'
nb_hidden = 30
reg_w = 8
reg_h = 8
reg_incr_w = 4
reg_incr_h = 4
nb_iterations = 30

regional_rbm = train_rbm(labels, nb_hidden, reg_w, reg_h, reg_incr_w, reg_incr_h, nb_iterations)
save_rbm(regional_rbm, filename)
"""

"""
labels = np.load('Corel_Dataset/labels.npy')
filename = 'global_rbm_corel.pkl'
nb_hidden = 15
reg_w = 18
reg_h = 12
reg_incr_w = 12
reg_incr_h = 12
nb_iterations = 5
height = 120
width = 180

global_rbm = train_rbm(labels, nb_hidden, reg_w, reg_h, reg_incr_w, reg_incr_h, nb_iterations)
save_rbm(global_rbm, filename)
"""

###############################################################################

"""
labels = np.load('Sowerby_Dataset/labels.npy')
filename = 'regional_rbm_sowerby.pkl'
nb_hidden = 50
reg_w = 6
reg_h = 4
reg_incr_w = 4
reg_incr_h = 1
nb_iterations = 30

regional_rbm = train_rbm(labels, nb_hidden, reg_w, reg_h, reg_incr_w,
                         reg_incr_h, nb_iterations)
save_rbm(regional_rbm, filename)
"""



labels = np.load('Sowerby_Dataset/labels.npy')
filename = 'global_rbm_sowerby.pkl'
nb_hidden = 10
reg_w = 8
reg_h = 8
reg_incr_w = 8
reg_incr_h = 8
nb_iterations = 20


global_rbm = train_rbm(labels, nb_hidden, reg_w, reg_h, reg_incr_w,
                        reg_incr_h, nb_iterations)
save_rbm(global_rbm, filename)

