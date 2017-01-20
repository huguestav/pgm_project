from mlp import MLP
from rbm import RBM
import numpy as np
from sklearn.model_selection import KFold
from inference import gibbs_sampling, info
from build_data import *

from rbm_hugues import train_rbm

# database = "Sowerby"
database = "Corel"
n_splits = 6
n_iter_sampling = 5

X = np.load("%s_Dataset/images_lab.npy" % database)
Y = np.load("%s_Dataset/labels.npy" % database)
nb_labels = len(np.unique(Y))
(_,height,width,p) = X.shape

kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
accuracies = []
mlp_accuracies = []

for train_index, test_index in kf.split(X):
    rbm_r = RBM(database, "regional")
    rbm_g = RBM(database, "global")
    mlp = MLP(database)
    rbm_r.train(train_index)
    rbm_g.train(train_index)
    mlp.train(train_index)

    X_test = X[test_index]
    X_test = build_data(X_test)
    Y_test = Y[test_index]
    accuracy = []
    mlp_accuracy = []
    info_r = info(database, "regional", rbm_r.w, width, height)
    info_g = info(database, "global", rbm_g.w, width, height)

    print("\nTesting...")
    for j in range(len(test_index)):
        print("Image %d/%d" % (j + 1, len(test_index)))
        X_inst = X_test[j].reshape(1, height, width, X_test.shape[3])
        X_inst = mlp.normalize(X_inst)
        X_inst = X_inst.reshape(width * height, X_inst.shape[3])
        Y_proba = mlp.model.predict_proba(X_inst)
        Y_guess = (np.argmax(Y_proba, axis=1).reshape(width * height, 1) == np.arange(nb_labels)) * 1
        Y_guess = Y_guess.reshape((height, width, nb_labels))

        Y_truth = Y_test[j].reshape(height * width)
        Y_guess_ = Y_guess.reshape((height * width, nb_labels))
        Y_guess_ = np.argmax(Y_guess_, axis=1)
        acc = np.sum(Y_guess_ == Y_truth) / float(width * height)

        mlp_accuracy.append(acc)

        for i in range(n_iter_sampling):
            rand_order = np.arange(width * height)
            np.random.shuffle(rand_order)
            Y_guess = gibbs_sampling(Y_guess, width * height, rand_order, info_r, info_g, Y_proba)

            Y_truth = Y_test[j].reshape(height * width)
            Y_guess_ = Y_guess.reshape((height * width, nb_labels))
            Y_guess_ = np.argmax(Y_guess_, axis=1)
            new_acc = np.sum(Y_guess_ == Y_truth) / float(width * height)

            if new_acc < acc:
                break
            else:
                acc = new_acc

        accuracy.append(acc)

    accuracies.append(np.mean(accuracy))
    mlp_accuracies.append(np.mean(mlp_accuracy))
    print "mcrf :"
    print(accuracies[-1])
    print ""
    print "mlp :"
    print(mlp_accuracies[-1])

print("Accuracies on each split :")
print "mcrf:"
print(accuracies)
print ""
print "mlp:"
print(mlp_accuracies)
print("Overall accuracy mcrf : %f" % round(np.mean(accuracies), 3))
print("Overall accuracy mlp: %f" % round(np.mean(mlp_accuracies), 3))
