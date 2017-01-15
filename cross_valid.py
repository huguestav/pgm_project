from mlp import MLP
from rbm import RBM
import numpy as np
from sklearn.model_selection import KFold
from inference import gibbs_sampling, info

database = "Sowerby"
n_splits = 3
n_iter_sampling = 2

X = np.load("%s_Dataset/images_lab.npy" % database)
Y = np.load("%s_Dataset/labels.npy" % database)
nb_labels = len(np.unique(Y))
(_,height,width,p) = X.shape

kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
accuracies = []

for train_index, test_index in kf.split(X):
	rbm_r = RBM(database, "regional")
	rbm_g = RBM(database, "global")
	mlp = MLP(database)
	rbm_r.train(train_index)
	rbm_g.train(train_index)
	mlp.train(train_index)
	
	X_test = X[test_index]
	Y_test = Y[test_index]
	accuracy = []
	info_r = info(database, "regional", rbm_r.w, width, height)
	info_g = info(database, "global", rbm_g.w, width, height)

	print("\nTesting...")
	for j in range(len(test_index)):
		print("Image %d/%d" % (j + 1, len(test_index)))
		X_inst = X_test[j].reshape(1, height, width, p)
		#X_inst = build_data(X_inst)
		X_inst = mlp.normalize(X_inst)
		X_inst = X_inst.reshape(width * height, X_inst.shape[3])
		Y_proba = mlp.model.predict_proba(X_inst)
		Y_guess = (np.argmax(Y_proba, axis=1).reshape(width * height, 1) == np.arange(nb_labels)) * 1
		Y_guess = Y_guess.reshape((height, width, nb_labels))

		Y_truth = Y_test[j].reshape(height * width)
		for i in range(n_iter_sampling):
			rand_order = np.arange(width * height)
			np.random.shuffle(rand_order)
			Y_guess = gibbs_sampling(Y_guess, width * height, rand_order, info_r, info_g, Y_proba)

		Y_guess_ = Y_guess.reshape((height * width, nb_labels))
		Y_guess_ = np.argmax(Y_guess_, axis=1)
		accuracy.append(np.sum(Y_guess_ == Y_truth) / float(width * height))

	accuracies.append(np.mean(accuracy))
	print(accuracies[-1])

print("Accuracies on each split :")
print(accuracies)
print("Overall accuracy : %f" % round(np.mean(accuracies), 3))
