#!/usr/bin/env python2

from build_data import build_data

import numpy as np
import sys
from parameters import param_mlp as params
from sklearn.neural_network import MLPClassifier
from skimage import color
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import pickle

class MLP():
	def __init__(self, database):
		self.database = database
		self.images = np.load("%s_Dataset/images_lab.npy" % database.title())
		self.labels = np.load("%s_Dataset/labels.npy" % database.title())
		self.nb_labels = len(np.unique(self.labels))

	def run_test(self, X, Y):
		Y_proba = self.model.predict_proba(X)
		Y_guess = np.argmax(Y_proba, axis=1)
		n, w, h = Y.shape
		return np.sum(Y_guess == Y) / float(n * w * h)

	def save(self):
		pickle.dump(self.model, open("models/mlp_%s_1.pfl" % database, "wb" ))
        	moments = {"mean": self.mean, "std": self.std}
        	filename = "models/mlp_moments_{dataname}.pkl".format(dataname=self.database)
        	pickle.dump(moments, open(filename, "wb" ))

	def normalize(self,X):
		return (X - self.mean) / self.std

	def train(self, subset = []):
		print("\n### Train MLP - %s" % self.database)
		Y = self.labels
		X = self.images
		if subset != []:
			Y = Y[subset]
			X = X[subset]
		X = X#build_data(images)
    		self.mean = np.mean(X, axis=(0,1,2))
    		self.std = np.std(X, axis=(0,1,2))
		X = self.normalize(X)
		(n_samples, height, width, p) = X.shape
		X = X.reshape(n_samples * height * width, p)
		Y = Y.reshape(n_samples * height * width, 1)

		# Train
		Y_train_vector = (Y == np.arange(self.nb_labels)) * 1
		mlp = MLPClassifier(	hidden_layer_sizes = params.hidden_layer_sizes,
					max_iter = params.max_iter,
					alpha = params.alpha,
					solver = params.solver,
					activation = params.activation,
					verbose = params.verbose,
					tol = params.tol,
					random_state = params.random_state,
					learning_rate_init = params.learning_rate_init)
		mlp.fit(X, Y_train_vector)
		print("Done.")
		self.model = mlp

def get_args():
	if len(sys.argv) < 2:
		exit("Usage : python " + sys.argv[0] + " <Corel|Sowerby>")
	database = sys.argv[1].lower()
	return database

if __name__ == "__main__":
	database = get_args()
	mlp = MLP(database)
	mlp.train()
	#mlp.save()
