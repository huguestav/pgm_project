import numpy as np
import pickle
import parameters
import sys

class RBM:
	def __init__(self, database, scale):
		self.database = database
		self.scale = scale
		param_dict = {	"corelregional":parameters.param_rbm_corel.reg,
				"corelglobal":parameters.param_rbm_corel.glob,
				"sowerbyregional":parameters.param_rbm_sowerby.reg,
				"sowerbyglobal":parameters.param_rbm_sowerby.glob
				}
		self.params = param_dict[database.lower() + scale]
		self.labels = np.load(self.params.labels)
		self.nb_labels = len(np.unique(self.labels))

	def convert_into_regions(self,img, reg_w, reg_h, reg_incr_w, reg_incr_h, width, height):

	    nb_reg_w = (width - reg_w) / reg_incr_w + 1
	    nb_reg_h = (height - reg_h) / reg_incr_h + 1
	    nb_reg = nb_reg_w * nb_reg_h
	    L_r = np.empty((nb_reg, reg_w * reg_h * self.nb_labels))

	    for i in range(nb_reg_w):
		p_wt = i * reg_incr_w
		p_wb = p_wt + reg_w
		for j in range(nb_reg_h):
		    p_hl = j * reg_incr_h
		    p_hr = p_hl + reg_h
		    size = reg_w * reg_h * self.nb_labels
		    L_r[i * nb_reg_h + j] = img[p_hl:p_hr,p_wt:p_wb].reshape(size)
	    return L_r

	def logistic(self,x):
		return (1. + np.tanh(x / 2.)) / 2.

	def fit(self,X, nb_hidden, batch_size, n_iter, learning_rate):
		np.random.seed(0)
		n_samples, nb_visible = X.shape
		w = np.random.normal(0, 0.01, (nb_hidden, nb_visible))
		intercept_h = np.zeros(nb_hidden)
		intercept_v = np.zeros(nb_visible)
		h_samples = np.zeros((batch_size, nb_hidden))
		n_batches = int(np.ceil(float(n_samples) / batch_size))
		slices = [ range(batch_size * i, min(batch_size * (i + 1), n_samples)) for i in range(n_batches)]

		for iteration in range(n_iter):
			print("Iteration " + str(iteration + 1) + "/" + str(n_iter))
			for batch_slice in slices:
				v_pos = X[batch_slice]
				h_pos = self.logistic(np.dot(v_pos, w.T) + intercept_h)
				v_neg = self.logistic(np.dot(h_samples, w) + intercept_v)
				v_neg = np.random.random(size=v_neg.shape) < v_neg
				h_neg = self.logistic(np.dot(v_neg, w.T) + intercept_h)

				lr = float(learning_rate) / batch_size
				update = np.dot(v_pos.T, h_pos).T - np.dot(v_neg.T, h_neg).T
				w += lr * update
				intercept_h += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
				intercept_v += lr * (v_pos.sum(axis=0) - v_neg.sum(axis=0))
				h_samples = np.random.uniform(size=h_neg.shape) < h_neg
		
		return w

	def train(self, subset = []):
	    print("\n### Train RBM - %s %s" % (self.database, self.scale))
	    params = self.params
	    labels = self.labels
	    nb_hidden = params.nb_hidden
	    reg_w = params.reg_w
	    reg_h = params.reg_h
	    reg_incr_w = params.reg_incr_w
	    reg_incr_h = params.reg_incr_h
	    nb_iterations = params.nb_iterations

	    if subset != []:
                labels = labels[subset]
	    (n_samples, height, width) = labels.shape
	    Y = labels.reshape(n_samples, height, width, 1)

	    # Vectorize Y
	    Y = Y.reshape(n_samples * width * height,1)
	    Y = (Y == np.arange(self.nb_labels)) * 1
	    Y = Y.reshape(n_samples, height, width, self.nb_labels)

	    # Regions
	    print("Transform regions...")
	    L_r = []
	    for i in range(n_samples):
		L_r.append(self.convert_into_regions(Y[i], reg_w, reg_h, reg_incr_w, reg_incr_h, width, height))
		print("Image %d/%d" % (i + 1, n_samples))
	    
	    print("Concatenate...")
	    L_r = np.concatenate(L_r)

	    # Model Training
	    print("Training...")
	    w = self.fit(L_r, nb_hidden = nb_hidden, batch_size=10, n_iter = nb_iterations, learning_rate=0.01)
	    print("Done.")

	    self.w = w

	def save(self):
	    pickle.dump(self.w, open("models/%s_rbm_%s.pkl" % (self.scale, self.database), "wb" ))

def get_args():
        if len(sys.argv) < 2:
                exit("Usage : python " + sys.argv[0] + " <Corel|Sowerby> <regional|global>")
        database = sys.argv[1].lower()
	scale = sys.argv[2].lower()
        return database, scale

if __name__ == "__main__":
	database, scale = get_args()
	rbm = RBM(database, scale)
	rbm.train()
	rbm.save()
