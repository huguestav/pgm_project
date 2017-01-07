class param_rbm_corel:
	class reg:
		labels = 'Corel_Dataset/labels.npy'
		nb_hidden = 30
		reg_w = 8
		reg_h = 8
		reg_incr_w = 4
		reg_incr_h = 4
		nb_iterations = 30
	class glob:
		labels = 'Corel_Dataset/labels.npy'
		nb_hidden = 15
		reg_w = 18
		reg_h = 12
		reg_incr_w = 12
		reg_incr_h = 12
		nb_iterations = 5

class param_rbm_sowerby:
	class reg:
		labels = 'Sowerby_Dataset/labels.npy'
		nb_hidden = 50
		reg_w = 6
		reg_h = 4
		reg_incr_w = 4
		reg_incr_h = 1
		nb_iterations = 30
	class glob:
		labels = 'Sowerby_Dataset/labels.npy'
		nb_hidden = 10
		reg_w = 8
		reg_h = 8
		reg_incr_w = 8
		reg_incr_h = 8
		nb_iterations = 20

class param_mlp:
	hidden_layer_sizes = (80,)
	max_iter = 100
	alpha = 1e-4
	solver = 'adam'
	activation = 'logistic'
	verbose = 10
	tol = 0.01
	random_state = 1
	learning_rate_init = .001
