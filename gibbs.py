import numpy as np
import matplotlib.pyplot as plt

def sample(p, L, nb_labels):
	shape = L.shape
	nb_components = np.prod(shape)
	L = L.reshape(nb_components)
	for i in range(nb_components):
		proba = []
		for y in range(nb_labels):
			temp = L[i]
			L[i] = y
			proba.append(p(L))
			L[i] = temp
		max_p = max(proba)
		min_p = min(proba)
		proba = (proba - min_p) / (max_p - min_p)
		norm = proba.sum()
		proba = proba / norm
		L[i] = np.random.choice(nb_labels, 1, list(proba))[0]
	return L.reshape(shape)

def f(l): return np.sum(l.reshape(9)*np.array([-1,1,1,1,-1,1,1,1,-1]))

v=[]
l = np.random.randint(0,7,(3,3))
for i in range(10000):
	if i % 1000 == 0:
		print(l, f(l))
	v.append(f(l))
	l = sample(f,l,7)
plt.plot(v)
plt.show()
