import numpy as np
import lom
import lom.experiments as experiments

def test_orm():
	np.random.seed(2)

	N = 100
	M = 100

	X = 2*np.array([M*[0,0,1,1,0,0], M*[1,1,0,0,0,0], M*[0,0,0,0,1,1]])-1
	
	X = np.concatenate(N*[X])

	orm = lom.Machine()

	orm.framework = 'numba'

	data = orm.add_matrix(val=X, sampling_indicator=False)

	layer1 = orm.add_layer(size=3, child=data, lbda_init=2., noise_model='or-link')

	orm.infer(convergence_window=20, no_samples=20,
			  convergence_eps=1e-3, burn_in_min=20,
			  burn_in_max=1000)

	print(np.mean( (X>0) == (layer1.output()>.5)))

	assert abs(1/(1+np.exp(-orm.layers[0].lbda())) - 1.) < 1e-3