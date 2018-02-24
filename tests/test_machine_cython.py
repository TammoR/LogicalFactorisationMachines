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

	orm.framework = 'cython'

	data = orm.add_matrix(val=X, sampling_indicator=False)

	layer1 = orm.add_layer(size=3, child=data, lbda_init=2., noise_model='or-link')

	orm.infer(convergence_window=20, no_samples=20,
			  convergence_eps=1e-3, burn_in_min=20,
			  burn_in_max=1000)

	print(np.mean( (X>0) == (layer1.output()>.5)))

	assert abs(1/(1+np.exp(-orm.layers[0].lbda())) - 1.) < 1e-3


def test_maxmachine():

	orm = lom.Machine()

	data = orm.add_matrix(np.array(50*[[-1,1,1],[1,-1,1]]),
						  sampling_indicator=False)
	layer=orm.add_layer(size=2, child=data, lbda_init=.9,
						noise_model='max-link')
	orm.infer(fix_lbda_iters=5,burn_in_min=100)

	assert np.abs(np.mean(layer.output() - .5*(data() +1))) < 1e-2


def test_tensorm():

	from scipy.special import logit
	import lom.lambda_updates_c_wrappers as sampling
	import lom.matrix_updates_c_wrappers as wrappers

	size = 2
	tensor, _, _, _ = experiments.generate_random_tensor(size, (3,4,5), noise=0.0)

	orm = lom.Machine()
	data = orm.add_matrix(2*tensor-1, sampling_indicator=False)
	layer = orm.add_tensorm_layer(
			child=data, size=size, 
			lbda_init=1.0,
			inits = 3*[.5],
			priors = [logit(x) for x in [.5, .5, .5]])

	# for factor_matrix in data.parents:
	#     factor_matrix.sampling_fct = wrappers.draw_tensorm_noparents_onechild_wrapper
	# layer.lbda.sampling_fct = sampling.draw_lbda_tensorm

	layer.auto_clean_up = False
	orm.infer(burn_in_min=20, no_samples=10)

	print('\n\n\n\n\n')
	print(np.mean((layer.output() > .5) == (tensor == 1)))

	assert np.all((layer.output() > .5) == (tensor == 1))