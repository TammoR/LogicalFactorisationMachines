#!/usr/bin/env python
"""
LOM

Various auxiliary functions

"""
import numpy as np
import wrappers
import cython_fcts as cf
import lom_sampling as sampling
import sys
import tempfile
import sklearn
from IPython.core.debugger import Tracer

# optional scipy dependencies
def expit(x):
	"""
	better implementation in scipy.special, 
	but can avoid dependency
	"""
	try:
		from scipy.special import expit
		return expit(x)
	except:
		return 1/(1+np.exp(-x))


def logit(x):
	"""
	better implementation in scipy.special, 
	but can avoid dependency
	"""
	try:
		from scipy.special import logit
		return logit(x)
	except:
		return np.log(float(x)/(1-x))


def logsumexp(a):
	"""
	better implementation in scipy.special, 
	but can avoid dependency
	"""    
	try:
		from scipy.special import logsumexp
		return logsumexp(a)
	except:
		a_max = np.max(a)
		out = np.log(np.sum(np.exp(a - a_max)))
		out += a_max
		return out


def maxmachine_relevance(layer, model_type='mcmc'):
	"""
	Return the expectec fraction of 1s that are modelled 
	by any latent dimension.
	This is similiar to the eigenvalues in PCA.
	"""

	# avoid unnecessary re-computation.
	if layer.eigenvals is not None:
		return layer.eigenvals


	if model_type is 'plugin': 
		alphas = layer.lbda()
		x = layer.child()
		# add clamped units
		z = np.concatenate([(layer.z.mean()+1)*.5,np.ones([layer.z().shape[0],1])], axis=1)
		u = np.concatenate([(layer.u.mean()+1)*.5,np.ones([layer.u().shape[0],1])], axis=1)
		N = z.shape[0]
		D = u.shape[0]
		L = z.shape[1]

		# # we need to argsort all z*u*alpha. This is of size NxDxL!
		# l_sorted = np.zeros([N,D,L], dtype=np.int8)
		# for n in range(N): # could be parallelised -> but no argsort in cython without extra pain
		#     for d in range(D):
		#         l_sorted[n,d,:] = np.argsort(alphas*u[d,:]*z[n,:])

		eigenvals = np.zeros(len(alphas)) # array for results
		idxs = np.where(layer.child()==1)
		idxs = zip(idxs[0],idxs[1]) # indices of n,d where x[n,d]=1
		no_ones = len(idxs)

		# iterate from largest to smallest alpha doesn't work
		for l in range(L):
			for n,d in idxs:
				eigenvals[l] += z[n,l]*u[d,l]*alphas[l] * np.prod(
					[1-z[n,l_prime]*u[d,l_prime] for l_prime in range(L) 
					 if z[n,l_prime]*u[d,l_prime]*alphas[l_prime] > z[n,l]*u[d,l]*alphas[l]])
		eigenvals /= len(idxs)

	elif model_type is 'mcmc':
		alpha_tr = layer.lbda.trace
		z_tr = layer.z.trace
		u_tr = layer.u.trace
		x = layer.child()
		tr_len = u_tr.shape[0]
		eigenvals = np.zeros(alpha_tr.shape[1])


		for tr_idx in range(tr_len):
			for l in range(u_tr.shape[2]):
				
				# deterministic prediction of current l
				x_pred = np.dot(z_tr[tr_idx,:,l:l+1]==1,
								u_tr[tr_idx,:,l:l+1].transpose()==1)

				# deterministic prediction of all l' > l
				x_pred_alt = np.zeros(x_pred.shape)
				for l_alt in range(u_tr.shape[2]):
					if alpha_tr[tr_idx,l_alt] > alpha_tr[tr_idx,l]:
						x_pred_alt += np.dot(z_tr[tr_idx,:,l_alt:l_alt+1]==1,
											 u_tr[tr_idx,:,l_alt:l_alt+1].transpose()==1)
						x_pred_alt = x_pred_alt > 0

				eigenvals[l] += alpha_tr[tr_idx,l] * np.sum(x[(x_pred==1) & (x_pred_alt !=1)] == 1)
				# import pdb; pdb.set_trace()

				# eigenvals[l] += alpha_tr[tr_idx,l] * np.sum(
				#     x[np.dot(z_tr[tr_idx,:,l:l+1]==1,
				#              u_tr[tr_idx,:,l:l+1].transpose()==1)]==1)
			eigenvals[-1] += alpha_tr[tr_idx, -1]

		eigenvals /= float(tr_len)*np.sum(x==1)

	layer.eigenvals = eigenvals
	return eigenvals

		
def maxmachine_forward_pass(u, z, alpha):
	"""
	compute probabilistic output for a single 
	latent dimension in maxmachine.
	"""
	x = np.zeros([z.shape[0], u.shape[0]])

	for l in np.argsort(-alpha[:-1]):
		x[x==0] += alpha[l]*np.dot(z[:,l:l+1],u[:,l:l+1].transpose())[x==0]

	x[x==0] = alpha[-1]

	return x


def compute_bp(q, n, N, tau=1):
	"""
	compute list of beta-binomial logit for 1...n draws with
	beta parameters a, b. Length of output is N
	and p(n>N) = -infinity.

	n = number of draws
	q = success probability per draw
	N = size of output (output gets logit(0)-padded)
	"""
	
	exp_bp = [(q*(n-k*tau)) / ((1-q)*(k*tau+1)) for k in range(n)]
	
	bp = [np.log(x) if (x > 0) else -np.infty for x in exp_bp]

	if N != n:
		bp_new = [-np.infty for i in range(N)]
		bp_new[:n-1] = bp
		bp = bp_new
		
	return np.array(bp, dtype=float)


def compute_bbp(n, a, b):
	"""
	compute list of beta-binomial logit for 1...n draws with
	beta parameters a, b.
	"""
	exp_bbp = [(float((n-k)*(k+a))/float((k+1)* (n-k+b-1))) for k in range(n+1)]
	bbp = [np.log(x) if (x > 0) else -np.infty for x in exp_bbp]
	return np.array(bbp, dtype=float)


def unique_ordered(seq):
	"""
	return unique list entries preserving order.
	"""
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]


		
def clean_up_codes(layer, noise_model):
	"""
	Remove redundant or all-zero latent dimensions
	from layer and adjust all attributes accordingly.
	Return True, if any dimension was removed, False otherwise.
	"""
	
	def remove_dimension(l_prime, layer):

		# update for tensorm link does not support parents
		# nor priors
		u = layer.u; z = layer.z; lbda = layer.lbda
		layer.size -=1
		u.val = np.delete(u.val, l_prime, axis=1)
		z.val = np.delete(z.val, l_prime, axis=1)

		if 'tensorm-link' in layer.noise_model:
			v = layer.v
			v.val = np.delete(v.val, l_prime, axis=1)

		else:	
			if layer.noise_model == 'max-link':
				lbda.val = np.delete(u.layer.lbda(), l_prime)
				layer.precompute_lbda_ratios()
			z.update_prior_config()
			u.update_prior_config()
			for iter_mat in [u,z]:
				if len(iter_mat.parents) != 0:
					for parent in iter_mat.parents:
						if parent.role == 'dim2':
							parent.val = np.delete(parent.val, l_prime, axis=0)
							parent.update_prior_config()

	reduction_applied = False
	# remove inactive codes
	l = 0
	while l < layer.size:
		if np.any([np.all(mat()[:,l]== -1) for mat in layer.child.parents]):
		# if np.all(layer.z()[:,l] == -1) or np.all(layer.u()[:,l] == -1):
			# print('remove zero dimension')
			remove_dimension(l, layer)
			reduction_applied = True
		l += 1

	if layer.noise_model == 'tensorm-link':
		return reduction_applied
						
	# remove duplicates
	l = 0
	while l < layer.size:
		l_prime = l+1
		while l_prime < layer.size:
			if (np.all(layer.u()[:,l] == layer.u()[:,l_prime]) or
			np.all(layer.z()[:,l] == layer.z()[:,l_prime])):
				# print('remove duplicate dimension')
				reduction_applied = True
				remove_dimension(l_prime, layer)
			l_prime += 1
		l += 1

	# clean by alpha threshold
	if layer.noise_model == 'max-link':
		l = 0
		while l < layer.size:
			if layer.lbda()[l] < 1e-3:
				# print('remove useless dimension')
				reduction_applied = True
				remove_dimension(l, layer)
			l += 1

	return reduction_applied

		
def reset_codes(layer, noise_model):
	"""
	Reset codes/assignemtns in redundant or unused latent dimensions
	to their initialisation. This can lead to better results and has a
	nonparametric flavor. _Not used in any of the current models_.
	"""
	z = layer.z
	u = layer.u

	has_reset = False
	if noise_model == 'tensorm-link':
		v = layer.v
		for l in range(u().shape[1]):
			for mat in [z(),u(),v()]:
				if np.all(mat[:,l]==-1):
					print('reset zeroed latent dimensions')
					has_reset = True
					for mat in [z,u,v]:
						mat()[:,l] = -1 # np.random.randint(0,2,size=mat.shape[0])
					break
		return has_reset

	else:
		# reset duplicates
		for l in range(u().shape[1]):
			for l_prime in range(l+1, u().shape[1]):
				if np.all(u()[:,l] == u()[:,l_prime]):
					print('reset duplicates')
					z()[:,l_prime] = -1
					u()[:,l_prime] = -1
					if noise_model is 'max-link': 
						#z.k[l_prime] = 0
						#u.k[l_prime] = 0

						u.j = np.array(np.count_nonzero(u()==1, 1), dtype=np.int32)
						z.j = np.array(np.count_nonzero(z()==1, 1), dtype=np.int32)

						u.k[l_prime] = 0 # = np.array(np.count_nonzero(u()==1, 0), dtype=np.int32)
						z.k[l_prime] = 0 # np.array(np.count_nonzero(z()==1, 0), dtype=np.int32)

		# reset zeroed codes
		for l in range(u().shape[1]):
			if np.all(u()[:,l] == -1):
				print('reset zeroed')
				z()[:,l] = -1
				# only needed for binomial / beta-binomial prior
				# TODO implement these prior for ormgachine
				if noise_model is 'max-link': 
					z.k[l] = 0
					z.j = np.array(np.count_nonzero(z()==1, 1), dtype=np.int32)

		# reset nested codes
		for l in range(u().shape[1]):
			for l_prime in range(u().shape[1]):
				if l == l_prime:
					continue
				# smaller code needs to have at least one 1.
				elif np.count_nonzero(u()[:,l_prime]==1) > 1 and np.all(u()[u()[:,l_prime]==1, l]==1):
					print('reset nesting '+str(l)+' '+str(l_prime) +
						  ' ' + str(np.count_nonzero(u()[:,l]==1)) +
						  ' ' + str(np.count_nonzero(u()[:,l_prime]==1)))
					u()[u()[:,l_prime]==1,l] = -1
					# z()[z()[:,l]==1,l_prime] = 1

					if noise_model is 'max-link':
						u.k = np.array(np.count_nonzero(u()==1, 0), dtype=np.int32)
						u.j = np.array(np.count_nonzero(u()==1, 1), dtype=np.int32)
						z.k = np.array(np.count_nonzero(z()==1, 0), dtype=np.int32)
						z.j = np.array(np.count_nonzero(z()==1, 1), dtype=np.int32)



def plot_matrix_ax(mat, ax, draw_cbar=True):
	"""
	wrapper for plotting a matrix of probabilities.
	attribues (optional) are used as xlabels
	"""

	if np.any(mat < 0): 
		print('rescaling matrix to probabilities')
		mat = .5*(mat+1)

	import matplotlib.pyplot as plt
	
	try:
		import seaborn as sns
		
		cmap = sns.cubehelix_palette(
		8, start=2, dark=0, light=1,
		reverse=False, as_cmap=True)

		cmap = sns.cubehelix_palette(
		4, start=2, dark=0, light=1,
		reverse=False, as_cmap=True)

		sns.set_style("whitegrid", {'axes.grid' : False})
			
	except:
		print('lala')
		cmap = 'gray_r'
		
	cax = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1)
	
	return ax, cax
	# ax.set_yticks([])

def plot_matrix(mat, figsize=(7,4), draw_cbar=True, vmin=0, vmax=1, cmap=None):
	"""
	wrapper for plotting a matrix of probabilities.
	attribues (optional) are used as xlabels
	"""

	if np.any(mat < 0): 
		print('rescaling matrix to probabilities')
		mat = .5*(mat+1)
	
	try:
		import seaborn as sns
		
		if cmap is None:
			
			cmap = sns.cubehelix_palette(
			8, start=2, dark=0, light=1,
			reverse=False, as_cmap=True)

			cmap = sns.cubehelix_palette(
			4, start=2, dark=0, light=1,
			reverse=False, as_cmap=True)

		sns.set_style("whitegrid", {'axes.grid' : False})

	except:
		print('lala')
		cmap = 'gray_r'
		
	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111)
	cax = ax.imshow(mat, aspect='auto', cmap=cmap, 
	                vmin=vmin, vmax=vmax, origin='upper')

	if draw_cbar is True:
		cbar = fig.colorbar(cax, orientation='vertical')
	
	return fig, ax
	# ax.set_yticks([])


def plot_codes(mat, attributes=None, order='relevance'):
	"""
	wrapper to plot factor matrices of factorisation models,
	ordered by the code relevance (alternatively by lbda)
	"""

	import matplotlib.pyplot as plt

	try:
		import seaborn as sns
		cmap = sns.cubehelix_palette(
		8, start=2, dark=0, light=1,
		reverse=False, as_cmap=True)
		sns.set_style("whitegrid", {'axes.grid' : False})
	except:
		print('seaborn import failed')
		cmap = 'gray_r'
	
	eigenvals = maxmachine_relevance(mat.layer)
	if order == 'relevance':
		l_idx = np.argsort(-np.array(eigenvals[:-1]))
	elif order == 'lbda':
		l_idx = np.argsort(-mat.layer.lbda()[:-1])

	fig = plt.figure(figsize=(7,4))
	ax_codes = fig.add_subplot(111)
	
	ax_codes.imshow(mat.mean().transpose()[l_idx,:], aspect='auto', cmap=cmap)

	ax_codes.set_yticks(range(mat().shape[1]))
	if attributes is not None:
		ax_codes.set_xticks(range(len(attributes)))
		xticklabels = ax_codes.set_xticklabels(list(attributes), rotation=90)

	yticklabels = ax_codes.set_yticklabels(
		[ r"$\nu={0:.1f}, $".format(100*eigenvals[i]) +
		  r"$\hat\lambda={0:.1f}$".format(100*mat.layer.lbda()[i]) 
		  for i in l_idx], rotation=0)

	return fig, ax_codes


def get_roc_auc(data, data_train, prediction):
	"""
	compute area under the roc curve
	"""

	zero_idx = np.where(data_train == 0)
	zero_idx = zip(list(zero_idx)[0],list(zero_idx)[1])
	auc = sklearn.metrics.roc_auc_score(
		[data[i,j]==1 for i,j in zero_idx], [prediction[i,j] for i,j in zero_idx])

	return auc


def predict_applicability_simple(data, dimensions=35, max_features=None):
	"""
	wrapper for a single layer maxmachine, meant to predict
	attribute applicability.
	"""

	# check input format
	if not -1 in np.unique(data):
		data = 2*data-1

	# sample hold-out data as test-set
	data_train = split_test_train(data)

	mm = maxmachine.Machine()
	mm_data = mm.add_matrix(val=np.array(data_train, dtype=np.int8),
				 sampling_indicator=False)
	
	layer = mm.add_layer(size=int(dimensions),
						  child=mm_data,
						  z_init='kmeans',
						  u_init='kmeans',
						  lbda_init=.9)
	layer.lbda.set_prior([10, 2])
	
	if max_features is not None:
		layer.u.set_prior('binomial', [.5, max_features], axis=1)

	layer.auto_clean_up = True

	mm.infer(no_samples=20, convergence_eps=5e-3, print_step=100)

	auc = get_roc_auc(data, data_train, layer.output())

	print('Test set area under ROC: '+str(auc))
	
	return layer
	

def split_test_train(data, p=.1):
	""" 
	In a binary matrix {-1,1}, set randomly 
	p/2 of the 1s and p/2 of the -1s to 0.
	This serves to create a test set for maxmachine/ormachine.
	"""
	import itertools
	
	if not -1 in np.unique(data):
		data = 2*data-1
		
	num_of_zeros = np.prod(data.shape)*p
	index_pairs = list(itertools.product(range(data.shape[0]), range(data.shape[1])))

	# randomly set indices unobserved
	if False:
		random_idx = np.random.choice(range(len(index_pairs)), num_of_zeros, replace=False)
		zero_idx = [index_pairs[i] for i in random_idx]

	# set same number applicable/non-applicable unobserved
	if True:
		true_index_pairs = [x for x in index_pairs if data[x]==1]
		false_index_pairs = [x for x in index_pairs if data[x]==-1]
		true_random_idx = np.random.choice(range(len(true_index_pairs)), 
										   int(num_of_zeros/2), replace=False)
		false_random_idx = np.random.choice(range(len(false_index_pairs)), 
											int(num_of_zeros/2), replace=False)
		zero_idx = [true_index_pairs[i] for i in true_random_idx] + [false_index_pairs[i] 
																	 for i in false_random_idx]

	data_train = data.copy()
	for i, j in zero_idx:
		data_train[i,j] = 0
		
	return data_train


def predict_applicability_fast(data,
							   N_sub = 1000,
							   dimensions=35,
							   max_features=None,
							   lbda_prior=None,
							   binom_prior_attr_sets=.5,
							   high_level_object_coding=None,
							   seed=1):
	"""
	wrapper for learning on a subsample and predicting on the whole data.
	lbda_prior - list: [a,b] parameters of beta prior
	"""

	np.random.seed(seed)
	old_stdout = sys.stdout
	
	L = dimensions # reassign for brevity in expressions
	data = check_binary_coding(data)
	data_train = split_test_train(data)
	
	# select subset at random
	if N_sub > data.shape[0]:
		N_sub = data.shape[0]

	subset_idx = np.random.choice(range(data.shape[0]), N_sub, replace=False)
	data_train_sub = data_train[subset_idx, :]

	# define model
	mm = maxmachine.Machine()
	data_layer = mm.add_matrix(val=data_train_sub, sampling_indicator=False)
	layer1 = mm.add_layer(size=int(L), child=data_layer, z_init=.1,
						  u_init='kmeans', noise_model='max-link', lbda_init=.95)
	if max_features is not None:
		layer1.u.set_prior('binomial',[binom_prior_attr_sets, max_features], axis=1)
	else:
		layer1.u.set_prior('binomial',[binom_prior_attr_sets], axis=1)   

	if lbda_prior is not None:
		layer1.lbda.set_prior(lbda_prior)
	layer1.auto_clean_up = True

	if high_level_object_coding is not None:
		high_level_object_coding = check_binary_coding(high_level_object_coding)
		layer2 = mm.add_layer(size=high_level_object_coding.shape[1],
							   child=layer1.z, 
							   noise_model='max-link',
							   lbda_init=.6, 
							   z_init=high_level_object_coding[subset_idx, :])
		layer2.z.set_sampling_indicator(False)

	# train

	print('Training on subsample...')
	sys.stdout = tempfile.TemporaryFile() # prevent printing (todo: write a decorator)
	mm.infer(no_samples=int(5e1), convergence_window=10,
			  convergence_eps=1e-2, burn_in_min=100, 
			  burn_in_max=int(3e3), fix_lbda_iters=10)
	sys.stdout = old_stdout

	# now run on full dataset with previous results as initialisation,
	# keep u fixed to learn z's

	L = layer1.u().shape[1]
	mm_2 = maxmachine.Machine()
	# define model architecture
	data_layer_2 = mm_2.add_matrix(val=data_train, sampling_indicator=False)
	layer1_2 = mm_2.add_layer(size=int(L), child=data_layer_2, z_init=0.0, 
							   u_init=2*(layer1.u.mean()>.5)-1,
							   noise_model='max-link', lbda_init=.9)
	# layer1_2.z.set_prior('binomial', [.5], axis=0)
	layer1_2.u.sampling_indicator = False
	layer1_2.auto_clean_up = True

	if high_level_object_coding is not None:
		layer2_2 = mm_2.add_layer(size=high_level_object_coding.shape[1],
								  child=layer1_2.z, 
								  noise_model='max-link', lbda_init=.6, 
								  z_init=high_level_object_coding)
		layer2_2.z.set_sampling_indicator(False)

	# train (i.e. adjust the z's and lbdas)
	print('Learning latent representation for all objects...')
	sys.stdout = tempfile.TemporaryFile()
	mm_2.infer(no_samples=int(10), convergence_window=5,
				convergence_eps=1e-2, burn_in_min=20, 
				burn_in_max=200, fix_lbda_iters=3)
	sys.stdout = old_stdout
	
	# now sample u and z
	layer1_2.u.sampling_indicator = True
	print('Drawing samples on the full dataset...')
	sys.stdout = tempfile.TemporaryFile()    
	mm_2.infer(no_samples=int(2e1), convergence_window=5,
				convergence_eps=5e-3, burn_in_min=10, 
				burn_in_max=50, fix_lbda_iters=3)
	sys.stdout = old_stdout
	
	roc_auc = get_roc_auc(data, data_train, layer1_2.output())
	print('Area under ROC curve: ' +  str(roc_auc))
	
	return layer1_2, roc_auc, data_train
	

def check_binary_coding(data):
	"""
	For MaxMachine and OrM, data and latent variables are
	in {-1,1}. Check and corret the coding here.
	"""

	if not -1 in np.unique(data):
		data = 2*data-1

	return np.array(data, dtype=np.int8)


def check_convergence_single_trace(trace, eps):
	"""
	compare mean of first and second half of a sequence,
	checking whether there difference is > epsilon.
	"""

	l = int(len(trace)/2)
	r1 = expit(np.mean(trace[:l]))
	r2 = expit(np.mean(trace[l:]))
	r = expit(np.mean(trace))
	
	if np.abs(r1-r2) < eps:
		return True
	else:
		return False       


def boolean_tensor_product(Z,U,V):
	"""
	Return the Boolean tensor product of three matrices
	that share their second dimension.
	"""

	N = Z.shape[0]
	D = U.shape[0]
	M = V.shape[0]
	L = Z.shape[1]
	X = np.zeros([N,D,M], dtype=bool)

	assert(U.shape[1]==L)
	assert(V.shape[1]==L)

	for n in range(N):
		for d in range(D):
			for m in range(M):
				if np.any([(Z[n,l] == True) and 
						   (U[d,l] == True) and 
						   (V[m,l] == True) 
						   for l in range(L)]):
					X[n,d,m] = True
	return X



def add_bernoulli_noise(X, p):
	
	X_intern = X.copy()
	
	for n in range(X.shape[0]):
		for d in range(X.shape[1]):
			for m in range(X.shape[2]):
				if np.random.rand() < p:
					X_intern[n,d,m] = ~X_intern[n,d,m]
					
	return X_intern



def add_bernoulli_noise_2d(X, p, seed=None):
	
	if seed is None:
		np.random.seed(np.random.randint(1e4))

	X_intern = X.copy()
	
	for n in range(X.shape[0]):
		for d in range(X.shape[1]):
			if np.random.rand() < p:
				X_intern[n,d] = ~X_intern[n,d]
					
	return X_intern



def add_bernoulli_noise_2d_biased(X, p_plus, p_minus, seed=None):
	
	if seed is None:
		np.random.seed(np.random.randint(1e4))

	X_intern = X.copy()
	
	for n in range(X.shape[0]):
		for d in range(X.shape[1]):
			if X_intern[n,d] == 1:
				p = p_plus
			if X_intern[n,d] == 0:
				continue
			elif X_intern[n,d] == -1:
				p = p_minus

			if np.random.rand() < p:
				X_intern[n,d] = -X_intern[n,d]
					
	return X_intern




def flatten(t):
	"""
	Generator flattening the structure
	 
	>>> list(flatten([2, [2, (4, 5, [7], [2, [6, 2, 6, [6], 4]], 6)]]))
	[2, 2, 4, 5, 7, 2, 6, 2, 6, 6, 4, 6]
	"""

	import collections
	for x in t:
		if not isinstance(x, collections.Iterable):
			yield x
		else:
			yield from flatten(x)



def intersect_dataframes(A, B):
    """
    given two dataframes, intersect rows and columns of both
    """
    
    joint_rows = set(A.index).intersection(B.index)
    A = A[A.index.isin(joint_rows)]
    B = B[B.index.isin(joint_rows)]
    
    joint_cols = set(A.columns).intersection(B.columns)
    A = A[list(joint_cols)]
    B = B[list(joint_cols)]
    
    A = A.sort_index()
    B = B.sort_index()
    
    assert np.all(A.index == B.index)
    assert np.all(A.columns == B.columns)
    
    print('\n\tNew shape is :'+str(mut.shape))
    
    return A, B			