#!/usr/bin/env python
"""
lom_sampling.py


"""
import numpy as np
import lom.matrix_updates_c_wrappers as wrappers
import lom._cython.matrix_updates as cf
import lom._cython.tensor_updates as cf_tensorm
import lom._numba.lambda_updates_numba as lambda_updates_numba

def draw_lbda_max(parm):
	"""
	Set MachienParameter instance lambda to its MLE/MAP.
	This should be cythonised, but contains nasty functions like argsort.
	"""

	z=parm.attached_matrices[0]
	u=parm.attached_matrices[1]
	x=parm.attached_matrices[0].child
	N = z().shape[0]
	D = u().shape[0]
	L = z().shape[1]

	mask = np.zeros([N,D], dtype=bool)
	l_list = range(L)
	
	predictions = [cf.predict_single_latent(u()[:,l], z()[:,l])==1 for l in l_list]

	TP = [np.count_nonzero(x()[predictions[l]] == 1) for l in range(L)]
	FP = [np.count_nonzero(x()[predictions[l]] == -1) for l in range(L)]

	for iter_index in range(L):

		# use Laplace rule of succession here, to avoid numerical issues
		l_pp_rate = [(tp+1)/float(tp+fp+2) for tp, fp in zip(TP, FP)]        

		# find l with max predictive power
		l_max_idx = np.argmax(l_pp_rate)
		l_max = l_list[l_max_idx]

		# assign corresponding alpha
		# mind: parm is of (fixed) size L, l_pp_rate gets smaller every iteration
		if parm.prior_config[0] == 0:
			# again, using Laplace rule of succession
			parm()[l_max] = l_pp_rate[l_max_idx]

		elif parm.prior_config[0] == 1:
			alpha = parm.prior_config[1][0]
			beta  = parm.prior_config[1][1]
			parm()[l_max] = ( ( TP[l_max_idx] + alpha - 1) /
							  float(TP[l_max_idx] + FP[l_max_idx] + alpha + beta - 2) )

		# remove the dimenson from l_list
		l_list = [l_list[i] for i in range(len(l_list)) if i != l_max_idx]

		# the following large binary arrays need to be computed L times -> precompute here
		temp_array = predictions[l_max] & ~mask
		temp_array1 = temp_array & (x()==1)
		temp_array2 = temp_array & (x()==-1)
		
		TP = [TP[l + (l >= l_max_idx)] - np.count_nonzero(temp_array1 & predictions[l_list[l]])
			   for l in range(len(l_list))]
		FP = [FP[l + (l >= l_max_idx)] - np.count_nonzero(temp_array2 & predictions[l_list[l]])
			   for l in range(len(l_list))]

		mask += predictions[l_max]==1
		
	assert len(l_list) == 0
	
	P_remain = np.count_nonzero(x()[~mask]==1)
	N_remain = np.count_nonzero(x()[~mask]==-1)

	if parm.prior_config[0] == 1:
		alpha = parm.prior_config[2][0]
		beta  = parm.prior_config[2][1]        
		p_new = (P_remain + alpha - 1)/float(P_remain + N_remain + alpha + beta - 2)
	elif parm.prior_config[0] == 0:
		p_new = (P_remain + 1)/float(P_remain + N_remain + 2)

	parm()[-1] = p_new

	# check that clamped lambda/alpha is the smallest
	if parm()[-1] != np.min(parm()):
		print('\nClamped lambda too large. Ok during burn-in, should not happen during sampling!\n')
		parm()[-1] = np.min(parm())
		
	# after updating lambda, ratios need to be precomputed
	# should be done in a lazy fashion
	parm.layer.precompute_lbda_ratios()
	


def draw_lbda_tensorm(parm):
	"""
	Update a Machine parameter to its MLE / MAP estima
	"""

	P = cf_tensorm.compute_p_tensorm(
			parm.layer.child(),
			parm.attached_matrices[0](),
			parm.attached_matrices[1](),
			parm.attached_matrices[2]())

	# effective number of observations (pre-compute for speedup TODO (not crucial))
	NMD = (np.prod(parm.layer.child().shape) -
			np.count_nonzero(parm.layer.child() == 0))

	# use lapalce succession
	parm.val = -np.log( ( (NMD+2) / (float(P+1)) ) - 1  )

	# don't use laplace succession
	# parm.val = -np.log( ( (NMD) / (float(P)) ) - 1  )


def draw_lbda_tensorm_indp_p(parm):
	"""
	Update lbda_p for independent TensOrM
	This is redundant with lbda_minus, but allows 
	for nice modularity
	"""

	TP, FP = cf_tensorm.compute_tp_fp_tensorm(
			parm.layer.child(),
			parm.attached_matrices[0](),
			parm.attached_matrices[1](),
			parm.attached_matrices[2]())


	# use beta prior
	# a = 100
	# b = 1
	# parm.val = np.max(
 #        [ - np.log( ( (TP + FP + a - 1) / ( float( TP + 1 + a + b -2 ) ) ) - 1  ),
 #         0 ] )
	parm.val = np.max([-np.log( ( (TP+FP+2) / (float(TP+1)) )-1 ), 0])
	# print('Positives: '+str(parm.val))


def draw_lbda_tensorm_indp_m(parm):
	"""
	Update lbda_m for independent TensOrM
	This is redundant with lbda_minus, but allows 
	for nice modularity
	"""

	TN, FN = cf_tensorm.compute_tn_fn_tensorm(
			parm.layer.child(),
			parm.attached_matrices[0](),
			parm.attached_matrices[1](),
			parm.attached_matrices[2]())

	# import pdb; pdb.set_trace()
	# use lapalce succession
	parm.val = np.max([-np.log( ( (TN+FN+2) / (float(TN+1)) )-1 ), 0])

	# print('Negatives: '+str(parm()))

def draw_lbda_or_balanced(parm):


	cf.compute_pred_accuracy(parm.attached_matrices[0].child(),
	                         parm.attached_matrices[0](),
	                         parm.attached_matrices[1](),
	                         parm.layer.pred_rates)

	TP, FP, TN, FN = parm.layer.pred_rates
	s = parm.balance_factor

	parm.val = ( TP + ( TN / s) ) / ( TP + FN + ( ( TN + FP ) / s ) )

	draw_lbda_or(parm)



def draw_lbda_or(parm):
	"""
	Update a Machine parameter to its MLE / MAP estimate
	"""

	if parm.layer.machine.framework == 'numba':
		lambda_updates_numba.draw_lbda_or_numba(parm)


	elif parm.layer.machine.framework == 'cython':

		# TODO: for some obscure reason this is faster than compute_P_parallel
		P = cf.compute_P(parm.attached_matrices[0].child(),
								  parm.attached_matrices[1](),
								  parm.attached_matrices[0]())

		# effectie number of observations (precompute for speedup TODO (not crucial))
		ND = (np.prod(parm.attached_matrices[0].child().shape) -\
						np.count_nonzero(parm.attached_matrices[0].child() == 0))

		# Flat prior
		if parm.prior_config[0] == 0:
			# use Laplace rule of succession
			parm.val = -np.log( ( (ND+2) / (float(P)+1) ) - 1  )
			#parm.val = np.max([0, np.min([1000, -np.log( (ND) / (float(P)-1) )])])

		# Beta prior
		elif parm.prior_config[0] == 1:
			alpha = parm.prior_config[1][0]
			beta  = parm.prior_config[1][1]
			parm.val = -np.log( (ND + alpha - 1) / (float(P) + alpha + beta -2) - 1 )        


def infer_sampling_fct_mat(mat):
	"""
	Assing appropriate sampling function as attribute, 
	depending on family status and noise-model.
	"""
	# first do some sanity checks, no of children etc. Todo
	if False and 'independent' in mat.layer.noise_model:
		if not np.any(mat.density_conditions) and not mat.parents:
			if mat.role == 'dim1':
				mat.sampling_fct = wrappers.draw_z_noparents_onechild_indpn_wrapper
			elif mat.role == 'dim2':
				mat.sampling_fct = wrappers.draw_u_noparents_onechild_indpn_wrapper
		else:
			raise StandardError('Appropriate sampling function for independent '+
								'noise model is not defined')
		return

	# elif mat.layer.noise_model == 'or-link':
	# assign different sampling fcts if matrix row/col density is constrained
	# matrix without child...
	if not mat.child:
		# ...and one parent
		if len(mat.parent_layers) == 1:                   
			if mat.role == 'dim1':
				mat.sampling_fct = wrappers.draw_z_oneparent_nochild_wrapper
			elif mat.role == 'dim2':
				mat.sampling_fct = wrappers.draw_u_oneparent_nochild_wrapper
		# ...and two parents
		if len(mat.parent_layers) == 2:
			if mat.role == 'dim1':
				mat.sampling_fct = wrappers.draw_z_twoparents_nochild_wrapper
			elif mat.role == 'dim2':
				mat.sampling_fct = wrappers.draw_u_twoparents_nochild_wrapper

	# matrix with one child...
	elif mat.child:

		# ... and no parent # like here, we don't need extra fcts for u/z
		if not mat.parents:
			if mat.layer.noise_model == 'max-link':
				mat.sampling_fct = wrappers.draw_noparents_onechild_maxmachine
			elif mat.layer.noise_model == 'or-link':
				mat.sampling_fct = wrappers.draw_noparents_onechild_wrapper   

		# ... and one parent 
		elif len(mat.parent_layers) == 1:
			if mat.role == 'dim1':
				if mat.layer.noise_model == 'max-link':
					mat.sampling_fct = wrappers.draw_z_oneparent_onechild_maxmachine
				elif mat.layer.noise_model == 'or-link':
					mat.sampling_fct = wrappers.draw_z_oneparent_onechild_wrapper
			elif mat.role == 'dim2':
				mat.sampling_fct = wrappers.draw_u_oneparent_onechild_wrapper       
		# ... and two parents (not implemented, throwing error)

		elif len(mat.parent_layers) == 2:
			if mat.role == 'dim1':
				mat.sampling_fct = wrappers.draw_z_twoparents_onechild_wrapper
			elif mat.role == 'dim2':
				mat.sampling_fct = wrappers.draw_u_twoparents_onechild_wrapper
		else:
			raise Warning('Sth is wrong with allocting sampling functions')


def infer_sampling_fct_lbda(self):
	"""
	For a MachineParameter, assing the appropriate 
	sampling/update function
	"""
	if 'or-link' in self.layer.noise_model:
		self.sampling_fct = draw_lbda_or    
	elif 'max-link' in self.noise_model:
		self.sampling_fct = draw_lbda_max
	else:
		raise StandardError('Can not infer appropriate samping function for lbda/mu')         

