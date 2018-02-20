#!/usr/bin/env python
#cython: profile=False, language_level=3, boundscheck=False, wraparound=False, cdivision=True
# #cython --compile-args=-fopenmp --link-args=-fopenmp --force -a
## for compilation run: python setup.py build_ext --inplace
"""
Logical Operator Machines
Cython functions for sampling in TensOrMachine
"""

cimport cython
from cython.parallel import prange, parallel
from libc.math cimport exp
from libc.math cimport log
from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc
from IPython.core.debugger import Tracer
from libcpp cimport bool as bool_t

cimport numpy as np
import numpy as np
import cython_fcts as cf

data_type = np.int8
ctypedef np.int8_t data_type_t


cdef int score_tensor_parents(
	data_type_t[:,:] child_n, 
	data_type_t[:] sibling1_n,
	data_type_t[:,:] sibling2,
	data_type_t[:,:] sibling3,
	int l) nogil:
	"""
	For a given latent matrix entry, compute the contributions
	to the logit posterior
	"""
	
	cdef int L = sibling1_n.shape[0]
	cdef int D = sibling2.shape[0]
	cdef int M = sibling3.shape[0]
	cdef bint alrdy_active
	cdef int score = 0
	
	# assert L == sibling2.shape[1]
	# assert L == sibling3.shape[1]
	# assert child_n.shape[0] == D
	# assert child_n.shape[1] == M

	for d in range(D):
		for m in range(M):
			if (sibling2[d,l] != 1) or (sibling3[m,l] !=1):
				continue

			alrdy_active = False
			for l_prime in range(L):
				if (sibling1_n[l_prime] == 1 and
					sibling2[d,l_prime] == 1 and
					sibling3[m,l_prime] == 1 and
					l_prime != l):
					alrdy_active = True
					break

			if (alrdy_active is False):
				score += child_n[d,m]

	return score

					
def draw_tensorm_noparents_onechild(
	data_type_t[:,:] sibling1,
	data_type_t[:,:] sibling2,
	data_type_t[:,:] sibling3,
	data_type_t[:,:,:] child,
	float lbda,
	float logit_prior):

	cdef float p, child_contribution
	cdef float prior = 0
	cdef int n, d, m, l
	cdef int N = sibling1.shape[0], L = sibling1.shape[1]
	cdef int D = sibling2.shape[0], 
	cdef int M = sibling3.shape[0],
	cdef data_type_t x_old

	for n in prange(N, schedule='dynamic', nogil=True):
		for l in range(L):
			child_contribution = lbda*score_tensor_parents(
									child[n,:,:], 
									sibling1[n,:], 
									sibling2[:,:],
									sibling3[:,:],
									l)

			p = sigmoid(child_contribution + logit_prior)
			sibling1[n,l] = swap_metropolised_gibbs_unified(p, sibling1[n,l])



def draw_tensorm_indp_noparents_onechild(
	data_type_t[:,:] sibling1,
	data_type_t[:,:] sibling2,
	data_type_t[:,:] sibling3,
	data_type_t[:,:,:] child,
	float lbda_p,
	float lbda_m):

	cdef float p, child_count
	cdef float prior = 0
	cdef int n, d, m, l
	cdef int N = sibling1.shape[0], L = sibling1.shape[1]
	cdef int D = sibling2.shape[0], 
	cdef int M = sibling3.shape[0],
	cdef data_type_t x_old

	for n in range(N):
		for l in range(L):
			child_count = score_tensor_parents(
									child[n,:,:], 
									sibling1[n,:], 
									sibling2[:,:],
									sibling3[:,:],
									l)

			p = 1/( 1 + ( 1 + exp(-child_count*lbda_m) ) /\
				( 1 + exp(child_count*lbda_p) ) )

			# sibling1[n,l] = cf.swap_gibbs_unified(p, sibling1[n,l])
			sibling1[n,l] = swap_metropolised_gibbs_unified(p, sibling1[n,l])



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long compute_p_tensorm(
	data_type_t[:,:,:] child,
	data_type_t[:,:] parent1,
	data_type_t[:,:] parent2,
	data_type_t[:,:] parent3): #nogil:
	""" 
	compute number of deterministically correct predictions 
	"""
	
	cdef long P = 0
	cdef int n, d, m
	cdef data_type_t out

	# assert child.shape[0] == parent1.shape[0]
	# assert child.shape[1] == parent2.shape[0]
	# assert child.shape[2] == parent3.shape[0]

	# for n in prange(child.shape[0], schedule='dynamic', nogil=True):
	for n in range(child.shape[0]):
		for d in range(child.shape[1]):
			for m in range(child.shape[2]):
				if child[n, d, m] == tensorm_single_output(parent1[n,:], 
				                                          parent2[d,:], 
				                                          parent3[m,:]):
					P += 1
	return P



cpdef compute_tp_fp_tensorm(
	data_type_t[:,:,:] child,
	data_type_t[:,:] parent1,
	data_type_t[:,:] parent2,
	data_type_t[:,:] parent3): #nogil:
	""" 
	compute TP/FP deterministically
	"""
	
	cdef long TP = 0
	cdef long FP = 0
	cdef int n, d, m
	cdef int prediction

	assert child.shape[0] == parent1.shape[0]
	assert child.shape[1] == parent2.shape[0]
	assert child.shape[2] == parent3.shape[0]

	# for n in prange(child.shape[0], schedule=dynamic, nogil=True):
	for n in range(child.shape[0]):
		for d in range(child.shape[1]):
			for m in range(child.shape[2]):
				prediction = tensorm_single_output(
								parent1[n,:], 
								parent2[d,:], 
								parent3[m,:])
				if prediction == 1:
					if child[n, d ,m] == 1:
						TP += 1
					elif child[n, d, m] == -1:
						FP += 1
	return TP, FP


cpdef compute_tn_fn_tensorm(
	data_type_t[:,:,:] child,
	data_type_t[:,:] parent1,
	data_type_t[:,:] parent2,
	data_type_t[:,:] parent3): #nogil:
	""" 
	compute TP/FP deterministically
	"""
	
	cdef long TN = 0
	cdef long FN = 0
	cdef int n, d, m
	cdef int prediction

	assert child.shape[0] == parent1.shape[0]
	assert child.shape[1] == parent2.shape[0]
	assert child.shape[2] == parent3.shape[0]

	# for n in prange(child.shape[0], schedule=dynamic, nogil=True):
	for n in range(child.shape[0]):
		for d in range(child.shape[1]):
			for m in range(child.shape[2]):
				prediction = tensorm_single_output(
								parent1[n,:], 
								parent2[d,:], 
								parent3[m,:])
				if prediction == -1:
					if child[n, d ,m] == -1:
						TN += 1
					elif child[n, d, m] == 1:
						FN += 1
	return TN, FN


@cython.boundscheck(False)
@cython.wraparound(False)
cdef data_type_t tensorm_single_output(
	data_type_t[:] z,
	data_type_t[:] u,
	data_type_t[:] v) nogil:
	"""
	for three vectors of same length, u and z, compute 2*min(1, u^T z)-1
	"""

	# assert u.shape[0] == z.shape[0]
	# assert u.shape[0] == v.shape[0]

	cdef int i
	for i in range(u.shape[0]):
		if u[i] == 1 and z[i] == 1 and v[i] == 1:
			return 1
	return -1



cpdef void probabilistic_output_tensorm(
	float[:,:,:] x,
	double[:,:] z,
	double[:,:] u,
	double[:,:] v,
	double lbda):
	cdef float p_dn, sgmd_lbda
	"""
	Output for or-link
	"""
	cdef int N = z.shape[0]
	cdef int D = u.shape[0]
	cdef int M = v.shape[0]
	cdef int L = v.shape[1]

	sgmd_lbda = cf.sigmoid(lbda)

	for d in range(D):
		for n in range(N):
			for m in range(M):
				p = 1
				for l in range(L):
					p = p * ( 1 - z[n,l]*u[d,l]*v[m,l] )
					x[n, d, m] = (sgmd_lbda * (1-p) + (p*(1-sgmd_lbda) ) )


cpdef void probabilistic_output_tensorm_indp(
	double[:,:,:] x,
	double[:,:] z,
	double[:,:] u,
	double[:,:] v,
	double lbda_p,
	double lbda_m):
	cdef float p_dn, sgmd_lbda_p, sgmd_lbda_m
	"""
	Output for or-link
	"""
	cdef int N = z.shape[0]
	cdef int D = u.shape[0]
	cdef int M = v.shape[0]
	cdef int L = v.shape[1]

	sgmd_lbda_p = cf.sigmoid(lbda_p)
	sgmd_lbda_m = cf.sigmoid(lbda_m)

	for d in range(D):
		for n in range(N):
			for m in range(M):
				p = 1
				for l in range(L):
					p = p * ( 1 - z[n,l]*u[d,l]*v[m,l] )
					x[n, d, m] = (sgmd_lbda_p * (1-p) + (p*(1-sgmd_lbda_m) ) )


## following functions are already defined in cython_fcts.pyx,
## but getting nogil error when called form there(? TODO)

cpdef float sigmoid(float x) nogil:
	cdef float p
	p = 1/(1+exp(-x))
	return p


@cython.cdivision(True)
cpdef int swap_metropolised_gibbs_unified(float p, data_type_t x) nogil:
	"""
	Given the p(x=1) and the current state of x \in {-1,1}.
	Draw new x according to metropolised Gibbs sampler
	"""
	cdef float alpha
	if x == 1:
		if p <= .5:
			alpha = 1 # TODO, can return -x here
		else:
			alpha = (1-p)/p
	else:
		if p >= .5:
			alpha = 1
		else:
			alpha = p/(1-p)
	if rand()/float(RAND_MAX) < alpha:
		return -x
	else:
		return x
