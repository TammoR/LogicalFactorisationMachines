import numpy as np
from lom.auxiliary_functions import all_columsn_are_disjoint
import lom._cython.matrix_updates as matrix_updates_cython
import lom._numba.matrix_updates_numba as numba_mu
from lom.auxiliary_functions import generate_orm_product


def test_ormachine_update_large_lambda_cython():
	"""
	Sample from U/Z with quasi deterministic lambda and correct product X.
	"""

	U, Z, X = generate_orm_product()

	Z_start = Z.copy()
	U_start = U.copy()
	
	matrix_updates_cython.draw_noparents_onechild(Z, U, X, 1000.0, 
											  np.ones(X.shape, dtype=np.int8))

	matrix_updates_cython.draw_noparents_onechild(U, Z, X.transpose(), 1000.0, 
											  np.ones(X.transpose().shape, dtype=np.int8))

	assert np.all(Z_start == Z)
	assert np.all(U_start == U)



def test_ormachine_update_small_lambda_cython():
	"""
	Sample from U/Z with negative lambda and correct product X.
	We expect some random flips
	"""

	U, Z, X = generate_orm_product()

	Z_start = Z.copy()
	U_start = U.copy()

	matrix_updates_cython.draw_noparents_onechild(
		Z, U, X, -10.0, 
		np.ones(X.shape, dtype=np.int8))

	matrix_updates_cython.draw_noparents_onechild(
		U, Z, X.transpose(), -10.0, 
		np.ones(X.transpose().shape, dtype=np.int8))

	assert not np.all(Z_start == Z)
	assert not np.all(U_start == U)



def test_ormachine_single_flip():

	U = np.array([[1,0],[0,1]], dtype=np.int8)
	Z = np.array([[1,0],[0,1]], dtype=np.int8)
	X = np.array(np.dot(Z==1, U.transpose()==1), dtype=np.int8)
	Z[0,0] = 0

	X = 2*X-1; U=2*U-1; Z=2*Z-1

	matrix_updates_cython.draw_noparents_onechild(
		Z, U, X, 1000.0, 
		np.ones(X.shape, dtype=np.int8))

	assert Z[0,0]==1



def test_posterior_scoring():

	from lom._cython.matrix_updates import score_no_parents_unified as cython_score
	from lom._numba.matrix_updates_numba import count_posterior_contributions_numba as numba_score

	U, Z, X = generate_orm_product()

	for n in range(Z.shape[0]):
		for l in range(Z.shape[1]):
			assert cython_score(X[n,:], Z[n,:], U, l) == numba_score(Z[n,:], U, X[n,:], l)

	for d in range(U.shape[0]):
		for l in range(U.shape[1]):
			assert cython_score(X[:,d], U[d,:], Z, l) == numba_score(U[d,:], Z, X[:,d], l)



def test_posterior_flips():
	from scipy.special import expit

	from lom._cython.matrix_updates import swap_metropolised_gibbs_unified as cython_flip
	from lom._numba.matrix_updates_numba import flip_metropolised_gibbs_numba as numba_flip

	cython_mean = np.mean([cython_flip(expit(3), 1) for x in range(100000)])
	numba_mean = np.mean([numba_flip(3, 1) for x in range(100000)])

	assert np.abs(cython_mean - numba_mean) < 0.01
