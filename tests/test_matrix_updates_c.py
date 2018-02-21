import numpy as np
from lom.auxiliary_functions import all_columsn_are_disjoint
import lom._cython.matrix_updates as matrix_updates


def generate_orm_product():
	"""
	Generate random matrix U, Z and their Boolean product X.
	returns: U, Z, X in {-1, 1} representation.
	Ascertain that different U[d,:] and Z[n,:] are disjoint.
	"""

	def disjoint_columns_mat(N=100, D=20, L=3):
		while True:
			mat = np.array(np.random.rand(N,L) > .5, dtype=np.int8)
			if all_columsn_are_disjoint(mat):
				return mat
	
	U = disjoint_columns_mat()
	Z = disjoint_columns_mat()

	X = np.array(np.dot(Z==1, U.transpose()==1), dtype=np.int8)

	# map to {-1, 0, 1} reprst.
	X = 2*X-1; U=2*U-1; Z=2*Z-1

	return U, Z, X


def test_ormachine_update_large_lambda():
	"""
	Sample from U/Z with quasi deterministic lambda and correct product X.
	"""

	U, Z, X = generate_orm_product()

	Z_start = Z.copy()
	U_start = U.copy()
	
	matrix_updates.draw_noparents_onechild(Z, U, X, 1000.0, 
											  np.ones(X.shape, dtype=np.int8))

	matrix_updates.draw_noparents_onechild(U, Z, X.transpose(), 1000.0, 
											  np.ones(X.transpose().shape, dtype=np.int8))

	assert np.all(Z_start == Z)
	assert np.all(U_start == U)



def test_ormachine_update_large_lambda():
	"""
	Sample from U/Z with negative lambda and correct product X.
	We expect some random flips
	"""

	U, Z, X = generate_orm_product()

	Z_start = Z.copy()
	U_start = U.copy()

	matrix_updates.draw_noparents_onechild(
		Z, U, X, -10.0, 
		np.ones(X.shape, dtype=np.int8))

	matrix_updates.draw_noparents_onechild(
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

	matrix_updates.draw_noparents_onechild(
		Z, U, X, 1000.0, 
		np.ones(X.shape, dtype=np.int8))

	assert Z[0,0]==1
