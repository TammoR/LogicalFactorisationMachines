import numpy as np
import lom._numba.matrix_updates_numba as numba_mu
from lom.auxiliary_functions import generate_orm_product

def test_ormachine_single_flip_numba():

	U = np.array([[1,0],[0,1]], dtype=np.int8)
	Z = np.array([[1,0],[0,1]], dtype=np.int8)
	X = np.array(np.dot(Z==1, U.transpose()==1), dtype=np.int8)
	U[0,0] = 0

	X = 2*X-1; U=2*U-1; Z=2*Z-1

	numba_mu.draw_Z_numba(
		U, Z, X.transpose(), 1000.0)

	assert U[0,0]==1	



def test_ormachine_update_large_lambda_numba():
	"""
	Sample from U/Z with quasi deterministic lambda and correct product X.
	"""

	U, Z, X = generate_orm_product()

	Z_start = Z.copy()
	U_start = U.copy()
	
	numba_mu.draw_Z_numba(Z, U, X, 1000.0)
	numba_mu.draw_Z_numba(U, Z, X.transpose(), 1000.0)

	assert np.all(Z_start == Z)
	assert np.all(U_start == U)



def test_ormachine_update_small_lambda_numba():
	"""
	Sample from U/Z with quasi deterministic lambda and correct product X.
	"""

	U, Z, X = generate_orm_product()

	Z_start = Z.copy()
	U_start = U.copy()
	
	numba_mu.draw_Z_numba(Z, U, X, 0.0)
	numba_mu.draw_Z_numba(U, Z, X.transpose(), 0.0)

	assert not np.all(Z_start == Z)
	assert not np.all(U_start == U)