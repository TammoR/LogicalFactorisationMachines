import numpy as np
import lom._numba.matrix_updates_numba as numba_mu
from lom.auxiliary_functions import generate_orm_product


def test_ormachine_single_flip_numba():

    U = np.array([[1, 0], [0, 1]], dtype=np.int8)
    Z = np.array([[1, 0], [0, 1]], dtype=np.int8)
    X = np.array(np.dot(Z == 1, U.transpose() == 1), dtype=np.int8)
    U[0, 0] = 0

    X = 2 * X - 1
    U = 2 * U - 1
    Z = 2 * Z - 1

    draw_OR_AND_2D = numba_mu.make_sampling_fct_onechild('OR_AND_2D')
    draw_OR_AND_2D(
        U, np.zeros(U.shape, dtype=np.int8), 
        Z, X.transpose(), 1000.0, 0)

    assert U[0, 0] == 1


def test_ormachine_update_large_lambda_numba():
    """
    Sample from U/Z with quasi deterministic lambda and correct product X.
    """

    np.random.seed(1)

    U, Z, X = generate_orm_product()

    Z_start = Z.copy()
    U_start = U.copy()

    draw_OR_AND_2D = numba_mu.make_sampling_fct_onechild('OR_AND_2D')
    draw_OR_AND_2D(Z, np.zeros(Z.shape, dtype=np.int8), U, X, 10000.0, 0.0)
    draw_OR_AND_2D(U, np.zeros(U.shape, dtype=np.int8), Z, X.transpose(), 10000.0, 0.0)

    assert np.all(Z_start == Z)
    assert np.all(U_start == U)


def test_ormachine_update_small_lambda_numba():
    """
    Sample from U/Z with quasi deterministic lambda and correct product X.
    """

    U, Z, X = generate_orm_product()

    Z_start = Z.copy()
    U_start = U.copy()

    draw_OR_AND_2D = numba_mu.make_sampling_fct_onechild('OR_AND_2D')
    draw_OR_AND_2D(Z, np.zeros(Z.shape, dtype=np.int8), U, X, 0.0, 0.0)
    draw_OR_AND_2D(U, np.zeros(U.shape, dtype=np.int8), Z, X.transpose(), 0.0, 0.0)

    assert not np.all(Z_start == Z)
    assert not np.all(U_start == U)


if __name__ == '__main__':
    
    test_ormachine_single_flip_numba()
    test_ormachine_update_small_lambda_numba()
    test_ormachine_update_large_lambda_numba()
