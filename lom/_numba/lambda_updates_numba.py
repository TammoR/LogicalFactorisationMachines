#!/usr/bin/env python
"""
Numba sampling routines
"""

import numpy as np
import numba
import math
from numba import jit, int8, int16, int32, float32, float64, prange


@jit('int8(int8[:], int8[:])', nogil=True, nopython=True)
def or_out(Z_n, U_d):
    """
    Compute OR-AND on two binary vectors in {-1, 1} representation
    """

    for l in range(Z_n.shape[0]):
        if Z_n[l] == 1 and U_d[l] == 1:
            return 1
    return -1


@jit('int8(int8[:], int8[:], int8[:])', nogil=True, nopython=True)
def or_out_3D(Z_n, U_d, V_m):
    """
    Compute OR-AND on two binary vectors in {-1, 1} representation
    """
    for l in range(Z_n.shape[0]):
        if Z_n[l] == 1 and U_d[l] == 1 and V_m[l] == 1:
            return 1
    return -1


@jit('int32(int8[:,:], int8[:,:], int8[:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_OR_AND_2D(Z, U, X):

    N, D = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            if or_out(Z[n, :], U[d, :]) == X[n, d]:
                count += 1

    return count


@jit('int32(int8[:,:], int8[:,:], int8[:,:], int8[:,:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_OR_AND_3D(Z, U, V, X):

    N, D, M = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            for m in range(M):
                if or_out_3D(Z[n, :], U[d, :], V[m, :]) == X[n, d, m]:
                    count += 1

    return count


def lbda_OR_AND(parm, K):
    """
    Set lambda in OR-AND machine to its MLE
    TODO: make for general arity
    """

    if K == 2:
        P = count_correct_predictions_OR_AND_2D(
            *[x.val for x in parm.layer.factors], parm.layer.child())

    elif K == 3:
        P = count_correct_predictions_OR_AND_3D(
            *[x.val for x in parm.layer.factors], parm.layer.child())

    ND = np.prod(parm.layer.child().shape) - np.sum(parm.layer.child() == 0)

    # Laplace rule of succession
    parm.val = -np.log(((ND + 2) / (float(P) + 1)) - 1)


if __name__ == '__main__':

    N = 100
    D = 100
    L = 2
    Z = np.array(np.random.rand(N, L) > .5, dtype=np.int8)
    U = np.array(np.random.rand(D, L) > .5, dtype=np.int8)
    X = np.array(np.dot(Z == 1, U.transpose() == 1), dtype=np.int8)
    X = 2 * X - 1
    U = 2 * U - 1
    Z = 2 * Z - 1  # map to {-1, 0, 1} reprst.
    Z_start = Z.copy()

    num_flips = 100
    n_flip = np.random.choice(range(N), num_flips, replace=False)
    d_flip = np.random.choice(range(D), num_flips, replace=False)

    for n, d in zip(n_flip, d_flip):
        X[n, d] *= -1

    assert count_correct_predictions_or_and_numba(Z, U, X) == N * D - num_flips
    print('assertion succeeded.')
