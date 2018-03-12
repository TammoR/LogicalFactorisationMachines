#!/usr/bin/env python
"""
Numba sampling routines
"""

import numpy as np
import numba
import math
from numba import jit, int8, int16, int32, float32, float64, prange
import lom._cython.matrix_updates as cython_mu
from scipy.special import expit


@jit('int8(float64, int8)', nopython=True, nogil=True)
def flip_metropolised_gibbs_numba(logit_p, z):
    """
    Given the logit probability of z=1
    flip z according to metropolised Gibbs
    """
    if z == 1 and logit_p <= 0:
        return -z

    elif z == -1 and logit_p >= 0:
        return -z

    else:
        # map from logit to [0,1]
        if math.exp(-z * logit_p) > np.random.rand():
            return -z
        else:
            return z


@jit('int8(float64, int8)', nopython=True, nogil=True)
def flip_metropolised_gibbs_numba_classic(p, z):
    """
    Given the *probability* of z=1
    flip z according to metropolised Gibbs
    """
    if z == 1:
        if p <= .5:
            return -z
            # alpha = 1 # TODO, can return -x here
        else:
            alpha = (1 - p) / p
    else:
        if p >= .5:
            return -z
            # alpha = 1
        else:
            alpha = p / (1 - p)
    if np.random.rand() < alpha:
        return -z
    else:
        return z


@jit('int16(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_score_OR_AND_2D(Z_n, U, X_n, l):
    """
    Return count of correct/incorrect explanations
    caused by setting Z[n,l] to 1, respecting
    explaining away dependencies
    TODO: should this be given a signature?
    """
    D, L = U.shape

    score = 0
    for d in range(D):
        if U[d, l] != 1:  # AND
            continue

        alrdy_active = False
        for l_prime in range(L):
            if (Z_n[l_prime] == 1) and\
               (U[d, l_prime] == 1) and\
               (l_prime != l):
                alrdy_active = True  # OR
                break

        if alrdy_active is False:
            score += X_n[d]

    return score


@jit('int16(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_score_OR_NAND_2D(Z_n, U, X_n, l):
    """
    Return count of correct/incorrect explanations
    caused by setting Z[n,l] to 1, respecting
    explaining away dependencies
    TODO: needs testing! Algorithm is based on setting z to 0
    """
    D, L = U.shape

    score = 0
    for d in range(D):
        # NAND check not needed. setting z_nl to zero makes it always true
        # This seems wrong TODO !!!!!

        # explaining away check
        alrdy_active = False
        for l_prime in range(L):
            if ((Z_n[l_prime] == 1) or (U[d, l_prime] == 1)) and\
               (l_prime != l):
                alrdy_active = True  # OR
                break

        if alrdy_active is False:
            score += X_n[d]

    return -score


@jit('int16(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)',
     nopython=True, nogil=True)
def posterior_score_OR_AND_3D(Z_n, U, V, X_n, l):
    """
    Return count of correct/incorrect explanations
    caused by setting Z[n,l] to 1, respecting
    explaining away dependencies
    TODO: should this be given a signature?
    """
    D, L = U.shape
    M, _ = V.shape

    score = 0
    for d in range(D):
        for m in range(M):
            if (U[d, l] != 1) or (V[m, l] != 1):  # AND
                continue

            alrdy_active = False
            for l_prime in range(L):
                if (Z_n[l_prime] == 1) and\
                    (U[d, l_prime] == 1) and\
                    (V[m, l_prime] == 1) and\
                        (l_prime != l):
                    alrdy_active = True  # OR
                    break

            if alrdy_active is False:
                score += X_n[d, m]

    return score


@jit('void(int8[:,:], int8[:,:], int8[:,:], float64)',
     nogil=True, nopython=True, parallel=True)
def draw_OR_AND_2D(Z, U, X, lbda):
    N, L = Z.shape
    for n in prange(N):
        for l in range(L):
            score = posterior_score_OR_AND_2D(Z[n, :], U, X[n, :], l)
            logit_p = lbda * score  # remain on logit scale [-inf, inf]
            Z[n, l] = flip_metropolised_gibbs_numba(logit_p, Z[n, l])
            #  Z[n,l] = flip_metropolised_gibbs_numba_classic(expit(logit_p), Z[n,l])


@jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:,:], float64)',
     nogil=True, nopython=True, parallel=True)
def draw_OR_AND_3D(Z, U, V, X, lbda):
    N, L = Z.shape
    for n in prange(N):
        for l in range(L):
            score = posterior_score_OR_AND_3D(Z[n, :], U, V, X[n, :, :], l)
            logit_p = lbda * score  # remain on logit scale [-inf, inf]
            Z[n, l] = flip_metropolised_gibbs_numba(logit_p, Z[n, l])


@jit('void(int8[:,:], int8[:,:], int8[:,:], float64)',
     nogil=True, nopython=True, parallel=True)
def draw_OR_NAND_2D(Z, U, X, lbda):
    N, L = Z.shape
    for n in prange(N):
        for l in range(L):
            score = posterior_score_OR_NAND_2D(Z[n, :], U, X[n, :], l)
            logit_p = lbda * score  # remain on logit scale [-inf, inf]
            Z[n, l] = flip_metropolised_gibbs_numba(logit_p, Z[n, l])


if __name__ == '__main__':

    N = 25
    D = 10
    L = 3

    Z = np.array(np.random.rand(N, L) > .5, dtype=np.int8)
    U = np.array(np.random.rand(D, L) > .3, dtype=np.int8)
    X = np.array(np.dot(Z == 1, U.transpose() == 1), dtype=np.int8)
    X = 2 * X - 1
    U = 2 * U - 1
    Z = 2 * Z - 1  # map to {-1, 0, 1} reprst.
    Z_start = Z.copy()

    draw_OR_AND_2D(Z, U, X, 1000.0)

    try:
        assert np.all(Z_start == Z)
    except:
        print('1: ', np.mean(Z_start == Z))

    assert not np.all(Z_start == Z)
