#!/usr/bin/env python
"""
Numba sampling routines
"""

import numpy as np
import numba
import math
from numba import jit, int8, int16, int32, float32, float64, prange
# import lom._cython.matrix_updates as cython_mu
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


@jit('int16(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_score_XOR_AND_2D(Z_n, U, X_n, l):
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

        # compute deltaXOR-AND
        num_active = np.int8(0)
        for l_prime in range(L):
            if (Z_n[l_prime] == 1) and\
               (U[d, l_prime] == 1) and\
               (l_prime != l):
               num_active += 1
               if num_active > 1:
                    break

        if num_active == 0:
            score += X_n[d]
        elif num_active == 1:
            score -= X_n[d]

    return score    


@jit('int16(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_score_XOR_NAND_2D(Z_n, U, X_n, l):
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

        # compute deltaXOR-NAND
        num_active = np.int8(0)
        for l_prime in range(L):
            if ((Z_n[l_prime] != 1) or (U[d, l_prime] != 1)) and\
               (l_prime != l):
               num_active += 1
               if num_active > 1:
                    break

        if num_active == 0:
            score += X_n[d]
        elif num_active == 1:
            score -= X_n[d]

    return -score


@jit('int16(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_score_OR_NAND_2D(Z_n, U, X_n, l):
    """
    Return count of correct/incorrect explanations
    caused by setting Z[n,l] to 1, respecting
    explaining away dependencies
    TODO: needs testing!
    """
    D, L = U.shape

    score = 0
    for d in range(D):
        if U[d, l] == -1:  # NAND
            continue

        alrdy_active = False
        for l_prime in range(L):
            if ((Z_n[l_prime] == -1) or (U[d, l_prime] == -1)) and\
               (l_prime != l):
                alrdy_active = True  # OR
                break

        if alrdy_active is False:
            score += X_n[d]

    return -score


@jit('int16(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_score_OR_XOR_2D(Z_n, U, X_n, l):
    """
    Return count of correct/incorrect explanations
    caused by setting Z[n,l] to 1, respecting
    explaining away dependencies
    TODO: needs testing!
    """
    D, L = U.shape

    score = 0
    for d in range(D):

        explained_away = False
        for l_prime in range(L):
            if (Z_n[l_prime] != U[d, l_prime]) and (l_prime != l):
                explained_away = True
                break

        if explained_away is False:
            score += X_n[d] * U[d, l]

    return -score



@jit('int16(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_score_NAND_XOR_2D(Z_n, U, X_n, l):

    D, L = U.shape
    score = 0
    for d in range(D):

        explained_away = False
        for l_prime in range(L):
            if (Z_n[l_prime] == U[d, l_prime]) and (l_prime != l):
                explained_away = True
                break

        if explained_away is False:
            score += X_n[d] * U[d, l]

    return score

@jit('int16(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_score_XOR_XOR_2D(Z_n, U, X_n, l):

    D, L = U.shape
    score = 0
    for d in range(D):
        num_active = np.int8(0)
        for l_prime in range(L):
            if (Z_n[l_prime] != U[d, l_prime]) and (l_prime != l):
                num_active += 1
            if num_active > 1:
                break

        if num_active == 0:
            score += - X_n[d] * U[d, l]
        elif num_active == 1:
            score += X_n[d] * U[d, l]

    return score


@jit('int16(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_score_XOR_NXOR_2D(Z_n, U, X_n, l):
    D, L = U.shape
    score = 0
    for d in range(D):
        num_active = np.int8(0)
        for l_prime in range(L):
            if (U[d, l_prime] == Z_n[l_prime]) and (l_prime != l):
                num_active += 1
            if num_active > 1:
                break

        if num_active == 0:
            score += X_n[d] * U[d, l]
        elif num_active == 1:
            score -= X_n[d] * U[d, l]

    return score


@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def OR_AND_product(u, z):
    for i in range(u.shape[0]):
        if u[i] == 1 and z[i] == 1:
            return 1
    return -1

@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def OR_NAND_product(u, z):
    for i in range(u.shape[0]):
        if (u[i] == 0) or (z[i] == 0):
            return 1
    return -1

@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def OR_XOR_product(u, z):
    for i in range(u.shape[0]):
        if u[i] != z[i]:
            return 1
    return -1

@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def NAND_XOR_product(u, z):  # = OR-NXOR
    for i in range(u.shape[0]):
        if u[i] == z[i]:
            return 1
    return -1

@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def XOR_AND_product(u, z):
    xor_count = np.int8(0)
    for i in range(u.shape[0]):
        if u[i] == 1 and z[i] == 1:
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1

@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def XOR_XOR_product(u, z):
    xor_count = np.int8(0)
    for i in range(u.shape[0]):
        if u[i] != z[i]:
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1    

@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def XOR_NXOR_product(u, z):
    xor_count = np.int8(0)
    for i in range(u.shape[0]):
        if u[i] == z[i]:
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1  

@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def XOR_NAND_product(u, z):
    xor_count = np.int8(0)
    for i in range(u.shape[0]):
        if (u[i] == 0) or (z[i]==0):
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1     

@jit('int32(int8[:], int8[:,:], int8[:], int64, int8[:], float32[:, :, :])', 
     nopython=True, nogil=True)
def posterior_score_MAX_AND_2D(Z_n, U, X_n, l, l_sorted, lbda_ratios):
    """
    Return count of correct/incorrect explanations
    caused by setting Z[n,l] to 1, respecting
    explaining away dependencies.
    """
    D, L = U.shape

    score = np.int32(0)
    for d in range(D):
        if U[d, l] != 1:  # AND
            continue

        alrdy_active = False

        # check older siblings
        for l_prime_idx in range(l):
            if (Z_n[l_sorted[l_prime_idx]] == 1) and\
               (U[d, l_sorted[l_prime_idx]] == 1):
                alrdy_active = True  # OR
                break

        if alrdy_active == True:
            continue

        # check younger siblings
        for l_prime_idx in range(l+1, L):
            if (Z_n[l_sorted[l_prime_idx]] == 1) and\
                (U[d, l_sorted[l_prime_idx]] == 1):
                if X_n[d] == 1:
                    score += lbda_ratios[0, l_sorted[l], l_sorted[l_prime_idx]]
                elif X_n[d] == -1:
                    score += lbda_ratios[1, l_sorted[l], l_sorted[l_prime_idx]]
                alrdy_active = True
                break

        if alrdy_active == True:
            continue

        # no siblings explain away -> compare to clamped unit
        if X_n[d] == 1:
            score += lbda_ratios[0, l_sorted[l], L]
        elif X_n[d] == -1:
            score += lbda_ratios[1, l_sorted[l], L]

    return score

@jit('void(int8[:,:], int8[:,:], int8[:,:], float64[:], int8[:], float32[:,:,:])',
     nogil=True, nopython=False, parallel=True)
def draw_MAX_AND_2D(Z, U, X, lbda, l_sorted, lbda_ratios):
    N, L = Z.shape
    for n in prange(N):
        for l in range(L):
            logit_p = posterior_score_MAX_AND_2D(
                        Z[n, :], U, X[n, :], l, l_sorted, lbda_ratios)
            Z[n, l] = flip_metropolised_gibbs_numba(logit_p, Z[n, l])


@jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:,:], float64)',
     nogil=True, nopython=True, parallel=True)
def draw_OR_AND_3D(Z, U, V, X, lbda):
    N, L = Z.shape
    for n in prange(N):
        for l in range(L):
            score = posterior_score_OR_AND_3D(Z[n, :], U, V, X[n, :, :], l)
            logit_p = lbda * score  # remain on logit scale [-inf, inf]
            Z[n, l] = flip_metropolised_gibbs_numba(logit_p, Z[n, l])


@jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:,:], float64, int8[:,:], int8[:,:], float64)',
     nogil=True, nopython=True, parallel=True)
def draw_OR_AND_3D_has_parent(Z, U, V, X, lbda, pa1, pa2, lbda_pa):
    N, L = Z.shape
    for n in prange(N):
        for l in range(L):
            score = posterior_score_OR_AND_3D(Z[n, :], U, V, X[n, :, :], l)
            logit_score = lbda * score  # remain on logit scale [-inf, inf]

            logit_parent_score = lbda_pa * \
                OR_AND_product(pa1[n, :], pa2[l, :])

            Z[n, l] = flip_metropolised_gibbs_numba(
                logit_score + logit_parent_score, Z[n, l])


@jit('void(int8[:,:], int8[:,:], int8[:,:], float64, int8[:,:], int8[:,:], float64)',
     nogil=True, nopython=False, parallel=True)
def draw_OR_AND_2D_has_parent(Z, U, X, lbda, pa1, pa2, lbda_pa):
    N, L = Z.shape
    for n in prange(N):
        for l in range(L):

            score = posterior_score_OR_AND_2D(Z[n, :], U, X[n, :], l)
            logit_score = lbda * score  # remain on logit scale [-inf, inf]

            logit_parent_score = lbda_pa * \
                OR_AND_product(pa1[n, :], pa2[l, :])

            Z[n, l] = flip_metropolised_gibbs_numba(
                logit_score + logit_parent_score, Z[n, l])


def get_posterior_score_fct(model):

    if model == 'OR_AND_2D':
        posterior_score_fct = posterior_score_OR_AND_2D
    elif model == 'OR_NAND_2D':
        posterior_score_fct = posterior_score_OR_NAND_2D
    elif model == 'OR_XOR_2D':
        posterior_score_fct = posterior_score_OR_XOR_2D
    elif model == 'NAND_XOR_2D':
        posterior_score_fct = posterior_score_NAND_XOR_2D
    elif model == 'XOR_AND_2D':
        posterior_score_fct = posterior_score_XOR_AND_2D        
    elif model == 'XOR_XOR_2D':
        posterior_score_fct = posterior_score_XOR_XOR_2D
    elif model == 'XOR_NXOR_2D':
        posterior_score_fct = posterior_score_XOR_NXOR_2D
    elif model == 'XOR_NAND_2D':
        posterior_score_fct = posterior_score_XOR_NAND_2D            
    else:
        print(model)
        raise NotImplementedError
    return posterior_score_fct


def get_parent_score_fct(model):

    if model == 'OR_AND_2D':
        return OR_AND_product
    if model == 'OR_NAND_2D':
        return OR_NAND_product
    if model == 'OR_XOR_2D':
        return OR_XOR_product
    if model == 'NAND_XOR_2D':
        return NAND_XOR_product
    if model == 'XOR_AND_2D':
        return XOR_AND_product                                
    if model == 'XOR_XOR_2D':
        return XOR_XOR_product
    if model == 'XOR_NXOR_2D':
        return XOR_NXOR_product                
    if model == 'XOR_NAND_2D':
        return XOR_NAND_product        
    else:
        print(model)
        raise NotImplementedError


###### USE CLOSURES!! (actually double closures.) ######
def make_sampling_fct(model): # maybe add score for parents

    posterior_score_fct = get_posterior_score_fct(model)

    @jit('void(int8[:,:], int8[:,:], int8[:,:], float64)',
     nogil=True, nopython=True, parallel=True)
    def sampling_fct(Z, U, X, lbda):
        N, L = Z.shape
        for n in prange(N):
            for l in range(L):
                logit_score = lbda * posterior_score_fct(Z[n, :], U, X[n, :], l)
                Z[n, l] = flip_metropolised_gibbs_numba(logit_score, Z[n, l])

    return sampling_fct


def make_sampling_fct_hasparents(model, parent_model):

    posterior_score_fct = get_posterior_score_fct(model)
    parent_posterior_score_fct = get_parent_score_fct(parent_model)

    @jit('void(int8[:,:], int8[:,:], int8[:,:], float64, int8[:,:], int8[:,:], float64)',
     nogil=True, nopython=True, parallel=True)
    def sampling_fct(Z, U, X, lbda, pa1, pa2, lbda_pa):
        N, L = Z.shape
        for n in prange(N):
            for l in range(L):
                logit_score = lbda * posterior_score_fct(Z[n, :], U, X[n, :], l)
                # parent_score = ...
                logit_parent_score = lbda_pa * parent_posterior_score_fct(pa1[n, :], pa2[l, :])

                Z[n, l] = flip_metropolised_gibbs_numba(
                    logit_score + logit_parent_score, Z[n, l])

    return sampling_fct    


def IBP_update(Z, U, X, lbda):

    return

    # # remove features that are used at most once (why does np.where suck so much?)
    # idx_to_keep = [i for i, val in enumerate(list((Z==1).sum(axis=0) <= 1)) if val == False]
    # if len(idx_to_remove) > 0:
    #     print('removed feature')
    #     Z = Z[:, idx_to_keep]
    #     U = U[:, idx_to_keep]

    # from lom._numba.lambda_updates_numba import or_and_out_2D
    # from scipy.special import expit
    # X_predict = or_and_out_2D(Z, U) # make a 1D version and do inside loop

    # N, L = Z.shape
    # for n in range(N):  # can't be parallel, sampling depends on L
    #     FN = np.sum( (X_predict[n, :] == 0) & (X[n, :] == 1) )
    #     TN = np.sum( (X_predict[n, :] == 0) & (X[n, :] == 0) )
    #     if sample_new_dimension(FN, TN, lbda, alpha, L):
    #         L += 1
    #         add_dimension_idx.append(n)

    # for n, l in zip(add_dimension_idx, range(len(add_dimension_idx)):
    #     Z[n,l] = 1

    # for u in add_dimension_idx:
    #     for l in range(L):
    #         sample_U


#############################    

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
