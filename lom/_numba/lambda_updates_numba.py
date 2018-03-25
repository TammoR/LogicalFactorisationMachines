#!/usr/bin/env python
"""
Numba sampling routines
"""

import numpy as np
import numba
import math
from numba import jit, int8, int16, int32, float32, float64, prange

@jit('float64[:,:](float64[:,:], float64[:,:])', 
     nogil=True, nopython=False, parallel=True)
def or_and_out_2D_fuzzy(Z, U):
    N = Z.shape[0]
    D = U.shape[0]
    X = np.zeros([N, D], dtype=np.float)
    for n in prange(N):
        for d in prange(D):
            X[n, d] = or_and_out_vec_2D_fuzzy(Z[n,:], U[d,:])
    return X

@jit('float64(float64[:], float64[:])', nogil=True, nopython=True)
def or_and_out_vec_2D_fuzzy(Z_n, U_d):
    """
    Compute probability of emitting a zero for fuzzy vectors under OR-AND logic.
    """
    out = 1

    for l in range(Z_n.shape[0]):
        out *= 1 - Z_n[l]*U_d[l] 
    return 1 - 2 * out  # map to [-1,1]

@jit('float64[:,:](float64[:,:], float64[:,:], float64[:,:])', 
     nogil=True, nopython=False, parallel=True)
def or_and_out_3D_fuzzy(Z, U, V):
    N = Z.shape[0]
    D = U.shape[0]
    M = V.shape[0]
    X = np.zeros([N, D, M], dtype=np.float)
    for n in prange(N):
        for d in prange(D):
            for m in range(M):
                X[n, d, m] = or_and_out_vec_3D_fuzzy(Z[n, :], U[d, :], V[m, :])
    return X

@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def or_and_out_vec_3D_fuzzy(Z_n, U_d, V_m):
    """
    Compute probability of emitting a zero for fuzzy vectors under OR-AND logic.
    """
    out = 1

    for l in range(Z_n.shape[0]):
        out *= 1 - (Z_n[l]*U_d[l]*V_m[l])
    return 1 - 2 * out  # map to [-1,1]


@jit('int8[:,:](int8[:,:], int8[:,:])', 
     nogil=True, nopython=False, parallel=True)
def or_and_out_2D(Z, U):
    N = Z.shape[0]
    D = U.shape[0]
    X = np.zeros([N, D], dtype=np.int8)
    for n in prange(N):
        for d in prange(D):
            X[n, d] = or_and_out_vec_2D(Z[n,:], U[d,:])
    return X



@jit('int8[:,:](int8[:,:], int8[:,:])', 
     nogil=True, nopython=False, parallel=True)
def or_xor_out_2D(Z, U):
    N = Z.shape[0]
    D = U.shape[0]
    X = np.zeros([N, D], dtype=np.int8)
    for n in prange(N):
        for d in prange(D):
            X[n, d] = or_xor_out_vec_2D(Z[n,:], U[d,:])
    return X

@jit('int8[:,:](int8[:,:], int8[:,:])', 
     nogil=True, nopython=False, parallel=True)
def nand_xor_out_2D(Z, U):
    N = Z.shape[0]
    D = U.shape[0]
    X = np.zeros([N, D], dtype=np.int8)
    for n in prange(N):
        for d in prange(D):
            X[n, d] = nand_xor_out_vec_2D(Z[n,:], U[d,:])
    return X

@jit('int8[:,:](int8[:,:], int8[:,:])', 
     nogil=True, nopython=False, parallel=True)
def xor_xor_out_2D(Z, U):
    N = Z.shape[0]
    D = U.shape[0]
    X = np.zeros([N, D], dtype=np.int8)
    for n in prange(N):
        for d in prange(D):
            X[n, d] = xor_xor_out_vec_2D(Z[n,:], U[d,:])
    return X

@jit('int8[:,:](int8[:,:], int8[:,:])', 
     nogil=True, nopython=False, parallel=True)
def xor_nxor_out_2D(Z, U):
    N = Z.shape[0]
    D = U.shape[0]
    X = np.zeros([N, D], dtype=np.int8)
    for n in prange(N):
        for d in prange(D):
            X[n, d] = xor_nxor_out_vec_2D(Z[n,:], U[d,:])
    return X            






@jit('int8[:,:](int8[:,:], int8[:,:], int8[:,:])', 
     nogil=True, nopython=False, parallel=True)
def or_and_out_3D(Z, U, V):
    N = Z.shape[0]
    D = U.shape[0]
    M = V.shape[0]
    X = np.zeros([N, D, M], dtype=np.int8)
    for n in prange(N):
        for d in prange(D):
            for m in range(M):
                X[n, d, m] = or_and_out_vec_3D(Z[n, :], U[d, :], V[m, :])
    return X

@jit('int8(int8[:], int8[:])', nogil=True, nopython=True)
def or_and_out_vec_2D(Z_n, U_d):
    """
    Compute OR-AND on two binary vectors in {-1, 1} representation
    """

    for l in range(Z_n.shape[0]):
        if Z_n[l] == 1 and U_d[l] == 1:
            return 1
    return -1


@jit('int8(int8[:], int8[:])', nogil=True, nopython=True)
def xor_and_out_vec_2D(Z_n, U_d):
    """
    Compute OR-AND on two binary vectors in {-1, 1} representation
    """

    or_active = np.int8(0)
    for l in range(Z_n.shape[0]):
        if Z_n[l] == 1 and U_d[l] == 1:
            or_active += 1
        if or_active > 1:
            return -1
    if or_active == 1:
        return 1
    else:
        return -1


@jit('int8(int8[:], int8[:])', nogil=True, nopython=True)
def xor_nand_out_vec_2D(Z_n, U_d):
    """
    Compute OR-AND on two binary vectors in {-1, 1} representation
    """
    or_active = np.int8(0)
    for l in range(Z_n.shape[0]):
        if Z_n[l] != 1 or U_d[l] != 1:
            or_active += 1
        if or_active > 1:
            return -1
    if or_active == 1:
        return 1
    else:
        return -1        


@jit('int8(int8[:], int8[:])', nogil=True, nopython=True)
def or_xor_out_vec_2D(Z_n, U_d):
    for l in range(Z_n.shape[0]):
        if Z_n[l] != U_d[l]:  # check XOR
            return 1
    return -1


@jit('int8(int8[:], int8[:])', nogil=True, nopython=True)
def nand_xor_out_vec_2D(Z_n, U_d):
    for l in range(Z_n.shape[0]):
        if Z_n[l] == U_d[l]:  # XOR is false
            return 1
    return -1

@jit('int8(int8[:], int8[:])', nogil=True, nopython=True)
def xor_xor_out_vec_2D(Z_n, U_d):
    xor_counter = np.int8(0)
    for l in range(Z_n.shape[0]):
        if Z_n[l] != U_d[l]:
            xor_counter += 1
        if xor_counter > 1:
            return -1
    if xor_counter == 1:
        return 1
    else:
        return -1

@jit('int8(int8[:], int8[:])', nogil=True, nopython=True)
def xor_nxor_out_vec_2D(Z_n, U_d):
    nxor_counter = np.int8(0)
    for l in range(Z_n.shape[0]):
        if Z_n[l] == U_d[l]:
            nxor_counter += 1
        if nxor_counter > 1:
            return -1
    if nxor_counter == 1:
        return 1
    else:
        return -1


@jit('int8(int8[:], int8[:])', nogil=True, nopython=True)
def or_nand_out_vec_2D(Z_n, U_d):
    """
    Compute OR-NAND on two binary vectors in {-1, 1} representation
    """

    for l in range(Z_n.shape[0]):
        if Z_n[l] == -1 or U_d[l] == -1:
            return 1
    return -1


@jit('int8(int8[:], int8[:], int8[:])', nogil=True, nopython=True)
def or_and_out_vec_3D(Z_n, U_d, V_m):
    """
    Compute OR-AND on two binary vectors in {-1, 1} representation
    """
    for l in range(Z_n.shape[0]):
        if Z_n[l] == 1 and U_d[l] == 1 and V_m[l] == 1:
            return 1
    return -1


@jit('int32(int8[:,:], int8[:,:], int8[:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_XOR_AND_2D(Z, U, X):

    N, D = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            if xor_and_out_vec_2D(Z[n, :], U[d, :]) == X[n, d]:
                count += 1

    return count


@jit('int32(int8[:,:], int8[:,:], int8[:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_XOR_NAND_2D(Z, U, X):

    N, D = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            if xor_nand_out_vec_2D(Z[n, :], U[d, :]) == X[n, d]:
                count += 1

    return count    


@jit('int32(int8[:,:], int8[:,:], int8[:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_OR_AND_2D(Z, U, X):

    N, D = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            if or_and_out_vec_2D(Z[n, :], U[d, :]) == X[n, d]:
                count += 1

    return count


@jit('int32(int8[:,:], int8[:,:], int8[:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_OR_XOR_2D(Z, U, X):

    N, D = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            if or_xor_out_vec_2D(Z[n, :], U[d, :]) == X[n, d]:
                count += 1

    return count


@jit('int32(int8[:,:], int8[:,:], int8[:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_NAND_XOR_2D(Z, U, X):

    N, D = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            if nand_xor_out_vec_2D(Z[n, :], U[d, :]) == X[n, d]:
                count += 1

    return count


@jit('int32(int8[:,:], int8[:,:], int8[:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_XOR_XOR_2D(Z, U, X):
    N, D = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            if xor_xor_out_vec_2D(Z[n, :], U[d, :]) == X[n, d]:
                count += 1

    return count


@jit('int32(int8[:,:], int8[:,:], int8[:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_XOR_NXOR_2D(Z, U, X):
    N, D = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            if xor_nxor_out_vec_2D(Z[n, :], U[d, :]) == X[n, d]:
                count += 1

    return count


@jit('int32(int8[:,:], int8[:,:], int8[:,:])',
     nogil=True, nopython=True, parallel=True)
def count_correct_predictions_OR_NAND_2D(Z, U, X):

    N, D = X.shape
    count = 0
    for n in prange(N):
        for d in prange(D):
            if or_nand_out_vec_2D(Z[n, :], U[d, :]) == X[n, d]:
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
                if or_and_out_vec_3D(Z[n, :], U[d, :], V[m, :]) == X[n, d, m]:
                    count += 1

    return count


@jit('float64[:, :](float64[:, :], float64[:, :], float64[:])',
     nogil=True, nopython=False, parallel=True)
def MAX_AND_fuzzy(Z, U, lbdas):
    N = Z.shape[0]
    D = U.shape[0]
    L = Z.shape[1]
    out = np.zeros([N, D]) # , dtype=np.float)
    for n in prange(N):
        for d in range(D):
            acc = 0  # accumulator for sum
            for l1 in range(L):
                temp1 = Z[n, l1] * U[d, l1] * lbdas[l1]
                # check for explaining away
                prod = 1
                for l2 in range(L):
                    if l1==l2:
                        continue
                    temp2 = Z[n, l2] * U[d, l2]
                    if temp2*lbdas[l2] > temp1:
                        prod *= 1 - temp2
                acc += temp1 * prod
            out[n, d] = acc
    return out


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


def lbda_XOR_clan(parm):

    if len(parm.layer.factors) == 2:
        if parm.layer.model == 'XOR-AND':
            P = count_correct_predictions_XOR_AND_2D(
                *[x.val for x in parm.layer.factors], parm.layer.child())

        elif parm.layer.model == 'XOR-NAND':
            P = count_correct_predictions_XOR_NAND_2D(
                *[x.val for x in parm.layer.factors], parm.layer.child())

        elif parm.layer.model == 'OR-XOR':
            P = count_correct_predictions_OR_XOR_2D(
                *[x.val for x in parm.layer.factors], parm.layer.child())

        elif parm.layer.model == 'NAND-XOR':
            P = count_correct_predictions_NAND_XOR_2D(
                *[x.val for x in parm.layer.factors], parm.layer.child())

        elif parm.layer.model == 'XOR-XOR':
            P = count_correct_predictions_XOR_XOR_2D(
                *[x.val for x in parm.layer.factors], parm.layer.child())

        elif parm.layer.model == 'XOR-NXOR':
            P = count_correct_predictions_XOR_NXOR_2D(
                *[x.val for x in parm.layer.factors], parm.layer.child())        

    else:
        raise NotImplementedError("Not implemented for 3D data.")                 

    ND = np.prod(parm.layer.child().shape) - np.sum(parm.layer.child() == 0)
    # Don't Laplace rule of succession
    parm.val = np.max([-np.log(((ND + 1) / (float(P))) - 1), 0])


def lbda_OR_NAND(parm, K):
    """
    Set lambda in OR-AND machine to its MLE
    TODO: make for general arity
    """

    if K == 2:
        P = count_correct_predictions_OR_NAND_2D(
            *[x.val for x in parm.layer.factors], parm.layer.child())

    else:
        raise NotImplementedError("Not implemented for 3D data.")

    ND = np.prod(parm.layer.child().shape) - np.sum(parm.layer.child() == 0)

    # Laplace rule of succession
    parm.val = -np.log(((ND + 2) / (float(P) + 1)) - 1)

@jit(parallel=True, nogil=True)
def predict_single_latent(z, u):
    """
    compute output matrix for a single latent dimension (deterministic).
    is equivalent to the product between to binary vectors.
    Returns in [0,1] mapping!
    """
    N = z.shape[0]
    D = u.shape[0]
    x = np.zeros([N, D], dtype=np.int8)

    for n in prange(N):
        for d in range(D):
            if (u[d] == 1) and (z[n] == 1):
                x[n,d] = 1
    return x


def lbda_MAX_AND(parm, K):
    """
    TODO: numba
    """

    if K != 2:
        raise NotImplementedError('Model not supported, yet.')

    z = parm.layer.factors[0]
    u = parm.layer.factors[1]
    x = parm.layer.child
    N, L = z().shape
    D = u().shape[0]

    mask = np.zeros([N, D], dtype=bool)
    l_list = range(L)

    predictions = [predict_single_latent(
                        z()[:, l], u()[:, l]) == 1 for l in l_list]

    TP = [np.count_nonzero(x()[predictions[l]] == 1) for l in range(L)]
    FP = [np.count_nonzero(x()[predictions[l]] == -1) for l in range(L)]

    for iter_index in range(L):

        # use Laplace rule of succession here, to avoid numerical issues
        l_pp_rate = [(tp + 1) / float(tp + fp + 2) for tp, fp in zip(TP, FP)]

        # find l with max predictive power
        l_max_idx = np.argmax(l_pp_rate)
        l_max = l_list[l_max_idx]

        # assign corresponding alpha
        parm()[l_max] = l_pp_rate[l_max_idx]

        # remove the dimenson from l_list
        l_list = [l_list[i] for i in range(len(l_list)) if i != l_max_idx]

        # the following large binary arrays need to be computed L times -> precompute here
        temp_array = predictions[l_max] & ~mask
        temp_array1 = temp_array & (x() == 1)
        temp_array2 = temp_array & (x() == -1)

        TP = [TP[l + (l >= l_max_idx)] - np.count_nonzero(temp_array1 & predictions[l_list[l]])
              for l in range(len(l_list))]
        FP = [FP[l + (l >= l_max_idx)] - np.count_nonzero(temp_array2 & predictions[l_list[l]])
              for l in range(len(l_list))]

        mask += predictions[l_max] == 1

    assert len(l_list) == 0

    P_remain = np.count_nonzero(x()[~mask] == 1)
    N_remain = np.count_nonzero(x()[~mask] == -1)

    p_new = (P_remain + 1 ) / float(P_remain + N_remain + 2 )

    parm()[-1] = p_new

    # check that clamped lambda/alpha is the smallest
    if parm()[-1] != np.min(parm()):
        # print('\nClamped lambda too large. '+
        #       'Ok during burn-in, should not happen during sampling!\n')
        parm()[-1] = np.min(parm())

    # after updating lambda, ratios need to be precomputed
    # should be done in a lazy fashion
    compute_lbda_ratios(parm.layer)        


def compute_lbda_ratios(layer):
    """
    TODO: speedup (cythonise and parallelise)
    precompute matrix of size [2,L+1,L+1],
    with log(lbda/lbda') / log( (1-lbda) / (1-lbda') )
    as needed for maxmachine gibbs updates.
    """

    L = layer.size + 1

    lbda_ratios = np.zeros([2, L, L], dtype=np.float32)

    for l1 in range(L):
        for l2 in range(l1 + 1):
            lratio_p = np.log(layer.lbda()[l1] / layer.lbda()[l2])
            lratio_m = np.log((1 - layer.lbda()[l1]) / (1 - layer.lbda()[l2]))
            lbda_ratios[0, l1, l2] = lratio_p
            lbda_ratios[0, l2, l1] = -lratio_p
            lbda_ratios[1, l1, l2] = lratio_m
            lbda_ratios[1, l2, l1] = -lratio_m

    layer.lbda_ratios = lbda_ratios    


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
