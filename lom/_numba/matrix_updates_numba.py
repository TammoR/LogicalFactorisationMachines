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
import lom._numba.lom_outputs as lom_outputs
import lom._numba.posterior_score_fcts as score_fcts


@jit('int8(float64, int8)', nopython=True, nogil=True)
def flip_metropolised_gibbs_numba(logit_p, z):
    """
    Given the logit probability of z=1
    flip z according to metropolised Gibbs
    """
    if z == 1 and logit_p <= 0:
        return -1

    elif z == -1 and logit_p >= 0:
        return 1

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


def get_posterior_score_fct(model):

    if model == 'OR_AND_2D':
        posterior_score_fct = score_fcts.posterior_score_OR_AND_2D
    elif model == 'OR_NAND_2D':
        posterior_score_fct = score_fcts.posterior_score_OR_NAND_2D
    elif model == 'OR_XOR_2D':
        posterior_score_fct = score_fcts.posterior_score_OR_XOR_2D
    elif model == 'NAND_XOR_2D':
        posterior_score_fct = score_fcts.posterior_score_NAND_XOR_2D
    elif model == 'XOR_AND_2D':
        posterior_score_fct = score_fcts.posterior_score_XOR_AND_2D
    elif model == 'XOR_XOR_2D':
        posterior_score_fct = score_fcts.posterior_score_XOR_XOR_2D
    elif model == 'XOR_NXOR_2D':
        posterior_score_fct = score_fcts.posterior_score_XOR_NXOR_2D
    elif model == 'XOR_NAND_2D':
        posterior_score_fct = score_fcts.posterior_score_XOR_NAND_2D
    elif model == 'OR_AND_3D':
        posterior_score_fct = score_fcts.posterior_score_OR_AND_3D
    elif model == 'OR_NAND_3D':
        posterior_score_fct = score_fcts.posterior_score_OR_NAND_3D
    elif model == 'OR_XOR_3D':
        posterior_score_fct = score_fcts.posterior_score_OR_XOR_3D
    elif model == 'NAND_XOR_3D':
        posterior_score_fct = score_fcts.posterior_score_NAND_XOR_3D
    elif model == 'XOR_AND_3D':
        posterior_score_fct = score_fcts.posterior_score_XOR_AND_3D
    elif model == 'XOR_XOR_3D':
        posterior_score_fct = score_fcts.posterior_score_XOR_XOR_3D
    elif model == 'XOR_NXOR_3D':
        posterior_score_fct = score_fcts.posterior_score_XOR_NXOR_3D
    elif model == 'XOR_NAND_3D':
        posterior_score_fct = score_fcts.posterior_score_XOR_NAND_3D
    elif model == 'OR_ALL_2D':
        posterior_score_fct = score_fcts.posterior_score_OR_ALL_2D
    elif model == 'OR_ALL_3D':
        posterior_score_fct = score_fcts.posterior_score_OR_ALL_3D
    else:
        print(model)
        raise NotImplementedError('Posterior sampling for ' + model + '.')
    return posterior_score_fct


def get_parent_score_fct(model):

    if model == 'OR_AND_2D':
        return lom_outputs.OR_AND_product
    if model == 'OR_NAND_2D':
        return lom_outputs.OR_NAND_product
    if model == 'OR_XOR_2D':
        return lom_outputs.OR_XOR_product
    if model == 'NAND_XOR_2D':
        return lom_outputs.NAND_XOR_product
    if model == 'XOR_AND_2D':
        return lom_outputs.XOR_AND_product
    if model == 'XOR_XOR_2D':
        return lom_outputs.XOR_XOR_product
    if model == 'XOR_NXOR_2D':
        return lom_outputs.XOR_NXOR_product
    if model == 'XOR_NAND_2D':
        return lom_outputs.XOR_NAND_product
    if model == 'OR_AND_3D':
        return lom_outputs.OR_AND_product_3d
    if model == 'OR_NAND_3D':
        return lom_outputs.OR_NAND_product_3d
    if model == 'OR_XOR_3D':
        return lom_outputs.OR_XOR_product_3d
    if model == 'NAND_XOR_3D':
        return lom_outputs.NAND_XOR_product_3d
    if model == 'XOR_AND_3D':
        return lom_outputs.XOR_AND_product_3d
    if model == 'XOR_XOR_3D':
        return lom_outputs.XOR_XOR_product_3d
    if model == 'XOR_NXOR_3D':
        return lom_outputs.XOR_NXOR_product_3d
    if model == 'XOR_NAND_3D':
        return lom_outputs.XOR_NAND_product_3d
    else:
        print(model)
        raise NotImplementedError


def make_sampling_fct_onechild(model):

    posterior_score_fct = get_posterior_score_fct(model)

    if model[-2:] == '2D':
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:],'
             'float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed, U, X, lbda, logit_prior):
            N, L = Z.shape
            for n in prange(N):
                for l in range(L):
                    if Z_fixed[n, l] == 1:
                        continue
                    logit_score = lbda *\
                        posterior_score_fct(Z[n, :], U, X[n, :], l)
                    Z[n, l] = flip_metropolised_gibbs_numba(
                        logit_score + logit_prior, Z[n, l])

    elif model[-2:] == '3D':
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:],' +
             'int8[:,:,:], float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed, U, V, X, lbda, logit_prior):
            N, L = Z.shape
            for n in prange(N):
                for l in range(L):
                    if Z_fixed[n, l] == 1:
                        continue
                    logit_score = lbda *\
                        posterior_score_fct(Z[n, :], U, V, X[n, :, :], l)
                    Z[n, l] = flip_metropolised_gibbs_numba(
                        logit_score + logit_prior, Z[n, l])

    return sampling_fct


def make_sampling_fct_onechild_oneparent(model, parent_model):

    posterior_score_fct = get_posterior_score_fct(model)
    parent_posterior_score_fct = get_parent_score_fct(parent_model)

    if model[-2:] == '2D':
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], ' +
             'float64, int8[:,:], int8[:,:], float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed, U, X, lbda, pa1, pa2, lbda_pa, logit_prior):
            N, L = Z.shape
            for n in prange(N):
                for l in range(L):
                    if Z_fixed[n, l] == 1:
                        continue
                    logit_score = lbda *\
                        posterior_score_fct(Z[n, :], U, X[n, :], l)
                    logit_parent_score = lbda_pa *\
                        parent_posterior_score_fct(pa1[n, :], pa2[l, :])

                    Z[n, l] = flip_metropolised_gibbs_numba(
                        logit_score + logit_parent_score + logit_prior, Z[n, l])

    elif model[-2:] == '3D':
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], int8[:,:,:], ' +
             'float64, int8[:,:], int8[:,:], float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed, U, V, X, lbda, pa1, pa2, lbda_pa, logit_prior):
            N, L = Z.shape
            for n in prange(N):
                for l in range(L):
                    if Z_fixed[n, l] == 1:
                        continue
                    logit_score = lbda *\
                        posterior_score_fct(Z[n, :], U, V, X[n, :], l)
                    logit_parent_score = lbda_pa *\
                        parent_posterior_score_fct(pa1[n, :], pa2[l, :])

                    Z[n, l] = flip_metropolised_gibbs_numba(
                        logit_score + logit_parent_score + logit_prior), Z[n, l]

    return sampling_fct


def make_sampling_fct_nochild_oneparent(parent_model):
    """
    Generate update function for factor matrices without children.
    In the general case this is a sampling version of the factor product.
    """

    parent_posterior_score_fct = get_parent_score_fct(parent_model)

    if parent_model[-2:] == '2D':
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed, pa1, pa2, lbda_pa, logit_prior):
            N, L = Z.shape
            for n in prange(N):
                for l in range(L):
                    if Z_fixed[n, l] == 1:
                        continue
                    logit_parent_score = lbda_pa *\
                        parent_posterior_score_fct(pa1[n, :], pa2[l, :])

                    Z[n, l] = flip_metropolised_gibbs_numba(
                        logit_parent_score + logit_prior, Z[n, l])

    elif parent_model[-2:] == '3D':
        @jit('void(int8[:,:,:], int8[:,:,:], int8[:,:], int8[:,:], float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed, pa1, pa2, pa3, lbda_pa, logit_prior):
            N, D, M = Z.shape
            for n in prange(N):
                for d in range(L):
                    for m in range(M):
                        if Z_fixed[n, d, m] == 1:
                            continue
                        logit_parent_score = lbda_pa *\
                            parent_posterior_score_fct(
                                pa1[n, :], pa2[d, :], pa3[m, :])

                        Z[n, d, m] = flip_metropolised_gibbs_numba(
                            logit_parent_score + logit_prior, Z[n, d, m])

    return sampling_fct


def make_sampling_fct_nochild_twoparents(parent_model_1, parent_model_2):

    parent_posterior_score_fct = get_parent_score_fct(parent_model_1)
    parent_posterior_score_fct = get_parent_score_fct(parent_model_2)

    if parent_model_1[-2:] == '2D' and parent_model_2[-2:] == '2D':
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], ' +
             'float64, int8[:,:], int8[:,:], float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed,
                         pa1_1, pa1_2, lbda_pa1,
                         pa2_1, pa2_2, lbda_pa2,
                         logit_prior):
            N, L = Z.shape
            for n in prange(N):
                for l in range(L):
                    if Z_fixed[n, l] == 1:
                        continue

                    logit_parent_score_1 = lbda_pa1 *\
                        parent_posterior_score_fct(pa1_1[n, :], pa1_2[l, :])

                    logit_parent_score_2 = lbda_pa2 *\
                        parent_posterior_score_fct(pa2_1[n, :], pa2_2[l, :])

                    Z[n, l] = flip_metropolised_gibbs_numba(
                        logit_parent_score_1 + logit_parent_score_2 + logit_prior,
                        Z[n, l])

    elif parent_model_1[-2:] == '3D' and parent_model_2[-2:] == '3D':
        @jit('void(int8[:,:,:], int8[:,:,:], int8[:,:], int8[:,:], int8[:,:], ' +
             'float64, int8[:,:], int8[:,:], int8[:,:], float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed,
                         pa1_1, pa1_2, pa1_3, lbda_pa1,
                         pa2_1, pa2_2, pa2_3, lbda_pa2,
                         logit_prior):
            N, D, M = Z.shape
            for n in prange(N):
                for d in range(D):
                    for m in range(M):
                        if Z_fixed[n, d, m] == 1:
                            continue

                        logit_parent_score_1 = lbda_pa1 *\
                            parent_posterior_score_fct(
                                pa1_1[n, :], pa1_2[d, :], pa1_3[m, :])

                        logit_parent_score_2 = lbda_pa2 *\
                            parent_posterior_score_fct(
                                pa2_1[n, :], pa2_2[d, :], pa2_3[m, :])

                        Z[n, d, m] = flip_metropolised_gibbs_numba(
                            logit_parent_score_1 + logit_parent_score_2 + logit_prior,
                            Z[n, d, m])

    return sampling_fct


def make_sampling_fct_onechild_twoparents(model, parent_model_1, parent_model_2):

    posterior_score_fct = get_posterior_score_fct(model)
    parent_posterior_score_fct = get_parent_score_fct(parent_model_1)
    parent_posterior_score_fct = get_parent_score_fct(parent_model_2)

    if model[-2:] == '2D':
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], float64,' +
             'int8[:,:], int8[:,:], float64,' +
             'int8[:,:], int8[:,:], float64,' +
             'float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed, U, X, lbda,
                         pa1_1, pa1_2, lbda_pa1,
                         pa2_1, pa2_2, lbda_pa2,
                         logit_prior):
            N, L = Z.shape
            for n in prange(N):
                for l in range(L):
                    if Z_fixed[n, l] == 1:
                        continue

                    logit_score = lbda *\
                        posterior_score_fct(Z[n, :], U, V, X[n, :], l)

                    logit_parent_score_1 = lbda_pa1 *\
                        parent_posterior_score_fct(pa1_1[n, :], pa1_2[l, :])

                    logit_parent_score_2 = lbda_pa2 *\
                        parent_posterior_score_fct(pa2_1[n, :], pa2_2[l, :])

                    Z[n, l] = flip_metropolised_gibbs_numba(
                        logit_score + logit_parent_score_1 +
                        logit_parent_score_2 + logit_prior,
                        Z[n, l])

    elif model[-2:] == '3D':
        raise NotImplementedError('3D tensors can not have children.')

    return sampling_fct


# def IBP_update(Z, U, X, lbda):

#     return

#     # # remove features that are used at most once (why does np.where suck so much?)
#     # idx_to_keep = [i for i, val in enumerate(list((Z==1).sum(axis=0) <= 1)) if val == False]
#     # if len(idx_to_remove) > 0:
#     #     print('removed feature')
#     #     Z = Z[:, idx_to_keep]
#     #     U = U[:, idx_to_keep]

#     # from lom._numba.lambda_updates_numba import or_and_out_2D
#     # from scipy.special import expit
#     # X_predict = or_and_out_2D(Z, U) # make a 1D version and do inside loop

#     # N, L = Z.shape
#     # for n in range(N):  # can't be parallel, sampling depends on L
#     #     FN = np.sum( (X_predict[n, :] == 0) & (X[n, :] == 1) )
#     #     TN = np.sum( (X_predict[n, :] == 0) & (X[n, :] == 0) )
#     #     if sample_new_dimension(FN, TN, lbda, alpha, L):
#     #         L += 1
#     #         add_dimension_idx.append(n)

#     # for n, l in zip(add_dimension_idx, range(len(add_dimension_idx)):
#     #     Z[n,l] = 1

#     # for u in add_dimension_idx:
#     #     for l in range(L):
#     #         sample_U
