#!/usr/bin/env python
"""
Numba sampling routines
"""
import numpy as np
import math
from numba import jit, prange
# import lom._cython.matrix_updates as cython_mu
import lom._numba.lom_outputs as lom_outputs
import lom._numba.posterior_score_fcts as score_fcts

# only needed for IBP
from lom.auxiliary_functions import logit, expit
from scipy.special import gammaln
from math import lgamma


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


@jit('int8(float64)', nopython=True, nogil=True)
def flip_gibbs_numba(p):
    """
    Given the probability of z=1
    flip z according to standard Gibbs sampler
    """
    if p > np.random.rand():
        return 1
    else:
        return -1


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
    elif model == 'OR_AND_dropout_2D':
        posterior_score_fct = score_fcts.posterior_scores_OR_AND_2D_dropout
    elif model == 'OR_AND_dropout_3D':
        posterior_score_fct = score_fcts.posterior_scores_OR_AND_3D_dropout
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
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:],' +
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


def make_sampling_fct_onechild_dropout(model):

    posterior_score_fct = get_posterior_score_fct(model)

    if model[-2:] == '2D':
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], float64, float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed, U, X, lbda, dropout_factor, logit_prior):
            N, L = Z.shape
            for n in prange(N):
                for l in range(L):
                    if Z_fixed[n, l] == 1:
                        continue
                    pos_score, neg_score = posterior_score_fct(Z[n, :], U, X[n, :], l)
                    logit_score = lbda * pos_score + dropout_factor * neg_score

                    Z[n, l] = flip_metropolised_gibbs_numba(
                        logit_score + logit_prior, Z[n, l])

    elif model[-2:] == '3D':
        @jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], int8[:,:,:], float64, float64, float64)',
             nogil=True, nopython=True, parallel=True)
        def sampling_fct(Z, Z_fixed, U, V, X, lbda, dropout_factor, logit_prior):
            N, L = Z.shape
            for n in prange(N):
                for l in range(L):
                    if Z_fixed[n, l] == 1:
                        continue
                    pos_score, neg_score = posterior_score_fct(
                        Z[n, :], U, V, X[n, :, :], l)
                    logit_score = lbda * pos_score + dropout_factor * neg_score
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
                for d in range(D):
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
                            logit_parent_score_1 +
                            logit_parent_score_2 +
                            logit_prior,
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
                        posterior_score_fct(Z[n, :], U, X[n, :], l)  # V?

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


def sample_2d_IBP(Z, U, X, lbda, q, alpha):
    """
    IBP update procedure for 2D OrMachine, drawing U and Z where
    U has flat prior and Z comes from IBP with concentration parameter alpha.
    Z[n,l], U[d,l], X[n,d], q: Bernoulli prior, alpha: IPB concentration parameter.
    """

    # lbda = lr.lbda.val
    # alpha = lr.alpha
    # X = lr.child()

    L_new_max = 3  # maxmimum number of new dishes to consider
    N, L = Z.shape  #
    D, _ = U.shape
    posterior_score_fct = get_posterior_score_fct('OR_AND_2D')

    # pre-compute scores for updating L
    # these are loglikelihood contributions of false negative/true negative
    # data points for a range of L'

    # For simple Bernoulli on U
    FN_factor = [np.log((expit(lbda) * (1 - 2 * (q**L_temp))) + (q**L_temp))
                 for L_temp in range(L_new_max)]
    TN_factor = [np.log((expit(-lbda) * (1 - 2 * (q**L_temp))) + (q**L_temp))
                 for L_temp in range(L_new_max)]

    for n in range(N):

        # how often is each dish ordered by other customers
        m = (Z[np.arange(N) != n, :] == 1).sum(axis=0)
        columns_to_keep = np.ones(L, dtype=bool)

        for l in range(L):

            # dishes that have already been ordered
            if m[l] > 0:
                # draw z[n,l] as usual
                logit_score = lbda * posterior_score_fct(
                    Z[n, :], U, X[n, :], l)
                logit_prior = logit(m[l] / N)
                Z[n, l] = flip_gibbs_numba(expit(logit_score +
                                                 logit_prior))
                # print(U[n, l])
                # print('score: ' + str(logit_score))
                # print('\n')
                # print('prior: ' + str(logit_prior))

            elif m[l] == 0:
                # mark columns for removal
                columns_to_keep[l] = False

        # remove marked columns
        Z = Z[:, columns_to_keep]
        U = U[:, columns_to_keep]
        L = columns_to_keep.sum()

        # draw number of new dishes (columns)
        # compute log probability of L' for a range of L' values
        # n_predict = [lom_outputs.OR_AND_product(Z[n, :], U[d, :]) for d in range(D)]
        # faster
        n_predict = lom_outputs.OR_AND_single_n(Z[n, :], U)
        # assert(np.all(n_predict_test==n_predict))

        # compute number of true negatives / false negatives
        TN = ((X[n, :] == -1) * (np.array(n_predict) == -1)).sum()
        FN = ((X[n, :] == 1) * (np.array(n_predict) == -1)).sum()

        lik_L_new = [TN * TN_factor[L_temp] + FN * FN_factor[L_temp]
                     for L_temp in range(L_new_max)]
        # L_new or L+L_new ??!
        prior_L_new = [(L_temp + L) * np.log(alpha / N) - (alpha / N) -
                       gammaln(L + L_temp + 1)
                       for L_temp in range(L_new_max)]
        log_L_new = [loglik + logprior
                     for loglik, logprior in zip(lik_L_new, prior_L_new)]
        # map to probabilities
        p_L_new = [np.exp(log_L_new[i] - np.max(log_L_new))
                   for i in range(L_new_max)]
        p_L_new /= np.sum(p_L_new)

        L_new = np.random.choice(range(L_new_max), p=p_L_new)

        if L_new > 0:

            # add new columns to Z
            Z = np.hstack([Z, np.full([N, L_new], fill_value=-1, dtype=np.int8)])
            Z[n, -L_new:] = 1
            U = np.hstack([U, 2 * np.zeros([D, L_new], dtype=np.int8) - 1])

            # sample the new hidden causes
            for l in list(range(L, L + L_new)):
                for d in range(D):
                    logit_score = lbda * posterior_score_fct(
                        U[d, :], Z, X[:, d], l)
                    U[d, l] = flip_gibbs_numba(expit(logit_score))

        L += L_new

        # if L_new > 0:
        #     print(L_new, Z.shape[0])

    return Z, U

# TODO: numba


# @jit(parallel=True)
# def sample_qL_q(Z, U, X, q, lbda, gamma):

#     N, D = X.shape

#     # g = np.zeros([N, D], dtype=np.int8)
#     # lom_outputs.compute_g(Z, U, g)
#     log_expit_plus_lbda = np.log(1 + np.exp(lbda))
#     log_expit_minus_lbda = np.log(1 + np.exp(-lbda))
#     q_max = 10

#     for d in prange(D):
#         q[0, d] = draw_q_d(X[:, d], Z, U[d, :],
#                            log_expit_plus_lbda,
#                            log_expit_minus_lbda,
#                            gamma,
#                            q_max)

#     return


@jit('int8(float64[:])', nopython=True, nogil=True)
def random_choice(p):
    """
    Return random index according to probabilities p.
    """

    rand_float = np.random.ranf()
    acc = 0
    for i in range(len(p)):
        acc += p[i]
        if rand_float < acc:
            return np.int8(i + 1)
    return len(p) + 1


# @jit('int8(int8[:], int8[:,:], int8[:], float64, float64, float64, int8)',
#      nopython=True, nogil=True)
# def draw_q_d(X_d, Z, U_d, p_lbda, m_lbda, gamma, q_max):

@jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], float64, float64)', nopython=True, nogil=True)
def sample_qL_q(Z, U, X, q, lbda, gamma):

    N, D = X.shape
    q_max = 6
    p_lbda = np.log(1 + np.exp(lbda))
    m_lbda = np.log(1 + np.exp(-lbda))

    logconditional = np.zeros((q_max - 1, D), dtype=np.float64)

    for d in prange(D):

        for q_new in prange(q_max - 1):  # shift by one to have q values and indices agree

            # compute logpriors
            logconditional[q_new, d] = (q_new + 1) * np.log(gamma) -\
                gamma - lgamma(q_new + 2)

            # compute loglikelihoods
            true_predictions = np.sum(np.dot((Z + 1) / 2, (U[d, :] + 1) / 2) >= q_new + 1)
            logconditional[q_new, d] = logconditional[q_new, d] + true_predictions * p_lbda +\
                (N - true_predictions) * m_lbda

        # overwrite log-conditional with normalised probability
        # (but avoid allocating new memory)
        log_p_max = np.max(logconditional)
        for q_new in prange(q_max - 1):
            logconditional[q_new, d] = np.exp(logconditional[q_new, d] - log_p_max)

        logconditional[:, d] /= np.sum(logconditional[:, d])  # normalise

        # import pdb; pdb.set_trace()
        q[0, d] = random_choice(logconditional[:, d])
        # q[0, d] = np.argmax(logconditional[:, d]) + 1


@jit('int16(int8[:], int8[:,:], int8[:], int16, int8[:,:])', nopython=True, nogil=True)
def posterior_score_fct_qL_Z(Z_n, U, X_n, l, q):
    """
    # TODO: numba
    """
    D, L = U.shape

    score = 0
    # We need q-1 sources active for Z_n to have an effect
    for d in range(D):
        if U[d, l] != 1:
            continue

        counter = 0  # count active sources
        # alrdy_active = False  # temp line
        for l_prime in range(L):
            if (Z_n[l_prime] == 1) and\
               (U[d, l_prime] == 1) and\
               (l_prime != l):
                # alrdy_active = True  # temp line
                counter += 1

            # no contribution of we have alrdy q source
            if counter == q[0, d]:
                break
        # if alrdy_active is False:
        #     score += X_n[d]

        if counter == q[0, d] - 1:
            score += X_n[d]

    return score


@jit('int16(int8[:], int8[:,:], int8[:], int16, int8)', nopython=True, nogil=True)
def posterior_score_fct_qL_U(U_d, Z, X_d, l, q_d):
    N, L = Z.shape

    score = 0
    # We need q-1 sources active for Z_n to have an effect
    for n in range(N):
        if Z[n, l] != 1:
            continue

        counter = 0  # count active sources
        for l_prime in range(L):
            if (U_d[l_prime] == 1) and\
               (Z[n, l_prime] == 1) and\
               (l_prime != l):
                counter += 1

            # no contribution of we have alrdy q sources
            if counter == q_d:
                break

        if counter == q_d - 1:
            score += X_d[n]

    return score


@jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], float64)', nopython=True, nogil=True)
def sample_qL_factors_Z(Z, U, X, q, lbda):
    """
    Need separate functions for U and Z because we have different q's
    for every Z[n,:], but the same q for every U[d,:]
    # TODO: numba
    """
    N, L = Z.shape
    for n in prange(N):
        for l in range(L):
            logit_score = posterior_score_fct_qL_Z(
                Z[n, :], U, X[n, :], l, q)
            Z[n, l] = flip_metropolised_gibbs_numba(
                logit_score, Z[n, l])
    return


@jit('void(int8[:,:], int8[:,:], int8[:,:], int8[:,:], float64)', nopython=True, nogil=True)
def sample_qL_factors_U(U, Z, X, q, lbda):
    """
    Need separate functions for U and Z because we have different q's
    for every Z[n,:], but the same q for every U[d,:]
    # TODO: numba
    """

    D, L = U.shape
    for d in prange(D):
        for l in range(L):
            logit_score = posterior_score_fct_qL_U(
                U[d, :], Z, X[d, :], l, q[0, d])
            U[d, l] = flip_metropolised_gibbs_numba(
                logit_score, U[d, l])
    return
