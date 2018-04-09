#!/usr/bin/env python
"""
Numba sampling routines
"""

import numpy as np
import numba
import math
from numba import jit, int8, int16, int32, float32, float64, prange
import lom._numba.lom_outputs as lom_outputs
import lom._numba.lom_outputs_fuzzy as lom_outputs_fuzzy


def get_scalar_output_function_2d(model, fuzzy=False):
    if fuzzy is False:
        if model == 'OR-AND':
            return lom_outputs.OR_AND_product
        if model == 'XOR-AND':
            return lom_outputs.XOR_AND_product
        if model == 'XOR-NAND':
            return lom_outputs.XOR_NAND_product
        if model == 'OR-XOR':
            return lom_outputs.OR_XOR_product
        if model == 'NAND-XOR':
            return lom_outputs.NAND_XOR_product
        if model == 'XOR-XOR':
            return lom_outputs.XOR_XOR_product
        if model == 'XOR-NXOR':
            return lom_outputs.XOR_NXOR_product
        if model == 'OR-NAND':
            return lom_outputs.OR_NAND_product
    else:
        if model == 'OR-AND':
            return lom_outputs_fuzzy.OR_AND_product_fuzzy
        if model == 'XOR-AND':
            return lom_outputs_fuzzy.XOR_AND_product_fuzzy
        if model == 'XOR-NAND':
            return lom_outputs_fuzzy.XOR_NAND_product_fuzzy
        if model == 'OR-XOR':
            return lom_outputs_fuzzy.OR_XOR_product_fuzzy
        if model == 'NAND-XOR':
            return lom_outputs_fuzzy.NAND_XOR_product_fuzzy
        if model == 'XOR-XOR':
            return lom_outputs_fuzzy.XOR_XOR_product_fuzzy
        if model == 'XOR-NXOR':
            return lom_outputs_fuzzy.XOR_NXOR_product_fuzzy
        if model == 'OR-NAND':
            return lom_outputs_fuzzy.OR_NAND_product_fuzzy


def get_scalar_output_function_3d(model, fuzzy=False):
    if fuzzy is False:
        if model == 'OR-AND':
            return lom_outputs.OR_AND_product_3d
        if model == 'XOR-AND':
            return lom_outputs.XOR_AND_product_3d
        if model == 'XOR-NAND':
            return lom_outputs.XOR_NAND_product_3d
        if model == 'OR-XOR':
            return lom_outputs.OR_XOR_product_3d
        if model == 'NAND-XOR':
            return lom_outputs.NAND_XOR_product_3d
        if model == 'XOR-XOR':
            return lom_outputs.XOR_XOR_product_3d
        if model == 'XOR-NXOR':
            return lom_outputs.XOR_NXOR_product_3d
        if model == 'OR-NAND':
            return lom_outputs.OR_NAND_product_3d
    else:
        if model == 'OR-AND':
            return lom_outputs_fuzzy.OR_AND_product_fuzzy_3d
        if model == 'XOR-AND':
            return lom_outputs_fuzzy.XOR_AND_product_fuzzy_3d
        if model == 'XOR-NAND':
            return lom_outputs_fuzzy.XOR_NAND_product_fuzzy_3d
        if model == 'OR-XOR':
            return lom_outputs_fuzzy.OR_XOR_product_fuzzy_3d
        if model == 'NAND-XOR':
            return lom_outputs_fuzzy.NAND_XOR_product_fuzzy_3d
        if model == 'XOR-XOR':
            return lom_outputs_fuzzy.XOR_XOR_product_fuzzy_3d
        if model == 'XOR-NXOR':
            return lom_outputs_fuzzy.XOR_NXOR_product_fuzzy_3d
        if model == 'OR-NAND':
            return lom_outputs_fuzzy.OR_NAND_product_fuzzy_3d


def make_output_function_2d(model):

    get_scalar_output_2d = get_scalar_output_function_2d(model, fuzzy=False)

    @jit('int8[:,:](int8[:,:], int8[:,:])',
         nogil=True, nopython=False, parallel=True)
    def output_function_2d(Z, U):
        N = Z.shape[0]
        D = U.shape[0]
        X = np.zeros([N, D], dtype=np.int8)
        for n in prange(N):
            for d in prange(D):
                X[n, d] = get_scalar_output_2d(Z[n, :], U[d, :])
        return X

    return output_function_2d


def make_output_function_3d(model):

    get_scalar_output_3d = get_scalar_output_function_3d(model, fuzzy=False)

    @jit('int8[:,:](int8[:,:], int8[:,:], int8[:,:])',
         nogil=True, nopython=False, parallel=True)
    def output_function_3d(Z, U, V):
        N = Z.shape[0]
        D = U.shape[0]
        M = V.shape[0]
        X = np.zeros([N, D, M], dtype=np.int8)
        for n in prange(N):
            for d in prange(D):
                for m in range(M):
                    X[n, d, m] = get_scalar_output_3d(Z[n, :], U[d, :], V[m, :])
        return X

    return output_function_3d


def make_output_function_2d_fuzzy(model):
    get_scalar_output_2d = get_scalar_output_function_2d(model, fuzzy=True)

    @jit('float64[:,:](float64[:,:], float64[:,:])',
         nogil=True, nopython=False, parallel=True)
    def output_function_2d(Z, U):
        N = Z.shape[0]
        D = U.shape[0]
        X = np.zeros([N, D], dtype=np.float64)
        for n in prange(N):
            for d in prange(D):
                X[n, d] = get_scalar_output_2d(Z[n, :], U[d, :])
        return X

    return output_function_2d


def make_output_function_3d_fuzzy(model):
    get_scalar_output_3d = get_scalar_output_function_3d(model, fuzzy=True)

    @jit('float64[:,:](float64[:,:], float64[:,:], float64[:,:])',
         nogil=True, nopython=False, parallel=True)
    def output_function_3d(Z, U, V):
        N = Z.shape[0]
        D = U.shape[0]
        M = V.shape[0]
        X = np.zeros([N, D, M], dtype=np.float64)
        for n in prange(N):
            for d in prange(D):
                for m in range(M):
                    X[n, d, m] = get_scalar_output_3d(Z[n, :], U[d, :], V[m, :])
        return X

    return output_function_3d


def make_correct_predictions_counter(model, dimensionality):
    """
    Generates function that counts the number of deterministically correct
    predictions with signature fct(factor0, factor1, ..., data)
    """

    if dimensionality == 2:

        output_fct = get_scalar_output_function_2d(model, fuzzy=False)

        @jit('int32(int8[:,:], int8[:,:], int8[:,:])',
             nogil=True, nopython=True, parallel=True)
        def correct_predictions_counter(Z, U, X):
            N, D = X.shape
            count = 0
            for n in prange(N):
                for d in prange(D):
                    if output_fct(Z[n, :], U[d, :]) == X[n, d]:
                        count += 1
            return count

    elif dimensionality == 3:

        output_fct = get_scalar_output_function_3d(model, fuzzy=False)

        @jit('int32(int8[:,:], int8[:,:], int8[:,:], int8[:,:,:])',
             nogil=True, nopython=True, parallel=True)
        def correct_predictions_counter(Z, U, V, X):
            N, D, M = X.shape
            count = np.int32(0)
            for n in prange(N):
                for d in prange(D):
                    for m in range(M):
                        if output_fct(Z[n, :], U[d, :], V[m, :]) == X[n, d, m]:
                            count += 1
            return count

    else:
        raise NotImplementedError(
            'Count correct predictions for dimensinalty > 3')

    return correct_predictions_counter


def make_lbda_update_fct(model, dimensionality):
    """
    Set lambda in OR-AND machine to its MLE
    TODO: make for general arity
    """

    if model == 'MAX-AND':
        return lbda_MAX_AND

    else:
        counter = make_correct_predictions_counter(model, dimensionality)

        def lbda_update_fct(parm):
            alpha, beta = parm.beta_prior

            # correct predictions, counting 0 prediction as false
            P = counter(*[x.val for x in parm.layer.factors], parm.layer.child())
            # number of data points that are to be predicted
            ND = np.prod(parm.layer.child().shape) - np.sum(parm.layer.child() == 0)
            parm.val = np.max([-np.log(((ND + alpha + beta) / (float(P) + alpha)) - 1), 0])
        return lbda_update_fct
