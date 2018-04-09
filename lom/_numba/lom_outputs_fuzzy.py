#!/usr/bin/env python
"""
Output functions for logical operator machine products
"""

import numpy as np
from numba import jit, prange  # int8, float64,

# fuzzy output functions mapping from scalar vectors of probabilities to
# to single data-point


# OR-AND
@jit('float64(float64[:], float64[:])', nogil=True, nopython=True)
def OR_AND_product_fuzzy(Z_n, U_d):
    """
    Compute probability of emitting a zero for fuzzy vectors under OR-AND logic.
    """
    out = 1
    for l in range(Z_n.shape[0]):
        out *= 1 - Z_n[l] * U_d[l]
    return 1 - out  # map to [-1,1], this is a shortcut here. it's correct.


@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def OR_AND_product_fuzzy_3d(Z_n, U_d, V_m):
    """
    Compute probability of emitting a zero for fuzzy vectors under OR-AND logic.
    """
    out = np.float64(1.0)
    for l in range(Z_n.shape[0]):
        out *= 1 - ( Z_n[l] * U_d[l] * V_m[l] )
    return 1 - out


# XOR-AND
@jit('float64(float64[:], float64[:])', nogil=True, nopython=True, parallel=False)
def XOR_AND_product_fuzzy(Z_n, U_d):
    out = np.float64(0.0)
    for l in prange(Z_n.shape[0]):
        temp = 1.0
        for l_prime in range(Z_n.shape[0]):
            if l != l_prime:
                temp *= 1 - Z_n[l_prime] * U_d[l_prime]
        out += Z_n[l] * U_d[l] * temp
    return out


@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def XOR_AND_product_fuzzy_3d(Z_n, U_d, V_m):
    out = np.float64(0.0)
    for l in prange(Z_n.shape[0]):
        temp = 1.0
        for l_prime in range(Z_n.shape[0]):
            if l != l_prime:
                temp *= 1 - Z_n[l_prime] * U_d[l_prime] * V_m[l_prime]
        out += Z_n[l] * U_d[l] * V_m[l] * temp
    return out


# XOR-NAND
@jit('float64(float64[:], float64[:])', nogil=True, nopython=True)
def XOR_NAND_product_fuzzy(Z_n, U_d):
    out = np.float64(0.0)
    for l in prange(Z_n.shape[0]):
        temp = 1.0
        for l_prime in range(Z_n.shape[0]):
            if l != l_prime:
                temp *= Z_n[l_prime] * U_d[l_prime]
        out += (1 - Z_n[l] * U_d[l]) * temp
    return out


@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def XOR_NAND_product_fuzzy_3d(Z_n, U_d, V_m):
    out = np.float64(0.0)
    for l in prange(Z_n.shape[0]):
        temp = 1.0
        for l_prime in range(Z_n.shape[0]):
            if l != l_prime:
                temp *= Z_n[l_prime] * U_d[l_prime] * V_m[l_prime]
        out += (1 - Z_n[l] * U_d[l] * V_m[l]) * temp
    return out


# OR-XOR
@jit('float64(float64[:], float64[:])', nogil=True, nopython=True)
def OR_XOR_product_fuzzy(Z_n, U_d):
    temp = np.float64(1)
    for l in range(Z_n.shape[0]):
        temp *= (Z_n[l] * U_d[l]) + (1 - Z_n[l]) * (1 - U_d[l])
    return 1 - temp


@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def OR_XOR_product_fuzzy_3d(Z_n, U_d, V_m):
    temp = np.float64(1)
    # this is hard to generalise to arbitrary D
    for l in range(Z_n.shape[0]):
        temp *= 1 - Z_n[l] * (1 - U_d[l]) * (1 - V_m[l]) +\
            U_d[l] * (1 - V_m[l]) * (1 - Z_n[l]) +\
            V_m[l] * (1 - Z_n[l]) * (1 - U_d[l])
    return 1 - temp


# NAND-XOR
@jit('float64(float64[:], float64[:])', nogil=True, nopython=True)
def NAND_XOR_product_fuzzy(Z_n, U_d):
    temp = np.float64(1)
    for l in range(Z_n.shape[0]):
        temp *= Z_n[l] * (1 - U_d[l]) + U_d[l] * (1 - Z_n[l])
    return 1 - temp


@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def NAND_XOR_product_fuzzy_3d(Z_n, U_d, V_m):
    temp = np.float64(1)
    for l in range(Z_n.shape[0]):
        temp *= Z_n[l] * (1 - U_d[l]) * (1 - V_m[l]) +\
                V_m[l] * (1 - Z_n[l]) * (1 - U_d[l]) +\
                U_d[l] * (1 - V_m[l]) * (1 - Z_n[l])

    return 1 - temp


# XOR_XOR
@jit('float64(float64[:], float64[:])', nogil=True, nopython=True)
def XOR_XOR_product_fuzzy(Z_n, U_d):
    out = np.float64(0.0)
    for l in prange(Z_n.shape[0]):
        temp = 1.0
        for l_prime in range(Z_n.shape[0]):
            if l != l_prime:
                temp *= Z_n[l_prime] * U_d[l_prime] +\
                    (1 - Z_n[l_prime]) * (1 - U_d[l_prime])
        out += temp * ((1 - Z_n[l]) * U_d[l] + (1 - U_d[l]) * Z_n[l])
    return out


@jit('float64(float64, float64, float64)', nogil=True, nopython=True)
def p_XOR_fuzzy_3d(z, u, v):
    """
    Compute XOR probability given p(x), p(u), p(z)
    """
    return  3 * z * u * v - 2 * (z * u + u * v + z * v) + z + u + v


@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def XOR_XOR_product_fuzzy_3d(Z_n, U_d, V_m):
    out = np.float64(0.0)
    for l in prange(Z_n.shape[0]):
        temp = 1.0
        for l_prime in range(Z_n.shape[0]):
            if l != l_prime:
                temp *= 1 - p_XOR_fuzzy_3d(
                    Z_n[l_prime], U_d[l_prime], V_m[l_prime])
        out += temp * p_XOR_fuzzy_3d(Z_n[l], U_d[l], V_m[l])
    return out


# XOR_NXOR
@jit('float64(float64[:], float64[:])', nogil=True, nopython=True)
def XOR_NXOR_product_fuzzy(Z_n, U_d):
    out = np.float64(0.0)
    for l in prange(Z_n.shape[0]):
        temp = 1.0
        for l_prime in range(Z_n.shape[0]):
            if l != l_prime:
                temp *= Z_n[l_prime] * (1 - U_d[l_prime]) +\
                    (1 - Z_n[l_prime]) * U_d[l_prime]
        out += temp * ((Z_n[l] * U_d[l]) + (1 - U_d[l]) * (1 - Z_n[l]))
    return out


@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def XOR_NXOR_product_fuzzy_3d(Z_n, U_d, V_m):
    out = np.float64(0.0)
    for l in prange(Z_n.shape[0]):
        temp = 1.0
        for l_prime in range(Z_n.shape[0]):
            if l != l_prime:
                temp *= p_XOR_fuzzy_3d(
                    Z_n[l_prime], U_d[l_prime], V_m[l_prime])
        out += temp * (1 - p_XOR_fuzzy_3d(Z_n[l], U_d[l], V_m[l]))
    return out


# OR_NAND
@jit('float64(float64[:], float64[:])', nogil=True, nopython=True)
def OR_NAND_product_fuzzy(Z_n, U_d):
    temp = np.float64(1)
    for l in range(Z_n.shape[0]):
        temp *= Z_n[l] * U_d[l]
    return 1 - temp


@jit('float64(float64[:], float64[:], float64[:])', nogil=True, nopython=True)
def OR_NAND_product_fuzzy_3d(Z_n, U_d, V_m):
    temp = np.float64(1)
    for l in range(Z_n.shape[0]):
        temp *= Z_n[l] * U_d[l] * V_m[l]
    return 1 - temp


# MAX_AND
@jit('float64[:, :](float64[:, :], float64[:, :], float64[:])',
     nogil=True, nopython=False, parallel=True)
def MAX_AND_product_fuzzy(Z, U, lbdas):
    N = Z.shape[0]
    D = U.shape[0]
    L = Z.shape[1]
    out = np.zeros([N, D])  # , dtype=np.float)
    for n in prange(N):
        for d in range(D):
            acc = 0  # accumulator for sum
            for l1 in range(L):
                temp1 = Z[n, l1] * U[d, l1] * lbdas[l1]
                # check for explaining away
                prod = 1
                for l2 in range(L):
                    if l1 == l2:
                        continue
                    temp2 = Z[n, l2] * U[d, l2]
                    if temp2 * lbdas[l2] > temp1:
                        prod *= 1 - temp2
                acc += temp1 * prod
            out[n, d] = acc
    return out    

