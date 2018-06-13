#!/usr/bin/env python
"""
Output functions for logical operator machine products
"""

import numpy as np
import numba
from numba import jit, prange  # int8, int16, int32, float32, float64


# Non-fuzzy output functions mapping from vectors with element in {-1,1}
# to single data point, also in {-1, 1}

# number of actives sources
@jit('void(int8[:,:], int8[:,:], int8[:,:])', nopython=True, nogil=True)
def compute_g(Z, U, g):
    N, L = Z.shape
    D, _ = U.shape
    for n in prange(N):
        for d in prange(D):
            for l in range(L):
                if Z[n, l] == U[d, l] == 1:
                    g[n, d] += 1


# qL-AND
@jit('int8(int8[:], int8[:], int8)', nopython=True, nogil=True)
def qL_AND_product(u, z, q):
    counter = np.int8(0)
    for l in range(u.shape[0]):
        if u[l] == 1 and z[l] == 1:
            counter += 1
        if counter == q:
            return 1
    else:
        return -1

# OR-AND
@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def OR_AND_product(u, z):
    for l in range(u.shape[0]):
        if u[l] == 1 and z[l] == 1:
            return 1
    return -1


@jit('int8(int8[:], int8[:], int8[:])', nogil=True, nopython=True)
def OR_AND_product_3d(Z_n, U_d, V_m):
    for l in range(Z_n.shape[0]):
        if Z_n[l] == 1 and U_d[l] == 1 and V_m[l] == 1:
            return 1
    return -1


@jit('int8[:,:](int8[:], int8[:])', nopython=False, nogil=True, parallel=True)
def OR_AND_product_expand(z, u):
    """
    Generate 2D matrix from single latent dimension
    """
    N = z.shape[0]
    D = u.shape[0]
    X = np.zeros([N, D], dtype=np.int8)

    for n in prange(N):
        for d in prange(D):
            if u[d] == 1 and z[n] == 1:
                X[n, d] = 1
    return X


# OR-NAND
@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def OR_NAND_product(u, z):
    for l in range(u.shape[0]):
        if (u[l] == -1) or (z[l] == -1):
            return 1
    return -1


@jit('int8(int8[:], int8[:], int8[:])', nogil=True, nopython=True)
def OR_NAND_product_3d(u, z, v):
    for l in range(u.shape[0]):
        if (u[l] == -1) or (z[l] == -1) or (v[l] == -1):
            return 1
    return -1


# OR-XOR
@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def OR_XOR_product(u, z):
    for l in range(u.shape[0]):
        if u[l] != z[l]:
            return 1
    return -1


@jit('int8(int8[:], int8[:], int8[:])', nogil=True, nopython=True)
def OR_XOR_product_3d(u, z, v):
    for l in range(u.shape[0]):
        if u[l] + z[l] + v[l] == -1:  # = -K+2
            return 1
    return -1


# NAND-XOR
@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def NAND_XOR_product(u, z):  # = OR-NXOR
    for l in range(u.shape[0]):
        if u[l] == z[l]:
            return 1
    return -1


@jit('int8(int8[:], int8[:], int8[:])', nopython=True, nogil=True)
def NAND_XOR_product_3d(u, z, v):
    for l in range(u.shape[0]):
        if u[l] + z[l] + v[l] != -1:
            return 1
    return -1


# XOR-AND
@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def XOR_AND_product(u, z):
    xor_count = np.int8(0)
    for l in range(u.shape[0]):
        if u[l] == 1 and z[l] == 1:
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1


@jit('int8(int8[:], int8[:], int8[:])', nopython=True, nogil=True)
def XOR_AND_product_3d(u, z, v):
    xor_count = np.int8(0)
    for l in range(u.shape[0]):
        if u[l] == 1 and z[l] == 1 and v[l] == 1:
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1


# XOR-XOR
@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def XOR_XOR_product(u, z):
    xor_count = np.int8(0)
    for l in range(u.shape[0]):
        if u[l] != z[l]:
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1


@jit('int8(int8[:], int8[:], int8[:])', nopython=True, nogil=True)
def XOR_XOR_product_3d(u, z, v):
    xor_count = np.int8(0)
    for l in range(u.shape[0]):
        if u[l] + z[l] + v[l] == -1:
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1


# XOR-NXOR
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


@jit('int8(int8[:], int8[:], int8[:])', nopython=True, nogil=True)
def XOR_NXOR_product_3d(u, z, v):
    xor_count = np.int8(0)
    for l in range(u.shape[0]):
        if u[l] + z[l] + v[l] != -1:
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1


# XOR-NAND
@jit('int8(int8[:], int8[:])', nopython=True, nogil=True)
def XOR_NAND_product(u, z):
    xor_count = np.int8(0)
    for l in range(u.shape[0]):
        if (u[l] != 1) or (z[l] != 1):
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1


@jit('int8(int8[:], int8[:], int8[:])', nopython=True, nogil=True)
def XOR_NAND_product_3d(u, z, v):
    xor_count = np.int8(0)
    for l in range(u.shape[0]):
        if (u[l] != 1) or (z[l] != 1) or (v[l] != 1):
            xor_count += 1
        if xor_count > 1:
            return -1
    if xor_count == 1:
        return 1
    else:
        return -1


# MAX-AND
# TODO: jit it
def MAX_AND_product_2d(factors, lbdas):

    out = np.zeros([f.shape[0] for f in factors])
    for l_idx in np.argsort(lbdas[:-1]):
        temp = lbdas[l_idx] * OR_AND_product_expand(
            *[f[:, l_idx] for f in factors])
        out[out == 0] = temp[out == 0]

    return out
