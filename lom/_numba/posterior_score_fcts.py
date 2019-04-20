#!/usr/bin/env python
"""
Posterior score functions for logical operator machines
"""

import numpy as np
from numba import jit
from numba.types import int64, int16


#OR-AND dropout
@jit('UniTuple(int16, 2)(int8[:], int8[:,:], int8[:], int16)', nopython=True, nogil=True)
def posterior_scores_OR_AND_2D_dropout(Z_n, U, X_n, l):
    """
    Return count of correct/incorrect explanations
    of 0/1 separately as caused by setting Z[n,l] to 1, respecting
    explaining away dependencies
    """
    D, L = U.shape
    pos_score = int16(0)
    neg_score = int16(0)

    for d in range(D):
        if U[d, l] != 1:  # AND
            continue

        alrdy_active = False
        for l_prime in range(L):
            if (Z_n[l_prime] == 1) and \
                    (U[d, l_prime] == 1) and \
                    (l_prime != l):
                alrdy_active = True  # OR
                break

        if alrdy_active is False:
            if X_n[d] == -1:
                neg_score += 1
            elif X_n[d] == 1:
                pos_score += 1

    return pos_score, neg_score


@jit('UniTuple(int64, 2)(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)',
     nopython=True, nogil=True)
def posterior_score_OR_AND_3D_dropout(Z_n, U, V, X_n, l):
    """
    Return count of correct/incorrect explanations
    of 0/1 separately as caused by setting Z[n,l] to 1, respecting
    explaining away dependencies
    """
    D, L = U.shape
    M, _ = V.shape
    pos_score = int16(0)
    neg_score = int16(0)

    for d in range(D):
        for m in range(M):
            if (U[d, l] != 1) or (V[m, l] != 1):  # AND
                continue

            alrdy_active = False
            for l_prime in range(L):
                if (Z_n[l_prime] == 1) and \
                        (U[d, l_prime] == 1) and \
                        (V[m, l_prime] == 1) and \
                        (l_prime != l):
                    alrdy_active = True  # OR
                    break

            if alrdy_active is False:
                if X_n[d, m] == -1:
                    neg_score += 1
                elif X_n[d, m] == 1:
                    pos_score += 1

    return pos_score, neg_score


# OR-AND
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


@jit('int64(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)',
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

    score = int64(0)
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


# XOR-AND
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


@jit('int64(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)', nopython=True, nogil=True)
def posterior_score_XOR_AND_3D(Z_n, U, V, X_n, l):
    """
    Return count of correct/incorrect explanations
    caused by setting Z[n,l] to 1, respecting
    explaining away dependencies
    TODO: should this be given a signature?
    """
    D, L = U.shape
    M, _ = V.shape

    score = int64(0)
    for d in range(D):
        for m in range(M):
            if (U[d, l] != 1) or (V[m, l] != 1):  # AND
                continue

            # compute deltaXOR-AND
            num_active = np.int8(0)
            for l_prime in range(L):
                if (Z_n[l_prime] == 1) and\
                        (U[d, l_prime] == 1) and\
                        (V[m, l_prime] == 1) and\
                        (l_prime != l):
                    num_active += 1
                    if num_active > 1:
                        break

            if num_active == 0:
                score += X_n[d, m]
            elif num_active == 1:
                score -= X_n[d, m]

    return score


# XOR-NAND
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


@jit('int64(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)', nopython=True, nogil=True)
def posterior_score_XOR_NAND_3D(Z_n, U, V, X_n, l):

    D, L = U.shape
    M, _ = V.shape

    score = int64(0)
    for d in range(D):
        for m in range(M):
            if U[d, l] != 1 or V[m, l] != 1:  # AND
                continue

            # compute deltaXOR-NAND
            num_active = np.int8(0)
            for l_prime in range(L):
                if ((Z_n[l_prime] != 1) or
                    (U[d, l_prime] != 1) or
                    (V[m, l_prime] != 1)) and\
                        (l_prime != l):
                    num_active += 1
                    if num_active > 1:
                        break

            if num_active == 0:
                score += X_n[d, m]
            elif num_active == 1:
                score -= X_n[d, m]

    return -score
    raise NotImplementedError


# OR-NAND
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


@jit('int64(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)', nopython=True, nogil=True)
def posterior_score_OR_NAND_3D(Z_n, U, V, X_n, l):

    M, _ = V.shape
    D, L = U.shape

    score = int64(0)
    for d in range(D):
        for m in range(M):
            if (U[d, l] == -1) or (V[m, l] == -1):  # NAND
                continue

            alrdy_active = False
            for l_prime in range(L):
                if ((Z_n[l_prime] == -1) or
                        (U[d, l_prime] == -1) or
                        (V[m, l_prime] == -1)) and\
                        (l_prime != l):
                    alrdy_active = True  # OR
                    break

            if alrdy_active is False:
                score += X_n[d, m]

    return -score


# OR-XOR
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


@jit('int64(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)', nopython=True, nogil=True)
def posterior_score_OR_XOR_3D(Z_n, U, V, X_n, l):

    D, L = U.shape
    M, _ = V.shape

    score = int64(0)
    for d in range(D):
        for m in range(M):
            if U[d, l] == 1 and V[m, l] == 1:  # XOR cant be changed by z_nl
                continue

            explained_away = False
            for l_prime in range(L):
                if (Z_n[l_prime] + U[d, l_prime] + V[m, l_prime] == -1) and\
                        (l_prime != l):
                    explained_away = True
                    break

            if explained_away is False:
                score += X_n[d, m] * U[d, l] * V[m, l]  # very elegant ;)

    return score


# NAND-XOR
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


@jit('int64(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)', nopython=True, nogil=True)
def posterior_score_NAND_XOR_3D(Z_n, U, V, X_n, l):

    M, _ = V.shape
    D, L = U.shape
    score = int64(0)
    for d in range(D):
        for m in range(M):
            if U[d, l] == 1 and V[m, l] == 1:  # XOR cant be changed by z_nl
                continue

            explained_away = False
            for l_prime in range(L):
                if (Z_n[l_prime] + U[d, l_prime] + V[m, l_prime] != -1) and\
                        (l_prime != l):
                    explained_away = True
                    break

            if explained_away is False:
                score += X_n[d, m] * U[d, l] * V[m, l]

    return -score


# XOR-XOR
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
            score -= X_n[d] * U[d, l]
        elif num_active == 1:
            score += X_n[d] * U[d, l]

    return score


@jit('int64(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)', nopython=True, nogil=True)
def posterior_score_XOR_XOR_3D(Z_n, U, V, X_n, l):

    M, _ = V.shape
    D, L = U.shape

    score = int64(0)
    for d in range(D):
        for m in range(M):
            if U[d, l] == 1 and V[m, l] == 1:  # XOR cant be changed by z_nl
                continue

            num_active = np.int8(0)
            for l_prime in range(L):
                if (Z_n[l_prime] + U[d, l_prime] + V[m, l_prime] == -1) and\
                        (l_prime != l):
                    num_active += 1
                if num_active > 1:
                    break

            if num_active == 0:
                score += X_n[d, m] * U[d, l] * V[m, l]
            elif num_active == 1:
                score -= X_n[d, m] * U[d, l] * V[m, l]

    return score


# XOR-NXOR
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


@jit('int64(int8[:], int8[:,:], int8[:,:], int8[:,:], int16)', nopython=True, nogil=True)
def posterior_score_XOR_NXOR_3D(Z_n, U, V, X_n, l):

    M, _ = V.shape
    D, L = U.shape

    score = int64(0)
    for d in range(D):
        for m in range(M):
            if U[d, l] == 1 and V[m, l] == 1:  # NXOR cant be changed by z_nl
                continue

            num_active = np.int8(0)
            for l_prime in range(L):
                if (U[d, l_prime] + Z_n[l_prime] + V[m, l_prime] != -1) and\
                        (l_prime != l):
                    num_active += 1
                if num_active > 1:
                    break

            if num_active == 0:
                score -= X_n[d, m] * U[d, l] * V[m, l]
            elif num_active == 1:
                score += X_n[d, m] * U[d, l] * V[m, l]

    return score
