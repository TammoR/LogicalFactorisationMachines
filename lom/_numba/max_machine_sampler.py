
import numpy as np
import numba
import math
from numba import jit, int8, int16, int32, float32, float64, prange
# import lom._cython.matrix_updates as cython_mu
from scipy.special import expit
from lom._numba.matrix_updates_numba import flip_metropolised_gibbs_numba
from lom._numba.lom_outputs import OR_AND_product_expand

# @jit('void(int8[:,:], int8[:,:], int8[:,:], float64[:], int8[:], float32[:,:,:])',
#      nogil=True, nopython=False, parallel=True)
def draw_MAX_AND_2D(Z, Z_fixed, U, X, lbda, l_sorted, lbda_ratios):
    N, L = Z.shape
    for n in prange(N):
        for l in range(L):
            if Z_fixed[n, l] == 1:
                continue
            logit_p = posterior_score_MAX_AND_2D(
                        Z[n, :], U, X[n, :], l, l_sorted, lbda_ratios)
            Z[n, l] = flip_metropolised_gibbs_numba(logit_p, Z[n, l])


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

    predictions = [OR_AND_product_expand(z()[:, l], u()[:, l]) == 1 
        for l in l_list]

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

    p_new = (P_remain + 1) / float(P_remain + N_remain + 2)

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


### The following functions aren't needed atm.
def maxmachine_relevance(layer, model_type='mcmc'):
    """
    Return the expectec fraction of 1s that are modelled 
    by any latent dimension.
    This is similiar to the eigenvalues in PCA.
    """

    # avoid unnecessary re-computation.
    if layer.eigenvals is not None:
        return layer.eigenvals


    if model_type is 'plugin': 
        alphas = layer.lbda()
        x = layer.child()
        # add clamped units
        z = np.concatenate([(layer.z.mean()+1)*.5,np.ones([layer.z().shape[0],1])], axis=1)
        u = np.concatenate([(layer.u.mean()+1)*.5,np.ones([layer.u().shape[0],1])], axis=1)
        N = z.shape[0]
        D = u.shape[0]
        L = z.shape[1]

        # # we need to argsort all z*u*alpha. This is of size NxDxL!
        # l_sorted = np.zeros([N,D,L], dtype=np.int8)
        # for n in range(N): # could be parallelised -> but no argsort in cython without extra pain
        #     for d in range(D):
        #         l_sorted[n,d,:] = np.argsort(alphas*u[d,:]*z[n,:])

        eigenvals = np.zeros(len(alphas)) # array for results
        idxs = np.where(layer.child()==1)
        idxs = zip(idxs[0],idxs[1]) # indices of n,d where x[n,d]=1
        no_ones = len(idxs)

        # iterate from largest to smallest alpha doesn't work
        for l in range(L):
            for n,d in idxs:
                eigenvals[l] += z[n,l]*u[d,l]*alphas[l] * np.prod(
                    [1-z[n,l_prime]*u[d,l_prime] for l_prime in range(L) 
                     if z[n,l_prime]*u[d,l_prime]*alphas[l_prime] > z[n,l]*u[d,l]*alphas[l]])
        eigenvals /= len(idxs)

    elif model_type is 'mcmc':
        alpha_tr = layer.lbda.trace
        z_tr = layer.z.trace
        u_tr = layer.u.trace
        x = layer.child()
        tr_len = u_tr.shape[0]
        eigenvals = np.zeros(alpha_tr.shape[1])


        for tr_idx in range(tr_len):
            for l in range(u_tr.shape[2]):
                
                # deterministic prediction of current l
                x_pred = np.dot(z_tr[tr_idx,:,l:l+1]==1,
                                u_tr[tr_idx,:,l:l+1].transpose()==1)

                # deterministic prediction of all l' > l
                x_pred_alt = np.zeros(x_pred.shape)
                for l_alt in range(u_tr.shape[2]):
                    if alpha_tr[tr_idx,l_alt] > alpha_tr[tr_idx,l]:
                        x_pred_alt += np.dot(z_tr[tr_idx,:,l_alt:l_alt+1]==1,
                                             u_tr[tr_idx,:,l_alt:l_alt+1].transpose()==1)
                        x_pred_alt = x_pred_alt > 0

                eigenvals[l] += alpha_tr[tr_idx,l] * np.sum(x[(x_pred==1) & (x_pred_alt !=1)] == 1)
                # import pdb; pdb.set_trace()

                # eigenvals[l] += alpha_tr[tr_idx,l] * np.sum(
                #     x[np.dot(z_tr[tr_idx,:,l:l+1]==1,
                #              u_tr[tr_idx,:,l:l+1].transpose()==1)]==1)
            eigenvals[-1] += alpha_tr[tr_idx, -1]

        eigenvals /= float(tr_len)*np.sum(x==1)

    layer.eigenvals = eigenvals
    return eigenvals

        
def maxmachine_forward_pass(u, z, alpha):
    """
    compute probabilistic output for a single 
    latent dimension in maxmachine.
    """
    x = np.zeros([z.shape[0], u.shape[0]])

    for l in np.argsort(-alpha[:-1]):
        x[x==0] += alpha[l]*np.dot(z[:,l:l+1],u[:,l:l+1].transpose())[x==0]

    x[x==0] = alpha[-1]

    return x


