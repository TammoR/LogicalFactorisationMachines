#!/usr/bin/env python
"""
Functions to automatize experiments

"""

import numpy as np
import lib
import sys
import tempfile
import sklearn
from IPython.core.debugger import Tracer

import lom
import lom.matrix_updates_c_wrappers as wrappers
import lom.lambda_updates_c_wrappers as sampling
import lom._cython.matrix_updates as cf

def generate_random_tensor(L, dims, noise=0, density=.5):
    """
    Generate a tensor X of low rank structure from the Boolean product
    of random matrices with dimension [N,L], [D,L], [M,L].
    the density of the matrices is chosen such that the expected density
    of of X equals density.
    Every bit in X is flipped with probability noise.
    """

    # compute required factor density
    d_f = lambda d_t, L, K: np.power(1-np.power(1-d_t,1/L),1/K)
    factor_density = d_f(density, L, 3)

    N, D, M = dims

    Z = np.array(np.random.rand(N,L)>(1-factor_density), dtype=bool)
    U = np.array(np.random.rand(D,L)>(1-factor_density), dtype=bool)
    V = np.array(np.random.rand(M,L)>(1-factor_density), dtype=bool)

    f_truth = [Z,U,V]

    X = lib.boolean_tensor_product(Z,U,V)

    X_noisy = lib.add_bernoulli_noise(X, p = noise)

    return X_noisy, X, f_truth, factor_density



def dbtf_reconstruct(X, L, hyperparms=[1,.3,50]):
    """
    Calling dbtf to compute Boolean tensor factorisation of X
    for L latent dimensions.
    Returns reconstruction and tuple of factor matrices Z, U, V
    hyerparameters is a tuple of 
        (initial factor sets, initial density, convergence iters)
    """

    import subprocess
    import os
    assert set(np.unique(X)) == {True, False}

    # files are written to and read from this directory
    dbtf_dir = "/Users/trukat/dphil_projects/modules/dbtf-1.0/"
    os.chdir(dbtf_dir)

    N, D, M = X.shape

    # write tensor X to text file
    sparse_tensor = np.array(np.where(X==1)).transpose()
    np.savetxt('./data/mytensor.tensor', sparse_tensor, 
           delimiter=',', fmt='%d',
           header="tensor size: "+str(N)+"-by-"+str(D)+"-by-"+str(M)+"\ntensor base index: 0")

    hyperparms = [str(x) for x in hyperparms]
    shell_command = ['./dbtf_submit_simple.sh']+[str(dim) for dim in X.shape]+[str(L)]+hyperparms

    # run dbtf algorithm
    print('Calling dbtf script: ', shell_command)
    subprocess.call(shell_command)

    # load results
    factors_dbtf_sparse = [np.loadtxt('./output/sample_factor'+str(i)+'_sparse.txt',
                                     delimiter=',',dtype=np.int) for i in range(1,4)]

    # convert results to dense arrays
    f_dbtf = []
    dims = [N,D,M]
    for f, dim in zip(factors_dbtf_sparse,dims):
        f_dbtf.append(np.zeros([dim,L], dtype=bool))
        for (i,j) in f:
            f_dbtf[-1][i,j] = True

    X_recon = lib.boolean_tensor_product(*f_dbtf)

    return X_recon, f_dbtf


def tensorm_reconstruct(X, L, hyperparms=[.5,1.0], return_layer=False):

    if not X.dtype == np.int8:
        X = np.array(X, dtype=np.int8)

    if np.all([y in [0,1] for y in np.unique(X)]):
        X = 2*X-1

    p_init = hyperparms[0]
    lbda_init = hyperparms[1]

    orm = lom.Machine()
    data = orm.add_matrix(X, sampling_indicator=False)
    layer = orm.add_tensorm_layer(
        child=data, size=L, 
        lbda_init=lbda_init,
        inits = 3*[p_init])

    # assign the correct updating functions
    for factor_matrix in data.parents:
        factor_matrix.sampling_fct = wrappers.draw_tensorm_noparents_onechild_wrapper
    layer.lbda.sampling_fct = sampling.draw_lbda_tensorm

    orm.infer(burn_in_min=1000, no_samples=50)

    X_recon = layer.output(recon_model='mc', force_computation=True)
    X_recon_plugin = layer.output(recon_model='plugin', force_computation=True)
    f_tensorm = (layer.z.mean(), layer.u.mean(), layer.v.mean())

    if return_layer is True:
        return layer
    else:
        return X_recon, f_tensorm, X_recon_plugin


def tensorm_reconstruct_indp(X, L, hyperparms=[0.5,1.0]):

    if not X.dtype == np.int8:
        X = np.array(X, dtype=np.int8)

    if np.all([y in [0,1] for y in np.unique(X)]):
        X = 2*X-1

    p_init = hyperparms[0]
    lbda_init = hyperparms[1]

    orm = lom.Machine()
    data = orm.add_matrix(X, sampling_indicator=False)
    layer = orm.add_tensorm_layer(
        child=data, size=L, 
        lbda_init=lbda_init,
        inits = 3*[p_init],
        noise_model='tensorm-link-indp')

    # assign the correct updating functions
    for factor_matrix in data.parents:
        factor_matrix.sampling_fct = wrappers.draw_tensorm_indp_noparents_onechild_wrapper

    layer.lbda_p.sampling_fct = sampling.draw_lbda_tensorm_indp_p
    layer.lbda_m.sampling_fct = sampling.draw_lbda_tensorm_indp_m
    layer.lbda = (layer.lbda_p, layer.lbda_m)

    orm.infer(burn_in_min=1000, no_samples=50)

    X_recon = layer.output(recon_model='mc', force_computation=True)
    X_recon_plugin = layer.output(recon_model='plugin', force_computation=True)
    f_tensorm = (layer.z.mean(), layer.u.mean(), layer.v.mean())

    return X_recon, f_tensorm, X_recon_plugin


def split_tensor_train_test(tensor, split=.1, balanced=False):
    """
    works with missing data. scales reasonably well large data, avoiding use of np.where
    """

    num_split = int(np.sum(tensor != 0)*split)


    index_generator = lambda: tuple([np.random.randint(dim) for dim in tensor.shape])
    # rand_tensor_idx = [tensor.shape for i in range(num_split)]
    rand_tensor_idx = np.zeros([len(tensor.shape), num_split], dtype=np.int)
    i = 0


    if balanced is False:
        while i < num_split:
            idx = index_generator()
            if (tensor[idx] != 0) and (idx not in rand_tensor_idx):
                rand_tensor_idx[:,i] = idx
                i += 1

    else:
        print('Balanced split is not optimised!')
        previous = -1
        while i < num_split:
            idx = index_generator()
            if (tensor[idx] != 0) and (idx not in rand_tensor_idx) and (tensor[idx] != previous):
                rand_tensor_idx[:,i] = idx
                previous *= -1
                i += 1

    # nz_idx = list(zip(*[np.where(tensor != 0)[i] for i in range(3)]))
    # num_data_points = len(nz_idx)
    # rand_idx = np.random.choice(range(num_data_points), replace=False, size=int(split*num_data_points))
    # rand_tensor_idx = [nz_idx[i] for i in rand_idx]

    test_mask = np.zeros(tensor.shape, dtype=bool)

    for idx in rand_tensor_idx.transpose():
        test_mask[tuple(idx)] = True

    tensor = np.array(tensor, dtype=np.int8)
    tensor[test_mask] = 0

    return tensor, test_mask



def split_tensor_train_test_old(tensor, split=.1):
    """
    Input: tensor with 8bit ints [-1,1] and test proportion split (float)
    Returns: integer tensor [-1,0,1] with elements randomly labeled as unobserved (0)
    """

    assert tensor.dtype == np.int8

    num_data_points = np.prod(tensor.shape)

    rand_idx = np.random.choice(range(num_data_points), replace=False, size=int(split*num_data_points))
    rand_tensor_idx = [x for i,x in enumerate(zip(*np.where(tensor))) if i in rand_idx]

    test_mask = np.zeros(tensor.shape, dtype=bool)

    for idx in rand_tensor_idx:
        test_mask[idx] = True
        
    tensor = np.array(tensor, dtype=np.int8)
    tensor[test_mask] = 0

    return tensor, test_mask
