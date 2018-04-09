#!/usr/bin/env python
"""
lom_sampling.py


"""
import numpy as np
import lom.matrix_update_wrappers as wrappers
import lom._cython.matrix_updates as cf
# import lom._cython.tensor_updates as cf_tensorm
import lom._numba.lambda_updates_numba as lambda_updates_numba


def get_update_fct(parm):

    print('Assigning update function: ' + parm.layer.__repr__())

    if parm.sampling_fct is not None:
        return parm.sampling_fct

    if parm.layer.model == 'MAX-AND' and parm.layer.dimensionality == 2:
        import lom._numba.max_machine_sampler as mm_sampler
        def MAX_AND_2D_lbda(parm):
            mm_sampler.lbda_MAX_AND(parm, K=2)
        return MAX_AND_2D_lbda

    else:
        return lambda_updates_numba.make_lbda_update_fct(
            parm.layer.model, parm.layer.dimensionality)


### Following functions are not needed anymore. Keep for reference for the moment
### TODO

def draw_lbda_or(parm):
    """
    Update a Machine parameter to its MLE / MAP estimate
    """

    if parm.layer.machine.framework == 'numba':
        lambda_updates_numba.draw_lbda_or_numba(parm)

    elif parm.layer.machine.framework == 'cython':

        # TODO: for some obscure reason this is faster than compute_P_parallel
        P = cf.compute_P(parm.attached_matrices[0].child(),
                         parm.attached_matrices[1](),
                         parm.attached_matrices[0]())

        # effectie number of observations (precompute for speedup TODO (not crucial))
        ND = (np.prod(parm.attached_matrices[0].child().shape) -
              np.count_nonzero(parm.attached_matrices[0].child() == 0))

        # Flat prior
        if parm.prior_config[0] == 0:
            # use Laplace rule of succession
            parm.val = -np.log(((ND + 2) / (float(P) + 1)) - 1)
            # parm.val = np.max([0, np.min([1000, -np.log( (ND) /
            # (float(P)-1) )])])

        # Beta prior
        elif parm.prior_config[0] == 1:
            alpha = parm.prior_config[1][0]
            beta = parm.prior_config[1][1]
            parm.val = -np.log((ND + alpha - 1) / (float(P) + alpha + beta - 2) - 1)


def draw_lbda_tensorm_indp_p(parm):
    """
    Update lbda_p for independent TensOrM
    This is redundant with lbda_minus, but allows 
    for nice modularity
    """

    TP, FP = cf_tensorm.compute_tp_fp_tensorm(
        parm.layer.child(),
        parm.attached_matrices[0](),
        parm.attached_matrices[1](),
        parm.attached_matrices[2]())

    # use beta prior
    # a = 100
    # b = 1
    # parm.val = np.max(
 #        [ - np.log( ( (TP + FP + a - 1) / ( float( TP + 1 + a + b -2 ) ) ) - 1  ),
 #         0 ] )
    parm.val = np.max([-np.log(((TP + FP + 2) / (float(TP + 1))) - 1), 0])
    # print('Positives: '+str(parm.val))


def draw_lbda_tensorm_indp_m(parm):
    """
    Update lbda_m for independent TensOrM
    This is redundant with lbda_minus, but allows 
    for nice modularity
    """

    TN, FN = cf_tensorm.compute_tn_fn_tensorm(
        parm.layer.child(),
        parm.attached_matrices[0](),
        parm.attached_matrices[1](),
        parm.attached_matrices[2]())

    # import pdb; pdb.set_trace()
    # use lapalce succession
    parm.val = np.max([-np.log(((TN + FN + 2) / (float(TN + 1))) - 1), 0])

    # print('Negatives: '+str(parm()))


def draw_lbda_or_balanced(parm):

    cf.compute_pred_accuracy(parm.attached_matrices[0].child(),
                             parm.attached_matrices[0](),
                             parm.attached_matrices[1](),
                             parm.layer.pred_rates)

    TP, FP, TN, FN = parm.layer.pred_rates
    s = parm.balance_factor

    parm.val = (TP + (TN / s)) / (TP + FN + ((TN + FP) / s))

    draw_lbda_or(parm)