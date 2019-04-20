#!/usr/bin/env python
"""
wrappers for calls to numba sampling functions.
"""

import numpy as np
import lom._numba.matrix_updates_numba as numba_mu
from scipy.special import logit


def get_sampling_fct(mat):
    """
    Assign matrix update function to mat, depending on its
    model and parent/child layers.
    The default architecture is a single child layer an no parents.
    Other architecture are only partially supported
    """

    # use pre-assigned sampling function if present
    if mat.sampling_fct is not None:
        return mat.sampling_fct

    # determine order of child dimensions such that first
    # child dimension and mat dimension are aligned.
    mod, p1mod, p2mod = get_relatives_models(mat)

    if mod is not None:
        transpose_order = tuple(
            [mat.child_axis] +
            [s.child_axis for s in mat.siblings])

    msg = 'Assigning sampling functions - '
    if mod is not None:
        msg += '\tchild: ' + str(mod)
    if p1mod is not None:
        msg += '\tparent1: ' + str(p1mod)
    if p2mod is not None:
        msg += '\tparent2: ' + str(p2mod)

    print(msg)

    if 'OR_AND_dropout_' in mod:

        sample = numba_mu.make_sampling_fct_onechild_dropout(mod)

        def lfm_sampler_dropout(mat):

            transpose_order = tuple([mat.child_axis] + [s.child_axis for s in mat.siblings])
            logit_bernoulli_prior = np.float64(logit(mat.bernoulli_prior))

            sample(
                mat(),
                mat.fixed_entries,
                *[x() for x in mat.siblings],
                mat.layer.child().transpose(transpose_order),
                mat.layer.lbda(),
                mat.layer.dropout_factor,
                logit_bernoulli_prior)

        return lfm_sampler_dropout

        # if mod == 'OR_AND_dropout_2D':
        #     temp = 0
        #
        # elif mod == 'OR_AND_dropout_3D':
        #     temp = 0

    # assign functions for MAX-AND model
    elif mod == 'MAX_AND_2D':
        assert p1mod is None
        assert p2mod is None
        from lom._numba.max_machine_sampler import draw_MAX_AND_2D

        def MAX_AND_2D(mat):
            l_order = np.array(np.argsort(-mat.layer.lbda()[:-1]), dtype=np.int8)
            draw_MAX_AND_2D(
                mat(),
                mat.fixed_entries,
                mat.siblings[0](),
                mat.layer.child().transpose(transpose_order),
                mat.layer.lbda(),
                l_order,
                mat.layer.lbda_ratios)

        return MAX_AND_2D

    elif mod == 'qL_AND_2D':

        def q_sampler(q):
            numba_mu.sample_qL_q(
                q.layer.factors[0](),
                q.layer.factors[1](),
                q.layer.child(),
                q(),
                q.layer.lbda(),
                q.layer.gamma)

        mat.layer.q.sampling_fct = q_sampler

        if mat.child_axis == 0:
            def qL_sampler_Z(mat):
                numba_mu.sample_qL_factors_Z(
                    mat(),
                    mat.siblings[0](),
                    mat.layer.child().transpose(transpose_order),
                    mat.layer.q(),
                    mat.layer.lbda())

            return qL_sampler_Z

        elif mat.child_axis == 1:
            def qL_sampler_U(mat):
                numba_mu.sample_qL_factors_U(
                    mat(),
                    mat.siblings[0](),
                    mat.layer.child().transpose(transpose_order),
                    mat.layer.q(),
                    mat.layer.lbda())

            return qL_sampler_U

    # assign sampling functions for IBP.
    # Here all updates happen in sampling for the first factor.
    elif mod == 'OR_AND_IBP_2D':

        if mat.layer.alpha is None:
            print('Setting IBP parameter alpha to 3.')
            mat.layer.alpha = 3

        # IBP only on the first factor
        if mat.child_axis == 0:

            def IBP_sampler(mat):
                """
                References are overwritten in the function,
                need to return arrays.
                """
                mat.val, mat.siblings[0].val = numba_mu.sample_2d_IBP(
                    mat(),
                    mat.siblings[0](),
                    mat.layer.child().transpose(transpose_order),
                    mat.layer.lbda(),
                    mat.siblings[0].bernoulli_prior,
                    mat.layer.alpha)

            return IBP_sampler

        # For U, sample as usual
        elif mat.child_axis == 1:
            # return lambda mat: None
            sample = numba_mu.make_sampling_fct_onechild('OR_AND_2D')
            logit_bernoulli_prior = np.float64(logit(mat.bernoulli_prior))

            def LOM_sampler(mat):
                # numba_mu.draw_OR_AND_2D(
                sample(
                    mat(),
                    mat.fixed_entries,
                    *[x() for x in mat.siblings],
                    mat.layer.child().transpose(transpose_order),
                    mat.layer.lbda(),
                    logit_bernoulli_prior)

            return LOM_sampler

    # assign functions for all classical LFMs
    else:
        logit_bernoulli_prior = np.float64(logit(mat.bernoulli_prior))

        # standard case: one child, no parents
        if p1mod is None and mod is not None:
            sample = numba_mu.make_sampling_fct_onechild(mod)

            def LOM_sampler(mat):
                # numba_mu.draw_OR_AND_2D(
                sample(
                    mat(),
                    mat.fixed_entries,
                    *[x() for x in mat.siblings],
                    mat.layer.child().transpose(transpose_order),
                    mat.layer.lbda(),
                    logit_bernoulli_prior)

            return LOM_sampler

        # one child, one parent
        elif p1mod is not None and mod is not None and p2mod is None:
            sample = numba_mu.make_sampling_fct_onechild_oneparent(
                mod, p1mod)

            def LOM_sampler_hasparents(mat):
                sample(
                    mat(),
                    mat.fixed_entries,
                    *[x() for x in mat.siblings],
                    mat.layer.child().transpose(transpose_order),
                    mat.layer.lbda(),
                    *[x() for x in mat.parents[0].factors],
                    mat.parents[0].lbda(),
                    logit_bernoulli_prior)

            return LOM_sampler_hasparents

        # no child, one parent
        elif mod is None and p1mod is not None and p2mod is None:
            sample = numba_mu.make_sampling_fct_nochild_oneparent(p1mod)

            def LOM_sampler_hasparents(mat):
                sample(
                    mat(),
                    mat.fixed_entries,
                    *[x() for x in mat.parents[0].factors],
                    mat.parents[0].lbda(),
                    logit_bernoulli_prior)

            return LOM_sampler_hasparents

        # no child, two parents
        elif mod is None and p1mod is not None and p2mod is not None:
            sample = numba_mu.make_sampling_fct_nochild_twoparents(p1mod, p2mod)

            def LOM_sampler_hasparents(mat):
                sample(
                    mat(),
                    mat.fixed_entries,
                    *[x() for x in mat.parents[0].factors],
                    mat.parents[0].lbda(),
                    *[x() for x in mat.parents[1].factors],
                    mat.parents[1].lbda(),
                    logit_bernoulli_prior)

            return LOM_sampler_hasparents

        # one child, two parents
        elif mod is not None and p1mod is not None and p2mod is not None:
            sample = numba_mu.make_sampling_fct_onechild_twoparents(
                mod, p1mod, p2mod)

            def LOM_sampler_hasparents(mat):
                sample(
                    mat(),
                    mat.fixed_entries,
                    *[x() for x in mat.siblings],
                    mat.layer.child().transpose(transpose_order),
                    mat.layer.lbda(),
                    *[x() for x in mat.parents[0].factors],
                    mat.parents[0].lbda(),
                    *[x() for x in mat.parents[1].factors],
                    mat.parents[1].lbda(),
                    logit_bernoulli_prior)

            return LOM_sampler_hasparents

        else:
            import pdb
            pdb.set_trace()
            raise NotImplementedError("More than one parent not supported.")


def get_relatives_models(mat):
    """
    Return tuple (model, parent1_model, parent2_model) in 'OR_AND_2D' format
    """
    if mat.model is not None:
        model = mat.model + '-' + str(mat.layer.dimensionality) + 'D'
        model = model.replace('-', '_')
    else:
        model = None

    parent1_model = None
    parent2_model = None
    if len(mat.parents) > 0:
        assert len(mat.parents[0].factors) == 2
        parent1_model = (mat.parents[0].model + '-2D').replace('-', '_')
    if len(mat.parents) > 1:
        assert len(mat.parents[1].factors) == 2
        parent2_model = (mat.parents[1].model + '-2D').replace('-', '_')
    elif len(mat.parents) > 2:
        raise NotImplementedError("More than two parents are not supported.")

    return model, parent1_model, parent2_model


# Following functions aren't used. keep for reference
#
# def draw_balanced_or(mat):
#     """
#     mat() and child need to share their first dimension. otherwise transpose.
#     """
#     transpose_order = tuple([mat.child_axis] + [mat.sibling.child_axis])
#
#     cf.draw_balanced_or(
#         mat(),
#         mat.sibling(),
#         mat.child.transpose(transpose_order),
#         mat.lbda(),
#         mat.lbda() * (1 / mat.lbda.balance_factor)
#     )
