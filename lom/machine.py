#!/usr/bin/env python
"""
Logical Operate Machine

This module implements classes for sampling form hierarchical binary
matrix factorisation models.

A machine consists of multiple matrices that have mutual
relationship akin to nodes in a graphical model.

The minimal example is a standard matrix factorisation model
with a data matrix 'data', and its two parents 'z' (objects x latent) and
'u' (features x latent). 'z' and 'u' are siblings, 'data' is their child.
All matrices are instances of the MachineMatrix class and expose their
family relationes as attributes, e.g.: z.child == data;
data.parents == [u,z], etc.

A further abstraction combines each pair of siblings into layers
(instances of MachineLayer), together with an additional set of
parameters 'lbda' (instances of MachineParameter).

During inference each matrix can be held fixed or can be
sampled elementwise according to its full conditional proability.

This framework allows for the definition of flexible hierarchies, e.g.
for a two-layer hierarchical factorisation model:

m = machine()
data = m.add_matrix( some_data, sampling_indicator=False )
layer1 = m.add_layer( size1, child = data )
layer2 = m.add_layer( size2, cild = layer1.z )

Prior distributions can be specified. For all instances of machine_matrix.
Some combinations are not implemented, yet.
    # iid bernoulli prior on all matrix entries
    # z.set_prior ('binomial', .5)

    # binomial prior across rows of z
    # z.set_prior ( 'bernoulli', .5, axis = 0 )

    # binomial prior across columns of z, with K draws to enforce sparsity.
    z.set_prior ( 'bernoulli', [.5, K] , axis = 1 )

For the MaxMachine, a beta prior can be specified on the dispersion parameter, e.g.
    # set beta(1,1) prior layer1.lbda.set_prior([1,1])

Finally, Machine.infer() draws samples from all matrices and updates
each layers parameters until convergence and saves the following samples
into each members' trace.


feature dimensions: d = 1...D
object dimensions:  n = 1...N
latent dimensions:  l = 1...L

latent object matrix:  z [NxL]
latent feature matrix: u [DxL]
additional parameters: lbda [L]

Implemented classes are:
- Trace
- MachineParameter

- machine
- machine_matrix
- machine_parameter
- trace

TODOs
- uncluter the family setup
- generalise everyting to arbitrary arity
- .member should return list in proper odering
- Need simple ways of accessing all children etc in well defined order
- layers and matrices should have names e.g. orm.layers[0].z.name = 'z'

"""

from __future__ import absolute_import, division, print_function  # for python2
from numpy.random import binomial
import numpy as np
import lom.auxiliary_functions as lib
import lom.auxiliary_functions as aux

import lom.matrix_update_wrappers as wrappers
import lom.matrix_update_wrappers as mat_wrappers

import lom.lambda_update_wrappers as sampling
import lom.lambda_update_wrappers as lbda_wrappers

# import lom._cython.matrix_updates as cf
# import lom._cython.tensor_updates as cf_tensorm
# from IPython.core.debugger import Tracer

__author__ = "Tammo Rukat"
__status__ = "Development"


class Trace():
    """
    abstract base class implementing methods posterior traces arrays.
    Inherited to MachineMatrix and MachineParameter. TODO MachineMatrix and MachineParameter
    should be instances of the same class (?).
    """

    def __call__(self):
        return self.val

    def allocate_trace_arrays(self, no_of_samples):
        no_of_samples = int(no_of_samples)
        if type(self.val) == np.ndarray:
            # nicer but no python2 compatible
            # self.trace = np.empty([no_of_samples, *self.val.shape], dtype=np.int8)

            self.trace = np.empty([no_of_samples] +
                                  [x for x in self.val.shape],
                                  dtype=self().dtype)
        else:  # not needed (?)
            self.trace = np.empty([no_of_samples], dtype=np.float32)

    def update_trace(self):
        self.trace[self.trace_index] = self.val
        self.trace_index += 1

    def mean(self):
        if 'trace' in dir(self):
            return np.mean(self.trace, axis=0)
        # if no trace is defined, return current state
        else:
            return self()

    def check_convergence(self, eps):
        """
        split trace in half and check difference between means < eps
        """

        if self.trace.ndim == 1:
            return lib.check_convergence_single_trace(self.trace, eps)

        # if we have multiple dispersion parameter every one of them needs to have converged
        elif self.trace.ndim == 2:
            return np.all(
                [lib.check_convergence_single_trace(self.trace[:, l], eps)
                 for l in range(self.trace.shape[1])])
        else:
            raise IndexError("Can not ascertain convergence of a matrix " +
                             "with more than 2 dimensions")


class MachineParameter(Trace):
    """
    Base class for parameters
    """

    def __init__(self, val, fixed=False):
        self.trace_index = 0
        self.sampling_fct = None
        self.val = val
        self.fixed = fixed

    def print_value(self):
        return ', '.join([str(round(lib.expit(np.mean(x)), 3))
                          for x in [self.val]])
        # if self.noise_model in ['or-link',
        #                         'tensorm-link',
        #                         'tensorm-link-indp',
        #                         'balanced-or']:
        #     return ', '.join([str(round(lib.expit(np.mean(x)), 3))
        #                       for x in [self.val]])
        # elif self.noise_model == 'independent':
        #     return ', '.join([str(round(lib.expit(x), 3))
        #                       for x in self.val])
        # elif self.noise_model == 'max-link':
        #     return '\t'.join([str("%.1f" % round(100 * x, 2))
        #                       for x in self.val])
        # else:
        #     raise SystemError
        # return ', '.join([str(str.format('{0:3f}',x)) for x in self.val])


class MachineMatrix(Trace):

    def __init__(self,
                 shape=None,
                 val=None,
                 child_axis=None,
                 fixed=False):
        self.trace_index = 0
        self.sampling_fct = None
        self.child_axis = child_axis
        self.parents = []

        if val is not None:
            shape = val.shape

        if type(fixed) is np.ndarray:
            assert fixed.shape == shape
        self.fixed = fixed

        # assign value if provided, otherwise bernoulli random
        if type(val) is np.ndarray:
            self.val = np.array(val, dtype=np.int8)
        elif type(val) is float:
            self.val = 2 * np.array(np.random.rand(*shape) > val,
                                    dtype=np.int8) - 1
        else:
            self.val = 2 * np.array(np.random.rand(*shape) > .05,
                                    dtype=np.int8) - 1

    def __call__(self):
        return self.val

    @property
    def siblings(self):
        siblings = [f for f in self.layer.factors if f is not self]
        return sorted(siblings, key=lambda f: f.child_axis)

    def set_to_map(self):
        self.val = np.array(self.mean() > 0, dtype=np.int8)
        self.val[self.val == 0] = -1


class MachineLayer():

    def __init__(self, factors, lbda, child, model='OR-AND'):

        self.factors = sorted(factors, key=lambda f: f.child_axis)

        self.lbda = lbda
        self.lbda.layer = self

        self.model = model
        self.child = child

        self.child.parents.append(self)
        self.size = factors[0]().shape[1]

        for factor in factors:
            factor.layer = self

        self.auto_clean_up = False
        self.auto_reset = False  # TODO get rid of

    def __repr__(self):
        return (self.model + '-' + str(len(self.factors)) +
                'D').replace('-', '_')

    # def assign_sampling_fcts(self):
    #     if self.model == 'OR-AND':
    #         if len(self.child().shape) == 2:
    #             for factor in self.factors:
    #                 factor.sampling_fct =\
    #                     mat_wrappers.draw_noparents_onechild_wrapper
    #             self.lbda.sampling_fct = lbda_wrappers.draw_lbda_or

    #         elif len(self.child().shape) == 3:
    #             for factor in self.factors:
    #                 factor.sampling_fct =\
    #                     mat_wrappers.draw_tensorm_noparents_onechild_wrapper

    #     else:
    #         raise ValueError("No valid model defined.")

    def output(self,
               technique='point_estimate',
               force_computation=False):
        """
        Valid techniques are:
            - 'point_estimate'
                output of the current state of factors
            - 'MC' TODO
                'probabilistic output from the MC trace'
            - 'Factor-MAP' TODO
                From the posterior MAP of factors
            - 'Factor-MEAN' TODO
                Computed from posterior mean of factors
        """
        K = len(self.factors)
        L = self.size

        if technique == 'point_estimate':

            out = np.zeros([x().shape[0] for x in self.factors], dtype=np.int8)
            outer_operator_name, inner_operator_name = self.model.split('-')

            outer_operator = aux.get_lop(outer_operator_name)
            inner_operator = aux.get_lop(inner_operator_name)

            outer_logic = np.zeros(L, dtype=bool)
            inner_logic = np.zeros(K, dtype=bool)

            for index, _ in np.ndenumerate(out):
                for l in range(L):
                    inner_logic[:] =\
                        [f()[index[i], l] == 1 for i, f in enumerate(self.factors)]
                    outer_logic[l] = inner_operator(inner_logic)
                out[index] = 2 * outer_operator(outer_logic) - 1

        elif technique == 'factor_mean':

            out = np.zeros([x().shape[0] for x in self.factors])
            outer_operator_name, inner_operator_name = self.model.split('-')

            outer_operator = aux.get_fuzzy_lop(outer_operator_name)
            inner_operator = aux.get_fuzzy_lop(inner_operator_name)

            outer_logic = np.zeros(L)
            inner_logic = np.zeros(K)

            for index, _ in np.ndenumerate(out):
                for l in range(L):
                    inner_logic[:] =\
                        [.5 * (f.mean()[index[i], l] + 1)
                         for i, f in enumerate(self.factors)]
                    outer_logic[l] = inner_operator(inner_logic)
                out[index] = outer_operator(outer_logic)

        return out

    def output_old(self, u=None, z=None, v=None,
                   recon_model='mc', force_computation=False):
        """
        propagate probabilities to child layer
        u and z are optional and intended for use
        when propagating through mutliple layers.
        outputs a probability of x being 1.
        """
        if (self.precomputed_output is not None) and (not force_computation):
            return self.precomputed_output

        if u is None:
            u = self.u.mean()
        if z is None:
            z = self.z.mean()

        L = z.shape[1]
        N = z.shape[0]
        D = u.shape[0]

        if self.noise_model == 'or-link' or self.noise_model == 'balanced-or':
            x = np.empty((N, D))

            if recon_model == 'plugin':
                cf.probabilistc_output(
                    x, .5 * (u + 1), .5 * (z + 1), self.lbda.mean(), D, N, L)

            elif recon_model == 'mc':
                from scipy.special import expit
                print('Computing MC estimate of data reconstruction')
                u_tr = self.u.trace
                z_tr = self.z.trace
                lbda_tr = self.lbda.trace
                trace_len = u_tr.shape[0]
                x = np.zeros([z_tr.shape[1], u_tr.shape[1]])

                for tr_idx in range(len(lbda_tr)):
                    det_prod = (
                        np.dot(u_tr[tr_idx, :, :] == 1,
                               z_tr[tr_idx, :, :].transpose() == 1)).transpose()
                    x[det_prod == 1] += expit(lbda_tr[tr_idx])
                    x[det_prod == 0] += 1 - expit(lbda_tr[tr_idx])
                x /= float(trace_len)

        elif self.noise_model == 'tensorm-link':
            if v is None:
                v = self.v.mean()

            M = v.shape[0]

            if recon_model == 'plugin':
                x = np.zeros((N, D, M), dtype=np.float32)
                print('Computing tensorm plugin reconstruction.')
                cf_tensorm.probabilistic_output_tensorm(
                    x, .5 * (z + 1), .5 * (u + 1), .5 * (v + 1),
                    self.lbda.mean())

            elif recon_model == 'mc':
                x = np.zeros((N, D, M), dtype=np.float32)
                from scipy.special import expit
                print('Computing MC estimate of data reconstruction')
                z_tr = self.z.trace
                u_tr = self.u.trace
                v_tr = self.v.trace
                lbda_tr = self.lbda.trace
                trace_len = u_tr.shape[0]
                x = np.zeros([z_tr.shape[1], u_tr.shape[1], v_tr.shape[1]])

                for tr_idx in range(len(lbda_tr)):
                    det_prod = lib.boolean_tensor_product(
                        z_tr[tr_idx, :, :],
                        u_tr[tr_idx, :, :],
                        v_tr[tr_idx, :, :])
                    x[det_prod == 1] += expit(lbda_tr[tr_idx])
                    x[det_prod == 0] += 1 - expit(lbda_tr[tr_idx])
                x /= float(trace_len)

            elif recon_model == 'map':
                x = np.zeros((N, D, M), dtype=bool)
                for n in range(N):
                    for d in range(D):
                        for m in range(M):
                            for l in range(L):
                                if ((z[n, l] > 0) and
                                        (u[d, l] > 0) and
                                        (v[m, l] > 0)):
                                    x[n, d, m] = True
                                    break

        elif self.noise_model == 'tensorm-link-indp':
            if v is None:
                v = self.v.mean()

            M = v.shape[0]
            x = np.zeros((N, D, M))

            if recon_model == 'mc':
                from scipy.special import expit
                print('Computing MC estimate of data reconstruction')
                z_tr = self.z.trace
                u_tr = self.u.trace
                v_tr = self.v.trace
                lbda_p_tr = self.lbda_p.trace
                lbda_m_tr = self.lbda_m.trace

                trace_len = u_tr.shape[0]
                x = np.zeros([z_tr.shape[1], u_tr.shape[1], v_tr.shape[1]])

                for tr_idx in range(len(lbda_p_tr)):
                    det_prod = lib.boolean_tensor_product(
                        z_tr[tr_idx, :, :],
                        u_tr[tr_idx, :, :],
                        v_tr[tr_idx, :, :])
                    x[det_prod == 1] += expit(lbda_p_tr[tr_idx])
                    x[det_prod == 0] += 1 - expit(lbda_m_tr[tr_idx])
                x /= float(trace_len)

            if recon_model == 'plugin':
                print('Computing tensorm plugin reconstruction.')
                cf_tensorm.probabilistic_output_tensorm_indp(
                    x, .5 * (z + 1), .5 * (u + 1), .5 * (v + 1),
                    self.lbda_p.mean(), self.lbda_m.mean())

            elif recon_model == 'map':
                f_tensorm = (self.z.mean() > 0,
                             self.u.mean() > 0,
                             self.v.mean() > 0)
                x = lib.boolean_tensor_product(*f_tensorm)

        elif self.noise_model == 'independent':
            cf.probabilistc_output_indpndt(
                x, .5 * (u + 1), .5 * (z + 1), self.lbda.mean()[1],
                self.lbda.mean()[0], D, N, L)

        elif self.noise_model is 'maxmachine_plugin':
            x = np.empty((N, D))
            # check that the background noise is smaller than any latent
            # dimension's noise
            if self.lbda.mean()[-1] != np.min(self.lbda.mean()):
                print('we have alphas < alpha[-1]')
            cf.probabilistic_output_maxmachine(
                x, .5 * (u + 1), .5 * (z + 1), self.lbda.mean(),
                np.zeros(len(self.lbda()), dtype=np.float64),
                np.zeros(len(self.lbda()), dtype=np.int32))

        elif self.noise_model == 'max-link':
            print('Computing MC estimate of data reconstruction')
            u_tr = self.u.trace
            z_tr = self.z.trace
            alpha_tr = self.lbda.trace
            x = np.zeros([N, D])
            trace_len = u_tr.shape[0]
            for tr_idx in range(trace_len):
                x += lib.maxmachine_forward_pass(u_tr[tr_idx, :, :] == 1,
                                                 z_tr[tr_idx, :, :] == 1,
                                                 alpha_tr[tr_idx, :])
            x /= trace_len

        else:
            raise ValueError(
                'Output function not defined for given noise model.')

        self.precomputed_output = x

        return x

    def log_likelihood(self):
        """
        Return log likelihood of the assoicated child, given the layer.
        TODO: implement for maxmachine
        """

        N = self.z().shape[0]
        D = self.u().shape[0]

        if 'or-link' in self.noise_model:

            P = cf.compute_P_parallel(self.lbda.attached_matrices[0].child(),
                                      self.lbda.attached_matrices[1](),
                                      self.lbda.attached_matrices[0]())

            return (-P * lib.logsumexp([0, -self.lbda.mean()]) -
                    (N * D - P) * lib.logsumexp([0, self.lbda.mean()]))

        elif 'independent' in self.noise_model:
            self.update_predictive_accuracy()
            TP, FP, TN, FN = self.pred_rates

            return (TP * lib.logsumexp([0, -self.lbda()]) -
                    FP * lib.logsumexp([0, self.lbda()]) -
                    TN * lib.logsumexp([0, -self.mu()]) -
                    FN * lib.logsumexp([0, self.lbda()]))

        else:
            print('Log likelihood computation not implemented for link.')

    def update_predictive_accuracy(self):
        """
        update values for TP/FP and TN/FN
        """
        cf.compute_pred_accuracy(self.child(), self.u(), self.z(), self.pred_rates)
        self.predictive_accuracy_updated = True

    def precompute_lbda_ratios(self):
        """
        TODO: speedup (cythonise and parallelise)
        precompute matrix of size [2,L+1,L+1],
        with log(lbda/lbda') / log( (1-lbda) / (1-lbda') )
        as needed for maxmachine gibbs updates.
        """

        if self.noise_model != 'max-link':
            return

        L = self.size + 1

        lbda_ratios = np.zeros([2, L, L], dtype=np.float32)

        for l1 in range(L):
            for l2 in range(l1 + 1):
                lratio_p = np.log(self.lbda()[l1] / self.lbda()[l2])
                lratio_m = np.log((1 - self.lbda()[l1]) / (1 - self.lbda()[l2]))
                lbda_ratios[0, l1, l2] = lratio_p
                lbda_ratios[0, l2, l1] = -lratio_p
                lbda_ratios[1, l1, l2] = lratio_m
                lbda_ratios[1, l2, l1] = -lratio_m

        self.lbda_ratios = lbda_ratios


class Machine():
    """
    Main class that bundles matrices, parameters and exposes
    inference methods.
    """

    def __init__(self):
        """
        Initialise lists for convenient access.
        """
        self.layers = []
        self.matrices = []

    @property
    def members(self):
        """
        Return all matrices from within and outside of layers
        """
        single_mats = self.matrices
        layer_mats = [f for layer in self.layers for f in layer.factors]

        return layer_mats + single_mats

    @property
    def lbdas(self):
        return [layer.lbda for layer in self.layers]

    def add_layer(self,
                  latent_size=None,
                  child=None,
                  shape=None,
                  model='OR-AND'):
        """
        information about children's parents need to be assigned.
        """

        # determine size of all members
        if child is None and shape is not None:
            child = MachineMatrix(shape=shape)
        elif shape is None and child is not None:
            shape = child().shape
        else:
            raise ValueError("Not enough shape information provided.")

        # some models require internal inversion of the data
        # apply and replace model type by canonical model. TODO
        # some model also require inversion of the factors.
        # inform the user here.
        if model in []:
            model = ''
            child.val = -child.val
            pass

        # initialise matrices/factors (use add_matrix)
        factors = [MachineMatrix(shape=(K, latent_size),
                                 child_axis=i)
                   for i, K in enumerate(shape)]

        # initialise lambdas (don't use add_parameter)
        if model == 'MAX':
            lbda_init = np.array([1.0 for i in range(latent_size + 1)])
        elif 'BALANCED' in model:
            lbda_init = np.array([1.0 for i in range(2)])
        else:
            lbda_init = 2.0
        lbda = MachineParameter(val=lbda_init)

        # initialise layer object
        layer = MachineLayer(factors, lbda, child, model)
        self.layers.append(layer)
        layer.machine = self

        return layer

    def add_matrix(self, val=None, shape=None, child_axis=None, fixed=False):

        if val is not None and val.dtype != np.int8:
            val = np.array(val, dtype=np.int8)

        mat = MachineMatrix(shape, val, child_axis, fixed)

        self.matrices.append(mat)

        return mat

    def burn_in(self,
                mats,
                lbdas,
                eps=1e-2,
                convergence_window=15,
                burn_in_min=0,
                burn_in_max=2000,
                print_step=10,
                fix_lbda_iters=0):
        """
        draw samples without saving to traces and check for convergence.
        we have an additional pre-burn-in phase where
        we do not check for convergence.
        Convergence is detected by comparing means of noise parameters.
        """

        # first sample without checking for convergence or saving lbda traces
        # this is a 'pre-burn-in-phase'
        pre_burn_in_iter = 0
        while True:
            # stop pre-burn-in if minimum numbers of burn-in iterations is reached
            if pre_burn_in_iter == burn_in_min:
                break

            pre_burn_in_iter += 1

            # print diagnostics
            if pre_burn_in_iter % print_step == 0:
                print('\r\titeration: ' +
                      str(pre_burn_in_iter) +
                      ' disperion.: ' +
                      '\t--\t '.join([x.print_value() for x in lbdas]),
                      end='')

            # draw samples
            [mat.sampling_fct(mat) for mat in np.random.permutation(mats)]
            if pre_burn_in_iter > fix_lbda_iters:
                [lbda.sampling_fct(lbda) for lbda in lbdas]

        # allocate array for lambda traces for burn in detection
        for lbda in lbdas:
            lbda.allocate_trace_arrays(convergence_window)
            lbda.trace_index = 0          # reset trace index

        # now cont. burn in and check for convergence
        burn_in_iter = 0
        while True:
            burn_in_iter += 1

            # print diagnostics
            if burn_in_iter % print_step == 0:
                print('\r\titeration: ' +
                      str(pre_burn_in_iter + burn_in_iter) +
                      ' recon acc.: ' +
                      '\t--\t '.join([x.print_value() for x in lbdas]),
                      end='')

            #  check convergence every convergence_window iterations
            if burn_in_iter % convergence_window == 0:
                # reset trace index
                for lbda in lbdas:
                    lbda.trace_index = 0
                # check convergence for all lbdas
                if np.all([x.check_convergence(eps=eps) for x in lbdas]):
                    print('\n\tconverged at reconstr. accuracy: ' +
                          '\t--\t'.join([x.print_value() for x in lbdas]))

                    # TODO: make this nice and pretty.
                    # check for dimensios to be removed and restart burn-in if
                    # layer.auto_clean_up is True
                    if np.any([lib.clean_up_codes(lr, lr.noise_model)
                               for lr in self.layers if lr.auto_clean_up is True]):
                        print('\tremove duplicate or useless latent ' +
                              'dimensions and restart burn-in. New L=' +
                              + str([lr.size for lr in self.layers]))
                        for lbda in lbdas:
                            # reallocate arrays for lbda trace
                            lbda.allocate_trace_arrays(convergence_window)
                            lbda.trace_index = 0

                    # alternatively, check for dimensions to be reset
                    elif np.any([lib.reset_codes(lr, lr.noise_model)
                                 for lr in self.layers
                                 if lr.auto_reset is True]):
                        for lbda in lbdas:
                            # reallocate arrays for lbda trace
                            lbda.allocate_trace_arrays(convergence_window)
                            lbda.trace_index = 0

                    else:
                        # save nu of burn in iters
                        self.burn_in_iters = burn_in_iter + pre_burn_in_iter
                        break

            # stop if max number of burn in inters is reached
            if (burn_in_iter + pre_burn_in_iter) > burn_in_max:
                print('\n\tmax burn-in iterations reached without convergence')
                # reset trace index
                for lbda in lbdas:
                    lbda.trace_index = 0
                self.burn_in_iters = burn_in_iter
                break

            # draw samples # shuffle(mats)
            [mat.sampling_fct(mat) for mat in np.random.permutation(mats)]
            [lbda.sampling_fct(lbda) for lbda in lbdas]
            [x.update_trace() for x in lbdas]

    def infer(self,
              no_samples=50,
              convergence_window=15,
              convergence_eps=5e-3,
              burn_in_min=30,
              burn_in_max=2000,
              print_step=10,
              fix_lbda_iters=5):
        """
        Infer matrices and parameters, starting with burn-in and subsequent
        sampling phase.
        """

        # create list of matrices to draw samples from
        mats = [mat for mat in self.members if not mat.fixed]
        # sort from large to small, does it affect convergence?
        # mats = sorted(mats, key=lambda x: x.val.shape[0], reverse=True) TODO

        # list of parameters to be updated
        lbdas = [lbda for lbda in self.lbdas if not lbda.fixed]

        for mat in mats:
            mat.sampling_fct = mat_wrappers.get_sampling_fct(mat)
        for lbda in lbdas:
            lbda.sampling_fct = lbda_wrappers.get_update_fct(lbda)
        # assign sampling function to each matrix and parameters
        # sampling functions are provided it wrappers.py and wrap Cython code
        # import pdb; pdb.set_trace()
        # for thing_to_update in mats + lbdas:
        #     if not thing_to_update.sampling_fct:
        #         print('Oo')
        #         thing_to_update.set_sampling_fct()

        # ascertain sure all trace indices are zero
        for mat in mats:
            mat.trace_index = 0
        for lbda in lbdas:
            if lbda is not None:
                lbda.trace_index = 0

        # burn in markov chain
        print('burning in markov chain...')
        self.burn_in(mats,
                     lbdas,
                     eps=convergence_eps,
                     convergence_window=convergence_window,
                     burn_in_min=burn_in_min,
                     burn_in_max=burn_in_max,
                     print_step=print_step,
                     fix_lbda_iters=fix_lbda_iters)

        # allocate memory to save samples
        print('allocating memory to save samples...')
        for mat in mats:
            mat.allocate_trace_arrays(no_samples)
        for lbda in lbdas:
            lbda.allocate_trace_arrays(no_samples)

        print('drawing samples...')
        for sampling_iter in range(1, no_samples + 1):

            # sample mats and write to trace # shuffle(mats)
            [mat.sampling_fct(mat) for mat in np.random.permutation(mats)]
            [mat.update_trace() for mat in mats]

            # sample lbdas and write to trace
            [lbda.sampling_fct(lbda) for lbda in lbdas]
            [lbda.update_trace() for lbda in lbdas]

            if sampling_iter % print_step == 0:
                print('\r\t' + 'iteration ' +
                      str(sampling_iter) +
                      '; recon acc.: ' +
                      '\t--\t'.join([x.print_value() for x in lbdas]),
                      end='')

        # some sanity checks
        for layer in self.layers:
            if False:  # and layer.noise_model == 'max-link':
                if layer.u.row_densities is not None:
                    assert np.all(layer.u.row_densities ==
                                  np.sum(layer.u() == 1, 1))
                if layer.u.col_densities is not None:
                    assert np.all(layer.u.row_densities ==
                                  np.sum(layer.u() == 1, 0))
                if layer.z.row_densities is not None:
                    assert np.all(layer.z.row_densities ==
                                  np.sum(layer.z() == 1, 1))
                if layer.z.col_densities is not None:
                    assert np.all(layer.z.row_densities ==
                                  np.sum(layer.z() == 1, 0))

        # set all parameters to MAP estimate
        # [mat.set_to_map() for mat in mats]
        # [lbda.update() for lbda in lbdas]

        print('\nfinished.')
