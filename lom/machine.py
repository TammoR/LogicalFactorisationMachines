#!/usr/bin/env python
"""
Logical Operate Machine

This module implements classes for sampling form hierarchical binary
matrix factorisation models.

A machine consists of multiple matrices that have mutual
relationship akin to nodes in a graphical model.

The minimal example is a standard matrix factorisation model
with a data matrix 'data', and its two parents 'z' (objects x latent) and
'u' (features x latent). 'z' and 'u' are siblings and part of layer,
'data' is the layer's child.
All matrices are instances of the MachineMatrix class and expose their
family relationes as attributes, e.g.: z.layer.child == data;
data.parents == [u,z], etc.

Each group of siblings is combined into layer
(instances of MachineLayer), together with an additional set of
parameters 'lbda' (instances of MachineParameter).

During inference each matrix can be held fixed or can be
sampled element-wise according to its full conditional probability.

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
- Need simple ways of accessing all children etc in well defined order
- layers and matrices should have names e.g. orm.layers[0].z.name = 'z'
"""

import numpy as np
from numpy.random import binomial
import lom.auxiliary_functions as lib
import lom.auxiliary_functions as aux

import lom.matrix_update_wrappers as wrappers
import lom.matrix_update_wrappers as mat_wrappers

import lom.lambda_update_wrappers as sampling
import lom.lambda_update_wrappers as lbda_wrappers

import lom._numba.lambda_updates_numba as lambda_updates_numba
import lom._numba.lom_outputs as lom_outputs
import lom._numba.lom_outputs_fuzzy as lom_outputs_fuzzy

import lom._numba.max_machine_sampler as mm


from numba import prange, jit

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

    def allocate_trace_arrays_IBP(self, no_of_samples):
        no_of_samples = int(no_of_samples)
        if type(self.val) == np.ndarray:

            self.trace = np.full([no_of_samples] +
                                 [self.val.shape[0]] +
                                 [self.val.shape[1] + 10],
                                 dtype=self().dtype, fill_value=-1)

    def update_trace_IBP(self):
        self.trace[self.trace_index,
                   :self.val.shape[0],
                   :self.val.shape[1]] = self.val
        self.trace_index += 1

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
        self.beta_prior = (1, 1)

    def print_value(self):
        #  TODO: clean up
        if self.layer.model == 'MAX-AND':
            return '\t'.join([str("%.1f" % round(100 * x, 2))
                              for x in self.val])
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
        self.bernoulli_prior = 0.5
        if val is not None:
            shape = val.shape

        # assign value if provided, otherwise bernoulli random
        if type(val) is np.ndarray:
            # avoid creation of new array.
            self.val = val  # np.array(val, dtype=np.int8)
        elif type(val) is float:
            self.val = 2 * np.array(np.random.rand(*shape) > val,
                                    dtype=np.int8) - 1
        else:
            self.val = 2 * np.array(np.random.rand(*shape) > .5,
                                    dtype=np.int8) - 1

        # fix the full matrix
        if type(fixed) is np.ndarray:
            assert fixed.shape == shape
        self.fixed = fixed

        # fix some matrix entries
        self.fixed_entries = np.zeros(self().shape, dtype=np.int8)

        # initialise layer to None
        self.layer = None

    def __call__(self):
        return self.val

    def show(self, technique='mean'):
        if technique == 'mean':
            aux.plot_matrix(self.mean())
        elif technique == 'map':
            aux.plot_matrix(self.mean() > .5)
        elif technique == 'state':
            aux.plot_matrix(self())
        else:
            raise ValueError('invalid technique')

    @property
    def model(self):
        """
        return the model of the corresponding layer.
        """
        if 'layer' in self.__dict__.keys() and self.layer is not None:
            return self.layer.model
        else:
            return None

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

        for factor in factors:
            factor.layer = self

        self.auto_clean = False
        self.auto_reset = False  # TODO get rid of

        self.prediction = None  # allow lazy computation of output.

        if model == 'MAX-AND':
            mm.compute_lbda_ratios(self)

        if model == 'OR-AND-IBP':
            self.alpha = None  # poisson hyperparameter for IBP

        if model == 'qL-AND':
            self.q = MachineMatrix(shape=(1, child().shape[1]), child_axis=1)
            self.q.val[:] = 1
            self.q.layer = self
            self.gamma = 1  # poissons hyperparmeters for q_d

    @property
    def z(self):
        return self.factors[0]

    @property
    def u(self):
        return self.factors[1]

    @property
    def v(self):
        return self.factors[2]

    @property
    def size(self):
        return self.factors[0]().shape[1]

    @property
    def L(self):
        return self.size

    @property
    def dimensionality(self):
        return len(self.factors)

    def __iter__(self):
        return iter(self.factors)

    def __repr__(self):
        return (self.model + '-' + str(len(self.factors)) +
                'D').replace('-', '_')

    def output(self,
               technique='factor_map',
               noisy_emission=False,
               lazy=False,
               map_to_probabilities=True):
        """
        Compute output matrix from posterior samples.
        Valid techniques are:
            - 'point_estimate'
                output of the current state of factors
            - 'MC' TODO
                'probabilistic output from the MC trace'
            - 'Factor-MAP' TODO
                From the posterior MAP of factors
            - 'Factor-MEAN'
                Computed from posterior mean of factors
        TODO: compute this in a lazy fashion
        Note, that outputs are always probabilities in (0,1)
        """

        # return precomputed value
        if type(self.prediction) is np.ndarray and lazy is True:
            print('returning previously computed value ' +
                  'under disregard of technique.')
            return self.prediction

        reset_name_to_IBP = False
        if self.model == 'OR-AND-IBP':
            reset_name_to_IBP = True
            self.model = 'OR-AND'

        # otherwise compute
        if self.model == 'MAX-AND':
            if technique == 'point_estimate':
                out = lom_outputs.MAX_AND_product_2d(
                    [x() for x in self.factors], self.lbda())
            elif technique == 'factor_map':
                out = lom_outputs.MAX_AND_product_2d(
                    [np.array(2 * (x.mean() > 0) - 1, dtype=np.int8)
                        for x in self.factors], self.lbda())
            elif technique == 'mc':
                out = np.zeros([x().shape[0] for x in self.factors])
                for t in range(self.lbda.trace.shape[0]):
                    out += lom_outputs.MAX_AND_product_2d(
                        [x.trace[t, :] for x in self.factors],
                        self.lbda.trace[t])
                out /= self.lbda.trace.shape[0]
            elif technique == 'factor_mean':
                out = lom_outputs_fuzzy.MAX_AND_product_fuzzy(
                    .5 * (self.z.mean() + 1),
                    .5 * (self.u.mean() + 1),
                    self.lbda.mean())

        elif self.model == 'qL-AND':
            N, D = self.child().shape
            out = np.zeros([N, D])
            for n in range(N):
                for d in range(D):
                    # TODO: Use MAP estimate
                    out[n, d] = lom_outputs.qL_AND_product(
                        self.u()[d, :],
                        self.z()[n, :],
                        self.q()[0, d])

        else:
            if technique == 'point_estimate':
                out = aux.lom_generate_data_fast(
                    [x() for x in self.factors], self.model)
                out = (1 + out) * .5  # map to probability of emitting a 1

            elif technique == 'factor_map':
                out = aux.lom_generate_data_fast(
                    [2 * (x.mean() > 0) - 1 for x in self.factors],
                    self.model)
                out = np.array(out == 1, dtype=np.int8)  # map to probability of emitting a 1

            elif technique == 'factor_mean':
                # output does not need to be mapped to probabilities
                out = aux.lom_generate_data_fast(
                    [(x.mean() + 1) * .5 for x in self.factors],  # map to (0,1)
                    self.model,
                    fuzzy=True)

            elif technique == 'factor_mean_old':
                out = aux.lom_generate_data_fuzzy(
                    [x.mean() for x in self.factors],
                    self.model)

            elif technique == 'mc':  # TODO numba
                out = np.zeros([x().shape[0] for x in self.factors])

                for t in range(self.lbda.trace.shape[0]):
                    out += aux.lom_generate_data_fast([x.trace[t, :]
                                                       for x in self.factors],
                                                      self.model)
                out /= self.lbda.trace.shape[0]
                out = (1 + out) * .5  # map to probability of emitting a 1

            # convert to probability of generating a 1
            if noisy_emission is True:
                out = out * aux.expit(self.lbda.mean()) +\
                    (1 - out) * aux.expit(-self.lbda.mean())

        self.prediction = out

        if reset_name_to_IBP is True:
            self.model = 'OR-AND-IBP'

        if map_to_probabilities is True:
            return out
        else:
            return 2 * out - 1


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
        self.anneal = False

    @property
    def members(self):
        """
        Return all matrices from within and outside of layers
        """
        single_mats = self.matrices
        layer_mats = [f for layer in self.layers for f in layer.factors]

        # q_counts for qL-AND model
        q_mat = [layer.q for layer in self.layers
                 if layer.model == 'qL-AND']

        return layer_mats + single_mats + q_mat

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

        # Convert model to canonical representation and invert data
        # if necessary, return indicators to track inversion.
        model, invert_data, invert_factors = aux.canonise_model(model, child)

        # determine size of all members
        if child is None and shape is not None:
            child = MachineMatrix(shape=shape)
        elif shape is None and child is not None:
            shape = child().shape
        else:
            raise ValueError("Not enough shape information provided.")

        # initialise matrices/factors (use add_matrix)
        factors = [MachineMatrix(shape=(K, latent_size),
                                 child_axis=i)
                   for i, K in enumerate(shape)]

        # initialise lambdas (don't use add_parameter)
        if model == 'MAX-AND':
            lbda_init = np.array([.8 for i in range(latent_size + 1)])
            lbda_init[-1] = .01
            # lbda_init = np.array([.91,.90,.89,.1])
        elif 'BALANCED' in model:
            lbda_init = np.array([1.0 for i in range(2)])
        else:
            lbda_init = .05
        lbda = MachineParameter(val=lbda_init)

        # initialise layer object
        layer = MachineLayer(factors, lbda, child, model)
        layer.invert_data = invert_data
        layer.invert_factors = invert_factors
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
            # update lambda
            if pre_burn_in_iter > fix_lbda_iters:
                if self.anneal is False:
                    [lbda.sampling_fct(lbda) for lbda in lbdas]

                # Anneal lambda for pre_burn_in_iter steps to
                # it's initially given value.
                elif self.anneal is True:
                    try:
                        assert fix_lbda_iters == 0
                    except:
                        raise ValueError('fix_lbda_iters should be zero for annealing.')
                    # pre-compute annealing steps
                    if pre_burn_in_iter == fix_lbda_iters + 1:
                        annealing_lbdas = [np.arange(
                            lbda() / burn_in_min,
                            lbda() + 2 * lbda() / burn_in_min,
                            lbda() / burn_in_min)
                            for lbda in lbdas]

                    for lbda_idx, lbda in enumerate(lbdas):
                        lbda.val = annealing_lbdas[lbda_idx][pre_burn_in_iter]

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
                    # check for dimensions to be removed and restart burn-in if
                    # layer.auto_clean_up is True
                    if np.any([aux.clean_up_codes(lr,
                                                  lr.auto_reset,
                                                  lr.auto_clean)
                               for lr in self.layers if
                               lr.auto_clean is True or
                               lr.auto_reset is True]):

                        for lbda in lbdas:
                            # reallocate arrays for lbda trace
                            lbda.allocate_trace_arrays(convergence_window)
                            lbda.trace_index = 0

                    else:
                        # save number of burn in iterations
                        self.burn_in_iters = burn_in_iter + pre_burn_in_iter
                        break

            # stop if maximum number of burn in iterations is reached
            if (burn_in_iter + pre_burn_in_iter) > burn_in_max:

                # clean up non-converged auto-reset dimensions
                for lr in self.layers:
                    if lr.auto_reset is True:
                        aux.clean_up_codes(lr, reset=False, clean=True)

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
              fix_lbda_iters=0):
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
            if mat.layer.model == 'OR-AND-IBP':
                mat.allocate_trace_arrays_IBP(no_samples)
            else:
                mat.allocate_trace_arrays(no_samples)
        for lbda in lbdas:
            lbda.allocate_trace_arrays(no_samples)

        print('drawing samples...')
        for sampling_iter in range(1, no_samples + 1):

            # sample mats and write to trace # shuffle(mats)
            [mat.sampling_fct(mat) for mat in np.random.permutation(mats)]

            if np.any([x.model == 'OR-AND-IBP' for x in mats]):
                [mat.update_trace_IBP() for mat in mats]
            else:
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
