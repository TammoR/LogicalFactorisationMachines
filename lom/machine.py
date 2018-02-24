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
    z.set_prior ( 'binomial', .5 ) 

    # binomial prior across rows of z
    z.set_prior ( 'bernoulli', .5, axis = 0 )

    # binomial prior across columns of z, with K draws to enforce sparsity.
    z.set_prior ( 'bernoulli', [.5, K] , axis = 1 )

For the MaxMachine, a beta prior can be specified on the dispersion parameter, e.g.
    # set beta(1,1) prior
    layer1.lbda.set_prior([1,1]) 

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

from __future__ import absolute_import, division, print_function # for python2 support
from numpy.random import binomial
import numpy as np
from random import shuffle
import lom.auxiliary_functions as lib
import lom.matrix_updates_c_wrappers as wrappers
import lom.lambda_updates_c_wrappers as sampling
import lom._cython.matrix_updates as cf
import lom._cython.tensor_updates as cf_tensorm

from IPython.core.debugger import Tracer

__author__ = "Tammo Rukat"
__status__ = "Development"


class Trace():
    """
    abstract base class implementing methods posterior traces arrays.
    Inherited to MachineMatrix and MachineParameter. TODO MachineMatrix and MachineParameter
    should be instances of the same class (?)
    """
    
    def __call__(self):
        return self.val
    

    def allocate_trace_arrays(self, no_of_samples):
        no_of_samples = int(no_of_samples)
        if type(self.val) == np.ndarray:
            # nicer but no python2 compatible
            # self.trace = np.empty([no_of_samples, *self.val.shape], dtype=np.int8)

            self.trace = np.empty([no_of_samples]+[x for x in self.val.shape], dtype=self().dtype)
        else: # not needed (?)
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
            return np.all([lib.check_convergence_single_trace(self.trace[:,l], eps)
                           for l in range(self.trace.shape[1])])

        else:
            raise IndexError("Can not ascertain convergence of a matrix with more than 2 dimensions")


    def set_sampling_fct(self, sampling_fct=None):
        """
        Provide a sampling function for any parameter or matrix
        or infer automatically.
        Both, MachineMatrix and MachineParameter inherit from Trace.
        """

        # sampling function already defined
        if self.sampling_fct is not None:
            return
        
        # user provides function
        if sampling_fct is not None:
            self.sampling_fct = sampling_fct

        # otherwise infer it
        else:
            if self.__class__.__name__ == 'MachineMatrix':
                sampling.infer_sampling_fct_mat(self)
            elif self.__class__.__name__ == 'MachineParameter':
                sampling.infer_sampling_fct_lbda(self)

            
class MachineParameter(Trace):
    """
    Base class for parameters
    """
    
    def __init__(self, val, attached_matrices=None, 
                 sampling_indicator=True, noise_model='or-link'):
        self.trace_index = 0
        self.sampling_fct = None
        self.val = val
        self.attached_matrices = attached_matrices
        self.correct_role_order() # TODO solve this more elegantly
        # make sure they are a tuple in order (observations/features)
        self.sampling_indicator = sampling_indicator
        self.noise_model = noise_model
        
        # assign matrices the parameter
        for mat in attached_matrices:
            mat.lbda = self
            # if not hasattr(mat, 'lbdas'):
            #     mat.lbdas = [self]
            # else:
            #     mat.lbdas.append(self)
        self.set_prior()


    def print_value(self):
        if self.noise_model in ['or-link', 'tensorm-link','tensorm-link-indp','balanced-or']:
            return ', '.join([str(round(lib.expit(np.mean(x)),3)) for x in [self.val]])
        elif self.noise_model == 'independent':
            return ', '.join([str(round(lib.expit(x),3)) for x in self.val])
        elif self.noise_model == 'max-link':
            return '\t'.join([str("%.1f"%round(100*x,2)) for x in self.val])
        else:
            raise SystemError
            #return ', '.join([str(str.format('{0:3f}',x)) for x in self.val])

            
    def set_prior(self, prior_parms=None, clamped_unit_prior_parms=None):
        """
        In analaogy to the set_prior method of machine_matrix,
        initialise all the necessary prior properties for posterior sampling.
        Only option here is a beta prior.
        Creates a property for the MachineParameter called .prior_config:
        list of [prior_type, [alpha, beta], [alpha_c, beta_c]] = [int, [float, float], [float, float]] 
        with
        prior_type = 0 -> Flat prior
        prior_type = 1 -> beta prior with parameters [alpha, beta] and [alpha_c, beta_c]
            on the clamped unit.
        """

        if (((prior_parms == [1,1]) or (prior_parms is None)) and
           ((clamped_unit_prior_parms == [1,1]) or (clamped_unit_prior_parms is None))):
            prior_type = 0
            self.prior_config = [0, None, None]
        else:
            prior_type = 1
            if prior_parms is None:
                prior_parms = [1,1]
            if clamped_unit_prior_parms is None:
                clamped_unit_prior_parms = prior_parms
            
            self.prior_config = [1, prior_parms, clamped_unit_prior_parms]

            
    def correct_role_order(self):
        """
        the assigned matrices have to be ordered (observations,features),
        i.e. (z,u)
        """
        roles = [x.role for x in self.attached_matrices]
        role_sort_idx = [i[0] for i in sorted(enumerate(roles), key=lambda x:x[1])]
        self.attached_matrices = [self.attached_matrices[i] for i in role_sort_idx]

        # import pdb; pdb.set_trace()
        # if self.attached_matrices[0].role == 'dim2'\
        # and self.attached_matrices[1].role == 'dim1':
        #     self.attached_matrices = (self.attached_matrices[1],self.attached_matrices[0])
        #     print('swapped role order')
        # elif self.attached_matrices[0].role == 'dim1'\
        # and self.attached_matrices[1].role == 'dim2':
        #     pass
        # else:
        #     raise ValueError('something is wrong with the role assignments')

            
class MachineMatrix(Trace):
    """
    base class for matrices
    """

    def __init__(self, val=None, shape=None, sibling=None, 
                 parents=None, child=None, lbda=None, p_init=.5,
                 role=None, sampling_indicator = True, parent_layers = None,
                 child_axis = None):
        """
        role (str): 'features (dim2)' or 'observations (dim1)' or 'data'. Try to infer if not provided
        """

        self.trace_index = 0
        self.sampling_fct = None
        
        # never use empty lists as default arguments. bad things will happen
        if parents is None:
            parents = []
        if parent_layers is None:
            parent_layers = []
        self.parent_layers = parent_layers

        # assign family
        self.parents = parents
        self.child = child
        self.sibling = sibling
        self.lbda = lbda
        self.child_axis = child_axis
        
        # ascertain that we have enough information to initiliase the matrix
        assert (val is not None) or (shape is not None and p_init is not None)

        # if value is given, assign
        if val is not None:
            if val.dtype != np.int8:
                self.val = np.array(val, dtype=np.int8)
            else:
                self.val = val

        # otherwise, if p_init is a matrix, assign it as value
        elif type(p_init) is np.ndarray:
            self.val = np.array(p_init, dtype=np.int8)            

        # otherwise, initialise iid random with p_init
        else:
            self.val = 2*np.array(binomial(n=1, p=p_init, size=shape), dtype=np.int8)-1

        # sanity check on input data
        if np.prod(self.val.shape) < 1e8:
            uniques = np.unique(self.val)
            assert np.all([y in [-1,0,1] for y in uniques])
            if 0 in uniques:
                print('Data contains zeros. Zeros are treated as missing values, while dat is coded as {-1, 1}.')     
        else:
            print('Skip checking of unique data point values due to size')

        self.family_setup()
        
        if role is None:
            self.infer_role()
        else:
            self.role = role
            
        # sampling indicator is boolean matrix of the same size, type=np.int8
        # of False is the whole matrix is not sampled.
        self.set_sampling_indicator(sampling_indicator)

        # initialise self.prior_config, which encapsulates
        # all information about the prior
        self.prior_config = [None for i in range(6)]
        self.set_prior()

    @property
    def get_siblings(self):

        siblings = [f for f in self.child.parents if f is not self]

        return sorted(siblings, key=lambda f: f.child_axis)

            
    def add_parent_layer(self, layer):
        if self.parent_layers is None:
            self.parent_layers = [layer]
        else:
            self.parent_layers.append(layer)
        

    def set_sampling_indicator(self, sampling_indicator):
        """
        matrix of same size, indiactor for every value whether it stays fixed
        or is updated. Could hardcode this, if all or no values are samples
        but speed advantage is negligible.
        """
        if sampling_indicator is True:
            self.sampling_indicator = np.ones(self().shape, dtype=np.int8)
        elif sampling_indicator is False:
            self.sampling_indicator = False
        elif type(sampling_indicator) is np.ndarray:
            assert sampling_indicator.shape == self().shape
            self.sampling_indicator = np.array(sampling_indicator, dtype=np.int8)

            
    def set_prior(self, prior_type=None, prior_parms=None, axis=None, prior_code=None):
        """
        prior_type (str): 'bernoulli', 'binomial', 'beta-binomial'
        prior_parms (list of mixed types):
            bernoulli: [p (float)]
            binomial: [q (float), N (int)]
            beta-bernoulli: [a (float) ,b (float), N (int)]
        Setting N different from the acutal size of the corresponding
        dimension enforces sparsit.

        the property .prior_code is
        0: no prior
        1: Bernoulli prior
        2: Binomial prior across columns
        3: Binomial prior across rows
        4: Binomial prior across rows and columns (not very reasonable)

        Every MachineMatrix exposes: .prior_type, .prior_parms, 
        and then: .row_densities, .col_densities, .binomial_prior_col, .binomial_prior_row, .bernoulli_prior
        """

        # initialise to None
        prior_code, logit_bernoulli_prior, row_binomial_prior,\
            col_binomial_prior, row_densities, col_densities = self.prior_config
        # parameters that generate the prior are kept as attribute
        # in case its needed for later updatign
        self.prior_parms = prior_parms
        self.prior_axis = axis
        
        if prior_type is None:
            prior_code = 0
        
        if (prior_type is not None) and (prior_type not in ['bernoulli', 'binomial', 'beta-binomial']):
            print('No proper prior specified. Choices are bernoulli, binomial, beta binomial.')
            print('Defaulting to flat priors')
            self.set_prior()

        if prior_type is 'bernoulli':
            prior_code = 1
            if type(prior_parms) is list:
                assert len(prior_parms) == 1
                prior_parms = prior_parms[0]
            else:
                assert type(prior_parms) is float
            logit_bernoulli_prior = lib.logit(prior_parms)

        if prior_type is 'binomial':
            # set number of draws to axis length, if not provided.
            if type(prior_parms) is list:
                if len(prior_parms) == 1:
                    prior_parms.append(self().shape[1-axis])
            elif type(prior_parms) is float:
                prior_parms = [prior_parms, self().shape[1-axis]]
            if axis is None:
                print('No axis for binomial prior provided. Defaulting to 0')
                axis = 0

            # set counts, summing the ones across rows/columns
            if axis == 0: # prior on each row, sum across columns
                if prior_code == 3:
                    prior_code = 4
                else:
                    prior_code = 2
                    
                row_densities = np.array(np.sum(self()==1, 1), dtype=np.int32)
                assert len(row_densities) == self().shape[0]
                # need to add +1 to also count all zero state.
                row_binomial_prior = lib.compute_bp(prior_parms[0], prior_parms[1]+1, self().shape[1-axis]+1)
                    
            elif axis == 1:
                if prior_code == 2:
                    prior_code = 4
                else:
                    prior_code = 3
                col_densities = np.array(np.sum(self()==1, 0), dtype=np.int32)
                assert len(col_densities) == self().shape[1]
                col_binomial_prior = lib.compute_bp(prior_parms[0], prior_parms[1]+1, self().shape[1-axis]+1)

            elif prior_type is 'beta-binomial':
                print('beta-binomial not yet implemented.') # works just like above

        self.prior_config = [prior_code, logit_bernoulli_prior, row_binomial_prior,
                             col_binomial_prior, row_densities, col_densities]

        
    def update_prior_config(self):
        """
        If the model dimensions change, binomial priors need to be updated.
        """
        # No updating for bernoulli priors
        if self.prior_config[0] < 2:
            return
        if (self.prior_config[0] == 2) or (self.prior_config == 4):
            row_densities = np.array(np.sum(self()==1, 1), dtype=np.int32)
            assert len(row_densities) == self().shape[0]
            row_binomial_prior = lib.compute_bp(self.prior_parms[0], self.prior_parms[1]+1,
                                                self().shape[1-self.prior_axis]+1)
            self.prior_config[2] = row_binomial_prior
            self.prior_config[4] = row_densities
            
        if (self.prior_config == 3)  or (self.prior_config == 4):
            col_densities = np.array(np.sum(self()==1, 0), dtype=np.int32)
            assert len(self.col_densities) == self().shape[1]
            col_binomial_prior = lib.compute_bp(self.prior_parms[0], self.prior_parms[1]+1,
                                                self().shape[1-self.prior_axis]+1)
            self.prior_config[3] = col_binomial_prior
            self.prior_config[5] = col_densities
            
 
    def family_setup(self):
        """
        Set family relationship of relatives
        every parent has only one child (no multiview), a child can have many parents.
        Every sibling has only one sibling
        """
        # register as child of all my parents
        if self.parents is not None:
            for parent in self.parents:
                # check whether tere is another child registered
                assert (parent.child is not None and parent.child != self)
                parent.child = self

        # register as parent of all my children
        if (self.child is not None) and (self not in self.child.parents):

            self.child.parents.append(self)

        # register as sibling of all my siblings
        if (self.sibling is not None) and (self != self.sibling.sibling):
            self.sibling.sibling = self
            
        # register as attached to my parameter
        if self.lbda is not None:
            if self.lbda.attached_matrices is None:
                self.lbda.attached_matrices = [self]
            elif type(self.lbda.attached_matrices == list) and (self not in self.lbda.attached_matrices):
                self.lbda.attached_matrices.append(self)

                
    def infer_role(self):
        """
        based on the dimensionality, infer whether self is a
        observation (z) or a feature matrix (u) or a data matrix (x)
        """
        
        # print('infer roles of matrix (data/observations/features)')
        
        if self.child is None:
            self.role = 'dim1'
            return 

        # if dimensions N=D, role can't be inferred
        if self.child().shape[0] == self.child().shape[1]:
            raise ValueError('Data dimensions are the same. Can not infer parents role.')
        
        if self().shape[0] == self.child().shape[0]:
            self.role = 'dim1'
        elif self().shape[0] == self.child().shape[1]:
            self.role = 'dim2'
        else:
            raise ValueError('dimension mismatch')

        
    def set_to_map(self):
        self.val = np.array(self.mean()>0, dtype=np.int8)
        self.val[self.val==0] = -1

            
class MachineLayer():
    """
    Essentially a container for (z,u,lbda) to allow
    convenient access to different layers for the sampling routine
    and the user.
    """
    
    def __init__(self, z, u, lbda, size, child, noise_model, v=None):
        # nice ways to solve the unknown number of matrices,
        # provide list or so. Can also avoid strange ('dim1','dim2',...) labels.
        # TODO
        self.z = z
        self.u = u
        if v is not None:
            self. v = v

        #self.lbdas = lbdas
        #for lbda in lbdas:
        #    lbda.layer = self

        if noise_model == 'tensorm-link-indp':
            self.lbda_p = lbda[0]
            self.lbda_m = lbda[1]
            self.lbda_p.layer = self
            self.lbda_m.layer = self
            self.z.lbda = (self.lbda_p, self.lbda_m)
            self.u.lbda = (self.lbda_p, self.lbda_m)
            self.v.lbda = (self.lbda_p, self.lbda_m)
        else:
            self.lbda = lbda
            self.lbda.layer = self

        self.size = size
        # register as layer of members
        self.z.layer = self
        self.u.layer = self
        # self.lbdas.layer = self # not working in independent noise implementation
        self.child = child
        self.child.add_parent_layer(self)
        self.noise_model = noise_model
        # Keep track of whehter P (OrM) or TP, FN, etc need updating
        self.auto_clean_up = False
        self.auto_reset = False
        self.precomputed_output = None
        self.eigenvals = None

        if noise_model == 'max-link':
            self.precompute_lbda_ratios()
        
        if 'independent' in noise_model or True: # we can compute predict accuracies for the original model, too.
            self.predictive_accuracy_updated = False # switch to update only if needed
            self.pred_rates = np.array([0,0,0,0], dtype=np.int) # TP, FP, TN, FN
            # fraction of ones in the data for unbiased inference of 0/1
            if 'unbias' in noise_model:
                self.child.log_bias = np.log(float(np.sum(self.child()==1))/np.sum(self.child()==-1))
            else:
                self.child.log_bias = 0

        if noise_model == 'balanced-or':
            self.pred_rates = np.array([0,0,0,0], dtype=np.int) # TP, FP, TN, FN            

                
    def __call__(self):
        return(self.z(), self.u(), self.lbda())

    
    def members(self):
        return [self.z, self.u]

    
    def child(self):
        assert self.z.child is self.u.child
        return self.z.child

    
    def output(self, u=None, z=None, v=None, recon_model='mc', force_computation=False):
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
                    x, .5*(u+1), .5*(z+1), self.lbda.mean(), D, N, L)

            elif recon_model == 'mc':
                from scipy.special import expit
                print('Computing MC estimate of data reconstruction')
                u_tr = self.u.trace
                z_tr = self.z.trace
                lbda_tr = self.lbda.trace
                trace_len = u_tr.shape[0]            
                x = np.zeros([z_tr.shape[1],u_tr.shape[1]])

                for tr_idx in range(len(lbda_tr)):
                    det_prod = (np.dot(u_tr[tr_idx,:,:]==1,z_tr[tr_idx,:,:].transpose()==1)).transpose()
                    x[det_prod==1] += expit(lbda_tr[tr_idx])
                    x[det_prod==0] += 1-expit(lbda_tr[tr_idx])
                x /= float(trace_len)

        elif self.noise_model == 'tensorm-link':
            if v is None:
                v = self.v.mean()

            M = v.shape[0]

            if recon_model == 'plugin':
                x = np.zeros((N, D, M), dtype=np.float32)
                print('Computing tensorm plugin reconstruction.')
                cf_tensorm.probabilistic_output_tensorm(
                    x, .5*(z+1), .5*(u+1), .5*(v+1), self.lbda.mean())

            elif recon_model == 'mc':
                x = np.zeros((N, D, M), dtype=np.float32)
                from scipy.special import expit
                print('Computing MC estimate of data reconstruction')
                z_tr = self.z.trace
                u_tr = self.u.trace
                v_tr = self.v.trace
                lbda_tr = self.lbda.trace
                trace_len = u_tr.shape[0]            
                x = np.zeros([z_tr.shape[1],u_tr.shape[1],v_tr.shape[1]])

                for tr_idx in range(len(lbda_tr)):
                    det_prod = lib.boolean_tensor_product(
                        z_tr[tr_idx,:,:],
                        u_tr[tr_idx,:,:],
                        v_tr[tr_idx,:,:])
                    x[det_prod==1] += expit(lbda_tr[tr_idx])
                    x[det_prod==0] += 1-expit(lbda_tr[tr_idx])
                x /= float(trace_len)

            elif recon_model == 'map':
                x = np.zeros((N, D, M), dtype=bool)
                for n in range(N):
                    for d in range(D):
                        for m in range(M):
                            for l in range(L):
                                if (z[n,l] > 0) and (u[d,l] > 0) and (v[m,l] > 0):
                                    x[n,d,m] = True
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
                x = np.zeros([z_tr.shape[1],u_tr.shape[1],v_tr.shape[1]])

                for tr_idx in range(len(lbda_p_tr)):
                    det_prod = lib.boolean_tensor_product(
                        z_tr[tr_idx,:,:],
                        u_tr[tr_idx,:,:],
                        v_tr[tr_idx,:,:])
                    x[det_prod==1] += expit(lbda_p_tr[tr_idx])
                    x[det_prod==0] += 1-expit(lbda_m_tr[tr_idx])
                x /= float(trace_len)

            if recon_model == 'plugin':
                print('Computing tensorm plugin reconstruction.')
                cf_tensorm.probabilistic_output_tensorm_indp(
                    x, .5*(z+1), .5*(u+1), .5*(v+1), 
                    self.lbda_p.mean(), self.lbda_m.mean())

            elif recon_model == 'map':
                f_tensorm = (self.z.mean()>0, self.u.mean()>0, self.v.mean()>0)
                x = lib.boolean_tensor_product(*f_tensorm)

        elif self.noise_model == 'independent':
            cf.probabilistc_output_indpndt(
                x, .5*(u+1), .5*(z+1), self.lbda.mean()[1], self.lbda.mean()[0], D, N, L)

        elif self.noise_model is 'maxmachine_plugin':
            x = np.empty((N, D))
            # check that the background noise is smaller than any latent dimension's noise
            if self.lbda.mean()[-1] != np.min(self.lbda.mean()):
                print('we have alphas < alpha[-1]')
            cf.probabilistic_output_maxmachine(
                x, .5*(u+1), .5*(z+1), self.lbda.mean(),
                np.zeros(len(self.lbda()), dtype=np.float64),
                np.zeros(len(self.lbda()), dtype=np.int32))
                #np.array(np.argsort(-self.lbda.mean()[:-1]), dtype=np.int32))

        elif self.noise_model == 'max-link':
            print('Computing MC estimate of data reconstruction')
            u_tr = self.u.trace
            z_tr = self.z.trace
            alpha_tr = self.lbda.trace
            x = np.zeros([N,D])
            trace_len = u_tr.shape[0]
            for tr_idx in range(trace_len):
                x += lib.maxmachine_forward_pass(u_tr[tr_idx,:,:]==1,
                                                 z_tr[tr_idx,:,:]==1,
                                                 alpha_tr[tr_idx,:])
            x /= trace_len

        else:
            raise StandardError('Output function not defined for given noise model.')

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

            return (-P*lib.logsumexp([0,-self.lbda.mean()]) -
                    (N*D-P)*lib.logsumexp([0,self.lbda.mean()]) )

        elif 'independent' in self.noise_model:
            self.update_predictive_accuracy()
            TP, FP, TN, FN = self.pred_rates

            return (-TP*lib.logsumexp([0,-self.lbda()])
                    -FP*lib.logsumexp([0,self.lbda()])
                    -TN*lib.logsumexp([0,-self.mu()])
                    -FN*lib.logsumexp([0,self.lbda()]))

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

        lbda_ratios = np.zeros([2,L,L], dtype=np.float32)

        for l1 in range(L):
            for l2 in range(l1+1):
                lratio_p = np.log(self.lbda()[l1]/self.lbda()[l2])
                lratio_m = np.log( (1-self.lbda()[l1])/ (1-self.lbda()[l2]) )
                lbda_ratios[0,l1,l2] = lratio_p
                lbda_ratios[0,l2,l1] = -lratio_p
                lbda_ratios[1,l1,l2] = lratio_m
                lbda_ratios[1,l2,l1] = -lratio_m

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
        self.members = []
        self.lbdas = []
        
        
    def add_layer(self, size=None, child=None, 
                  lbda_init=1.5, z_init='kmeans', u_init=.5, 
                  noise_model='or-link'):
        """
        Essentially wraps the necessary calls to add_parameter and add_matrix
        noise-mode: 'or-link', 'max-link'
        """

        # infer layer shape is not provided
        if (size is None) and (child is not None):
            if type(z_init) is np.ndarray:
              shape_z = (child().shape[0], z_init.shape[1])
              shape_u = (child().shape[1], z_init.shape[1])              
            elif type(u_init) is np.ndarray:
              shape_z = (child().shape[0], u_init.shape[1])
              shape_u = (child().shape[1], u_init.shape[1])              
        elif (size is not None) and (child is not None):
            shape_z = (child().shape[0], size)
            shape_u = (child().shape[1], size)    
        else:
            raise Warning('Can not infer layer size')

        # kmeans initialisation: one matrix is initialised to kmeans
        # centroids, the other to the corresponding one-hot representation
        if u_init is 'kmeans':
            try:
                from sklearn.cluster import KMeans
                assert child is not None
                km = KMeans(shape_u[1], random_state=2, n_init=10).fit(child())
                u_init=2*np.array(np.round(km.cluster_centers_ > 0),
                                  dtype=np.int8).transpose()-1
                # z_init = np.zeros(shape_z)
                # z_init[:] = -1
                # for row in range(shape_z[0]):
                #     z_init[row, kmeans.labels_[row]] = 1
                
            except:
                print('K means initialisation of u not raised exception. Resort to Bernoulli random')
                u_init = .5

        if z_init is 'kmeans':
            try:
                from sklearn.cluster import KMeans
                assert child is not None
                km = KMeans(shape_u[1], random_state=2, n_init=10).fit(child().transpose())
                z_init=2*np.array(np.round(km.cluster_centers_ > 0),
                                  dtype=np.int8).transpose()-1
                # u_init = np.zeros(shape_u)
                # u_init[:] = -1
                # print('yes')
                # for row in range(shape_u[0]):
                #     u_init[row, km.labels_[row]] = 1
            except:
                print('K means initialisation of z not possible. Resort to Bernoulli random')
                z_init = .5
            

        z = self.add_matrix(shape=shape_z, 
                            child=child, p_init=z_init,
                            role='dim1', child_axis=0)
        
        u = self.add_matrix(shape=shape_u, sibling=z, 
                            child=child, p_init=u_init,
                            role='dim2', child_axis=1)

        if 'or-link' in noise_model:
            lbda = self.add_parameter(attached_matrices=(z,u), 
                                      val=lbda_init,
                                      noise_model=noise_model)
            
        elif 'independent' in noise_model:
            print('All noise parameters are initialised with lbda_init')
            lbda = self.add_parameter(attached_matrices=(z,u),
                                      val=np.array(2*[lbda_init]),
                                      noise_model=noise_model)

        elif 'balanced-or' in noise_model:
            lbda = self.add_parameter(attached_matrices=(z,u),
                                      val=lbda_init,
                                      # val=np.array(2*[lbda_init]),
                                      noise_model=noise_model)

        elif noise_model == 'max-link': # experimental
            if type(lbda_init) is not list:
                lbda = self.add_parameter(attached_matrices=(z,u),
                                          val=np.array([lbda_init for i in range(size+1)]),
                                          noise_model=noise_model)
                lbda()[-1] = .01

            elif len(lbda_init) == size + 1:
                lbda = self.add_parameter(attached_matrices=(z,u), val=np.array(lbda_init), noise_model=noise_model)
            else:
                raise StandardError('lambda not properly initiliased for max-link')

        else:
            raise StandardError('No proper generative model specified.')

        layer = MachineLayer(z, u, lbda, size, child, noise_model)

        layer.machine = self
        self.layers.append(layer)

        return layer


    def add_tensorm_layer(
            self, 
            size = 3,
            child=None, 
            lbda_init=1.5,
            inits = [.5,.5,.5],
            priors = None,
            noise_model = 'tensorm-link'):
        """
        add layer for TensOrMachine factorizing 3D arrary into
        three 2D matrices
        Probably better to have a add_layer routine for every noise model
        instead of accepting different arguments. TODO
        noise model can be tensorm-link / tensorm-link-indp
        """

        shape_z, shape_u, shape_v = [(child().shape[i], size) for i in range(3)]

        z_init, u_init, v_init = inits
        

        u = self.add_matrix(shape=shape_u, 
                            child=child, p_init=u_init,
                            role='dim2', child_axis=1)

        z = self.add_matrix(shape=shape_z, sibling=u,
                            child=child, p_init=z_init,
                            role='dim1', child_axis=0)
        
        v = self.add_matrix(shape=shape_v, sibling=z, 
                            child=child, p_init=v_init,
                            role='dim3', child_axis=2)

        if priors is not None:
            z.logit_prior, u.logit_prior, v.logit_prior = priors
        else:
            z.logit_prior, u.logit_prior, v.logit_prior = 3*[0]

        if noise_model == 'tensorm-link':

            lbda = self.add_parameter(
                attached_matrices=(z,u,v),
                val = lbda_init,
                noise_model = noise_model)

            layer = MachineLayer(z, u, lbda, size, child, noise_model, v)


        elif noise_model == 'tensorm-link-indp':

            lbda_p = self.add_parameter(
                attached_matrices=(z,u,v),
                val = lbda_init,
                noise_model = noise_model)

            lbda_m = self.add_parameter(
                attached_matrices=(z,u,v),
                val = lbda_init,
                noise_model = noise_model)

            layer = MachineLayer(z, u, (lbda_p, lbda_m), size, child, noise_model, v)

        self.layers.append(layer)
        layer.machine = self

        for factor_matrix in layer.members()[0].child.parents:
            factor_matrix.sampling_fct = wrappers.draw_tensorm_noparents_onechild_wrapper
        layer.lbda.sampling_fct = sampling.draw_lbda_tensorm

        return layer
    

    def add_matrix(self, val=None, shape=None, sibling=None,
                   parents=None, child=None, lbda=None, p_init=.5,
                   role=None, sampling_indicator=True, child_axis=5):

        # ascertain the correct coding as [-1,0,1] integers, 
        # where 0 is unobserved.
        if val is not None and val.dtype != np.int8:
            val = np.array(val, dtype=np.int8)

        mat = MachineMatrix(val, shape, sibling, parents, child,
                            lbda, p_init, role, sampling_indicator,
                            None, child_axis)

        self.members.append(mat)
        return mat
        
        
    def add_parameter(self, val=2.0, attached_matrices=None, noise_model='or-link'):
        
        lbda = MachineParameter(val=val, attached_matrices=attached_matrices,
                                noise_model=noise_model)
        self.lbdas.append(lbda)
        return lbda    
    

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
        we have an additional pre-burn-in phase where we do not check for convergence.
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
                      str(pre_burn_in_iter+burn_in_iter) +
                      ' recon acc.: ' +
                       '\t--\t '.join([x.print_value() for x in lbdas]),
                      end='')
                  
            # check convergence every convergence_window iterations
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
                        print('\tremove duplicate or useless latent dimensions and restart burn-in. New L='
                              +str([lr.size for lr in self.layers]))
                        for lbda in lbdas:
                            # reallocate arrays for lbda trace
                            lbda.allocate_trace_arrays(convergence_window)
                            lbda.trace_index = 0

                    # alternatively, check for dimensions to be reset
                    elif np.any([lib.reset_codes(lr, lr.noise_model) 
                                 for lr in self.layers if lr.auto_reset is True]):
                        for lbda in lbdas:
                            # reallocate arrays for lbda trace
                            lbda.allocate_trace_arrays(convergence_window)
                            lbda.trace_index = 0                                

                    else:
                        self.burn_in_iters = burn_in_iter+pre_burn_in_iter # save nu of burn in iters
                        break

            # stop if max number of burn in inters is reached
            if (burn_in_iter+pre_burn_in_iter) > burn_in_max:
                print('\n\tmax burn-in iterations reached without convergence')
                # reset trace index
                for lbda in lbdas:
                    lbda.trace_index = 0
                self.burn_in_iters = burn_in_iter
                break
            
            # draw sampels # shuffle(mats)
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
        mats = [mat for mat in self.members if not mat.sampling_indicator is False]

        # sort from large to small, does it affect convergence?
        # mats = sorted(mats, key=lambda x: x.val.shape[0], reverse=True)

        # get list of parameters to sample from
        lbdas = []
        for mat in mats:
            lbdas += [mat.lbda]
            if len(mat.parents) > 0:
                lbdas += [mat.parents[0].lbda]
        # remove duplicates preserving order
        lbdas = list(lib.flatten(lbdas))
        lbdas = [x for x in lib.unique_ordered(lbdas) if x is not None]

        # assign sampling function to each matrix and parameters
        # sampling functions are provided it wrappers.py and wrap Cython code
        for thing_to_update in mats+lbdas:
            if not thing_to_update.sampling_fct:
                thing_to_update.set_sampling_fct()

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
        for sampling_iter in range(1, no_samples+1):

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
            if False: # and layer.noise_model == 'max-link':
                if layer.u.row_densities is not None:
                    assert np.all(layer.u.row_densities == np.sum(layer.u()==1,1))
                if layer.u.col_densities is not None:
                    assert np.all(layer.u.row_densities == np.sum(layer.u()==1,0))    
                if layer.z.row_densities is not None:
                    assert np.all(layer.z.row_densities == np.sum(layer.z()==1,1))
                if layer.z.col_densities is not None:
                    assert np.all(layer.z.row_densities == np.sum(layer.z()==1,0))
                    
        # set all parameters to MAP estimate
        # [mat.set_to_map() for mat in mats]
        # [lbda.update() for lbda in lbdas]
        
        print('\nfinished.')


