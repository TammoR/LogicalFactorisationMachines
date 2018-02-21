#!/usr/bin/env python
"""
MaxMachine

wrappers for easy calls to cython functions.
The fowllowing functions are assigned to instances
of MachineMatrix, to allow for simple update calls
of the form:
sampling_function(matrix)
"""

import numpy as np
from IPython.core.debugger import Tracer
import warnings

import lom._cython.matrix_updates as cf
import lom._cython.tensor_updates as cf_tensorm


def draw_balanced_or(mat):
    """
    mat() and child need to share their first dimension. otherwise transpose.
    """
    if '2' in mat.role:
        child = mat.child().transpose()
    else:
        child = mat.child()
    
    cf.draw_balanced_or(
        mat(),
        mat.sibling(),
        child,
        mat.lbda(),
        mat.lbda()*(1/mat.lbda.balance_factor)
        )


def draw_tensorm_noparents_onechild_wrapper(mat):

    # this can be done more nicely, and also for simple OrM 
    # to improve code reusage! TODO
    mat_idx = int(mat.role[-1])-1 # integer indicating (z,u,v)
    siblings = [x for i,x in enumerate(mat.lbda.attached_matrices) if i != mat_idx]
    # transpose data such that dimensions agree with parent matrices
    transpose_order = (mat_idx, 
                      int(siblings[0].role[-1])-1,
                      int(siblings[1].role[-1])-1)

    cf_tensorm.draw_tensorm_noparents_onechild(
        mat(),
        siblings[0](),
        siblings[1](),
        mat.child().transpose(transpose_order),
        mat.lbda(),
        mat.logit_prior
        )

def draw_tensorm_indp_noparents_onechild_wrapper(mat):

    # this can be done more nicely, and also for simple OrM 
    # to improve code reusage! TODO
    mat_idx = int(mat.role[-1])-1 # integer indicating (z,u,v)
    siblings = [x for i,x in enumerate(mat.lbda[0].attached_matrices) if i != mat_idx]
    # transpose data such that dimensions agree with parent matrices
    transpose_order = (mat_idx, 
                      int(siblings[0].role[-1])-1,
                      int(siblings[1].role[-1])-1)

    cf_tensorm.draw_tensorm_indp_noparents_onechild(
        mat(),
        siblings[0](),
        siblings[1](),
        mat.child().transpose(transpose_order),
        mat.lbda[0](),
        mat.lbda[1]()
        )


def draw_z_noparents_onechild_wrapper(mat):

    cf.draw_noparents_onechild(
        mat(),  # NxD
        mat.sibling(),  # sibling u: D x Lc
        mat.child(),  # child observation: N x Lc
        mat.lbda(),  # own parameter: double
        mat.sampling_indicator)
    
def draw_u_noparents_onechild_wrapper(mat):
    cf.draw_noparents_onechild(
        mat(), # NxD
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.sampling_indicator)

def draw_u_noparents_onechild_wrapper_single_thread(mat):
    cf.draw_noparents_onechild_single_thread(
        mat(), # NxD
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.sampling_indicator)    

def draw_z_oneparent_nochild_wrapper(mat):

    cf.draw_oneparent_nochild(
        mat(), # NxD
        mat.parents[0](), # parents obs: N x Lp
        mat.parents[1](), # parents feat: D x Lp
        mat.parents[0].lbda(), # parent lbdas:
        mat.prior_config,
        mat.sampling_indicator)
    
def draw_u_oneparent_nochild_wrapper(mat):
    
    cf.draw_oneparent_nochild(
        mat(), # NxD
        mat.parents[1](), # parents obs: N x Lp
        mat.parents[0](), # parents feat: D x Lp
        mat.parents[0].lbda(), # parent lbdas: K (KxL for MaxM)
        mat.prior_config,        
        mat.sampling_indicator)    
        
def draw_z_twoparents_nochild_wrapper(mat):

    cf.draw_twoparents_nochild(
        mat(), # NxD
        mat.parent_layers[0].z(), # parents obs: N x Lp
        mat.parent_layers[0].u(), # parents feat: D x Lp
        mat.parent_layers[0].u.lbda(), # parent lbda
        mat.parent_layers[1].z(), # parents obs: N x Lp
        mat.parent_layers[1].u(), # parents feat: D x Lp
        mat.parent_layers[1].u.lbda(), # parent lbda
        mat.prior_config,        
        mat.sampling_indicator)
    
def draw_u_twoparents_nochild_wrapper(mat):

    cf.draw_twoparents_nochild(
        mat(), # NxD
        mat.parent_layers[0].u(), # parents obs: N x Lp
        mat.parent_layers[0].z(), # parents feat: D x Lp
        mat.parent_layers[0].u.lbda(), # parent lbda
        mat.parent_layers[1].u(), # parents obs: N x Lp
        mat.parent_layers[1].z(), # parents feat: D x Lp
        mat.parent_layers[1].u.lbda(), # parent lbda
        mat.prior_config,        
        mat.sampling_indicator)        


def draw_z_oneparent_onechild_wrapper(mat):
    cf.draw_oneparent_onechild(
        mat(), # N x D
        mat.parents[0](), # parent obs: N x Lp
        mat.parents[1](), # parent features, D x Lp
        mat.parents[1].lbda(), # parent lbda
        mat.sibling(), # sibling u: D x Lc
        mat.child(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.prior_config,
        mat.sampling_indicator)

def draw_u_oneparent_onechild_wrapper(mat):
    cf.draw_oneparent_onechild(
        mat(), # NxD
        mat.parents[1](), # parent obs: N x Lp
        mat.parents[0](), # parent features, D x Lp
        mat.parents[1].lbda(), # parent lbda
        mat.sibling(), # sibling u: D x Lc
        mat.child().transpose(), # child observation: N x Lc
        mat.lbda(), # own parameter: double
        mat.prior_config,
        mat.sampling_indicator)

def draw_u_noparents_onechild_maxmachine(mat):

    # provide order of dimensions by assoicated noise
    l_statistic = mat.layer.lbda()[:-1]    
    l_order = np.array(np.argsort(-l_statistic), dtype=np.int32)
    
    cf.draw_noparents_onechild_maxmachine(
        mat(), 
        mat.sibling(),
        mat.child().transpose(),
        mat.lbda(),
        l_order,
        mat.prior_config,
        mat.layer.lbda_ratios)


def draw_z_noparents_onechild_maxmachine(mat):
    
    # provide order of dimensions by assoicated noise    
    l_order = np.array(np.argsort(-mat.layer.lbda()[:-1]), dtype=np.int32)
    
    cf.draw_noparents_onechild_maxmachine(
        mat(),
        mat.sibling(),
        mat.child(),
        mat.lbda(),
        l_order,
        mat.prior_config,
        mat.layer.lbda_ratios)
    
            
def draw_u_oneparent_onechild_maxmachine(mat):
    l_order = np.array(np.argsort(-mat.layer.lbda()[:-1]), dtype=np.int32)
    l_order_pa = np.array(np.argsort(-mat.parent_layers[0].lbda()[:-1]), dtype=np.int32)
    
    cf.draw_oneparent_onechild_maxmachine(
        mat(),
        mat.sibling(),
        mat.child().transpose,
        mat.lbda(),
        l_order,
        mat.prior_config,
        mat.parents[0](),
        mat.parents[1](),
        lib.logit(mat.parent_layers[0].lbda()), # TODO compute logit inside function
        l_order_pa,
        mat.layer.lbda_ratios)    
            
        
def draw_z_oneparent_onechild_maxmachine(mat):
    l_order = np.array(np.argsort(-mat.layer.lbda()[:-1]), dtype=np.int32)
    l_order_pa = np.array(np.argsort(-mat.parent_layers[0].lbda()[:-1]), dtype=np.int32)
    
    cf.draw_oneparent_onechild_maxmachine(
        mat(),
        mat.sibling(),
        mat.child(),
        mat.lbda(),
        l_order,
        mat.prior_config,
        mat.parents[1](),
        mat.parents[0](),
        lib.logit(mat.parent_layers[0].lbda()), # TODO compute logit inside function
        l_order_pa,
        mat.layer.lbda_ratios)
