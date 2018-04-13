#!/usr/bin/env python
"""
LOM

Various auxiliary functions

"""
import numpy as np
import sys
import tempfile
import sklearn
import itertools
from lom._numba import lambda_updates_numba
# import lom._cython.matrix_updates as cf
from numpy.random import binomial
from numba import jit


def expit(x):
    """
    better implementation in scipy.special,
    but can avoid dependency
    """
    try:
        from scipy.special import expit
        return expit(x)
    except ModuleNotFoundError:
        return 1 / (1 + np.exp(-x))


def logit(x):
    """
    better implementation in scipy.special,
    but can avoid dependency
    """
    try:
        from scipy.special import logit
        return logit(x)
    except ModuleNotFoundError:
        return np.log(float(x) / (1 - x))


def logsumexp(a):
    """
    better implementation in scipy.special,
    but can avoid dependency
    """
    try:
        from scipy.special import logsumexp
        return logsumexp(a)
    except ModuleNotFoundError:
        a_max = np.max(a)
        out = np.log(np.sum(np.exp(a - a_max)))
        out += a_max
        return out


def compute_bp(q, n, N, tau=1):
    """
    compute list of beta-binomial logit for 1...n draws with
    beta parameters a, b. Length of output is N
    and p(n>N) = -infinity.

    n = number of draws
    q = success probability per draw
    N = size of output (output gets logit(0)-padded)
    """

    exp_bp = [(q * (n - k * tau)) / ((1 - q) * (k * tau + 1)) for k in range(n)]

    bp = [np.log(x) if (x > 0) else -np.infty for x in exp_bp]

    if N != n:
        bp_new = [-np.infty for i in range(N)]
        bp_new[:n - 1] = bp
        bp = bp_new
    return np.array(bp, dtype=float)


def compute_bbp(n, a, b):
    """
    compute list of beta-binomial logit for 1...n draws with
    beta parameters a, b.
    """
    exp_bbp = [(float((n - k) * (k + a)) /
                float((k + 1) * (n - k + b - 1))) for k in range(n + 1)]
    bbp = [np.log(x) if (x > 0) else -np.infty for x in exp_bbp]
    return np.array(bbp, dtype=float)


def unique_ordered(seq):
    """
    return unique list entries preserving order.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def clean_up_codes(layer, reset=True, clean=False):
    """
    Remove redundant or all-zero latent dimensions
    from layer and adjust all attributes accordingly.
    Return True, if any dimension was removed, False otherwise.
    """

    if reset is True:
        cleaning_action = reset_dimension
    elif clean is True:
        cleaning_action = remove_dimension

    # import pdb; pdb.set_trace()

    reduction_applied = False
    # remove inactive codes
    l = 0
    while l < layer.size:  # need to use while loop because layer.size changes.
        if np.any([np.all(f()[:, l] == -1) for f in layer.factors]):
            cleaning_action(l, layer)
            reduction_applied = True
        l += 1

    # remove duplicates
    l = 0
    while l < layer.size:
        l_prime = l + 1
        while l_prime < layer.size:
            for f in layer.factors:
                if np.all(f()[:, l] == f()[:, l_prime]):
                    reduction_applied = True
                    cleaning_action(l_prime, layer)
                    break
            l_prime += 1
        l += 1

    if reduction_applied is True:
        if reset is True:
            print('\n\tre-initialise duplicate or useless latent ' +
                  'dimensions and restart burn-in. New L=' + str(layer.size))

        elif clean is True:
            print('\n\tremove duplicate or useless latent ' +
                  'dimensions and restart burn-in. New L=' + str(layer.size))

    return reduction_applied


def remove_dimension(l_prime, layer):

    # update for tensorm link does not support parents
    # nor priors
    # layer.size -= 1
    for f in layer.factors:
        f.val = np.delete(f.val, l_prime, axis=1)


def reset_dimension(l_prime, layer):
    for f in layer.factors:
        f.val[:, l_prime] = -1


def plot_matrix_ax(mat, ax, draw_cbar=True):
    """
    wrapper for plotting a matrix of probabilities.
    attribues (optional) are used as xlabels
    """

    if np.any(mat < 0):
        print('rescaling matrix to probabilities')
        mat = .5 * (mat + 1)

    try:
        import seaborn as sns

        cmap = sns.cubehelix_palette(
            8, start=2, dark=0, light=1,
            reverse=False, as_cmap=True)

        cmap = sns.cubehelix_palette(
            4, start=2, dark=0, light=1,
            reverse=False, as_cmap=True)

        sns.set_style("whitegrid", {'axes.grid': False})

    except:
        cmap = 'gray_r'

    cax = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    return ax, cax
    # ax.set_yticks([])


def plot_matrix(mat, figsize=(7, 4), draw_cbar=True, vmin=0, vmax=1, cmap=None):
    """
    wrapper for plotting a matrix of probabilities.
    attribues (optional) are used as xlabels
    """

    if np.any(mat < 0):
        print('rescaling matrix to probabilities')
        mat = .5 * (mat + 1)

    try:
        import seaborn as sns

        if cmap is None:

            cmap = sns.cubehelix_palette(
                8, start=2, dark=0, light=1,
                reverse=False, as_cmap=True)

            cmap = sns.cubehelix_palette(
                4, start=2, dark=0, light=1,
                reverse=False, as_cmap=True)

        sns.set_style("whitegrid", {'axes.grid': False})

    except:
        print('lala')
        cmap = 'gray_r'

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.imshow(mat, aspect='auto', cmap=cmap,
                    vmin=vmin, vmax=vmax, origin='upper')

    if draw_cbar is True:
        fig.colorbar(cax, orientation='vertical')

    return fig, ax
    # ax.set_yticks([])


def plot_codes(mat, attributes=None, order='relevance'):
    """
    wrapper to plot factor matrices of factorisation models,
    ordered by the code relevance (alternatively by lbda)
    """

    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        cmap = sns.cubehelix_palette(
            8, start=2, dark=0, light=1,
            reverse=False, as_cmap=True)
        sns.set_style("whitegrid", {'axes.grid': False})
    except:
        print('seaborn import failed')
        cmap = 'gray_r'

    eigenvals = maxmachine_relevance(mat.layer)
    if order == 'relevance':
        l_idx = np.argsort(-np.array(eigenvals[:-1]))
    elif order == 'lbda':
        l_idx = np.argsort(-mat.layer.lbda()[:-1])

    fig = plt.figure(figsize=(7, 4))
    ax_codes = fig.add_subplot(111)

    ax_codes.imshow(mat.mean().transpose()[l_idx, :], aspect='auto', cmap=cmap)

    ax_codes.set_yticks(range(mat().shape[1]))
    if attributes is not None:
        ax_codes.set_xticks(range(len(attributes)))
        xticklabels = ax_codes.set_xticklabels(list(attributes), rotation=90)

    yticklabels = ax_codes.set_yticklabels(
        [r"$\nu={0:.1f}, $".format(100 * eigenvals[i]) +
         r"$\hat\lambda={0:.1f}$".format(100 * mat.layer.lbda()[i])
         for i in l_idx], rotation=0)

    return fig, ax_codes


def get_roc_auc(data, data_train, prediction):
    """
    compute area under the roc curve
    """

    zero_idx = np.where(data_train == 0)
    zero_idx = zip(list(zero_idx)[0], list(zero_idx)[1])
    auc = sklearn.metrics.roc_auc_score(
        [data[i, j] == 1 for i, j in zero_idx], [prediction[i, j] for i, j in zero_idx])

    return auc


def predict_applicability_simple(data, dimensions=35, max_features=None):
    """
    wrapper for a single layer maxmachine, meant to predict
    attribute applicability.
    """

    # check input format
    if not -1 in np.unique(data):
        data = 2 * data - 1

    # sample hold-out data as test-set
    data_train = split_test_train(data)

    mm = maxmachine.Machine()
    mm_data = mm.add_matrix(val=np.array(data_train, dtype=np.int8),
                            sampling_indicator=False)

    layer = mm.add_layer(size=int(dimensions),
                         child=mm_data,
                         z_init='kmeans',
                         u_init='kmeans',
                         lbda_init=.9)
    layer.lbda.set_prior([10, 2])

    if max_features is not None:
        layer.u.set_prior('binomial', [.5, max_features], axis=1)

    layer.auto_clean_up = True

    mm.infer(no_samples=20, convergence_eps=5e-3, print_step=100)

    auc = get_roc_auc(data, data_train, layer.output())

    print('Test set area under ROC: ' + str(auc))

    return layer


def split_test_train(data, p=.1):
    """ 
    In a binary matrix {-1,1}, set randomly 
    p/2 of the 1s and p/2 of the -1s to 0.
    This serves to create a test set for maxmachine/ormachine.
    """
    import itertools

    if not -1 in np.unique(data):
        data = 2 * data - 1

    num_of_zeros = np.prod(data.shape) * p
    index_pairs = list(itertools.product(range(data.shape[0]), range(data.shape[1])))

    # randomly set indices unobserved
    if False:
        random_idx = np.random.choice(range(len(index_pairs)), num_of_zeros, replace=False)
        zero_idx = [index_pairs[i] for i in random_idx]

    # set same number applicable/non-applicable unobserved
    if True:
        true_index_pairs = [x for x in index_pairs if data[x] == 1]
        false_index_pairs = [x for x in index_pairs if data[x] == -1]
        true_random_idx = np.random.choice(range(len(true_index_pairs)),
                                           int(num_of_zeros / 2), replace=False)
        false_random_idx = np.random.choice(range(len(false_index_pairs)),
                                            int(num_of_zeros / 2), replace=False)
        zero_idx = [true_index_pairs[i] for i in true_random_idx] + [false_index_pairs[i]
                                                                     for i in false_random_idx]

    data_train = data.copy()
    for i, j in zero_idx:
        data_train[i, j] = 0

    return data_train


def predict_applicability_fast(data,
                               N_sub=1000,
                               dimensions=35,
                               max_features=None,
                               lbda_prior=None,
                               binom_prior_attr_sets=.5,
                               high_level_object_coding=None,
                               seed=1):
    """
    wrapper for learning on a subsample and predicting on the whole data.
    lbda_prior - list: [a,b] parameters of beta prior
    """

    np.random.seed(seed)
    old_stdout = sys.stdout

    L = dimensions  # reassign for brevity in expressions
    data = check_binary_coding(data)
    data_train = split_test_train(data)

    # select subset at random
    if N_sub > data.shape[0]:
        N_sub = data.shape[0]

    subset_idx = np.random.choice(range(data.shape[0]), N_sub, replace=False)
    data_train_sub = data_train[subset_idx, :]

    # define model
    mm = maxmachine.Machine()
    data_layer = mm.add_matrix(val=data_train_sub, sampling_indicator=False)
    layer1 = mm.add_layer(size=int(L), child=data_layer, z_init=.1,
                          u_init='kmeans', noise_model='max-link', lbda_init=.95)
    if max_features is not None:
        layer1.u.set_prior('binomial', [binom_prior_attr_sets, max_features], axis=1)
    else:
        layer1.u.set_prior('binomial', [binom_prior_attr_sets], axis=1)

    if lbda_prior is not None:
        layer1.lbda.set_prior(lbda_prior)
    layer1.auto_clean_up = True

    if high_level_object_coding is not None:
        high_level_object_coding = check_binary_coding(high_level_object_coding)
        layer2 = mm.add_layer(size=high_level_object_coding.shape[1],
                              child=layer1.z,
                              noise_model='max-link',
                              lbda_init=.6,
                              z_init=high_level_object_coding[subset_idx, :])
        layer2.z.set_sampling_indicator(False)

    # train

    print('Training on subsample...')
    sys.stdout = tempfile.TemporaryFile()  # prevent printing (todo: write a decorator)
    mm.infer(no_samples=int(5e1), convergence_window=10,
             convergence_eps=1e-2, burn_in_min=100,
             burn_in_max=int(3e3), fix_lbda_iters=10)
    sys.stdout = old_stdout

    # now run on full dataset with previous results as initialisation,
    # keep u fixed to learn z's

    L = layer1.u().shape[1]
    mm_2 = maxmachine.Machine()
    # define model architecture
    data_layer_2 = mm_2.add_matrix(val=data_train, sampling_indicator=False)
    layer1_2 = mm_2.add_layer(size=int(L), child=data_layer_2, z_init=0.0,
                              u_init=2 * (layer1.u.mean() > .5) - 1,
                              noise_model='max-link', lbda_init=.9)
    # layer1_2.z.set_prior('binomial', [.5], axis=0)
    layer1_2.u.sampling_indicator = False
    layer1_2.auto_clean_up = True

    if high_level_object_coding is not None:
        layer2_2 = mm_2.add_layer(size=high_level_object_coding.shape[1],
                                  child=layer1_2.z,
                                  noise_model='max-link', lbda_init=.6,
                                  z_init=high_level_object_coding)
        layer2_2.z.set_sampling_indicator(False)

    # train (i.e. adjust the z's and lbdas)
    print('Learning latent representation for all objects...')
    sys.stdout = tempfile.TemporaryFile()
    mm_2.infer(no_samples=int(10), convergence_window=5,
               convergence_eps=1e-2, burn_in_min=20,
               burn_in_max=200, fix_lbda_iters=3)
    sys.stdout = old_stdout

    # now sample u and z
    layer1_2.u.sampling_indicator = True
    print('Drawing samples on the full dataset...')
    sys.stdout = tempfile.TemporaryFile()
    mm_2.infer(no_samples=int(2e1), convergence_window=5,
               convergence_eps=5e-3, burn_in_min=10,
               burn_in_max=50, fix_lbda_iters=3)
    sys.stdout = old_stdout

    roc_auc = get_roc_auc(data, data_train, layer1_2.output())
    print('Area under ROC curve: ' + str(roc_auc))

    return layer1_2, roc_auc, data_train


def check_binary_coding(data):
    """
    For MaxMachine and OrM, data and latent variables are
    in {-1,1}. Check and corret the coding here.
    """

    if not -1 in np.unique(data):
        data = 2 * data - 1

    return np.array(data, dtype=np.int8)


def check_convergence_single_trace(trace, eps):
    """
    compare mean of first and second half of a sequence,
    checking whether there difference is > epsilon.
    """

    l = int(len(trace) / 2)
    r1 = expit(np.mean(trace[:l]))
    r2 = expit(np.mean(trace[l:]))
    r = expit(np.mean(trace))

    if np.abs(r1 - r2) < eps:
        return True
    else:
        return False


def boolean_tensor_product(Z, U, V):
    """
    Return the Boolean tensor product of three matrices
    that share their second dimension.
    """

    N = Z.shape[0]
    D = U.shape[0]
    M = V.shape[0]
    L = Z.shape[1]
    X = np.zeros([N, D, M], dtype=bool)

    assert(U.shape[1] == L)
    assert(V.shape[1] == L)

    for n in range(N):
        for d in range(D):
            for m in range(M):
                if np.any([(Z[n, l] == True) and
                           (U[d, l] == True) and
                           (V[m, l] == True)
                           for l in range(L)]):
                    X[n, d, m] = True
    return X


def add_bernoulli_noise(X, p):

    X_intern = X.copy()

    for n in range(X.shape[0]):
        for d in range(X.shape[1]):
            for m in range(X.shape[2]):
                if np.random.rand() < p:
                    X_intern[n, d, m] = ~X_intern[n, d, m]

    return X_intern


def add_bernoulli_noise_2d(X, p, seed=None):

    if seed is None:
        np.random.seed(np.random.randint(1e4))

    X_intern = X.copy()

    for n in range(X.shape[0]):
        for d in range(X.shape[1]):
            if np.random.rand() < p:
                X_intern[n, d] = ~X_intern[n, d]

    return X_intern


def add_bernoulli_noise_2d_biased(X, p_plus, p_minus, seed=None):

    if seed is None:
        np.random.seed(np.random.randint(1e4))

    X_intern = X.copy()

    for n in range(X.shape[0]):
        for d in range(X.shape[1]):
            if X_intern[n, d] == 1:
                p = p_plus
            if X_intern[n, d] == 0:
                continue
            elif X_intern[n, d] == -1:
                p = p_minus

            if np.random.rand() < p:
                X_intern[n, d] = -X_intern[n, d]

    return X_intern


def flatten(t):
    """
    Generator flattening the structure

    >>> list(flatten([2, [2, (4, 5, [7], [2, [6, 2, 6, [6], 4]], 6)]]))
    [2, 2, 4, 5, 7, 2, 6, 2, 6, 6, 4, 6]
    """

    import collections
    for x in t:
        if not isinstance(x, collections.Iterable):
            yield x
        else:
            yield from flatten(x)


def intersect_dataframes(A, B):
    """
    given two dataframes, intersect rows and columns of both
    """

    joint_rows = set(A.index).intersection(B.index)
    A = A[A.index.isin(joint_rows)]
    B = B[B.index.isin(joint_rows)]

    joint_cols = set(A.columns).intersection(B.columns)
    A = A[list(joint_cols)]
    B = B[list(joint_cols)]

    A = A.sort_index()
    B = B.sort_index()

    assert np.all(A.index == B.index)
    assert np.all(A.columns == B.columns)

    print('\n\tNew shape is :' + str(mut.shape))

    return A, B


def all_columsn_are_disjoint(mat):
    """
    Check whether places with 1s in all columns of mat are disjoint
    for all pairs of columns.
    """

    L = mat.shape[1]
    return not np.any([np.all(mat[mat[:, i] == 1, j] == 1)
                       for i, j in list(itertools.permutations(range(L), 2))])


def random_machine_matrix(p, shape):
    return 2 * np.array(binomial(n=1, p=p, size=shape), dtype=np.int8) - 1


def generate_orm_product(N=100, D=20, L=3):
    """
    Generate random matrix U, Z and their Boolean product X.
    returns: U, Z, X in {-1, 1} representation.
    Ascertain that different U[d,:] and Z[n,:] are disjoint.
    """

    def disjoint_columns_mat(K, L):
        while True:
            mat = np.array(np.random.rand(K, L) > .5, dtype=np.int8)
            if all_columsn_are_disjoint(mat):
                return mat

    U = disjoint_columns_mat(D, L)
    Z = disjoint_columns_mat(N, L)

    X = np.array(np.dot(Z == 1, U.transpose() == 1), dtype=np.int8)

    # map to {-1, 0, 1} reprst.
    X = 2 * X - 1
    U = 2 * U - 1
    Z = 2 * Z - 1

    return U, Z, X


def get_lop(name='OR'):
    """
    Return logical operators such that they can be applied
    to 1D arrays of arbitrary length.
    """

    @jit
    def OR(x):
        return np.any(x == 1)

    @jit
    def NOR(x):
        return ~(np.any(x == 1))

    @jit
    def AND(x):
        return np.all(x == 1)

    @jit
    def NAND(x):
        return ~(np.all(x == 1))

    @jit
    def XOR(x):
        return np.sum(x == 1) == 1

    @jit
    def NXOR(x):
        return ~(np.sum(x == 1) == 1)

    lops = [OR, NOR, AND, NAND, XOR, NXOR]

    for lop in lops:
        if lop.__name__ == name:
            return lop

    raise ValueError('Logical operator not defined.')


def get_fuzzy_lop(name='OR'):
    """
    Return logical operators such that they can be applied
    to 1D array of arbitrary length that contain probabilities.
    """

    @jit
    def AND(x):
        return np.prod(x)

    @jit
    def OR(x):
        return 1 - np.prod(1 - x)

    def XOR(x):
        return np.sum(
            [np.prod(
                [1 - x[i] for i in range(len(x)) if i != j] + [x[j]])
                for j in range(len(x))])

    @jit
    def NAND(x):
        return 1 - np.prod(x)

    @jit
    def NOR(x):
        return np.prod(1 - x)

    def NXOR(x):
        return 1 - np.sum(
            [np.prod(
                [1 - x[i] for i in range(len(x)) if i != j] + [x[j]])
                for j in range(len(x))])

    lops = [OR, NOR, AND, NAND, XOR, NXOR]

    for lop in lops:
        if lop.__name__ == name:
            return lop

    raise ValueError('Logical operator not defined.')


def lom_generate_data_fast(factors, model='OR-AND', fuzzy=False):
    """
    Factors and generated data are in [-1,1] mapping.
    """

    if model not in implemented_loms():
        print('Requested model output is not explicitly implemented.\n' +
              'Falling back to slower general implementation.')
        return lom_generate_data(factors, model)

    if len(factors) == 2:

        if fuzzy is False:
            out2D = lambda_updates_numba.make_output_function_2d(model)
            return out2D(*[np.array(f, dtype=np.int8) for f in factors])

        elif fuzzy is True:
            out2D = lambda_updates_numba.make_output_function_2d_fuzzy(model)
            return out2D(*[np.array(f, dtype=np.float64) for f in factors])

    elif len(factors) == 3 and model == 'OR-AND':
        if fuzzy is False:
            out3D = lambda_updates_numba.make_output_function_3d(model)
            return out3D(*[np.array(f, dtype=np.int8) for f in factors])

        elif fuzzy is True:
            out3D = lambda_updates_numba.make_output_function_3d_fuzzy(model)
            return out3D(*[np.array(f, dtype=np.float64) for f in factors])

    else:
        return lom_generate_data(factors, model)


def lom_generate_data(factors, model='OR-AND'):
    """
    Elegant way of generating data according to any LOM.
    Not very fast, however.
    See lom_generate_data_fast for a more performant implementation
    """

    K = len(factors)
    L = factors[0].shape[1]
    outer_operator_name, inner_operator_name = model.split('-')

    out = np.zeros([x.shape[0] for x in factors], dtype=np.int8)

    outer_operator = get_lop(outer_operator_name)
    inner_operator = get_lop(inner_operator_name)

    outer_logic = np.zeros(L, dtype=bool)
    inner_logic = np.zeros(K, dtype=bool)

    for index, _ in np.ndenumerate(out):
        for l in range(L):
            inner_logic[:] =\
                [f[index[i], l] == 1 for i, f in enumerate(factors)]
            outer_logic[l] = inner_operator(inner_logic)
        out[index] = 2 * outer_operator(outer_logic) - 1

    return out


def lom_generate_data_fuzzy(factors, model='OR-AND'):

    K = len(factors)
    L = factors[0].shape[1]
    outer_operator_name, inner_operator_name = model.split('-')

    out = np.zeros([x.shape[0] for x in factors])

    outer_operator = get_fuzzy_lop(outer_operator_name)
    inner_operator = get_fuzzy_lop(inner_operator_name)

    outer_logic = np.zeros(L)  # , dtype=bool)
    inner_logic = np.zeros(K)  # , dtype=bool)

    for index, _ in np.ndenumerate(out):
        for l in range(L):
            inner_logic[:] =\
                [.5 * (f[index[i], l] + 1)
                 for i, f in enumerate(factors)]
            outer_logic[l] = inner_operator(inner_logic)
        out[index] = outer_operator(outer_logic)

    return 2 * out - 1


def canonise_model(model, child):
    """
    Many of the possible Logical Operator Machines are equivalent,
    or equivalent after inversion of the data or the factors.
    Here the model is translated to its canonical form.
    """

    model_new = replace_equivalent_model(model)

    invert_data = False
    invert_factors = False

    # translate to canonical models with data/factor inversion
    # OR-AND family
    if model_new == 'OR-AND':
        pass
    elif model_new == 'AND-NAND':
        model_new = 'OR-AND'
        invert_data = True
    elif model_new == 'OR-NOR':
        model_new = 'OR-AND'
        invert_factors = True
    elif model_new == 'AND-OR':
        model_new = 'OR-AND'
        invert_data = True
        invert_factors = True

    # OR-NAND family
    elif model_new == 'OR-NAND':
        pass
    elif model_new == 'AND-AND':
        model_new = 'OR-NAND'
        invert_data = True
    elif model_new == 'OR-OR':
        model_new = 'OR-NAND'
        invert_factors = True
    elif model_new == 'AND-NOR':
        model_new = 'OR-NAND'
        invert_data = True
        invert_factors = True

    # XOR-AND family
    elif model_new == 'XOR-AND':
        pass
    elif model_new == 'XOR-NOR':
        model_new = 'XOR-AND'
        invert_factors = True
    elif model == 'NXOR-AND':
        model_new = 'XOR-AND'
        invert_data = True
    elif model == 'NXOR-NOR':
        model_new = 'XOR-AND'
        invert_data = True
        invert_factors = True

    # XOR-NAND family
    elif model_new == 'XOR-NAND':
        pass
    elif model_new == 'XOR-OR':
        model_new = 'XOR-NAND'
        invert_factors = True
    elif model_new == 'NXOR-NAND':
        model_new = 'XOR-NAND'
        invert_data = True
    elif model_new == 'NXOR-OR':
        model_new = 'XOR-NAND'
        invert_data = True
        invert_factors = True

    # AND-XOR family
    elif model_new == 'NAND-XOR':
        pass
    elif model_new == 'AND-XOR':
        model_new = 'NAND-XOR'
        invert_data = True

    # OR-XOR family
    elif model_new == 'OR-XOR':
        pass
    elif model_new == 'NOR-XOR':
        model_new = 'OR-XOR'
        invert_data = True

    # XOR-NXOR family
    elif model_new == 'XOR-NXOR':
        pass
    elif model_new == 'NXOR-NXOR':
        model_new = 'XOR-NXOR'
        invert_data = True

    # XOR-XOR family
    elif model_new == 'XOR-XOR':
        pass
    elif model_new == 'NXOR-XOR':
        model_new = 'XOR-XOR'
        invert_data = True

    elif model_new == 'MAX-AND':
        pass

    else:
        import pdb
        pdb.set_trace()
        raise NotImplementedError("Model not implemented.")

    # print output and invert data if needed.
    if invert_data is False and invert_factors is False:
        print('\n' + model + ' is treated as ' + model_new + '.\n')

    if invert_data is True and invert_factors is False:
        print('\n' + model + ' is treated as ' + model_new +
              ' with inverted data.\n')
        child.val *= -1

    if invert_data is False and invert_factors is True:
        print('\n' + model + ' is treated as ' + model_new +
              ' with inverted factors. (Invert yourself!)\n')

    if invert_data is True and invert_factors is True:
        print('\n' + model + ' is treated as ' + model_new +
              ' with inverted data and inverted factors. ' +
              ' (invert factors yourself!)\n')
        child.val *= -1

    # print warning for OR-NAND models.
    if model_new == 'OR-NAND':
        print(model_new + ' based models are reasonable only for' +
              'a single latent dimensions!\n')

    return model_new, invert_data, invert_factors


def canonical_loms(level='clans', mode='implemented'):
    """
    which: clans, families
    type: implemented, canonical
    """

    if mode == 'implemented':
        clans = ['OR-AND', 'OR-NAND', 'OR-XOR', 'NAND-XOR',
                 'XOR-NAND', 'XOR-AND', 'XOR-NXOR', 'XOR-XOR']
        families = ['NOR-AND', 'NOR-NAND', 'NOR-XOR', 'AND-XOR',
                    'NXOR-NAND', 'NXOR-AND', 'NXOR-NXOR', 'NXOR-XOR']
    elif mode == 'canonical':
        clans = ['AND-AND', 'AND-NAND', 'XOR-AND', 'XOR-NAND',
                 'AND-XOR', 'AND-NXOR', 'XOR-XOR', 'XOR-NXOR']
        families = ['OR-NAND', 'OR_AND', 'NXOR-AND', 'NXOR-NAND',
                    'OR-NXOR', 'OR-XOR', 'NXOR-XOR', 'NXOR-NXOR']
    else:
        raise ValueError

    if level == 'clans':
        return clans
    elif level == 'families':
        return clans + families
    else:
        raise ValueError


def implemented_loms():
    return ['OR-AND', 'OR-NAND', 'OR-XOR', 'NAND-XOR',
            'XOR-AND', 'XOR-XOR', 'XOR-NXOR', 'XOR-NAND']


def replace_equivalent_model(model, equivalent_pairs=None):

    # the following pairs area equivalent and the
    # left partner supports posterior inference
    # (this is not equivalent to the canonical representation)
    if equivalent_pairs is None:
        equivalent_pairs = [('OR-AND', 'NAND-NAND'),
                            ('OR-NOR', 'NAND-OR'),
                            ('AND-OR', 'NOR-NOR'),
                            ('AND-NAND', 'NOR-AND'),
                            ('OR-OR', 'NAND-NOR'),
                            ('OR-NAND', 'NAND-AND'),
                            ('AND-AND', 'NOR-NAND'),
                            ('AND-NOR', 'NOR-OR'),
                            ('NAND-XOR', 'OR-NXOR'),
                            ('AND-XOR', 'NOR-NXOR'),  # remove
                            ('OR-XOR', 'NAND-NXOR'),  # remove
                            ('NOR-XOR', 'AND-NXOR')]  # remove

    # replace model by its equivalent counterparts
    if model in [pair[1] for pair in equivalent_pairs]:
        model = [pair[0] for pair in equivalent_pairs if pair[1] == model][0]
    return model


def expected_density(model, L, K, f):
    """
    """

    def invert(x):
        return 1 - x

    def identity(x):
        return x

    if model == 'XOR-NXOR' or model == 'NXOR-NXOR':
        # need some extra treatment, XOR-XOR does not generalise via inversion
        pass
        if model == 'NXOR-NXOR':
            Inv = invert
        else:
            Inv = identity
        d = Inv(
            L * (((1 - (K * f * (1 - f)**(K - 1)))) *
                 (K * f * (1 - f)**(K - 1))**(L - 1))
        )
        return d
    else:
        model_group, Inv = get_lom_class(model)

    for i, indicator in enumerate(Inv):
        if indicator is True:
            Inv[i] = invert
        else:
            Inv[i] = identity

    if model_group == 'AND-AND':
        d = Inv[0](Inv[1](Inv[2](f)**K)**L)

    elif model_group == 'XOR-AND':
        d = Inv[0](L * invert(Inv[1](Inv[2](f)**K))**(L - 1) *
                   Inv[1](Inv[2](f)**K))

    elif model_group == 'AND-XOR':
        # d = (K * f * invert(f)**(K - 1))**L
        d = Inv[0](Inv[1](K * f * invert(f)**(K - 1))**L)

    elif model_group == 'XOR-XOR':
        d = Inv[0](L * (
            Inv[1](K * f * (invert(f))**(K - 1) *
                   (f**K + invert(f)**(K)) ** (L - 1))
        ))

    return d


def factor_density(machine, L, K, X):
    """
    X is the desired expected data density
    returns the corresponding factor density
    """
    from scipy.optimize import fsolve
    from scipy.optimize import least_squares


    def func(f_d):
        if f_d < 0:
            return 1e10
        if f_d > 1:
            return 1e10
        else:
            return expected_density(machine, L, K, f=f_d) - X

    # Plot it
    tau = np.linspace(0, 1, 201)

    # Use the numerical solver to find the roots
    best = 1e10
    for tau_initial_guess in [.01,.25,.5,.75,.99]:
        tau_solution = fsolve(func, tau_initial_guess, 
                              maxfev=int(1e6), full_output=False, 
                              xtol=1e-10, factor=10)
        if np.abs(func(tau_solution)[0]) < best:
            best = np.abs(func(tau_solution)[0])
            best_solution = tau_solution

    print("The solution is tau = " + str(best_solution))
    print("at which the value of the expression is " +
          str(func(best_solution) + X))

    if np.abs(func(best_solution)) > 1e-6:
        print('Solution does not exist. Returning closest value.')

    return best_solution[0]


def get_lom_class(machine):
    """
    Return corresponding class and tuple of inversion
    instructions.
    Inv: I_outer, I_hidden, I_inner
    """

    # AND-AND class
    if machine == 'AND-AND' or machine == 'NOR-NAND':
        Inv = [False, False, False]
        machine = 'AND-AND'
    elif machine == 'AND-NOR' or machine == 'NOR-OR':
        Inv = [False, False, True]
        machine = 'AND-AND'
    elif machine == 'OR-NAND' or machine == 'NAND-AND':
        Inv = [True, False, False]
        machine = 'AND-AND'
    elif machine == 'OR-OR' or machine == 'NAND-NOR':
        Inv = [True, False, True]
        machine = 'AND-AND'
    elif machine == 'AND-NAND' or machine == 'NOR-AND':
        Inv = [False, True, False]
        machine = 'AND-AND'
    elif machine == 'AND-OR' or machine == 'NOR-NOR':
        Inv = [False, True, True]
        machine = 'AND-AND'
    elif machine == 'OR-AND' or machine == 'NAND-NAND':
        Inv = [True, True, False]
        machine = 'AND-AND'
    elif machine == 'OR-NOR' or machine == 'NAND-OR':
        Inv = [True, True, True]
        machine = 'AND-AND'

    elif machine == 'XOR-AND':
        Inv = [False, False, False]
        machine = 'XOR-AND'
    elif machine == 'XOR-NOR':
        Inv = [False, False, True]
        machine = 'XOR-AND'
    elif machine == 'NXOR-AND':
        Inv = [True, False, False]
        machine = 'XOR-AND'
    elif machine == 'NXOR-NOR':
        Inv = [True, False, True]
        machine = 'XOR-AND'
    elif machine == 'XOR-NAND':
        Inv = [False, True, False]
        machine = 'XOR-AND'
    elif machine == 'XOR-OR':
        Inv = [False, True, True]
        machine = 'XOR-AND'
    elif machine == 'NXOR-NAND':
        Inv = [True, True, False]
        machine = 'XOR-AND'
    elif machine == 'NXOR-OR':
        Inv = [True, True, True]
        machine = 'XOR-AND'

    elif machine == 'AND-XOR' or machine == 'NOR-NXOR':
        Inv = [False, False, False]
        machine = 'AND-XOR'
    elif machine == 'OR-NXOR' or machine == 'NAND-XOR':
        Inv = [True, False, False]
        machine = 'AND-XOR'
    elif machine == 'AND-NXOR' or machine == 'NOR-XOR':
        Inv = [False, True, False]
        machine = 'AND-XOR'
    elif machine == 'OR-XOR' or machine == 'NAND-NXOR':
        Inv = [True, True, False]
        machine = 'AND-XOR'

    elif machine == 'XOR-XOR':
        Inv = [False, False, False]
        machine = 'XOR-XOR'
    elif machine == 'NXOR-XOR':
        Inv = [True, False, False]
        machine = 'XOR-XOR'
    elif machine == 'XOR-NXOR':
        Inv = [False, True, False]
        machine = 'XOR-XOR'
    elif machine == 'NXOR-NXOR':
        Inv = [True, True, False]
        machine = 'XOR-XOR'

    try:
        return machine, Inv
    except:
        import pdb
        pdb.set_trace()
