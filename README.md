# Logical Factorisation Machines
[![Build Status](https://travis-ci.org/TammoR/LogicalFactorisationMachines.svg?branch=master)](https://travis-ci.org/TammoR/LogicalOperatorMachines)

[ICML 2018 paper on tensor decomposition](http://proceedings.mlr.press/v80/rukat18a.html)

[ICML 2017 paper on matrix factorisation](http://proceedings.mlr.press/v70/rukat17a.html)

This package generalises the deprecated [OrMachine](https://github.com/TammoR/ormachine) package.
If you are looking for an implementation of Boolean Matrix Factorisation or Boolean Tensor Factorisation, 
you should use Logical Factorisation Machines with the default model `OR-AND`.

This requires Python 3 and the [numba](numba.pydata.org) package.
The easiest way is to use the [Anaconda Python distribution](https://www.anaconda.com/download).
See [here](https://pypi.python.org/pypi/numba) numba installation instructions.

For installation go to the cloned directory and do `pip install .`.

## Basic usage example

All (optional) steps can be ignored.

```
import lom
import lom.auxiliary_functions as aux

# generate toy data
N = 20
D = 20
L = 3
Z = np.array(np.random.rand(N, L) > .5, dtype=np.int8)
U = np.array(np.random.rand(D, L) > .5, dtype=np.int8)
X = aux.lom_generate_data([2 * Z-1, 2 * U-1], model='OR-AND') # take Boolean product
X_noisy = aux.add_bernoulli_noise_2d(X, p=.1) # add noise

# initialise model
orm = lom.Machine()
data = orm.add_matrix(X_noisy, fixed=True)
layer = orm.add_layer(latent_size=3, child=data, model='OR-AND')

# initialise factors (optional)
layer.factors[0].val = np.array( 2*(np.random.rand(N, L) > .5) - 1, dtype=np.int8)

# Fix particular entries (1s in fixed_entries matrix) (optional)
layer.factors[1].fixed_entries = np.zeros(layer.factors[1]().shape, dtype=np.int8)
layer.factors[1].fixed_entries[0,:] = 1

# Set priors beta prior on sigmoid(lambda) (optional)
layer.lbda.beta_prior = (1,1)

# Set iid bernoulli priors on factor matrix entries (optional)
layer.factors[1].bernoulli_prior = .5

# Use annealing to improve convergence (optional, not needed in general).
orm.anneal = True
layer.lbda.val = 3.0 # if annealing: target temperature, otherwise initial value

# run inference
orm.infer(burn_in_min=100, burn_in_max=1000, no_samples=50)

# inspect the factor mean
[layer.factors[i].show() for i in range(len(layer.factors))]

# inspect the reconstruction
fig, ax = aux.plot_matrix(X_noisy)
ax.set_title('Input data')

fig, ax = aux.plot_matrix(layer.output(technique='factor_map'))
ax.set_title('Reconstruction')

fig, ax = aux.plot_matrix(X)
ax.set_title('Noisefree data')
```