import numpy as np
import lom
import lom.experiments as experiments
import itertools
import lom.auxiliary_functions as aux


def generate_random_2D_data(N=100, D=20, L=5):

    Z = np.array(np.random.rand(N, L) > .5, dtype=np.int8)
    U = np.array(np.random.rand(D, L) > .5, dtype=np.int8)
    X = np.array(2 * (np.dot(Z, U.transpose()) > 0) - 1, dtype=np.int8)
    X = aux.add_bernoulli_noise_2d(X, .05)

    return X


def test_ibp():

    X = generate_random_2D_data()

    orm = lom.Machine()
    data = orm.add_matrix(X, fixed=True)
    layer = orm.add_layer(latent_size=1, child=data, model='OR-AND-IBP')

    orm.infer(burn_in_min=200)

    assert np.mean((2*layer.output()-1) == X) > .9

test_ibp()