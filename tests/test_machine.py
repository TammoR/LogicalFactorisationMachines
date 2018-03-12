import numpy as np
import lom
import lom.experiments as experiments


def test_orm():

    np.random.seed(3)

    N = 100
    M = 5

    X = 2 * np.array([M * [0, 0, 1, 1, 0, 0],
                      M * [1, 1, 0, 0, 0, 0],
                      M * [0, 0, 0, 0, 1, 1]]) - 1

    X = np.concatenate(N * [X])

    orm = lom.Machine()

    orm.framework = 'numba'

    data = orm.add_matrix(val=X, fixed=True)  # , sampling_indicator=False)

    layer1 = orm.add_layer(latent_size=3,
                           child=data,
                           model='OR-AND')

    orm.infer(convergence_window=20, no_samples=20,
              convergence_eps=1e-3, burn_in_min=20,
              burn_in_max=1000)

    print(np.mean((X > 0) == (layer1.output() > .5)))

    assert abs(1 / (1 + np.exp(-orm.layers[0].lbda())) - 1.) < 1e-3


def test_tensorm():

    np.random.seed(1)
    size = 2
    tensor, _, _, _ = experiments.generate_random_tensor(size, (3, 4, 5),
                                                         noise=0)

    orm = lom.Machine()
    data = orm.add_matrix(2 * tensor - 1, fixed=True)
    layer = orm.add_layer(child=data, latent_size=size, model='OR-AND')

    orm.infer(burn_in_min=20, no_samples=10)

    assert np.all((layer.output() > .5) == (tensor == 1))


if __name__ == '__main__':

    test_orm()
    test_tensorm()
