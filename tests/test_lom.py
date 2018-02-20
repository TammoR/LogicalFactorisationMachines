import numpy as np
import src.lom as lom


def test_orm():
    np.random.seed(2)

    N = 100
    M = 100

    X = 2*np.array([M*[0,0,1,1,0,0], M*[1,1,0,0,0,0], M*[0,0,0,0,1,1]])-1
    
    X = np.concatenate(N*[X])

    orm = lom.Machine()

    data = orm.add_matrix(val=X, sampling_indicator=False)

    layer1 = orm.add_layer(size=3, child=data, lbda_init=2., noise_model='or-link')

    orm.infer(convergence_window=50, no_samples=200,
              convergence_eps=1e-3, burn_in_min=100,
              burn_in_max=10000)

    assert abs(1/(1+np.exp(-orm.layers[0].lbda())) - 1.) < 1e-3


def test_maxmachine():

    orm = lom.Machine()

    data = orm.add_matrix(np.array(50*[[-1,1,1],[1,-1,1]]),
                          sampling_indicator=False)
    layer=orm.add_layer(size=2, child=data, lbda_init=.9,
                        noise_model='max-link')
    orm.infer(fix_lbda_iters=5,burn_in_min=100)

    assert np.abs(np.mean(layer.output() - .5*(data() +1))) < 1e-2


if __name__ == "__main__":
    test_orm()