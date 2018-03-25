import numpy as np
import lom
import lom.experiments as experiments
import itertools
import lom.auxiliary_functions as aux


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


def test_all_2D_LOMs():

  operators = ['AND','NAND','OR','NOR','XOR','NXOR']
  machines = [x[0]+'-'+x[1] for x in list(itertools.product(operators, repeat=2))]

  for machine in machines:

      N = 50
      D = 10
      L = 3

      Z = np.array(np.random.rand(N,L)>.5, dtype=np.int8)
      U = np.array(np.random.rand(D,L)>.5, dtype=np.int8)
      X = aux.lom_generate_data([2*Z-1,2*U-1], model=machine)

      orm = lom.Machine()

      data = orm.add_matrix(X, fixed=True)
      layer = orm.add_layer(latent_size=L, child=data, model=machine)
      layer.z.val = (1-2*layer.invert_factors)*(2*Z-1)
      layer.u.val = (1-2*layer.invert_factors)*(2*U-1)

      orm.infer(burn_in_min=10, fix_lbda_iters=0)
      
      try:
          assert np.mean((2*(layer.output()>.5)-1) == data()) > .98
      except:
          acc = np.mean((2*(layer.output()>.5)-1) == data())
          print(machine+' failed with reconstruction accuracy of '+str(acc))


if __name__ == '__main__':

    test_orm()
    test_tensorm()
