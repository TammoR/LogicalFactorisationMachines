import numpy as np
import lom
import lom.experiments as experiments
import itertools
import lom.auxiliary_functions as aux


def test_orm():

  np.random.seed(3)

  N = 10
  M = 5
  L = 3

  X = 2 * np.array([M * [0, 0, 1, 1, 0, 0],
                    M * [1, 1, 0, 0, 0, 0],
                    M * [0, 0, 0, 0, 1, 1]]) - 1

  X = np.concatenate(N * [X])
  N, D = X.shape

  orm = lom.Machine()

  orm.framework = 'numba'

  data = orm.add_matrix(val=X, fixed=True)  # , sampling_indicator=False)

  layer1 = orm.add_layer(latent_size=L,
                         child=data,
                         model='OR-AND')

  # layer1.factors[0].val = np.array(2*np.ones([N,L])-1, dtype=np.int8)
  # layer1.factors[1].val = np.array(2*np.ones([D,L])-1, dtype=np.int8)

  orm.infer(convergence_window=20, no_samples=20,
            convergence_eps=1e-3, burn_in_min=20,
            burn_in_max=1000)

  assert abs(1 / (1 + np.exp(-orm.layers[0].lbda())) - 1.) < 1e-2


def test_all_2D_LOMs():

  operators = ['AND', 'NAND', 'OR', 'NOR', 'XOR', 'NXOR']
  # operators = ['OR', 'AND']
  machines = [x[0] + '-' + x[1] for x in list(itertools.product(operators, repeat=2))]

  for machine in aux.canonical_loms():  # machines:

    N = 50
    D = 10
    L = 3

    Z = np.array(np.random.rand(N, L) > .5, dtype=np.int8)
    U = np.array(np.random.rand(D, L) > .5, dtype=np.int8)
    X = aux.lom_generate_data([2 * Z - 1, 2 * U - 1], model=machine)

    orm = lom.Machine()

    data = orm.add_matrix(X, fixed=True)
    layer = orm.add_layer(latent_size=L, child=data, model=machine)
    layer.z.val = (1 - 2 * layer.invert_factors) * (2 * Z - 1)
    layer.u.val = (1 - 2 * layer.invert_factors) * (2 * U - 1)

    # we initialise with ground truth, hence set lbda large to avoid effectively
    # random initialisation
    layer.lbda.val = 3.0

    orm.infer(burn_in_min=10, fix_lbda_iters=2)

    try:
      assert np.mean((2 * (layer.output(technique='factor_mean') > .5) - 1) ==
                     data()) > .98
    except:
      acc = np.mean((2 * (layer.output(technique='factor_mean') > .5) - 1) == data())
      print(machine + ' failed with reconstruction accuracy of ' + str(acc))
      raise ValueError()


def test_all_3D_LOMs():

  operators = ['AND', 'NAND', 'OR', 'NOR', 'XOR', 'NXOR']
  # operators = ['OR', 'AND']
  machines = [x[0] + '-' + x[1] for x in list(itertools.product(operators, repeat=2))]

  for machine in aux.canonical_loms():  # machines:

    N = 50
    D = 10
    L = 3

    Z = np.array(np.random.rand(N, L) > .5, dtype=np.int8)
    U = np.array(np.random.rand(D, L) > .5, dtype=np.int8)
    V = np.array(np.random.rand(D, L) > .5, dtype=np.int8)
    X = aux.lom_generate_data([2 * Z - 1, 2 * U - 1, 2 * V - 1], model=machine)

    orm = lom.Machine()

    data = orm.add_matrix(X, fixed=True)
    layer = orm.add_layer(latent_size=L, child=data, model=machine)
    layer.z.val = (1 - 2 * layer.invert_factors) * (2 * Z - 1)
    layer.u.val = (1 - 2 * layer.invert_factors) * (2 * U - 1)
    layer.v.val = (1 - 2 * layer.invert_factors) * (2 * V - 1)

    # we initialise with ground truth, hence set lbda large to avoid effectively
    # random initialisation
    layer.lbda.val = 3.0

    orm.infer(burn_in_min=10, fix_lbda_iters=2)

    try:
      assert np.mean((2 * (layer.output(technique='factor_map') > .5) - 1) ==
                     data()) > .98
      assert np.mean((2 * (layer.output(technique='factor_mean') > .5) - 1) ==
                     data()) > .98
    except:
      acc = np.mean((2 * (layer.output(technique='factor_mean') > .5) - 1) == data())
      print(machine + ' failed with reconstruction accuracy of ' + str(acc))
      # import pdb; pdb.set_trace()
      raise ValueError()


def test_maxmachine():

  # generate toy data
  A = 2 * np.array([[0, 0, 0, 0, 0, 1, 1]]) - 1
  B = 2 * np.array([[0, 0, 1, 1, 1, 1, 0]]) - 1
  C = 2 * np.array([[1, 1, 1, 1, 0, 0, 0]]) - 1
  X = np.concatenate(100 * [C] + 100 * [B] + 100 * [A])  # + 100 *[2*((A==1) + (B==1))-1])
  # X = np.concatenate(100*[C]+50*[2*np.array([[0,0,0,1,1,1,1]])-1])
  for i in range(X.shape[0]):
    for j in range(0, X.shape[1]):
      if np.random.rand() > .98:  # .9
        X[i, j] = -X[i, j]
    # heterosced noise
    if True:
      for j in range(4, X.shape[1]):
        if np.random.rand() > .95:  # .75
          X[i, j] = -X[i, j]

  machine = 'MAX-AND'

  orm = lom.Machine()
  L = 3

  data = orm.add_matrix(X, fixed=True)
  layer = orm.add_layer(latent_size=L, child=data, model=machine)

  # we initialise with ground truth, hence set lbda large to avoid effectively
  # random initialisation

  layer.u.val = np.array([[1, -1, -1],
                          [1, -1, -1],
                          [1, 1, -1],
                          [1, 1, -1],
                          [-1, 1, 1],
                          [-1, 1, 1],
                          [-1, -1, 1]], dtype=np.int8)

  layer.z.val = np.array(np.concatenate([100 * [[1, -1, -1]],
                                         100 * [[-1, 1, -1]],
                                         100 * [[-1, -1, 1]]]), dtype=np.int8)

  layer.lbda.val = np.array([.99 for x in range(L + 1)])

  orm.infer(burn_in_min=50, burn_in_max=200, fix_lbda_iters=20, no_samples=100)

  assert np.mean((layer.output(technique='mc') > .5) == (X == 1)) > .8


def test_densities():

  machines = aux.canonical_loms(level='clans', mode='implemented')

  for machine in machines:

    d = aux.expected_density(machine_class, L=3, K=2, f=.5)

    N = 200
    D = 200
    L = 3
    Z = np.array(np.random.rand(N, L) > .5, dtype=np.int8)
    U = np.array(np.random.rand(D, L) > .5, dtype=np.int8)
    X = aux.lom_generate_data_fast([2 * Z - 1, 2 * U - 1], model=machine)

    X_train, train_mask = experiments.split_train_test(X, split=.1)

    try:
      assert (np.abs(np.mean(X == 1) - d)) < 5e-2
    except:
      print(np.abs(np.mean(X == 1)))
      print(d)
      print(machine)


if __name__ == '__main__':

  # test_orm()
  # test_all_2D_LOMs()
  # test_all_3D_LOMs()
  # test_maxmachine()
  test_densities()
