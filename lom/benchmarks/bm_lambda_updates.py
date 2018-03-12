"""
Benchmark lambda updates, which amount to computing the
positive predictive value.
Compare to reference performance on 2017 macbook pro.
"""


import time
import numpy as np
import lom
import lom.auxiliary_functions as aux
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_data(N=100, D=100):

    np.random.seed(2)

    L = 5

    U = np.array(2 * (np.random.rand(D, L) > .5) - 1, dtype=np.int8)
    Z = np.array(2 * (np.random.rand(N, L) > .5) - 1, dtype=np.int8)
    X = np.array(2 * np.dot(Z == 1, U.transpose() == 1) - 1, dtype=np.int8)

    X[int(X.shape[0] / 2):, :] *= -1

    orm = lom.Machine()

    data = orm.add_matrix(val=X, sampling_indicator=False)

    layer = orm.add_layer(size=3,
                          child=data,
                          lbda_init=2.,
                          noise_model='or-link')

    layer.z.val = Z
    layer.u.val = U

    return layer


def time_lambda(layer):

    lom.lambda_update_wrappers.draw_lbda_or(layer.lbda)
    reps = 20

    print('start')
    start = time.time()
    for i in range(reps):
        lom.lambda_update_wrappers.draw_lbda_or(layer.lbda)

    end = time.time()

    return (end - start) / reps


if __name__ == "__main__":

    sizes = []
    numba_times = []
    cython_times = []
    out = []

    for D in [int(10**x) for x in range(2, 4)]:
        for N in [int(10**x) for x in range(2, 4)]:
            print(N, D)

            layer = generate_data(N, D)

            try:
                layer.machine.framework = 'cython'
                cython_time = time_lambda(layer)

                out.append([np.log10(cython_time),
                            'cython',
                            int(np.log10(N * D))])
            except:
                print('cython sampling not supported.')

            layer.machine.framework = 'numba'
            numba_time = time_lambda(layer)

            out.append([np.log10(numba_time),
                        'numba',
                        int(np.log10(N * D))])

    out = pd.DataFrame(
        out,
        columns=['log10 time in s', 'framework', 'log10 number of data-points'])

    # out.to_csv('./bm_lambda_mac_reference.csv')

    reference = pd.read_csv('./bm_lambda_mac_reference.csv', index_col=0)

    out = pd.concat([out, reference])

    sns.pointplot(
        data=out,
        x='log10 number of data-points',
        y='log10 time in s',
        hue='framework')
    plt.savefig('./lambda_bm.png')
    plt.show()
