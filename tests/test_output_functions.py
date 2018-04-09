import numpy as np
import lom
import lom.experiments as experiments
import itertools
import lom.auxiliary_functions as aux
import lom._numba.matrix_updates_numba as mupd
import lom._numba.lambda_updates_numba as lupd


def scalar_output_python(model, z, u):
    """
    Return output (True, False) for two binary vectors (-1, 1) given
    a machine architecture.
    Here in elegant but poorly scalable python
    """

    op1 = aux.get_lop(model.split('-')[0])
    op2 = aux.get_lop(model.split('-')[1])

    out = op1(np.array([op2(np.array([z[l], u[l]])) for l in range(len(z))]))

    return 2 * out - 1  # map to {-1,1}


def scalar_output_python_fuzzy(model, z, u):
    """
    Return output (True, False) for two binary vectors (-1, 1) given
    a machine architecture.
    Here in elegant but poorly scalable python
    """

    op1 = aux.get_fuzzy_lop(model.split('-')[0])
    op2 = aux.get_fuzzy_lop(model.split('-')[1])

    out = op1(np.array([op2(np.array([z[l], u[l]])) for l in range(len(z))]))

    return out


def test_all_scalar_output():
    """
    Test numba implementations against the simple python implementation
    """

    # generate random data
    L = 3

    for model in aux.canonical_loms():
        for randiter in range(10):

            z = 2 * np.array(np.random.rand(L) > .5, dtype=np.int8) - 1
            u = 2 * np.array(np.random.rand(L) > .5, dtype=np.int8) - 1

            numba_fct = lupd.get_scalar_output_function_2d(model, fuzzy=False)

            try:
                assert numba_fct(z, u) == scalar_output_python(model, z, u)
            except:
                raise ValueError(
                    'Scalar output function for ' +
                    model + ' failed.')


def test_all_scalar_output_fuzzy():
    """
    Test output for fuzzy factors, i.e. means in [0,1]
    """

    L = 3

    for model in aux.canonical_loms():
        for randiter in range(3):

            z = np.random.rand(L)
            u = np.random.rand(L)

            numba_fct = lupd.get_scalar_output_function_2d(model, fuzzy=True)
            numba_out = numba_fct(z, u)
            python_out = scalar_output_python_fuzzy(model, z, u)

            try:
                assert abs(numba_out - python_out) < 1e-12
            except:
                import pdb
                pdb.set_trace()
                raise ValueError(
                    'Fuzzy scalar output function for ' +
                    model + ' failed.')


if __name__ == '__main__':
    test_all_scalar_output()
    test_all_scalar_output_fuzzy()
