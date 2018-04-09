import numpy as np
import lom
import lom._numba.lambda_updates_numba as lambda_updates_numba
from lom.auxiliary_functions import generate_orm_product


def test_lambda_update_or():

    model = 'OR-AND'

    U, Z, X = generate_orm_product()

    orm = lom.Machine()

    data = orm.add_matrix(val=X, fixed=True)

    layer = orm.add_layer(latent_size=3, child=data, model=model)

    layer.factors[0].val = Z
    layer.factors[1].val = U

    assert np.all(np.dot(Z == 1, U.transpose() == 1) == (data() == 1))

    lbda_update_fct = lambda_updates_numba.make_lbda_update_fct(model, 2)
    lbda_update_fct(layer.lbda)

    ND = np.prod(X.shape)

    print('\n')
    print(ND)

    assert layer.lbda() == -np.log(((ND + 2) / (ND + 1)) - 1)


if __name__ == "__main__":

    test_lambda_update_or()
