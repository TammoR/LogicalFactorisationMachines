import numpy as np
import lom
import lom._numba.lambda_updates_numba as lambda_updates_numba
from lom.auxiliary_functions import generate_orm_product


def test_lambda_update_or():

	U, Z, X = generate_orm_product()

	orm = lom.Machine()

	data = orm.add_matrix(val=X, sampling_indicator=False)

	layer = orm.add_layer(size=3, child=data, lbda_init=2., 
	                       noise_model='or-link')

	layer.members()[0].val = Z
	layer.members()[1].val = U

	lambda_updates_numba.draw_lbda_or_numba(layer.lbda)

	assert layer.lbda() == -np.log( ( ( 10000 + 2 ) / ( 10000 + 1 ) ) - 1  )


if __name__ == "__main__":
	
	test_lambda_update_or()