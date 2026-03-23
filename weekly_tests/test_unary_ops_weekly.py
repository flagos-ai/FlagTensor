import pytest

from ctests.test_CUTENSOR_OP_ABS import test_abs_correctness as _test_abs_correctness
from ctests.test_CUTENSOR_OP_ACOS import test_acos_correctness as _test_acos_correctness
from ctests.test_CUTENSOR_OP_ACOSH import test_acosh_correctness as _test_acosh_correctness
from ctests.test_CUTENSOR_OP_ADD import test_add_correctness as _test_add_correctness
from ctests.test_CUTENSOR_OP_ASIN import test_asin_correctness as _test_asin_correctness
from ctests.test_CUTENSOR_OP_ASINH import test_asinh_correctness as _test_asinh_correctness
from ctests.test_CUTENSOR_OP_ATAN import test_atan_correctness as _test_atan_correctness
from ctests.test_CUTENSOR_OP_ATANH import test_atanh_correctness as _test_atanh_correctness
from ctests.test_CUTENSOR_OP_CEIL import test_ceil_correctness as _test_ceil_correctness
from ctests.test_CUTENSOR_OP_CONJ import test_conj_correctness as _test_conj_correctness
from ctests.test_CUTENSOR_OP_COS import test_cos_correctness as _test_cos_correctness
from ctests.test_CUTENSOR_OP_COSH import test_cosh_correctness as _test_cosh_correctness
from ctests.test_CUTENSOR_OP_EXP import test_exp_correctness as _test_exp_correctness
from ctests.test_CUTENSOR_OP_FLOOR import test_floor_correctness as _test_floor_correctness
from ctests.test_CUTENSOR_OP_IDENTITY import test_identity_correctness as _test_identity_correctness
from ctests.test_CUTENSOR_OP_LOG import test_log_correctness as _test_log_correctness
from ctests.test_CUTENSOR_OP_MAX import test_max_correctness as _test_max_correctness
from ctests.test_CUTENSOR_OP_MIN import test_min_correctness as _test_min_correctness
from ctests.test_CUTENSOR_OP_MISH import test_mish_correctness as _test_mish_correctness
from ctests.test_CUTENSOR_OP_MUL import test_mul_correctness as _test_mul_correctness
from ctests.test_CUTENSOR_OP_NEG import test_neg_correctness as _test_neg_correctness
from ctests.test_CUTENSOR_OP_RCP import test_rcp_correctness as _test_rcp_correctness
from ctests.test_CUTENSOR_OP_RELU import test_relu_correctness as _test_relu_correctness
from ctests.test_CUTENSOR_OP_SIGMOID import test_sigmoid_correctness as _test_sigmoid_correctness
from ctests.test_CUTENSOR_OP_SIN import test_sin_correctness as _test_sin_correctness
from ctests.test_CUTENSOR_OP_SINH import test_sinh_correctness as _test_sinh_correctness
from ctests.test_CUTENSOR_OP_SOFT_PLUS import test_soft_plus_correctness as _test_soft_plus_correctness
from ctests.test_CUTENSOR_OP_SOFT_SIGN import test_soft_sign_correctness as _test_soft_sign_correctness
from ctests.test_CUTENSOR_OP_SQRT import test_sqrt_correctness as _test_sqrt_correctness
from ctests.test_CUTENSOR_OP_SWISH import test_swish_correctness as _test_swish_correctness
from ctests.test_CUTENSOR_OP_TAN import test_tan_correctness as _test_tan_correctness
from ctests.test_CUTENSOR_OP_TANH import test_tanh_correctness as _test_tanh_correctness


test_abs_weekly = pytest.mark.abs(_test_abs_correctness)
test_acos_weekly = pytest.mark.acos(_test_acos_correctness)
test_acosh_weekly = pytest.mark.acosh(_test_acosh_correctness)
test_add_weekly = pytest.mark.add(_test_add_correctness)
test_asin_weekly = pytest.mark.asin(_test_asin_correctness)
test_asinh_weekly = pytest.mark.asinh(_test_asinh_correctness)
test_atan_weekly = pytest.mark.atan(_test_atan_correctness)
test_atanh_weekly = pytest.mark.atanh(_test_atanh_correctness)
test_ceil_weekly = pytest.mark.ceil(_test_ceil_correctness)
test_conj_weekly = pytest.mark.conj(_test_conj_correctness)
test_cos_weekly = pytest.mark.cos(_test_cos_correctness)
test_cosh_weekly = pytest.mark.cosh(_test_cosh_correctness)
test_exp_weekly = pytest.mark.exp(_test_exp_correctness)
test_floor_weekly = pytest.mark.floor(_test_floor_correctness)
test_identity_weekly = pytest.mark.identity(_test_identity_correctness)
test_log_weekly = pytest.mark.log(_test_log_correctness)
test_max_weekly = pytest.mark.max(_test_max_correctness)
test_min_weekly = pytest.mark.min(_test_min_correctness)
test_mish_weekly = pytest.mark.mish(_test_mish_correctness)
test_mul_weekly = pytest.mark.mul(_test_mul_correctness)
test_neg_weekly = pytest.mark.neg(_test_neg_correctness)
test_rcp_weekly = pytest.mark.rcp(_test_rcp_correctness)
test_relu_weekly = pytest.mark.relu(_test_relu_correctness)
test_sigmoid_weekly = pytest.mark.sigmoid(_test_sigmoid_correctness)
test_sin_weekly = pytest.mark.sin(_test_sin_correctness)
test_sinh_weekly = pytest.mark.sinh(_test_sinh_correctness)
test_soft_plus_weekly = pytest.mark.soft_plus(_test_soft_plus_correctness)
test_soft_sign_weekly = pytest.mark.soft_sign(_test_soft_sign_correctness)
test_sqrt_weekly = pytest.mark.sqrt(_test_sqrt_correctness)
test_swish_weekly = pytest.mark.swish(_test_swish_correctness)
test_tan_weekly = pytest.mark.tan(_test_tan_correctness)
test_tanh_weekly = pytest.mark.tanh(_test_tanh_correctness)
