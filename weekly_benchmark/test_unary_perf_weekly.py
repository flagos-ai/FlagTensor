import pytest

from benchmark.test_CUTENSOR_OP_ABS_perf import test_abs_perf as _test_abs_perf
from benchmark.test_CUTENSOR_OP_ACOS_perf import test_acos_perf as _test_acos_perf
from benchmark.test_CUTENSOR_OP_ACOSH_perf import test_acosh_perf as _test_acosh_perf
from benchmark.test_CUTENSOR_OP_ADD_perf import test_add_perf as _test_add_perf
from benchmark.test_CUTENSOR_OP_ASIN_perf import test_asin_perf as _test_asin_perf
from benchmark.test_CUTENSOR_OP_ASINH_perf import test_asinh_perf as _test_asinh_perf
from benchmark.test_CUTENSOR_OP_ATAN_perf import test_atan_perf as _test_atan_perf
from benchmark.test_CUTENSOR_OP_ATANH_perf import test_atanh_perf as _test_atanh_perf
from benchmark.test_CUTENSOR_OP_CEIL_perf import test_ceil_perf as _test_ceil_perf
from benchmark.test_CUTENSOR_OP_CONJ_perf import test_conj_perf as _test_conj_perf
from benchmark.test_CUTENSOR_OP_COS_perf import test_cos_perf as _test_cos_perf
from benchmark.test_CUTENSOR_OP_COSH_perf import test_cosh_perf as _test_cosh_perf
from benchmark.test_CUTENSOR_OP_EXP_perf import test_exp_perf as _test_exp_perf
from benchmark.test_CUTENSOR_OP_FLOOR_perf import test_floor_perf as _test_floor_perf
from benchmark.test_CUTENSOR_OP_IDENTITY_perf import test_identity_perf as _test_identity_perf
from benchmark.test_CUTENSOR_OP_LOG_perf import test_log_perf as _test_log_perf
from benchmark.test_CUTENSOR_OP_MAX_perf import test_max_perf as _test_max_perf
from benchmark.test_CUTENSOR_OP_MIN_perf import test_min_perf as _test_min_perf
from benchmark.test_CUTENSOR_OP_MISH_perf import test_mish_perf as _test_mish_perf
from benchmark.test_CUTENSOR_OP_MUL_perf import test_mul_perf as _test_mul_perf
from benchmark.test_CUTENSOR_OP_NEG_perf import test_neg_perf as _test_neg_perf
from benchmark.test_CUTENSOR_OP_RCP_perf import test_rcp_perf as _test_rcp_perf
from benchmark.test_CUTENSOR_OP_RELU_perf import test_relu_perf as _test_relu_perf
from benchmark.test_CUTENSOR_OP_SIGMOID_perf import test_sigmoid_perf as _test_sigmoid_perf
from benchmark.test_CUTENSOR_OP_SIN_perf import test_sin_perf as _test_sin_perf
from benchmark.test_CUTENSOR_OP_SINH_perf import test_sinh_perf as _test_sinh_perf
from benchmark.test_CUTENSOR_OP_SOFT_PLUS_perf import test_soft_plus_perf as _test_soft_plus_perf
from benchmark.test_CUTENSOR_OP_SOFT_SIGN_perf import test_soft_sign_perf as _test_soft_sign_perf
from benchmark.test_CUTENSOR_OP_SQRT_perf import test_sqrt_perf as _test_sqrt_perf
from benchmark.test_CUTENSOR_OP_SWISH_perf import test_swish_perf as _test_swish_perf
from benchmark.test_CUTENSOR_OP_TAN_perf import test_tan_perf as _test_tan_perf
from benchmark.test_CUTENSOR_OP_TANH_perf import test_tanh_perf as _test_tanh_perf


def _mark_perf(op_marker, func):
    return pytest.mark.performance(getattr(pytest.mark, op_marker)(func))


test_abs_perf_weekly = _mark_perf("abs", _test_abs_perf)
test_acos_perf_weekly = _mark_perf("acos", _test_acos_perf)
test_acosh_perf_weekly = _mark_perf("acosh", _test_acosh_perf)
test_add_perf_weekly = _mark_perf("add", _test_add_perf)
test_asin_perf_weekly = _mark_perf("asin", _test_asin_perf)
test_asinh_perf_weekly = _mark_perf("asinh", _test_asinh_perf)
test_atan_perf_weekly = _mark_perf("atan", _test_atan_perf)
test_atanh_perf_weekly = _mark_perf("atanh", _test_atanh_perf)
test_ceil_perf_weekly = _mark_perf("ceil", _test_ceil_perf)
test_conj_perf_weekly = _mark_perf("conj", _test_conj_perf)
test_cos_perf_weekly = _mark_perf("cos", _test_cos_perf)
test_cosh_perf_weekly = _mark_perf("cosh", _test_cosh_perf)
test_exp_perf_weekly = _mark_perf("exp", _test_exp_perf)
test_floor_perf_weekly = _mark_perf("floor", _test_floor_perf)
test_identity_perf_weekly = _mark_perf("identity", _test_identity_perf)
test_log_perf_weekly = _mark_perf("log", _test_log_perf)
test_max_perf_weekly = _mark_perf("max", _test_max_perf)
test_min_perf_weekly = _mark_perf("min", _test_min_perf)
test_mish_perf_weekly = _mark_perf("mish", _test_mish_perf)
test_mul_perf_weekly = _mark_perf("mul", _test_mul_perf)
test_neg_perf_weekly = _mark_perf("neg", _test_neg_perf)
test_rcp_perf_weekly = _mark_perf("rcp", _test_rcp_perf)
test_relu_perf_weekly = _mark_perf("relu", _test_relu_perf)
test_sigmoid_perf_weekly = _mark_perf("sigmoid", _test_sigmoid_perf)
test_sin_perf_weekly = _mark_perf("sin", _test_sin_perf)
test_sinh_perf_weekly = _mark_perf("sinh", _test_sinh_perf)
test_soft_plus_perf_weekly = _mark_perf("soft_plus", _test_soft_plus_perf)
test_soft_sign_perf_weekly = _mark_perf("soft_sign", _test_soft_sign_perf)
test_sqrt_perf_weekly = _mark_perf("sqrt", _test_sqrt_perf)
test_swish_perf_weekly = _mark_perf("swish", _test_swish_perf)
test_tan_perf_weekly = _mark_perf("tan", _test_tan_perf)
test_tanh_perf_weekly = _mark_perf("tanh", _test_tanh_perf)
