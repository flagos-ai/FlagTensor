# Standalone unary kernel for C++ wrapper (loaded via TritonJIT).
# op_mode selects the operation:
#   0=abs, 1=acos, 2=acosh, 3=asin, 4=asinh, 5=atan, 6=atanh,
#   7=ceil, 8=cos, 9=cosh, 10=exp, 11=floor, 12=identity,
#   13=log, 14=mish, 15=neg, 16=rcp, 17=relu, 18=sigmoid,
#   19=sin, 20=sinh, 21=soft_plus, 22=soft_sign, 23=sqrt,
#   24=swish, 25=tan, 26=tanh, 27=conj
#
# All float32 computation, output cast to input dtype.

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def unary_kernel(
    in_ptr, out_ptr, n_elements,
    op_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)

    # Compute scalar result based on op_mode
    if op_mode == 0:   # abs
        y = tl.abs(xf)
    elif op_mode == 1:  # acos
        y = libdevice.acos(xf)
    elif op_mode == 2:  # acosh
        y = libdevice.acosh(xf)
    elif op_mode == 3:  # asin
        y = libdevice.asin(xf)
    elif op_mode == 4:  # asinh
        y = libdevice.asinh(xf)
    elif op_mode == 5:  # atan
        y = libdevice.atan(xf)
    elif op_mode == 6:  # atanh
        y = libdevice.atanh(xf)
    elif op_mode == 7:  # ceil
        y = tl.ceil(xf)
    elif op_mode == 8:  # cos
        y = tl.cos(xf)
    elif op_mode == 9:  # cosh
        y = 0.5 * (tl.exp(xf) + tl.exp(-xf))
    elif op_mode == 10:  # exp
        y = tl.exp(xf)
    elif op_mode == 11:  # floor
        y = tl.floor(xf)
    elif op_mode == 12:  # identity
        y = xf
    elif op_mode == 13:  # log
        y = tl.log(xf)
    elif op_mode == 14:  # mish
        abs_x = tl.abs(xf)
        softplus = tl.log(1.0 + tl.exp(-abs_x)) + tl.where(xf > 0, xf, 0.0)
        exp_neg_twice = tl.exp(-2.0 * softplus)
        tanh_sp = (1.0 - exp_neg_twice) / (1.0 + exp_neg_twice)
        y = xf * tanh_sp
    elif op_mode == 15:  # neg
        y = -xf
    elif op_mode == 16:  # rcp
        y = 1.0 / xf
    elif op_mode == 17:  # relu
        y = tl.where(xf > 0, xf, 0.0)
    elif op_mode == 18:  # sigmoid
        y = 1.0 / (1.0 + tl.exp(-xf))
    elif op_mode == 19:  # sin
        y = tl.sin(xf)
    elif op_mode == 20:  # sinh
        y = 0.5 * (tl.exp(xf) - tl.exp(-xf))
    elif op_mode == 21:  # soft_plus
        abs_x = tl.abs(xf)
        y = tl.log(1.0 + tl.exp(-abs_x)) + tl.where(xf > 0, xf, 0.0)
    elif op_mode == 22:  # soft_sign
        y = xf / (tl.abs(xf) + 1.0)
    elif op_mode == 23:  # sqrt
        y = tl.sqrt(xf)
    elif op_mode == 24:  # swish
        y = xf / (1.0 + tl.exp(-xf))
    elif op_mode == 25:  # tan
        y = libdevice.tan(xf)
    elif op_mode == 26:  # tanh
        exp_neg_twice = tl.exp(-2.0 * xf)
        y = (1.0 - exp_neg_twice) / (1.0 + exp_neg_twice)
    elif op_mode == 27:  # conj (passthrough for real, conjugate for complex)
        y = xf
    else:
        y = xf  # fallback identity

    # Cast back to original dtype if not float32
    y = y.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)
