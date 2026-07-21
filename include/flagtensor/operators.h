#pragma once
#include <optional>
#include "torch/torch.h"

namespace flagtensor {

// === Unary operators (28 ops) ================================================
at::Tensor abs(const at::Tensor &input);
at::Tensor acos(const at::Tensor &input);
at::Tensor acosh(const at::Tensor &input);
at::Tensor asin(const at::Tensor &input);
at::Tensor asinh(const at::Tensor &input);
at::Tensor atan(const at::Tensor &input);
at::Tensor atanh(const at::Tensor &input);
at::Tensor ceil(const at::Tensor &input);
at::Tensor conj(const at::Tensor &input);
at::Tensor cos(const at::Tensor &input);
at::Tensor cosh(const at::Tensor &input);
at::Tensor exp(const at::Tensor &input);
at::Tensor floor(const at::Tensor &input);
at::Tensor identity(const at::Tensor &input);
at::Tensor log(const at::Tensor &input);
at::Tensor mish(const at::Tensor &input);
at::Tensor neg(const at::Tensor &input);
at::Tensor rcp(const at::Tensor &input);
at::Tensor relu(const at::Tensor &input);
at::Tensor sigmoid(const at::Tensor &input);
at::Tensor sin(const at::Tensor &input);
at::Tensor sinh(const at::Tensor &input);
at::Tensor soft_plus(const at::Tensor &input);
at::Tensor soft_sign(const at::Tensor &input);
at::Tensor sqrt(const at::Tensor &input);
at::Tensor swish(const at::Tensor &input);
at::Tensor tan(const at::Tensor &input);
at::Tensor tanh(const at::Tensor &input);

// === Binary operators (4 ops) ================================================
at::Tensor add(const at::Tensor &a, const at::Tensor &b);
at::Tensor mul(const at::Tensor &a, const at::Tensor &b);
at::Tensor max(const at::Tensor &a, const at::Tensor &b);
at::Tensor min(const at::Tensor &a, const at::Tensor &b);

// === Contraction operators (3 ops) ===========================================
// General tensor contraction: C = A * B  (with optional transpose)
at::Tensor contraction(const at::Tensor &a, const at::Tensor &b,
                       bool trans_a = false, bool trans_b = false);
// Three-input tensor contraction with element-wise trinary reduction
at::Tensor contraction_trinary(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c);
// Generic element-wise trinary operator
at::Tensor elementwise_trinary(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c);

// === Sparse operators (1 op) =================================================
// Block-sparse tensor contraction
at::Tensor block_sparse_contraction(const at::Tensor &a, const at::Tensor &b,
                                    const at::Tensor &a_block_desc, const at::Tensor &b_block_desc);

}  // namespace flagtensor