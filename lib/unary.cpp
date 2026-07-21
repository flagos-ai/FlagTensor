#include "flagtensor/operators.h"
#include "flagtensor/backend_utils.h"
#include "flagtensor/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flagtensor {
using namespace triton_jit;

namespace {

at::Tensor unary_op(const at::Tensor& input, int op_mode) {
  TORCH_CHECK(input.is_cuda(), "input must be on CUDA device");
  int64_t n_elements = input.numel();
  if (n_elements == 0) return at::empty_like(input);

  at::Tensor out = at::empty_like(input);
  at::Tensor in_contig = input.contiguous();

  int64_t tile_size = 1024;
  int num_warps = 8;
  int num_stages = 1;
  uint64_t grid_x = (static_cast<uint64_t>(n_elements) + tile_size - 1) / tile_size;

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_triton_src_path() / "unary_kernel.py"), "unary_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream(out.device());
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f(raw_stream,
    grid_x, 1, 1,
    num_warps, num_stages,
    in_contig, out, n_elements,
    /* op_mode = */ op_mode,
    /* BLOCK_SIZE = */ tile_size);
  return out;
}

}  // anonymous namespace

at::Tensor abs(const at::Tensor& input)     { return unary_op(input, 0); }
at::Tensor acos(const at::Tensor& input)    { return unary_op(input, 1); }
at::Tensor acosh(const at::Tensor& input)   { return unary_op(input, 2); }
at::Tensor asin(const at::Tensor& input)    { return unary_op(input, 3); }
at::Tensor asinh(const at::Tensor& input)   { return unary_op(input, 4); }
at::Tensor atan(const at::Tensor& input)    { return unary_op(input, 5); }
at::Tensor atanh(const at::Tensor& input)   { return unary_op(input, 6); }
at::Tensor ceil(const at::Tensor& input)    { return unary_op(input, 7); }
at::Tensor cos(const at::Tensor& input)     { return unary_op(input, 8); }
at::Tensor cosh(const at::Tensor& input)    { return unary_op(input, 9); }
at::Tensor exp(const at::Tensor& input)     { return unary_op(input, 10); }
at::Tensor floor(const at::Tensor& input)   { return unary_op(input, 11); }
at::Tensor identity(const at::Tensor& input){ return unary_op(input, 12); }
at::Tensor log(const at::Tensor& input)     { return unary_op(input, 13); }
at::Tensor mish(const at::Tensor& input)    { return unary_op(input, 14); }
at::Tensor neg(const at::Tensor& input)     { return unary_op(input, 15); }
at::Tensor rcp(const at::Tensor& input)     { return unary_op(input, 16); }
at::Tensor relu(const at::Tensor& input)    { return unary_op(input, 17); }
at::Tensor sigmoid(const at::Tensor& input) { return unary_op(input, 18); }
at::Tensor sin(const at::Tensor& input)     { return unary_op(input, 19); }
at::Tensor sinh(const at::Tensor& input)    { return unary_op(input, 20); }
at::Tensor soft_plus(const at::Tensor& input){return unary_op(input, 21); }
at::Tensor soft_sign(const at::Tensor& input){return unary_op(input, 22); }
at::Tensor sqrt(const at::Tensor& input)    { return unary_op(input, 23); }
at::Tensor swish(const at::Tensor& input)   { return unary_op(input, 24); }
at::Tensor tan(const at::Tensor& input)     { return unary_op(input, 25); }
at::Tensor tanh(const at::Tensor& input)    { return unary_op(input, 26); }
at::Tensor conj(const at::Tensor& input)    { return unary_op(input, 27); }

}  // namespace flagtensor
