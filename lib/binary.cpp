#include "flagtensor/operators.h"
#include "flagtensor/backend_utils.h"
#include "flagtensor/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flagtensor {
using namespace triton_jit;

namespace {

at::Tensor binary_op(const at::Tensor& a, const at::Tensor& b, int op_mode) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be on CUDA device");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "input dtypes must match");

  // Broadcast shapes
  auto out_shape = at::infer_size(a.sizes(), b.sizes());
  at::Tensor a_b = a.broadcast_to(out_shape).contiguous();
  at::Tensor b_b = b.broadcast_to(out_shape).contiguous();
  int64_t n_elements = a_b.numel();
  if (n_elements == 0) return at::empty(out_shape, a.options());

  at::Tensor out = at::empty(out_shape, a.options());

  int64_t tile_size = 1024;
  int num_warps = 8;
  int num_stages = 1;
  uint64_t grid_x = (static_cast<uint64_t>(n_elements) + tile_size - 1) / tile_size;

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_triton_src_path() / "binary_kernel.py"), "binary_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream(out.device());
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f(raw_stream,
    grid_x, 1, 1,
    num_warps, num_stages,
    a_b, b_b, out, n_elements,
    /* op_mode = */ op_mode,
    /* BLOCK_SIZE = */ tile_size);
  return out;
}

}  // anonymous namespace

at::Tensor add(const at::Tensor& a, const at::Tensor& b) { return binary_op(a, b, 0); }
at::Tensor mul(const at::Tensor& a, const at::Tensor& b) { return binary_op(a, b, 1); }
at::Tensor max(const at::Tensor& a, const at::Tensor& b) { return binary_op(a, b, 2); }
at::Tensor min(const at::Tensor& a, const at::Tensor& b) { return binary_op(a, b, 3); }

}  // namespace flagtensor
