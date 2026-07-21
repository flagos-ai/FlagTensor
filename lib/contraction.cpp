#include "flagtensor/operators.h"
#include "flagtensor/backend_utils.h"
#include "flagtensor/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flagtensor {
using namespace triton_jit;

at::Tensor contraction(const at::Tensor& a, const at::Tensor& b, bool trans_a, bool trans_b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be on CUDA device");

  int64_t M, N, K;
  int64_t stride_am, stride_ak, stride_bk, stride_bn;

  if (!trans_a) {
    M = a.size(0); K = a.size(1);
    stride_am = a.stride(0); stride_ak = a.stride(1);
  } else {
    M = a.size(1); K = a.size(0);
    stride_am = a.stride(1); stride_ak = a.stride(0);
  }

  if (!trans_b) {
    TORCH_CHECK(b.size(0) == K, "inner dimensions must match");
    N = b.size(1);
    stride_bk = b.stride(0); stride_bn = b.stride(1);
  } else {
    TORCH_CHECK(b.size(1) == K, "inner dimensions must match");
    N = b.size(0);
    stride_bk = b.stride(1); stride_bn = b.stride(0);
  }

  at::Tensor out = at::empty({M, N}, a.options());
  int64_t stride_cm = out.stride(0), stride_cn = out.stride(1);

  constexpr int BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32, GROUP_M = 8;
  int num_warps = 4, num_stages = 3;
  uint64_t grid_x = utils::cdiv(M, BLOCK_M) * utils::cdiv(N, BLOCK_N);

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_triton_src_path() / "contraction_kernel.py"), "contraction_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream(out.device());
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f(raw_stream,
    grid_x, 1, 1,
    num_warps, num_stages,
    a, b, out,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
  return out;
}

at::Tensor contraction_trinary(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "inputs must be on CUDA device");
  // contraction_trinary: a * b + c
  at::Tensor ab = contraction(a, b, false, false);
  at::Tensor out = at::empty_like(ab);
  int64_t n_elements = ab.numel();
  if (n_elements == 0) return out;

  at::Tensor c_b = c.broadcast_to(ab.sizes()).contiguous();
  at::Tensor ab_c = ab.contiguous();

  int64_t tile_size = 1024;
  uint64_t grid_x = (static_cast<uint64_t>(n_elements) + tile_size - 1) / tile_size;

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_triton_src_path() / "contraction_kernel.py"),
      "elementwise_trinary_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream(out.device());
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f(raw_stream, grid_x, 1, 1, 8, 1,
    ab_c, c_b, at::zeros({1}, a.options()),  // dummy c -> uses ab*c+0  pattern
    out, n_elements,
    /* BLOCK_SIZE = */ tile_size);
  // NOTE: kernel signature is (a, b, c, out, n, BLOCK_SIZE) computing a*b+c
  // For contraction_trinary we want ab + c, so pass ab as a, c as b, zeros as c

  // Fix: actually elementwise_trinary_kernel computes a*b + c
  // So we need to pass ab as a, a tensor of ones as b, and c as c
  // Let's reuse the kernel differently:
  at::Tensor ones = at::ones({n_elements}, a.options());
  f(raw_stream, grid_x, 1, 1, 8, 1,
    ab_c, ones, c_b, out, n_elements, tile_size);
  return out;
}

at::Tensor elementwise_trinary(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "inputs must be on CUDA device");
  auto out_shape = at::infer_size(at::infer_size(a.sizes(), b.sizes()), c.sizes());
  at::Tensor a_b = a.broadcast_to(out_shape).contiguous();
  at::Tensor b_b = b.broadcast_to(out_shape).contiguous();
  at::Tensor c_b = c.broadcast_to(out_shape).contiguous();
  int64_t n_elements = a_b.numel();
  at::Tensor out = at::empty(out_shape, a.options());

  int64_t tile_size = 1024;
  uint64_t grid_x = (static_cast<uint64_t>(n_elements) + tile_size - 1) / tile_size;

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_triton_src_path() / "contraction_kernel.py"),
      "elementwise_trinary_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream(out.device());
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f(raw_stream, grid_x, 1, 1, 8, 1,
    a_b, b_b, c_b, out, n_elements, tile_size);
  return out;
}

}  // namespace flagtensor