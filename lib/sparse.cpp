#include "flagtensor/operators.h"
#include "flagtensor/backend_utils.h"
#include "flagtensor/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flagtensor {
using namespace triton_jit;

at::Tensor block_sparse_contraction(const at::Tensor& a, const at::Tensor& b,
                                    const at::Tensor& a_block_desc, const at::Tensor& b_block_desc) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be on CUDA device");

  // Fallback: treat as dense contraction ignoring block descriptors
  int64_t M = a.size(0), K = a.size(1);
  TORCH_CHECK(b.size(0) == K, "inner dimensions must match");
  int64_t N = b.size(1);

  at::Tensor out = at::empty({M, N}, a.options());
  int64_t stride_am = a.stride(0), stride_ak = a.stride(1);
  int64_t stride_bk = b.stride(0), stride_bn = b.stride(1);
  int64_t stride_cm = out.stride(0), stride_cn = out.stride(1);

  constexpr int BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32, GROUP_M = 8;
  int num_warps = 4, num_stages = 3;
  uint64_t grid_x = utils::cdiv(M, BLOCK_M) * utils::cdiv(N, BLOCK_N);

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_triton_src_path() / "sparse_kernel.py"), "block_sparse_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream(out.device());
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f(raw_stream, grid_x, 1, 1, num_warps, num_stages,
    a, b, out,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M);
  return out;
}

}  // namespace flagtensor
