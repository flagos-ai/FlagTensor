#include "flagtensor/utils.h"
#include <dlfcn.h>

namespace flagtensor::utils {

std::filesystem::path get_path_of_this_library() {
  static const std::filesystem::path cached_path = []() {
    Dl_info dl_info;
    if (dladdr(reinterpret_cast<void*>(&get_path_of_this_library), &dl_info) && dl_info.dli_fname) {
      return std::filesystem::canonical(dl_info.dli_fname);
    } else {
      throw std::runtime_error("cannot get the path of liboperators.so");
    }
  }();
  return cached_path;
}

std::filesystem::path get_triton_src_path() {
  const static std::filesystem::path triton_src_dir = []() {
    // Check installed path: <libdir>/flagtensor/triton_src
    std::filesystem::path installed_path =
        get_path_of_this_library().parent_path() / "flagtensor" / "triton_src";
    if (std::filesystem::exists(installed_path)) {
      return installed_path;
    }
    // Fallback: <libdir>/triton_src (development layout)
    std::filesystem::path dev_path =
        std::filesystem::path(__FILE__).parent_path().parent_path() / "lib" / "triton_src";
    if (std::filesystem::exists(dev_path)) {
      return std::filesystem::canonical(dev_path);
    }
    return installed_path;
  }();
  return triton_src_dir;
}

int64_t next_power_of_2(int64_t n) {
  if (n <= 1) return 1;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}

bool broadcastable_to(at::IntArrayRef s1, at::IntArrayRef s2) {
  size_t r1 = s1.size();
  if (r1 == 0) return true;
  size_t r2 = s2.size();
  if (r2 == 0) return false;
  if (r1 > r2) return false;
  size_t d = r2 - r1;
  for (size_t i = 0; i < r1; ++i) {
    if (s1[i] == 1 || s1[i] == s2[d + i]) continue;
    return false;
  }
  return true;
}

int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

}  // namespace flagtensor::utils