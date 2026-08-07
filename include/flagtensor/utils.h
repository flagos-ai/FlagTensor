#pragma once
#include <filesystem>
#include <string>
#include "torch/torch.h"

namespace flagtensor::utils {

std::filesystem::path get_path_of_this_library();
std::filesystem::path get_triton_src_path();
int64_t next_power_of_2(int64_t n);
bool broadcastable_to(at::IntArrayRef s1, at::IntArrayRef s2);
int cdiv(int a, int b);

}  // namespace flagtensor::utils
