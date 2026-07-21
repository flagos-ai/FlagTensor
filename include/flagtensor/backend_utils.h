#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include "torch/torch.h"

namespace flagtensor {
namespace backend {

using StreamType = c10::cuda::CUDAStream;
using RawStreamType = CUstream;

inline StreamType getCurrentStream(const at::Device& device) {
  return c10::cuda::getCurrentCUDAStream(device.index());
}

inline StreamType getCurrentStream() {
  return c10::cuda::getCurrentCUDAStream();
}

inline RawStreamType getRawStream(const StreamType& stream) {
  return stream.stream();
}

inline at::DeviceType getBackendDeviceType() {
  return at::kCUDA;
}

inline at::Device getCurrentDevice() {
  return at::Device(at::kCUDA, at::cuda::current_device());
}

inline at::Device getDefaultDevice(int index = 0) {
  return at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(index));
}

}  // namespace backend
}  // namespace flagtensor