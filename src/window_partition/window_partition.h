#pragma once
#include <torch/types.h>

namespace cuda_playground {

std::tuple<at::Tensor, at::Tensor, at::Tensor> window_partition_cuda(const at::Tensor &window_ids);


inline std::tuple<at::Tensor, at::Tensor, at::Tensor> window_partition(const at::Tensor &window_ids) {
  if (!window_ids.device().is_cuda()) {
    AT_ERROR("window_partition is not implemented on CPU yet!");
  }

  return window_partition_cuda(window_ids);
}

} // cuda_playground