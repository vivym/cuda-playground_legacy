#pragma once
#include <torch/types.h>

namespace cuda_playground {

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::tuple<float, float, float>> window_partition_cuda(const at::Tensor &window_ids);

std::tuple<float, float> sort_perf_cuda(const at::Tensor &window_ids1, const at::Tensor &window_ids2);


inline std::tuple<at::Tensor, at::Tensor, at::Tensor, std::tuple<float, float, float>> window_partition(const at::Tensor &window_ids) {
  if (!window_ids.device().is_cuda()) {
    AT_ERROR("window_partition is not implemented on CPU yet!");
  }

  return window_partition_cuda(window_ids);
}

std::tuple<float, float> sort_perf(const at::Tensor &window_ids1, const at::Tensor &window_ids2) {
  if (!window_ids1.device().is_cuda() || !window_ids2.device().is_cuda()) {
    AT_ERROR("window_partition is not implemented on CPU yet!");
  }

  return sort_perf_cuda(window_ids1, window_ids2);
}

} // cuda_playground