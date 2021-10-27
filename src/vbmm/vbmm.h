#pragma once
#include <torch/types.h>

namespace cuda_playground {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, float, float, float, float> vbmm_cuda(
    bool transA, bool transB,
    const std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int> &mA,
    const std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int> &mB);

std::vector<at::Tensor> batched_gemm_cuda(
    const std::vector<at::Tensor>& mat1,
    const std::vector<at::Tensor>& mat2);

at::Tensor sgemm_test(const at::Tensor &mat1, const at::Tensor &mat2);

at::Tensor dgemm_test(const at::Tensor &mat1, const at::Tensor &mat2);

at::Tensor sgemm_cublas(const at::Tensor &mat1, const at::Tensor &mat2);

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, float, float, float, float> vbmm(
    bool transA, bool transB,
    const std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int> &mA,
    const std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int> &mB) {
  if (!std::get<0>(mA).device().is_cuda()) {
    AT_ERROR("window_partition is not implemented on CPU yet!");
  }

  return vbmm_cuda(transA, transB, mA, mB);
}

} // cuda_playground
