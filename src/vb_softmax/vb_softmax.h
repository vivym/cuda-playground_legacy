#pragma once
#include <torch/types.h>

namespace cuda_playground {

at::Tensor vb_softmax(const std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int> &mat, int dim);

} // cuda_playground
