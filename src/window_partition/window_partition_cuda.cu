#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

namespace cuda_playground {

std::tuple<at::Tensor, at::Tensor, at::Tensor> window_partition_cuda(const at::Tensor &window_ids) {
  at::cuda::CUDAGuard device_guard(window_ids.device());

  auto num_items = window_ids.size(0);

  at::Tensor sorted_indices = at::arange(num_items, window_ids.options().dtype(at::kLong));
  at::Tensor output_window_ids = at::empty({num_items,}, window_ids.options());
  at::Tensor window_sizes = at::empty({num_items,}, window_ids.options());

  auto window_ids_ptr = window_ids.data_ptr<int>();
  auto sorted_indices_ptr = sorted_indices.data_ptr<int64_t>();
  auto output_window_ids_ptr = output_window_ids.data_ptr<int>();
  auto window_sizes_ptr = window_sizes.data_ptr<int>();

  thrust::sort_by_key(thrust::device, sorted_indices_ptr, sorted_indices_ptr + num_items, window_ids_ptr);

  thrust::device_vector<int> ones(num_items);
  thrust::fill(thrust::device, ones.begin(), ones.end(), 1);

  auto output_end = thrust::reduce_by_key(
    thrust::device,
    window_ids_ptr, window_ids_ptr + num_items,
    ones.begin(),
    output_window_ids_ptr,
    window_sizes_ptr);
  
  auto num_windows = output_end.second - window_sizes_ptr;
  std::initializer_list<at::indexing::TensorIndex> slice = {at::indexing::Slice(0, num_windows)};
  return {sorted_indices, output_window_ids.index(slice), window_sizes.index(slice)};
}

} // cuda_playground