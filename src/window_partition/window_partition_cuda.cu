#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

namespace cuda_playground {

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::tuple<float, float, float>> window_partition_cuda(const at::Tensor &window_ids) {
  at::cuda::CUDAGuard device_guard(window_ids.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto num_items = window_ids.size(0);

  float time_new_tensor = 0;
  float time_sort = 0;
  float time_reduce_by_key = 0;
  cudaEvent_t event_start, event_stop;

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, stream);

  // at::Tensor sorted_indices = at::arange(num_items, window_ids.options().dtype(at::kLong));
  at::Tensor sorted_indices = at::arange(num_items, window_ids.options());
  at::Tensor output_window_ids = at::empty({num_items,}, window_ids.options());
  at::Tensor window_sizes = at::empty({num_items,}, window_ids.options());
  at::Tensor ones = at::ones({num_items,}, window_ids.options());

  cudaEventRecord(event_stop, stream);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&time_new_tensor, event_start, event_stop);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);

  auto window_ids_ptr = window_ids.data_ptr<int32_t>();
  auto sorted_indices_ptr = sorted_indices.data_ptr<int32_t>();
  auto output_window_ids_ptr = output_window_ids.data_ptr<int32_t>();
  auto window_sizes_ptr = window_sizes.data_ptr<int32_t>();
  auto ones_ptr = ones.data_ptr<int32_t>();

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, stream);

  thrust::sort_by_key(thrust::device, window_ids_ptr, window_ids_ptr + num_items, sorted_indices_ptr);

  cudaEventRecord(event_stop, stream);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&time_sort, event_start, event_stop);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, stream);

  auto output_end = thrust::reduce_by_key(
    thrust::device,
    window_ids_ptr, window_ids_ptr + num_items,
    ones_ptr,
    output_window_ids_ptr,
    window_sizes_ptr);

  cudaEventRecord(event_stop, stream);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&time_reduce_by_key, event_start, event_stop);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);
  
  auto num_windows = output_end.second - window_sizes_ptr;
  std::initializer_list<at::indexing::TensorIndex> slice = {at::indexing::Slice(0, num_windows)};
  return {sorted_indices, output_window_ids.index(slice), window_sizes.index(slice), {time_new_tensor, time_sort, time_reduce_by_key}};
}

std::tuple<float, float> sort_perf_cuda(const at::Tensor &window_ids1, const at::Tensor &window_ids2) {
  at::cuda::CUDAGuard device_guard(window_ids1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto num_items = window_ids1.size(0);

  at::Tensor sorted_indices = at::arange(num_items, window_ids1.options());

  auto window_ids1_ptr = window_ids1.data_ptr<int32_t>();
  auto window_ids2_ptr = window_ids2.data_ptr<int32_t>();
  auto sorted_indices_ptr = sorted_indices.data_ptr<int32_t>();

  float time_sort = 0;
  float time_sort_by_key = 0;
  cudaEvent_t event_start, event_stop;

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, stream);

  thrust::sort(thrust::device, window_ids1_ptr, window_ids1_ptr + num_items);

  cudaEventRecord(event_stop, stream);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&time_sort, event_start, event_stop);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, stream);

  thrust::sort_by_key(thrust::device, window_ids2_ptr, window_ids2_ptr + num_items, sorted_indices_ptr);

  cudaEventRecord(event_stop, stream);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&time_sort_by_key, event_start, event_stop);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);

  return {time_sort, time_sort_by_key};
}

} // cuda_playground
