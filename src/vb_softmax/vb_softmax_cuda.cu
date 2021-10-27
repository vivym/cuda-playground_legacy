#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

namespace cuda_playground {

/// Allocator for Thrust to re-route its internal device allocations
/// to the THC allocator
class ThrustAllocator {
public:
  typedef char value_type;

  char* allocate(std::ptrdiff_t size) {
    return static_cast<char*>(c10::cuda::CUDACachingAllocator::raw_alloc(size));
  }

  void deallocate(char* p, size_t size) {
    c10::cuda::CUDACachingAllocator::raw_delete(p);
  }
};

int get_num_threads(int num_items) {
  const int thread_sizes[5] = {32, 64, 128, 256, 512};
  for (int i = 0; i != 5; i ++) {
    if (num_items <= thread_sizes[i]) {
      return thread_sizes[i];
    }
  }
  return thread_sizes[4];
}

template <typename scalar_t>
__global__ void vb_softmax_kernel(scalar_t *values, int *pool_sizes, int *pool_offsets, int batch_size, scalar_t *out_values) {
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  for (int index = tid + blkid * blksz, step = blksz * gridsz; index < batch_size; index += step) {
    int pool_offset = pool_offsets[index];
    scalar_t *values_i = values + pool_offset;
    scalar_t *out_values_i = out_values + pool_offset;
    const int pool_size = pool_sizes[index];
    const int stride = 1;

    // at::GenericPackedTensorAccessor<float, 1, at::RestrictPtrTraits, int32_t> accessor(values_i, &pool_size, &stride);
    // at::GenericPackedTensorAccessor<float, 1, at::RestrictPtrTraits, int32_t> out_accessor(out_values_i, &pool_size, &stride);

    scalar_t mx = -1e30;  // TODO: -std::numeric_limits<scalar_t>::infinity()
    scalar_t exp_sum = 0;

    for (int j = 0; j < pool_size; j ++) {
      mx = max(mx, values_i[j]);  // TODO: max<scalar_t>
    }

    for (int j = 0; j < pool_size; j ++) {
      auto v = __expf(values_i[j] - mx);    // TODO: __exp<scalar_t>
      out_values_i[j] = v;
      exp_sum += v;
    }

    for (int j = 0; j < pool_size; j ++) {
      out_values_i[j] *= 1.0 / exp_sum;
    }
  }
}

__global__ void generate_sizes_kernel(int *m_cumsum, int *m, int *n, int batch_size, int *sizes) {
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  for (int index = tid + blkid * blksz, step = blksz * gridsz; index < batch_size; index += step) {
    const int m_i = m[index];
    const int n_i = n[index];
    const int stride = 1;
    int *sizes_i = sizes + m_cumsum[index];
    // at::GenericPackedTensorAccessor<int, 1, at::RestrictPtrTraits, int32_t> accessor(sizes_i, &m_i, &stride);
    for (int j = 0; j < m_i; j ++) {
      sizes_i[j] = n_i;
    }
  }
}

std::tuple<at::Tensor, at::Tensor> get_offsets(const at::Tensor &m, const at::Tensor &n, int batch_size, const at::TensorOptions &options) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto m_ptr = m.data_ptr<int>();
  auto n_ptr = n.data_ptr<int>();

  at::Tensor m_cumsum = at::empty({batch_size}, options);
  auto m_cumsum_ptr = m_cumsum.data_ptr<int>();
  thrust::exclusive_scan(policy, m_ptr, m_ptr + batch_size, m_cumsum_ptr);

  int m_sum = thrust::reduce(policy, m_ptr, m_ptr + batch_size);
  at::Tensor sizes = at::empty({m_sum}, options);
  auto sizes_ptr = sizes.data_ptr<int>();

  {
    const int block_size = get_num_threads(batch_size);
    const int grid_size = (batch_size + block_size - 1) / block_size;

    generate_sizes_kernel<<<grid_size, block_size, 0, stream>>>(m_cumsum_ptr, m_ptr, n_ptr, batch_size, sizes_ptr);
  }

  at::Tensor offsets = at::empty({m_sum}, options);
  thrust::exclusive_scan(policy, sizes_ptr, sizes_ptr + m_sum, offsets.data_ptr<int>());

  return {sizes, offsets};
}

at::Tensor vb_softmax(const std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int> &mat, int dim) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

  at::Tensor values, offsets, m, n;
  int batch_size;
  std::tie(values, offsets, m, n, batch_size) = mat;

  at::Tensor pool_sizes, pool_offsets;
  std::tie(pool_sizes, pool_offsets) = get_offsets(m, n, batch_size, offsets.options());

  const int num_pool = pool_sizes.size(0);
  
  at::Tensor out_values = at::empty(values.sizes(), values.options());

  {
    const int block_size = get_num_threads(num_pool);
    const int grid_size = (num_pool + block_size - 1) / block_size;

    vb_softmax_kernel<float><<<block_size, grid_size, 0, stream>>>(
        values.data_ptr<float>(),
        pool_sizes.data_ptr<int>(),
        pool_offsets.data_ptr<int>(),
        num_pool,
        out_values.data_ptr<float>());
  }

  return out_values;
}

} // cuda_playground
