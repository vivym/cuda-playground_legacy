#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <magma_auxiliary.h>
#include <magma_svbatched.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

namespace cuda_playground {

template <typename _Arg, typename _Result>
struct plus_scalar : public thrust::unary_function<_Arg, _Result> {
  plus_scalar(_Result scalar) {
    this->scalar_ = scalar;
  }

  __host__ __device__
  _Result operator() (_Arg x) {
    return this->scalar_ + x;
  }

private:
  _Result scalar_;
};

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, float, float, float, float> vbmm_cuda(
    bool transA, bool transB,
    const std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int> &mA,
    const std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int> &mB) {
  float time_mC_offsets = 0;
  float time_mC_size = 0;
  float time_plus_addr = 0;
  float time_gemm = 0;
  cudaEvent_t event_start, event_stop;
  
  int batch_size = 0;
  at::Tensor mA_values, mA_offsets, mA_m, mA_n;
  at::Tensor mB_values, mB_offsets, mB_m, mB_n;
  at::Tensor mC_values, mC_offsets, mC_m, mC_n;

  std::tie(mA_values, mA_offsets, mA_m, mA_n, batch_size) = mA;
  std::tie(mB_values, mB_offsets, mB_m, mB_n, batch_size) = mB;

  auto magma_transA = MagmaNoTrans;
  auto magma_transB = MagmaNoTrans;
  if (transA) {
    magma_transA = MagmaTrans;
    std::swap(mA_m, mA_n);
  }
  if (transB) {
    magma_transB = MagmaTrans;
    std::swap(mB_m, mB_n);
  }

  mC_m = mA_m.clone();
  mC_n = mB_n.clone();

  mC_offsets = at::empty({batch_size,}, mA_offsets.options());

  auto mA_values_ptr = mA_values.data_ptr<float>();
  auto mB_values_ptr = mB_values.data_ptr<float>();
  auto mA_offsets_ptr = mA_offsets.data_ptr<int>();
  auto mB_offsets_ptr = mB_offsets.data_ptr<int>();
  auto mC_offsets_ptr = mC_offsets.data_ptr<int>();

  auto d_m = mA_m.data_ptr<int>();
  auto d_k = mA_n.data_ptr<int>();
  auto d_n = mB_n.data_ptr<int>();

  at::cuda::CUDAGuard device_guard(mA_values.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, stream);

  thrust::transform(thrust::device, d_m, d_m + batch_size, d_n, mC_offsets_ptr, thrust::multiplies<int>());
  int mC_size = thrust::reduce(thrust::device, mC_offsets_ptr, mC_offsets_ptr + batch_size, 0);
  thrust::exclusive_scan(
    thrust::device,
    mC_offsets_ptr, mC_offsets_ptr + batch_size,
    mC_offsets_ptr
  );

  cudaEventRecord(event_stop, stream);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&time_mC_offsets, event_start, event_stop);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, stream);

  // std::cout << "mC_size: " << mC_size << std::endl << std::flush;

  cudaEventRecord(event_stop, stream);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&time_mC_size, event_start, event_stop);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);

  mC_values = at::empty({mC_size}, mA_values.options());
  auto mC_values_ptr = mC_values.data_ptr<float>();

  at::Tensor dA_array, dB_array, dC_array;

  dA_array = at::empty({batch_size,}, mA_offsets.options().dtype(at::kLong));
  dB_array = at::empty({batch_size,}, mA_offsets.options().dtype(at::kLong));
  dC_array = at::empty({batch_size,}, mA_offsets.options().dtype(at::kLong));

  float **dA_array_ptr = reinterpret_cast<float **>(dA_array.data_ptr<int64_t>());
  float **dB_array_ptr = reinterpret_cast<float **>(dB_array.data_ptr<int64_t>());
  float **dC_array_ptr = reinterpret_cast<float **>(dC_array.data_ptr<int64_t>());

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, stream);

  thrust::transform(thrust::device, mA_offsets_ptr, mA_offsets_ptr + batch_size, dA_array_ptr, plus_scalar<int, float *>(mA_values_ptr));
  thrust::transform(thrust::device, mB_offsets_ptr, mB_offsets_ptr + batch_size, dB_array_ptr, plus_scalar<int, float *>(mB_values_ptr));
  thrust::transform(thrust::device, mC_offsets_ptr, mC_offsets_ptr + batch_size, dC_array_ptr, plus_scalar<int, float *>(mC_values_ptr));

  cudaEventRecord(event_stop, stream);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&time_plus_addr, event_start, event_stop);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);

  magma_device_t device;
  magma_queue_t queue;
  magma_getdevice(&device);
  magma_queue_create(device, &queue);

  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, stream);

  auto ldda = d_n;
  auto lddb = d_k;
  if (transA) {
    ldda = d_k;
  }
  if (transB) {
    lddb = d_n;
  }

  magmablas_sgemm_vbatched(
      magma_transA, magma_transB,
      d_n, d_m, d_k,
      1.0f,   // alpha
      dB_array_ptr,
      ldda,
      dA_array_ptr,
      lddb,
      0.f,    // beta
      dC_array_ptr,
      d_n,
      batch_size,
      queue
  );

  magma_queue_destroy(queue);

  cudaEventRecord(event_stop, stream);
  cudaEventSynchronize(event_stop);
  cudaEventElapsedTime(&time_gemm, event_start, event_stop);
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_stop);

  return {mC_values, mC_offsets, mC_m, mC_n, batch_size, time_mC_offsets, time_mC_size, time_plus_addr, time_gemm};
}

} // cuda_playground
