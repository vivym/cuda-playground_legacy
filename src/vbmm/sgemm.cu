#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <magma_auxiliary.h>
#include <magma_svbatched.h>
#include <magmablas_s.h>
#include <magmablas_d.h>
#include <magma_copy.h>
#include <iostream>
#include <cublas_v2.h>

namespace cuda_playground {

at::Tensor dgemm_test(const at::Tensor &mat1, const at::Tensor &mat2) {
  magma_device_t device;
  magma_queue_t queue;
  magma_getdevice(&device);
  magma_queue_create(device, &queue);

  auto m = mat1.size(0);
  auto k = mat1.size(1);
  auto n = mat2.size(1);

  at::Tensor mat3 = at::empty({m, n}, mat1.options());

  magmablas_dgemm(
    MagmaNoTrans, MagmaNoTrans,
    n, m, k,
    1.0f,
    mat2.data_ptr<double>(), n,
    mat1.data_ptr<double>(), k,
    0.0f,
    mat3.data_ptr<double>(), n,
    queue
  );

  magma_queue_sync(queue);
  magma_queue_destroy(queue);

  return mat3;
}

at::Tensor sgemm_test(const at::Tensor &mat1, const at::Tensor &mat2) {
  magma_device_t device;
  magma_queue_t queue;
  magma_getdevice(&device);
  magma_queue_create(device, &queue);

  auto m = mat1.size(0);
  auto k = mat1.size(1);
  auto n = mat2.size(1);

  at::Tensor mat3 = at::empty({m, n}, mat1.options());
  at::Tensor mat4 = at::empty({m, n}, mat1.options());

  magmablas_sgemm(
    MagmaNoTrans, MagmaNoTrans,
    n, m, k,
    1.0f,
    mat2.data_ptr<float>(), n,
    mat1.data_ptr<float>(), k,
    0.0f,
    mat3.data_ptr<float>(), n,
    queue
  );

  cublasHandle_t handle;
  cublasCreate(&handle);

  const float a = 1.0, b = 0.0;
  cublasSgemm(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,
    &a,
    mat2.data_ptr<float>(), n,
    mat1.data_ptr<float>(), k,
    &b,
    mat4.data_ptr<float>(), n
  );

  cublasDestroy(handle);

  std::cout << "magma_sgemm:" << std::endl;
  auto m3_ = mat3.cpu();
  for(int i = 0; i < m;i ++) {
    for (int j = 0; j < n; j ++) {
      std::cout << *(m3_.data_ptr<float>() + i * m3_.size(1) + j) << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << "magma_sgemm:" << std::endl;
  auto m4_ = mat4.cpu();
  for(int i = 0; i < m;i ++) {
    for (int j = 0; j < n; j ++) {
      std::cout << *(m4_.data_ptr<float>() + i * m4_.size(1) + j) << "\t";
    }
    std::cout << std::endl;
  }

  magma_queue_sync(queue);
  magma_queue_destroy(queue);

  return mat3;
}

at::Tensor sgemm_cublas(const at::Tensor &mat1, const at::Tensor &mat2) {
  auto m = mat1.size(0);
  auto k = mat1.size(1);
  auto n = mat2.size(1);

  at::Tensor mat3 = at::empty({m, n}, mat1.options());

  cublasHandle_t handle;
  cublasCreate(&handle);

  const float a = 1.0, b = 0.0;
  cublasSgemm(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,
    &a,
    mat2.data_ptr<float>(), n,
    mat1.data_ptr<float>(), k,
    &b,
    mat3.data_ptr<float>(), n
  );

  cublasDestroy(handle);

  return mat3;
}

} // cuda_playground
