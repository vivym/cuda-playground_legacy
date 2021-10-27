#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <magma_auxiliary.h>
#include <magma_svbatched.h>
#include <magmablas_s.h>
#include <magma_copy.h>
#include <iostream>

// Pulled from magma test code
#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )

namespace cuda_playground {

std::vector<at::Tensor> batched_gemm_cuda_vanilla_impl(
    const std::vector<at::Tensor>& mat1,
    const std::vector<at::Tensor>& mat2) {
    
    std::vector<at::Tensor> res(mat1.size());

    for (size_t i = 0; i < mat1.size(); i++) {
        // auto stream = at::cuda::getStreamFromPool();
        // at::cuda::CUDAStreamGuard guard(stream);
        res[i] = at::mm(mat1[i], mat2[i]);
    }

    return res;
}

std::vector<at::Tensor> batched_gemm_cuda_magma_impl(
    const std::vector<at::Tensor>& mat1,
    const std::vector<at::Tensor>& mat2) {

    real_Double_t magma_time;

    magma_device_t device;
    magma_queue_t queue;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    auto batch_count = mat1.size();
    std::vector<magma_int_t> m(batch_count + 1), n(batch_count + 1), k(batch_count + 1);
    // std::vector<magma_int_t> ldda(batch_count + 1), lddb(batch_count + 1), lddc(batch_count + 1);
    std::vector<float const *> hA_array(batch_count), hB_array(batch_count);
    std::vector<float *> hC_array(batch_count);
    std::vector<at::Tensor> matC(batch_count);

    magma_time = magma_sync_wtime(queue);
    for (decltype(batch_count) i = 0; i < batch_count; i++) {
        m[i] = mat1[i].size(0);
        k[i] = mat1[i].size(1);
        n[i] = mat2[i].size(1);
        // ldda[i] = lddc[i] = m[i]
        // lddb[i] = k[i]

        auto m1 = mat1[i];
        auto m2 = mat2[i];

        hA_array[i] = (float const*) m1.data_ptr();
        hB_array[i] = (float const*) m2.data_ptr();
        auto mC = at::zeros({m[i], n[i]}, mat1[i].options());
        matC[i] = mC;
        hC_array[i] = (float*) mC.data_ptr();
    }
    std::cout << "buf init:" << magma_sync_wtime(queue) - magma_time << std::endl;

    float const* * dA_array;
    float const* * dB_array;
    float ** dC_array;
    magma_int_t *d_m, *d_n, *d_k;

    magma_time = magma_sync_wtime(queue);
    magma_malloc((void**)&dA_array, sizeof(float*) * batch_count);
    magma_malloc((void**)&dB_array, sizeof(float*) * batch_count);
    magma_malloc((void**)&dC_array, sizeof(float*) * batch_count);
    magma_malloc((void**)&d_m, sizeof(float*) * (batch_count + 1));
    magma_malloc((void**)&d_n, sizeof(float*) * (batch_count + 1));
    magma_malloc((void**)&d_k, sizeof(float*) * (batch_count + 1));

    magma_setvector(batch_count, sizeof(float*), hA_array.data(), 1, dA_array, 1, queue);
    magma_setvector(batch_count, sizeof(float*), hB_array.data(), 1, dB_array, 1, queue);
    magma_setvector(batch_count, sizeof(float*), hC_array.data(), 1, dC_array, 1, queue);
    magma_setvector(batch_count + 1, sizeof(magma_int_t), m.data(), 1, d_m, 1, queue);
    magma_setvector(batch_count + 1, sizeof(magma_int_t), n.data(), 1, d_n, 1, queue);
    magma_setvector(batch_count + 1, sizeof(magma_int_t), k.data(), 1, d_k, 1, queue);

    std::cout << "device array init:" << magma_sync_wtime(queue) - magma_time << std::endl;

    magma_time = magma_sync_wtime(queue);
    magmablas_sgemm_vbatched(
        MagmaNoTrans, MagmaNoTrans,
        // MagmaTrans, MagmaTrans,
        d_n, d_m, d_k,
        1.0f,   // alpha
        dB_array,
        d_n,
        dA_array,
        d_k,
        0.f,    // beta
        dC_array,
        d_n,
        batch_count,
        queue
    );
    std::cout << "magmablas_sgemm_vbatched array init:" << magma_sync_wtime(queue) - magma_time << std::endl;

    magma_free(dA_array);
    magma_free(dB_array);
    magma_free(dC_array);

    magma_queue_sync(queue);
    magma_queue_destroy(queue);

    return matC;
}

std::vector<at::Tensor> batched_gemm_cuda(
    const std::vector<at::Tensor>& mat1,
    const std::vector<at::Tensor>& mat2) {
    
    return batched_gemm_cuda_magma_impl(mat1, mat2);
    /*
    if (alg == BatchedGEMMAlgorithm::Vanilla) {
        return batched_gemm_cuda_vanilla_impl(mat1, mat2);
    }
    else if (alg == BatchedGEMMAlgorithm::MAGMA) {
        return batched_gemm_cuda_magma_impl(mat1, mat2);
    } else {
        AT_ERROR("Not implemented!");
    }
    */
}

} // namespace cuda_playground
