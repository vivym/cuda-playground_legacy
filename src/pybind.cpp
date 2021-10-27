#include <torch/extension.h>
#include "window_partition/window_partition.h"
#include "vbmm/vbmm.h"
#include "vb_softmax/vb_softmax.h"

namespace cuda_playground {

PYBIND11_MODULE(_cuda_playground, m) {
  m.def("window_partition", &window_partition, "Window Partition");
  m.def("sort_perf", &sort_perf, "Sort Perf");

  m.def("vbmm", &vbmm, "Variable-size batched gemm");
  m.def("batched_gemm_cuda", &batched_gemm_cuda, "batched_gemm_cuda");
  m.def("sgemm_test", &sgemm_test, "sgemm_test");
  m.def("dgemm_test", &dgemm_test, "dgemm_test");
  m.def("sgemm_cublas", &sgemm_cublas, "sgemm_cublas");

  m.def("vb_softmax", &vb_softmax, "vb_softmax");
}

} // cuda_playground
