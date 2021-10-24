#include <torch/extension.h>
#include "window_partition/window_partition.h"

namespace cuda_playground {

PYBIND11_MODULE(_cuda_playground, m) {
  m.def("window_partition", &window_partition, "Window Partition");
  m.def("sort_perf", &sort_perf, "Sort Perf");
}

} // cuda_playground
