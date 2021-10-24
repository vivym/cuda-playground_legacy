cmake_minimum_required(VERSION 3.15...3.21)

project(
  cuda_playground
  LANGUAGES C CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TORCH_EXTENSION_NAME _cuda_playground)

find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# find_package(CUDAToolkit)

set(CMAKE_CXXFLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

include(FetchContent)
FetchContent_Declare(
  Thrust
  GIT_REPOSITORY  https://github.com/NVIDIA/thrust.git
  GIT_TAG         1.14.0
)

FetchContent_MakeAvailable(Thrust)

if (SKBUILD)
  message(STATUS "This project is built using scikit-build")
endif()

add_library(${TORCH_EXTENSION_NAME} MODULE
  src/pybind.cpp
  src/window_partition/window_partition_cuda.cu
)
python_extension_module(${TORCH_EXTENSION_NAME})
target_link_libraries(${TORCH_EXTENSION_NAME} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_include_directories(${TORCH_EXTENSION_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS ${TORCH_EXTENSION_NAME} DESTINATION cuda_playground)
