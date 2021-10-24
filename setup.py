import os
from setuptools import find_packages

from skbuild import setup
from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
import torch

TORCH_ROOT = os.path.dirname(torch.__file__)
# TORCH_ROOT = torch.utils.cmake_prefix_path


def get_version():
    return "0.1.0"


setup(
    name="cuda-playground",
    version=get_version(),
    author="Ming Yang (ymviv@qq.com)",
    zip_safe=False,
    packages=["cuda_playground"],
    cmake_args=[f"-DCMAKE_PREFIX_PATH={TORCH_ROOT}"]
)
