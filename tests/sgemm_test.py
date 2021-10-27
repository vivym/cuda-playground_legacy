from typing import Sequence
import unittest
import time
from pathlib import Path
import random

from tqdm import tqdm
import numpy as np
import torch

from cuda_playground.vbmm import sgemm_test, dgemm_test, sgemm_cublas

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def batched_gemm_python(mat1: Sequence[torch.Tensor], mat2: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    return [torch.mm(m1, m2) for m1, m2 in zip(mat1, mat2)]


class SimpleTestCase(unittest.TestCase):
    def test(self):
        mat1 = torch.randn(3, 2, dtype=torch.float32, device=torch.device("cuda"))
        mat2 = torch.randn(2, 3, dtype=torch.float32, device=torch.device("cuda"))

        mat1 = torch.as_tensor([
            [1, 2],
            [3, 4],
            [5, 6],
        ], dtype=torch.float32, device=torch.device("cuda"))
        mat2 = torch.as_tensor([
            [1, 2, 3],
            [4, 5, 6],
        ], dtype=torch.float32, device=torch.device("cuda"))

        mat1 += 0.2
        mat2 += 0.5

        r0 = dgemm_test(mat1.double(), mat2.double())

        print("sum", mat1.sum(), mat2.sum())
        r1 = sgemm_test(mat1, mat2)
        print("sum", mat1.sum(), mat2.sum())
        # r2 = mat1 @ mat2
        r2 = torch.mm(mat1, mat2)
        print("sum", mat1.sum(), mat2.sum())
        r3 = sgemm_cublas(mat1, mat2)
        print("sum", mat1.sum(), mat2.sum())
        r4 = mat1.double() @ mat2.double()
        r5 = mat1 @ mat2

        print((r0 - r1.double()).abs().sum())
        print((r0 - r2.double()).abs().sum())
        print((r0 - r3.double()).abs().sum())
        print((r0 - r4.double()).abs().sum())
        print((r0 - r5.double()).abs().sum())

        print(r1, r1.shape)
        print(r2, r2.shape)
        print(r3, r3.shape)
        print(r1 - r2)
        print(torch.allclose(r1, r2))


if __name__ == "__main__":
    unittest.main()
