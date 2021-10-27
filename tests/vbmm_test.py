from typing import Sequence
import unittest
import time
from pathlib import Path
import random

from tqdm import tqdm
import numpy as np
import torch

from cuda_playground.vbmm import vbmm, batched_gemm_cuda
from cuda_playground_25.batched_gemm import batched_gemm


def batched_gemm_python(mat1: Sequence[torch.Tensor], mat2: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    return [torch.mm(m1, m2) for m1, m2 in zip(mat1, mat2)]


class SimpleTestCase(unittest.TestCase):
    def test(self):
        batch_size = 2048
        mat_size_min = 10
        mat_size_max = 256
        mat_k = 64
        device = torch.device("cuda")

        mat1, mat2 = [], []
        for _ in range(batch_size):
            m = torch.randn(random.randint(mat_size_min, mat_size_max), mat_k, device=device)
            mat1.append(m.contiguous())
            mat2.append(m.t().contiguous())

        test_m = torch.as_tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32, device=device)
        mat1, mat2 = [test_m.contiguous()], [test_m.t().contiguous()]
        
        r0 = batched_gemm(mat1, mat2)
        r1 = batched_gemm_cuda(mat1, mat2)
        r2 = batched_gemm_python(mat1, mat2)

        print(r0, r2)

        print("test1")
        for m0, m2 in zip(r0, r2):
            self.assertTrue(torch.allclose(m0, m2), (m0, m2, m0 - m2))

        print("test2")
        for m1, m2 in zip(r1, r2):
            self.assertTrue(torch.allclose(m1, m2), (m1, m2, m1 - m2))


if __name__ == "__main__":
    unittest.main()
