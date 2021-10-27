import unittest
import time
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

from cuda_playground.vbmm import vbmm

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SimpleTestCase(unittest.TestCase):
    def test(self):
        batch_size = 1000
        m = torch.randint(1, 128, (batch_size,), dtype=torch.int32, device=torch.device("cuda"))
        n = torch.randint(1, 128, (batch_size,), dtype=torch.int32, device=torch.device("cuda"))
        k = torch.full((batch_size,), 128, dtype=torch.int32, device=torch.device("cuda"))

        mA_sizes = m * k
        mA = torch.randn((mA_sizes.sum().item(),), dtype=torch.float32, device=torch.device("cuda"))
        mB_sizes = k * n
        mB = torch.randn((mB_sizes.sum().item(),), dtype=torch.float32, device=torch.device("cuda"))
        mC_sizes = m * n
        mC = torch.randn((mC_sizes.sum().item(),), dtype=torch.float32, device=torch.device("cuda"))

        print("mA_size", mA_sizes.sum().item())
        print("mB_size", mB_sizes.sum().item())
        print("mC_size", mC_sizes.sum().item())

        mA_offsets = torch.zeros(batch_size, dtype=torch.int32, device=torch.device("cuda"))
        mA_offsets[1:] = torch.cumsum(mA_sizes, dim=0)[:-1]
        mB_offsets = torch.zeros(batch_size, dtype=torch.int32, device=torch.device("cuda"))
        mB_offsets[1:] = torch.cumsum(mB_sizes, dim=0)[:-1]
        mC_offsets = torch.zeros(batch_size, dtype=torch.int32, device=torch.device("cuda"))
        mC_offsets[1:] = torch.cumsum(mC_sizes, dim=0)[:-1]

        mC_, mC_offsets_, m_, n_, batch_size_ = vbmm(
            (mA, mA_offsets, m, k, batch_size),
            (mB, mB_offsets, k, n, batch_size),
        )

        self.assertEqual(batch_size, batch_size_, "batch_size")
        self.assertTrue((m_ == m).all(), "m")
        self.assertTrue((n_ == n).all(), "n")
        self.assertTrue((mC_offsets_ == mC_offsets).all(), "mC_offsets")

        for i in range(batch_size):
            m1 = mA[mA_offsets[i]:mA_offsets[i] + mA_sizes[i]].reshape(m[i], k[i])
            m2 = mB[mB_offsets[i]:mB_offsets[i] + mB_sizes[i]].reshape(k[i], n[i])

            m3 = m1 @ m2

            m3_ = mC_[mC_offsets[i]:mC_offsets[i] + mC_sizes[i]].reshape(m[i], n[i])

            self.assertTrue(torch.allclose(m3, m3_, atol=1e-4, rtol=1e-5), (m3, m3_, (m3 - m3_).abs().max(), i))


if __name__ == "__main__":
    unittest.main()
