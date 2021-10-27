from typing import Sequence
import unittest
import time
from pathlib import Path
import random

from tqdm import tqdm
import numpy as np
import torch

from cuda_playground.vb_softmax import vb_softmax

torch.manual_seed(0)


class SimpleTestCase(unittest.TestCase):
    def test(self):
        device = torch.device("cuda")

        values = torch.as_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.float32, device=device)
        offsets = torch.as_tensor([0, 6], dtype=torch.int32, device=device)
        m = torch.as_tensor([2, 3], dtype=torch.int32, device=device)
        n = torch.as_tensor([3, 4], dtype=torch.int32, device=device)
        batch_size = 2

        m1 = values[:6].reshape(2, 3)
        m2 = values[6:].reshape(3, 4)

        r1 = vb_softmax((values, offsets, m, n, batch_size), -1)

        self.assertTrue(torch.allclose(r1[:6].reshape(2, 3), m1.softmax(dim=-1)))
        self.assertTrue(torch.allclose(r1[6:].reshape(3, 4), m2.softmax(dim=-1)))


class BunchTestCase(unittest.TestCase):
    def test(self):
        device = torch.device("cuda")
        batch_size = 1024

        m = torch.randint(1, 128, (batch_size,), dtype=torch.int, device=device)
        n = torch.randint(1, 128, (batch_size,), dtype=torch.int, device=device)

        mat_sizes = m * n
        values = torch.randn((mat_sizes.sum().item(),), dtype=torch.float32, device=device)
        offsets = torch.zeros((batch_size,), dtype=torch.int, device=device)
        offsets[1:] = mat_sizes[:-1].cumsum(dim=0)

        r = vb_softmax((values, offsets, m, n, batch_size), -1)

        for i in range(batch_size):
            mat = values[offsets[i]:offsets[i] + mat_sizes[i]].reshape(m[i], n[i])
            r_i = r[offsets[i]:offsets[i] + mat_sizes[i]].reshape(m[i], n[i])
            self.assertTrue(torch.allclose(r_i, mat.softmax(dim=-1)))


if __name__ == "__main__":
    unittest.main()
