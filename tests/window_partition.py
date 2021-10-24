import unittest
import time
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

from cuda_playground.window_partition import window_partition, sort_perf


class SimpleTestCase(unittest.TestCase):
    def test(self):
        window_ids = torch.randint(0, 100, (5000,), dtype=torch.int32, device=torch.device("cuda"))

        sorted_indices, output_window_ids, window_sizes, _ = window_partition(window_ids.clone())

        for i in range(100):
            self.assertTrue(window_sizes[i].item() == (window_ids == i).sum().item(), f"window_sizes[{i}]")

        self.assertTrue((window_ids[sorted_indices.long()] == torch.sort(window_ids).values).all())


class RealTestCase(unittest.TestCase):
    def do_sparse_transformer(self, data, window_sizes1, window_sizes2):
        time_window_ids = 0
        time_window_partition = 0
        time_reorder_features = 0
        time_new_tensor, time_sort, time_reduce_by_key = 0, 0, 0
        time_sort_perf1, time_sort_perf2 = 0, 0
        for stage in range(3):
            for layer in range(2):
                features, coords = data[stage][layer]

                # sda

                start_time = time.time()
                window_coords = torch.div(coords, window_sizes1[None, :], rounding_mode="floor")
                window_ids = window_coords[:, 0] * 32 * 1024 * 1024 + \
                             window_coords[:, 1] * 1024 * 1024 + \
                             window_coords[:, 2] * 1024 + \
                             window_coords[:, 3]
                time_window_ids += time.time() - start_time

                window_ids1, window_ids2 = window_ids.clone(), window_ids.clone()
                time_sort_perf1_, time_sort_perf2_ = sort_perf(window_ids1, window_ids2)
                time_sort_perf1 += time_sort_perf1_
                time_sort_perf2 += time_sort_perf2_

                start_time = time.time()
                sorted_indices, output_window_ids, _, (time_new_tensor_, time_sort_, time_reduce_by_key_) = window_partition(window_ids)
                time_window_partition += time.time() - start_time
                time_new_tensor += time_new_tensor_
                time_sort += time_sort_
                time_reduce_by_key += time_reduce_by_key_

                start_time = time.time()
                features = features[sorted_indices.long()]
                time_reorder_features += time.time() - start_time

                # lda

                start_time = time.time()
                window_coords = coords % window_sizes2[None, :]
                window_ids = window_coords[:, 0] * 32 * 1024 * 1024 + \
                             window_coords[:, 1] * 1024 * 1024 + \
                             window_coords[:, 2] * 1024 + \
                             window_coords[:, 3]
                time_window_ids += time.time() - start_time

                start_time = time.time()
                sorted_indices, output_window_ids, _, (time_new_tensor_, time_sort_, time_reduce_by_key_) = window_partition(window_ids)
                time_window_partition += time.time() - start_time
                time_new_tensor += time_new_tensor_
                time_sort += time_sort_
                time_reduce_by_key += time_reduce_by_key_

                start_time = time.time()
                features = features[sorted_indices.long()]
                time_reorder_features += time.time() - start_time

        return (
            time_window_ids * 1000, time_window_partition * 1000, time_reorder_features * 1000,
            time_new_tensor, time_sort, time_reduce_by_key,
            time_sort_perf1, time_sort_perf2,
        )

    def test(self):
        window_sizes1 = torch.as_tensor([1, 6, 4, 4], dtype=torch.int32, device=torch.device("cuda"))
        window_sizes2 = torch.as_tensor([1, 2, 8, 8], dtype=torch.int32, device=torch.device("cuda"))
        root_dir = Path("datasets") / "test_tensor"
        for split in ["test_tensor_batch_size1", "test_tensor_batch_size4"]:
            tensor_dir = root_dir / split
            times = []
            for filename in tqdm(list(tensor_dir.glob("sample_*"))):
                data = torch.load(filename, map_location="cuda")
                times.append(self.do_sparse_transformer(data, window_sizes1, window_sizes2))
            times = np.asarray(times[10:])
            print(np.mean(times, axis=0))


if __name__ == "__main__":
    unittest.main()
