import time

import numpy as np
import torch

from ._cuda_playground import window_partition, sort_perf

__all__ = ["window_partition", "sort_perf"]


def main():
    window_ids = torch.randint(0, 100, (5000,), dtype=torch.int32, device=torch.device("cuda"))
    print(window_ids.shape)
    print(window_ids[:100])

    for i in range(10):
        sorted_indices, output_window_ids, window_sizes = window_partition(window_ids)

    times = []
    for i in range(100):
        start = time.time()
        sorted_indices, output_window_ids, window_sizes = window_partition(window_ids)
        times.append(time.time() - start)

    print(np.mean(times))
    """
    sorted_indices, output_window_ids, window_sizes = window_partition(window_ids)
    print(window_ids[:100])
    print(sorted_indices[:100])
    """
    print(output_window_ids)

    print(sorted_indices.shape)
    print(output_window_ids.shape)
    print(window_sizes.shape)


if __name__ == "__main__":
    main()
