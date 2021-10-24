from pathlib import Path

import torch


def check(data):
    print(type(data), len(data))
    print(type(data[0]), len(data[0]))
    print(type(data[0][0]), len(data[0][0]))
    print(type(data[0][0][0]), data[0][0][0].shape)
    print(type(data[0][0][1]), data[0][0][1].shape)


def main():
    root_dir = Path("datasets") / "test_tensor"
    for split in ["test_tensor_batch_size1", "test_tensor_batch_size4"]:
        tensor_dir = root_dir / split
        for filename in tensor_dir.glob("sample_*"):
            data = torch.load(filename, map_location="cpu")
            check(data)
            return


if __name__ == "__main__":
    main()
