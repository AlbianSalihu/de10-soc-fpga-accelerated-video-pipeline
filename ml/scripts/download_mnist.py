from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets

def main()->int:
    parser = argparse.ArgumentParser(description="Download MNIST dataset into a local directory")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ml/data",
        help="Root directory where the MNIST will be downloaded (default: ml/data).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download_mnist] Downloading MNIST into: {data_dir}")

    datasets.MNIST(root=str(data_dir), train=True, download=True)
    datasets.MNIST(root=str(data_dir), train=False, download=True)

    mnist_dir = data_dir / "MNIST"
    if mnist_dir.exists():
        print(f"[download_mnist] Done. Dataset folder: {mnist_dir}")
    else:
        print("[download_mnist] Done. (MNIST folder not found where expected, but download completed)")

    return 0

if __name__=="__main__":
    raise SystemExit(main())