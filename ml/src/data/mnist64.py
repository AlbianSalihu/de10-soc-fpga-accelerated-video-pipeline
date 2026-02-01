from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import torch 
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class MNIST64Config:
    data_dir: str = "ml/data"
    image_size: int = 64
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    val_ratio: float = 0.1
    seed: int = 1234
    normalize: bool = True
    augment: bool = False

def _build_transform(image_size: int, normalize: bool, augment: bool):
    """Returns (train_transform, test_transform).

    Args:
        image_size (int): The size of the image
        normalize (bool): Whether to normalize or not
        augment (bool): Whether to augment or not
    """

    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)

    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if normalize:
        base.append(transforms.Normalize(mean=mnist_mean, std=mnist_std))

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0,1),
                scale=(0.9, 1.1),
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mnist_mean, std=mnist_std) if normalize else transforms.Lambda(lambda x: x),
        ])
    else:
        train_transform = transforms.Compose(base)

    test_transform = transforms.Compose(base)
    return train_transform, test_transform

def get_datasets(cfg: MNIST64Config):
    """Returns: train_ds, val_ds and test_ds

    Args:
        cfg (MNIST64Config): config class
    """
    data_dir = Path(cfg.data_dir).expanduser().resolve()
    train_transform, test_transform = _build_transform(cfg.image_size, cfg.normalize, cfg.augment)

    full_train = datasets.MNIST(
        root = str(data_dir),
        train = True,
        download = False,
        transform = train_transform, 
    )

    test_ds = datasets.MNIST(
        root = str(data_dir),
        train = False,
        download = False, 
        transform = test_transform,
    )

    val_size = int(len(full_train) * cfg.val_ratio)
    train_size = len(full_train) - val_size
    if val_size <= 0 or train_size <= 0: 
        raise ValueError(f"Invalid val_ratio={cfg.val_ratio}; train={train_size}, val={val_size}")
    
    gen = torch.Generator().manual_seed(cfg.seed)

    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=gen)

    return train_ds, val_ds, test_ds

def get_dataloaders(cfg: MNIST64Config)->Tuple[DataLoader, DataLoader, DataLoader]:
    """ 
    Args:
        cfg (MNIST64Config): config class

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train_loader, val_loader, test_loader
    """

    train_ds, val_ds, test_ds = get_datasets(cfg)

    persistent = cfg.persistent_workers and cfg.num_workers>0

    train_loader = DataLoader(
        train_ds, 
        batch_size = cfg.batch_size,
        shuffle = True, 
        num_workers = cfg.num_workers,
        pin_memory = cfg.pin_memory,
        persistent_workers = persistent,
        drop_last = False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = cfg.batch_size,
        shuffle = False, 
        num_workers = cfg.num_workers,
        pin_memory = cfg.pin_memory,
        persistent_workers = persistent,
        drop_last = False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size = cfg.batch_size,
        shuffle = False, 
        num_workers = cfg.num_workers,
        pin_memory = cfg.pin_memory,
        persistent_workers = persistent,
        drop_last = False,
    )

    return train_loader, val_loader, test_loader

def show_sample(cfg: MNIST64Config, split: str="train", index: int = 0) -> None:
    """Display one resized MNIST sample (64x64) using matplotlib

    Args:
        cfg (MNIST64Config): config class
        split (str, optional): train | val | test. Defaults to "train".
        index (int, optional): what image to show by id. Defaults to 0.
    """
    train_ds, val_ds, test_ds = get_datasets(cfg)
    if split == "train":
        ds = train_ds
    elif split == "val":
        ds = val_ds
    elif split == "test":
        ds = test_ds
    else:
        raise ValueError("split must be 'train' 'val' or 'test'")
    
    x, y = ds[index]

    img = x.clone()
    if cfg.normalize:
        mean = torch.tensor([0.1307]).view(1,1,1)
        std = torch.tensor([0.3081]).view(1,1,1)
        img = img * std + mean

    img =  img.squeeze(0).detach().cpu().numpy()

    plt.figure()
    plt.title(f"MNIST64 sample | split {split}")
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()
