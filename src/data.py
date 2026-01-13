"""
Data Loading Utilities for FashionMNIST

Dataset: 10 classes, 60k train + 10k test, grayscale 28×28 images
    x ∈ [0, 1]^{1×28×28}, y ∈ {0, ..., 9}
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def seed_all(seed: int = 0):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fashion_mnist(batch_size: int = 128, root: str = "./data",
                       limit_train: int | None = None, limit_test: int | None = None):
    """Load FashionMNIST with optional size limits."""
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.FashionMNIST(root=root, train=False, download=True, transform=tfm)

    if limit_train is not None:
        idx = torch.randperm(len(train))[:limit_train]
        train = Subset(train, idx.tolist())
    if limit_test is not None:
        idx = torch.randperm(len(test))[:limit_test]
        test = Subset(test, idx.tolist())

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train, test, train_loader, test_loader


def dataset_to_tensors(dataset, max_items: int | None = None):
    """Materialize dataset to tensors: X ∈ ℝ^{N×1×28×28}, Y ∈ {0,...,9}^N."""
    xs, ys = [], []
    n = len(dataset) if max_items is None else min(len(dataset), max_items)
    for i in range(n):
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y
