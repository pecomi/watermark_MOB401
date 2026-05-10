import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
    data_dir,
    batch_size,
    num_workers,
    seed,
    train_subset=None,
    dataset="mnist",
):
    if dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ]
        )
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        transform = transforms.ToTensor()
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if train_subset is not None and train_subset < len(train_set):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_set), generator=generator)[:train_subset].tolist()
        train_set = Subset(train_set, indices)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
