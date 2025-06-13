from typing import Callable

import torch
import torchvision.datasets
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

"""
This file provides useful functions to load, transform and manipulate popular computer vision datasets such as MNIST, FashionMNIST, CIFAR10 and SVHN
"""


_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def get_fashion_mnist_dataset(root: str = './data') -> (Dataset, Dataset):
    """
    Loads the FashionMNIST dataset with standard normalization.
    """
    train_dataset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=_TRANSFORM)
    val_dataset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=_TRANSFORM)
    return train_dataset, val_dataset


def get_cifar10_dataset(root: str = './data') -> (Dataset, Dataset):
    """
    Loads the CIFAR10 dataset with standard normalization.
    """
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=cifar_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=cifar_transform)
    return train_dataset, val_dataset


def get_mnist_dataset(root: str = './data') -> (Dataset, Dataset):
    """
    Loads the MNIST dataset with standard normalization.
    """
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=_TRANSFORM)
    val_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=_TRANSFORM)
    return train_dataset, val_dataset


def get_svhn_dataset(root: str = './data') -> (Dataset, Dataset):
    """
    Loads the SVHN dataset with standard normalization.
    """
    train_dataset = torchvision.datasets.SVHN(root=root, train=True, download=True, transform=_TRANSFORM)
    val_dataset = torchvision.datasets.SVHN(root=root, train=False, download=True, transform=_TRANSFORM)
    return train_dataset, val_dataset


def reduce_dataset(dataset: Dataset, num_samples: int) -> Dataset:
    """Reduces a dataset to a random subset of the specified size.

    Args:
        dataset (Dataset): The original dataset.
        num_samples (int): Number of samples to keep.

    Raises:
        AttributeError: If num_samples is greater than the dataset size.

    Returns:
        Dataset: A subset of the original dataset.
    """
    if num_samples > len(dataset):
        raise AttributeError
    indices = torch.randperm(len(dataset))[:num_samples]
    return Subset(dataset, indices)


def split_dataset(dataset: Dataset, split_proportion: float = 0.8) -> tuple[Dataset, Dataset]:
    """
    Splits a dataset into two subsets according to the given proportion.

    Args:
        dataset (Dataset): The original dataset.
        split_proportion (float, optional): Proportion of samples in the first subset. Defaults to 0.8.

    Returns:
        tuple: (subset1, subset2)
    """
    indices = torch.randperm(len(dataset))
    dataset1_indices, dataset2_indices = (
        indices[:int(len(dataset) * split_proportion)], indices[int(len(dataset) * split_proportion):]
    )
    return Subset(dataset, dataset1_indices), Subset(dataset, dataset2_indices)


def _extract_labels(dataset):
    """
    This is an helper function to extract labels from a dataset

    """
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return dataset.tensors[1]
    if hasattr(dataset, "targets"):
        return dataset.targets.clone().detach()

    if hasattr(dataset, "labels"):
        return torch.tensor(dataset.labels)


def filter_labels(dataset: Dataset, labels_to_keep: list[int]) -> tuple[Dataset, Dataset]:
    """ Filters a dataset to only keep samples with specified labels.

    Args:
        dataset (Dataset): The original dataset.
        labels_to_keep (list[int]): List of labels to keep.

    Returns:
        Dataset: A subset of the original dataset containing only the specified labels.
    """
    labels = _extract_labels(dataset)
    mask = torch.isin(labels, torch.tensor(labels_to_keep))
    indices = torch.where(mask)[0]
    return Subset(dataset, indices)


def map_labels(dataset: Dataset, label_map: dict[int, int]) -> Dataset:
    """
    Maps the labels of a dataset according to a specified mapping.

    Args:
        dataset (Dataset): The original dataset.
        label_map (dict[int, int]): Dictionary mapping old labels to new labels.

    Returns:
        Dataset: A dataset with remapped labels.
    """
    return LabelMapDataset(dataset, label_map)


class LabelMapDataset(Dataset):
    """
    A Dataset class that remaps labels according to a given mapping
    """
    def __init__(self, dataset: Dataset, label_map: dict[int, int]) -> None:
        self._dataset = dataset
        self._label_map = label_map

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        x, y = self._dataset[index]
        y = self._label_map[y]
        return x, y
