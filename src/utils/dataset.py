from typing import Callable

import torch
import torchvision.datasets
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def get_fashion_mnist_dataset(root: str = './data') -> (Dataset, Dataset):
    train_dataset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=_TRANSFORM)
    val_dataset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=_TRANSFORM)
    return train_dataset, val_dataset


def get_cifar10_dataset(root: str = './data') -> (Dataset, Dataset):
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=cifar_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=cifar_transform)
    return train_dataset, val_dataset


def get_mnist_dataset(root: str = './data') -> (Dataset, Dataset):
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=_TRANSFORM)
    val_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=_TRANSFORM)
    return train_dataset, val_dataset


def get_svhn_dataset(root: str = './data') -> (Dataset, Dataset):
    train_dataset = torchvision.datasets.SVHN(root=root, train=True, download=True, transform=_TRANSFORM)
    val_dataset = torchvision.datasets.SVHN(root=root, train=False, download=True, transform=_TRANSFORM)
    return train_dataset, val_dataset


def reduce_dataset(dataset: Dataset, num_samples: int) -> Dataset:
    if num_samples > len(dataset):
        raise AttributeError
    indices = torch.randperm(len(dataset))[:num_samples]
    return Subset(dataset, indices)


def split_dataset(dataset: Dataset, split_proportion: float = 0.8) -> tuple[Dataset, Dataset]:
    indices = torch.randperm(len(dataset))
    dataset1_indices, dataset2_indices = (
        indices[:int(len(dataset) * split_proportion)], indices[int(len(dataset) * split_proportion):]
    )
    return Subset(dataset, dataset1_indices), Subset(dataset, dataset2_indices)


def _extract_labels(dataset):
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return dataset.tensors[1]
    if hasattr(dataset, "targets"):
        return dataset.targets.clone().detach()

    if hasattr(dataset, "labels"):
        return torch.tensor(dataset.labels)


def filter_labels(dataset: Dataset, labels_to_keep: list[int]) -> tuple[Dataset, Dataset]:
    labels = _extract_labels(dataset)
    mask = torch.isin(labels, torch.tensor(labels_to_keep))
    indices = torch.where(mask)[0]
    return Subset(dataset, indices)


def map_labels(dataset: Dataset, label_map: dict[int, int]) -> Dataset:
    return LabelMapDataset(dataset, label_map)


class LabelMapDataset(Dataset):
    def __init__(self, dataset: Dataset, label_map: dict[int, int]) -> None:
        self._dataset = dataset
        self._label_map = label_map

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        x, y = self._dataset[index]
        y = self._label_map[y]
        return x, y
