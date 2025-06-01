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
