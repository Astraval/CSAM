import torch
import torchvision.datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms


_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] , std=[0.5] )
    ])
def get_fashion_mnist_dataset(root: str = './data') -> (Dataset, Dataset):

    train_dataset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=_TRANSFORM)
    val_dataset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=_TRANSFORM)
    return train_dataset, val_dataset


def get_mnist_dataset(root: str = './data') -> (Dataset, Dataset):
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=_TRANSFORM)
    val_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=_TRANSFORM)
    return train_dataset, val_dataset