from abc import ABC, abstractmethod

import torch

from src.cert import Safebox


class VolumeFunction(ABC):

    def compute_volume(self, model:torch.nn.Sequential) -> torch.Tensor:
        volume = 0.0
        for layer in model:
            if isinstance(layer, Safebox.BDense) or isinstance(layer, Safebox.BConv2d):
                volume += self._compute_volume(layer.W_u, layer.W_l)
                volume += self._compute_volume(layer.b_u, layer.b_l)
        n_params = 0
        for layer in model:
            if isinstance(layer, Safebox.BDense) or isinstance(layer, Safebox.BConv2d):
                n_params += layer.W_c.numel() + layer.b_c.numel()
        return volume/n_params
    @abstractmethod
    def _compute_volume(self, p_u: torch.Tensor, p_l: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def compute_hypercube_length(self, target_volume: float, n_params: int) -> float:
        raise NotImplementedError

    def __call__(self, model: torch.nn.Sequential, *args, **kwargs) -> torch.Tensor:
        return self.compute_volume(model)


class LogVolume(VolumeFunction):
    def __init__(self, epsilon: float = 1e-8):
        self._epsilon = epsilon

    def _compute_volume(self, p_u: torch.Tensor, p_l: torch.Tensor) -> torch.Tensor:
        volume = torch.log(p_u + p_l + self._epsilon).sum()
        return volume

    def compute_hypercube_length(self, target_volume: float, n_params: int) -> float:
        return torch.exp(torch.tensor([target_volume/n_params])).item()/2.0

