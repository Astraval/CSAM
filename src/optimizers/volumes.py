from abc import ABC, abstractmethod

import torch

from src.cert import Safebox

"""
This file provides with useful methods to compute the certified volume of interval parameters boxes
"""

class VolumeFunction(ABC):

    def compute_volume(self, model:torch.nn.Sequential) -> torch.Tensor:
        """
        Computes the average volume of all interval parameters in the model.

        Args:
            model (torch.nn.Sequential): The interval bounded model .

        Returns:
            torch.Tensor: The average volume per parameter.
        """
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
        """
        Abstract method to compute the volume for a given pair of upper and lower bounds.

        Args:
            p_u (torch.Tensor): Upper bound.
            p_l (torch.Tensor): Lower bound.

        Returns:
            torch.Tensor: The computed volume.
        """
        raise NotImplementedError
    
    @abstractmethod
    def compute_hypercube_length(self, target_volume: float, n_params: int) -> float:
        """
        Computes the length of the side of an hypercube with a given target volume and number of parameters.

        Args:
            target_volume (float): The desired total volume.
            n_params (int): Number of parameters.

        Returns:
            float: The length of the side of the hypercube.
        """
        raise NotImplementedError

    def __call__(self, model: torch.nn.Sequential, *args, **kwargs) -> torch.Tensor:
        """
        Allows it to be called as a function to compute the volume.

        Args:
            model (torch.nn.Sequential): The interval bounded model.

        Returns:
            torch.Tensor: The average volume per parameter.
        """
        return self.compute_volume(model)


class LogVolume(VolumeFunction):
    def __init__(self, epsilon: float = 1e-8):
        self._epsilon = epsilon

    def _compute_volume(self, p_u: torch.Tensor, p_l: torch.Tensor) -> torch.Tensor:
        """compute the log volume given upper and lower bounds

        Args:
            p_u (torch.Tensor): upper bound
            p_l (torch.Tensor): lower bound

        Returns:
            torch.Tensor: the volume
        """
        volume = torch.log(p_u + p_l + self._epsilon).sum()
        return volume

    def compute_hypercube_length(self, target_volume: float, n_params: int) -> float:
        """
        Computes the length of the side of an hypercube for a given target log-volume and number of parameters.

        Args:
            target_volume (float): The desired total log-volume.
            n_params (int): Number of parameters.

        Returns:
            float: The side length of the hypercube.
        """
        return torch.exp(torch.tensor([target_volume/n_params])).item()/2.0

