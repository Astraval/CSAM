import copy
from abc import ABC, abstractmethod

import torch.nn
from src.cert import Safebox


class ConstrainedVolumeOptimizer(ABC):
    def __init__(self, model: torch.nn.Sequential):
        self._interval_model = Safebox.modelToBModel(model)

    @abstractmethod
    def set_volume_constrain(self, volume: float):
        raise NotImplementedError

    @abstractmethod
    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100, batch_size: int = 64, lr: float = 1e-4):
        raise NotImplementedError

