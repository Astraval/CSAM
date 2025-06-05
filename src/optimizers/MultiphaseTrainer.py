from abc import ABC, abstractmethod
from typing import Callable

import torch.nn

from src.optimizers.Trainer import Trainer


class MultiphaseTrainer:
    def __init__(self, trainer: Trainer,
                 quiet: bool = False):
        self._trainer = trainer
        self._quiet = quiet

    @abstractmethod
    def train(self, *trainer_args,**trainer_kwargs):
        raise NotImplementedError

    def _print(self, *messages, **kwargs):
        if not self._quiet:
            print(*messages, **kwargs)

    def result(self) -> torch.nn.Sequential:
        return self._trainer.result()
