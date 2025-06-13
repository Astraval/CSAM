from abc import ABC, abstractmethod
from typing import Callable

import torch.nn

from src.optimizers.Trainer import Trainer


class MultiphaseTrainer:
    """
    This is an abstract class for trainers that use multiphase training strategies.
    """

    def __init__(self, trainer: Trainer,
                 quiet: bool = False):
        """Initializes the MultiphaseTrainer.

        Args:
            trainer (Trainer): The trainer used for each phase.
            quiet (bool, optional): If True, suppresses training output. Defaults to False.
        """
        self._trainer = trainer
        self._quiet = quiet

    @abstractmethod
    def train(self, *trainer_args,**trainer_kwargs) -> bool:
        """Abstract method to run the multiphase training process.

        Args:
            *trainer_args: Arguments for the underlying trainer's train method.
            **trainer_kwargs: Additional keyword arguments for the underlying trainer's train method.

        Returns:
            bool: True if training completes successfully, False otherwise.
        """
        raise NotImplementedError

    def _print(self, *messages, **kwargs):
        """Prints messages to the console if not in quiet mode.

        Args:
            *messages: Messages to print.
            **kwargs: Additional keyword arguments for the print function.
        """
        if not self._quiet:
            print(*messages, **kwargs)

    def result(self) -> torch.nn.Sequential:
        """Returns the result from the underlying trainer.

        Returns:
            torch.nn.Sequential: The trained model.
        """
        return self._trainer.result()
