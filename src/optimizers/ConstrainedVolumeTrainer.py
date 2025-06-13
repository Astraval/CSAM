import copy
from abc import ABC, abstractmethod

import torch.nn
from tqdm import tqdm

from src.cert import Safebox
from src.optimizers.Trainer import Trainer


class ConstrainedVolumeTrainer(Trainer, ABC):
    """
    Base class for trainers that optimize interval neural network models with a constraint on the safe box volume.
    """
    def __init__(self, model: torch.nn.Sequential, quiet: bool = False, device: str = "cpu"):
        super().__init__(quiet=quiet, device=device)
        self._current_val_dataset = None
        self._interval_model: torch.nn.Sequential = Safebox.modelToBModel(model)
        self._interval_model = self._interval_model.to(device)
        self._previous_min_acc : float | None = None
        self._previous_min_acc_wait : float = 20

    @abstractmethod
    def set_volume_constrain(self, epsilon: float):
        """Abstract method to set the safe box volume constraint for the interval model.

        Args:
            epsilon (float): The desired volume constraint.
        """
        raise NotImplementedError

    def _print(self, *messages, **kwargs):
        """Prints messages to the console if not in quiet mode.

        Args:
            *messages: Messages to print.
            **kwargs: Additional arguments for the print function.
        """
        if not self._quiet:
            print(*messages, **kwargs)

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100,
              batch_size: int = 64, lr: float = 1e-4, **kwargs) -> bool:
        """Trains the interval model on the training dataset.

        Args:
            train_dataset (torch.utils.data.Dataset): The dataset used for training.
            val_dataset (torch.utils.data.Dataset): The dataset used for validation.
            loss_obj (float): The target loss value to reach.
            max_iters (int, optional): Maximum number of training iterations. Defaults to 100.
            batch_size (int, optional): Number of samples per training batch. Defaults to 64.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            **kwargs: Additional arguments for the training loop.

        Returns:
            bool: True if training completes successfully, otherwise False (for example if we suffered from bad initialization).
        """
        self._interval_model.train()
        self._current_val_dataset = val_dataset
        return super().train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss_obj=loss_obj, max_iters=max_iters,
            batch_size=batch_size, lr=lr, **kwargs
        )

    def result(self) -> torch.nn.Sequential:
        """Returns a copy of the trained interval (safe box) model.

        Returns:
            torch.nn.Sequential: The trained interval model.
        """
        return copy.deepcopy(self._interval_model)

    def _evaluate_min_val_acc(self,
                              val_dataset: torch.utils.data,
                              num_samples: int = 64) -> float:
        """Computes the minimum certified accuracy on the validation set using the bounded interval model.

        Args:
            val_dataset (torch.utils.data.Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to use for evaluation. Defaults to 64.

        Returns:
            float: The minimum certified accuracy.
        """
        X, y = next(iter(torch.utils.data.DataLoader(val_dataset, batch_size=num_samples, shuffle=True)))
        self._interval_model.eval()
        with torch.no_grad():
            X = X.unsqueeze(-1)
            X = X.expand(*X.shape[:-1], 2)
            X, y = X.to(self._device), y.to(self._device)
            y_pred = self._interval_model(X)
            min_acc = Safebox.min_acc(y, y_pred)
        return min_acc.item()

    def evaluate_min_val_acc(self,
                             val_dataset: torch.utils.data,
                             num_samples: int = 64) -> float:
        """This is the public method to compute the minimum certified accuracy on the validation set.

        Args:
            val_dataset (torch.utils.data.Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to use for evaluation. Defaults to 64.

        Returns:
            float: The minimum certified accuracy.
        """
        return self._evaluate_min_val_acc(val_dataset, num_samples)

    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        """Performs one optimization step and returns the batch loss and the current minimum certified validation accuracy.

        Args:
            X (torch.Tensor): Batch of input data.
            y (torch.Tensor): Corresponding labels for the batch.
            lr (float, optional): Learning rate for this step. Defaults to 1e-4.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the batch loss (float) and a dictionary with the current minimum certified validation accuracy under the key "min_val_acc".
        """
        self._interval_model.train()
        loss, infos = self._optimize_step(X, y, lr=lr, **kwargs)
        if self._previous_min_acc is None or self._previous_min_acc_wait <= 0:
            self._previous_min_acc = round(self._evaluate_min_val_acc(self._current_val_dataset, 2000), 4)
            self._previous_min_acc_wait = 20  # every 20 iterations evaluate min acc
            min_acc = self._previous_min_acc
        else:
            self._previous_min_acc_wait -= 1
            min_acc = self._previous_min_acc
        return loss, {"min_val_acc": min_acc} | infos

    @abstractmethod
    def _optimize_step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        """Abstract method to perform the actual optimization step for the bounded interval model

        Args:
            X (torch.Tensor): Batch of input data.
            y (torch.Tensor): Corresponding labels for the batch.
            lr (float, optional): Learning rate for this step. Defaults to 1e-4.
            **kwargs: Additional arguments if required.
        """
        raise NotImplementedError
