import copy

import torch.nn

from src.optimizers.Trainer import Trainer
from src.utils.evaluation import evaluate_accuracy


class SimpleTrainer(Trainer):
    """ Basic trainer for standard neural networks
    """

    def __init__(self, model: torch.nn.Sequential, quiet: bool = False, device: str = "cpu", acc_evaluation_steps: int= 60):
        """  Initializes the SimpleTrainer.

        Args:
            model (torch.nn.Sequential): The neural network model to be trained.
            quiet (bool, optional): If True, suppresses training output. Defaults to False.
            device (str, optional): The device to use for training ('cpu' or 'cuda'). Defaults to "cpu".
            acc_evaluation_steps (int, optional): Number of steps between validation accuracy evaluations. Defaults to 60.
        """
        super().__init__(quiet=quiet, device=device)
        self._acc_evaluation_steps = acc_evaluation_steps
        self._val_dataset = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._model = copy.deepcopy(model).to(device)
        self._best_model = self._model
        self._best_acc: float = -1.0
        self._acc_wait_steps = 0
        self._current_acc: float = -1.0

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100,
              batch_size: int = 64, lr: float = 1e-4, **kwargs) -> bool:
        """ Trains the model on the train_dataset.

        Args:
            train_dataset (torch.utils.data.Dataset): The dataset used for training.
            val_dataset (torch.utils.data.Dataset): The dataset used for validation.
            loss_obj (float): The target loss value to reach or optimize.
            max_iters (int, optional): Maximum number of training iterations. Defaults to 100.
            batch_size (int, optional): Number of samples per training batch. Defaults to 64.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            **kwargs: Additional keyword arguments for the training loop.


        Returns:
            bool: True if training completes successfully, otherwise False.
        """
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._val_dataset = val_dataset
        return super().train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss_obj=loss_obj,
            batch_size=batch_size,
            lr=lr,
            max_iters=max_iters,
            **kwargs
        )

    def _evaluate_accuracy(self, num_samples: int = 64) -> float:
        """
        Computes the model's accuracy on the validation dataset.

        Args:
            num_samples (int, optional): Number of samples to use for evaluation. Defaults to 64.

        Returns:
            float: The computed accuracy on the validation set.
        """
        self._model.eval()
        with torch.no_grad():
            return evaluate_accuracy(self._val_dataset, self._model, num_samples=num_samples, device=self._device)

    def result(self) -> torch.nn.Sequential:
        """
        Returns a copy of the best model found during training (highest validation accuracy).

        Returns:
            torch.nn.Sequential: The best model.
        """
        return copy.deepcopy(self._best_model)

    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        """
        Performs one optimization step and returns the batch loss and current validation accuracy.

        Args:
            X (torch.Tensor): Batch of input data.
            y (torch.Tensor): Corresponding labels for the batch.
            lr (float, optional): Learning rate for this step. Defaults to 1e-4.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the batch loss (float) and a dictionary with the current validation accuracy "val_acc".
        """
        self._model.train()
        self._model.zero_grad()
        self._optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(self._model(X), y)
        loss.backward()
        self._optimizer.step()
        if self._acc_wait_steps <= 0:
            self._acc_wait_steps = self._acc_evaluation_steps
            self._current_acc = self._evaluate_accuracy(num_samples=len(self._val_dataset))
            if self._current_acc > self._best_acc:
                self._best_acc = self._current_acc
                self._best_model = copy.deepcopy(self._model)
        else:
            self._acc_wait_steps -= 1
        return loss.item(), {"val_acc":round(self._current_acc, 4)}