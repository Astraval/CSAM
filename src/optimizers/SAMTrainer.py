import copy

import torch

from src.optimizers.Trainer import Trainer
from src.optimizers.sam import SAM
from src.utils.evaluation import evaluate_accuracy


class SAMTrainer(Trainer):
    """
    Trainer for neural networks using Sharpness-Aware Minimization (SAM) method.
    """
    def __init__(self, model: torch.nn.Sequential, quiet: bool = False, device: str = "cpu"):
        """Initializes the SAMTrainer.

        Args:
            model (torch.nn.Sequential): The neural network model to be trained.
            quiet (bool, optional): If True, suppresses training output. Defaults to False.
            device (str, optional): The device to use for training ('cpu' or 'cuda'). Defaults to "cpu".
        """
        super().__init__(quiet=quiet, device=device)
        self._base_optimizer: torch.optim.Optimizer | None = None
        self._optimizer: SAM | None = None
        self._model = copy.deepcopy(model)
        self._val_dataset : torch.utils.data.Dataset | None = None

    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        """Performs one SAM optimization step and returns the batch loss and current validation accuracy.

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
        loss = torch.nn.CrossEntropyLoss()(self._model(X), y)
        loss.backward()
        self._optimizer.first_step(zero_grad=True)
        loss = torch.nn.CrossEntropyLoss()(self._model(X), y)
        loss.backward()
        self._optimizer.second_step(zero_grad=True)
        accuracy = self._evaluate_accuracy()
        return loss.item(), {"val_acc": round(accuracy, 4)}

    def result(self) -> torch.nn.Sequential:
        """Returns a copy of the trained model.

        Returns:
            torch.nn.Sequential: The trained model.
        """
        return copy.deepcopy(self._model)
    
    def _evaluate_accuracy(self, num_samples: int = 64) -> float:
        """Computes the model's accuracy on the validation dataset.

        Args:
            num_samples (int, optional): Number of samples to use for evaluation. Defaults to 64.

        Returns:
            float: The computed accuracy on the validation set.
        """
        self._model.eval()
        with torch.no_grad():
            return evaluate_accuracy(self._val_dataset, self._model, num_samples=num_samples, device=self._device)


    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100,
              batch_size: int = 64, lr: float = 1e-4,
              rho=0.05, adaptive=False, **kwargs) -> torch.nn.Sequential:
        """
        Trains the model using the SAM optimizer.

        Args:
            train_dataset (torch.utils.data.Dataset): The dataset used for training.
            val_dataset (torch.utils.data.Dataset): The dataset used for validation.
            loss_obj (float): The target loss value to reach or optimize.
            max_iters (int, optional): Maximum number of training iterations. Defaults to 100.
            batch_size (int, optional): Number of samples per training batch. Defaults to 64.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            rho (float, optional): Sets the radius of the neighborhood around each parameter. Defaults to 0.05.
            adaptive (bool, optional): Whether to use adaptive SAM or not. Defaults to False.
            **kwargs: Additional arguments for the training loop.

        Returns:
            torch.nn.Sequential: The trained model.
        """
        self._base_optimizer = torch.optim.Adam
        self._optimizer = SAM(self._model.parameters(),self._base_optimizer,rho,adaptive,**kwargs)
        self._val_dataset = val_dataset
        return super().train(
            train_dataset,
            val_dataset,
            loss_obj, max_iters,
            batch_size, lr,
        )