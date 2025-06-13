from abc import ABC, abstractmethod

import torch.nn
from tqdm import tqdm

class Trainer(ABC):
    """
    Abstract base class for neural network trainers.
    """

    def __init__(self, quiet: bool = False, device: str = "cpu"):
        """
        Initializes the Trainer.

        Args:
            quiet (bool, optional): If True, suppresses training output and progress bars. Defaults to False.
            device (str, optional): The device to use for training ('cpu' or 'cuda'). Defaults to "cpu".
        """
        self._quiet = quiet
        self._device = device

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
              batch_size: int = 64, lr: float = 1e-4, n_iter_eval_stop: int = 100,
              hard_min_loss_stop: float = 2.2, **kwargs) -> bool:
        """Runs the training loop, calling the step method for each batch.

        Args:
            train_dataset (torch.utils.data.Dataset): The dataset used for training.
            val_dataset (torch.utils.data.Dataset): The dataset used for validation.
            loss_obj (float): The target loss value to reach at the end of training.
            max_iters (int, optional): Maximum number of training iterations. Defaults to 100.
            batch_size (int, optional): Number of samples per training batch. Defaults to 64.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            n_iter_eval_stop (int, optional): Number of iterations before checking for bad initialization. Defaults to 100.
            hard_min_loss_stop (float, optional): This is the loss threshold for detecting bad initialization. Defaults to 2.2.
            **kwargs: Additional keyword arguments for the training loop.

        Returns:
            bool: True if training completes successfully,False otherwise (for example if we suffered from bad initialization)
        """
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        data_iter = iter(dataloader)
        progress_bar = tqdm(range(max_iters), disable=self._quiet)
        loss = 0
        initial_loss = 0
        for k in progress_bar:
            X, y = next(data_iter, (None, None))
            if X is None:
                data_iter = iter(dataloader)
                X, y = next(data_iter)
            X, y = X.to(self._device), y.to(self._device)
            loss, info_dict = self.step(X, y, lr=lr, **kwargs)
            if k == 0:
                initial_loss = loss
            progress_bar.set_postfix({
                                         "loss": round(loss, 4),
                                     } | info_dict)
            if loss < loss_obj:
                self._print("Loss objective reached. Stop training.")
                return True
            # interrupt training because of bad initialization
            if k > n_iter_eval_stop and loss > hard_min_loss_stop:
                self._print("Bad Initialization detected. Interrupted Training.")
                return False

        return True

    @abstractmethod
    def result(self) -> torch.nn.Sequential:
        """Abstract method to return the trained model.

        Returns:
            torch.nn.Sequential: The trained model.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        """Abstract method to perform one optimization step.

        Args:
            X (torch.Tensor): Batch of input data.
            y (torch.Tensor): Corresponding labels for the batch.
            lr (float, optional): Learning rate for this step. Defaults to 1e-4.
            **kwargs: Additional arguments if required.

        Returns:
            tuple: A tuple containing the batch loss (float) and a dictionary with additional information.
        """
        raise NotImplementedError
