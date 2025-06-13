import torch.nn

from src.cert import Safebox
from src.optimizers.ConstrainedVolumeTrainer import ConstrainedVolumeTrainer


class HypercubeTrainer(ConstrainedVolumeTrainer):
    """
    Trainer for interval bounded neural networks where the safe box is an hypercube (which means that all interval lenghts are equal)
    """
    def __init__(self, model: torch.nn.Sequential, device: str = "cpu", quiet: bool = False):
        """Initializes the HypercubeTrainer.

        Args:
            model (torch.nn.Sequential): The bounded neural network to train.
            device (str, optional): The device to use for training ('cpu' or 'cuda'). Defaults to "cpu".
            quiet (bool, optional): If True, suppresses training output. Defaults to False.
        """
        super().__init__(model, device=device, quiet=quiet)
        self._optimizer = None

    def set_volume_constrain(self, epsilon: float):
        """Sets the interval length (epsilon) for all parameters in the interval model.

        Args:
            epsilon (float): The lenght of the interval for all parameters.
        """
        Safebox.assign_epsilon(self._interval_model, epsilon)

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100, batch_size: int = 64, lr: float = 1e-4, **kwargs):
        """Trains the interval model, only updating the centers of the intervals.

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
        for layer in self._interval_model:
            if isinstance(layer, Safebox.BDense) or isinstance(layer, Safebox.BConv2d):
                layer.W_c.requires_grad = True  # only enable to center updates
                layer.b_c.requires_grad = True
                layer.W_u.requires_grad = False
                layer.b_u.requires_grad = False
                layer.W_l.requires_grad = False
                layer.b_l.requires_grad = False

        self._optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, self._interval_model.parameters()), lr=lr
        )  # only chose centers in optimizer
        self._optimizer.zero_grad()
        return super().train(
            train_dataset, val_dataset, loss_obj, max_iters=max_iters, batch_size=batch_size, lr=lr, **kwargs
        )

    def _optimize_step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        """Performs one optimization step, computes the worst case (max) loss over the interval, 
        backpropagates, and updates only the center parameters.

        Args:
            X (torch.Tensor): Batch of input data.
            y (torch.Tensor): Corresponding labels for the batch.
            lr (float, optional): Learning rate for this step. Defaults to 1e-4.
            **kwargs: Additional arguments.

        Returns:
            tuple: A tuple containing the batch max loss and an empty dictionary.
        """
        self._optimizer.zero_grad()
        self._interval_model.zero_grad()
        self._interval_model.train()
        X = X.unsqueeze(-1)
        X = X.expand(*X.shape[:-1], 2)
        y_pred = self._interval_model(X)
        max_loss = Safebox.max_loss(y, y_pred)
        max_loss.backward()
        self._optimizer.step()
        return max_loss.item(), {}
