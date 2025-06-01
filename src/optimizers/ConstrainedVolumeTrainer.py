import copy
from abc import ABC, abstractmethod

import torch.nn
from tqdm import tqdm

from src.cert import Safebox
from src.optimizers.Trainer import Trainer


class ConstrainedVolumeTrainer(Trainer, ABC):
    def __init__(self, model: torch.nn.Sequential, quiet: bool = False, device: str = "cpu"):
        super().__init__(quiet=quiet, device=device)
        self._current_val_dataset = None
        self._interval_model: torch.nn.Sequential = Safebox.modelToBModel(model)
        self._interval_model = self._interval_model.to(device)

    @abstractmethod
    def set_volume_constrain(self, epsilon: float):
        raise NotImplementedError

    def _print(self, *messages, **kwargs):
        if not self._quiet:
            print(*messages, **kwargs)

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100,
              batch_size: int = 64, lr: float = 1e-4, **kwargs) -> torch.nn.Sequential:
        self._interval_model.train()
        self._current_val_dataset=val_dataset
        return super().train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss_obj=loss_obj, max_iters=max_iters,
            batch_size=batch_size, lr=lr, **kwargs
        )

    def result(self) -> torch.nn.Sequential:
        return copy.deepcopy(self._interval_model)

    def _evaluate_min_val_acc(self,
                              val_dataset: torch.utils.data,
                              num_samples: int = 64) -> float:
        X, y = next(iter(torch.utils.data.DataLoader(val_dataset, batch_size=num_samples, shuffle=True)))
        self._interval_model.eval()
        with torch.no_grad():
            X = X.unsqueeze(-1)
            X = X.expand(*X.shape[:-1], 2)
            X, y = X.to(self._device), y.to(self._device)
            y_pred = self._interval_model(X)
            min_acc = Safebox.min_acc(y, y_pred)
        return min_acc.item()

    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        self._interval_model.train()
        loss, infos = self._optimize_step(X, y, lr=lr, **kwargs)
        min_acc = round(self._evaluate_min_val_acc(self._current_val_dataset, 64), 4)
        return loss, {"min_val_acc": min_acc} | infos

    @abstractmethod
    def _optimize_step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        raise NotImplementedError
