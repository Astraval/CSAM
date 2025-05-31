import copy
from abc import ABC, abstractmethod

import torch.nn
from tqdm import tqdm

from src.cert import Safebox


class ConstrainedVolumeOptimizer(ABC):
    def __init__(self, model: torch.nn.Sequential, quiet: bool = False):
        self._current_volume = None
        self._interval_model: torch.nn.Sequential = Safebox.modelToBModel(model)
        self._quiet = quiet

    @abstractmethod
    def _set_volume_constrain(self, volume: float):
        raise NotImplementedError

    def set_volume_constrain(self, volume: float):
        self._current_volume = volume

    def _print(self, *messages, **kwargs):
        if not self._quiet:
            print(*messages, **kwargs)

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100,
              batch_size: int = 64, lr: float = 1e-4, **kwargs) -> torch.nn.Sequential:
        self._interval_model.train()
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size == batch_size, shuffle=True)
        data_iter = iter(dataloader)
        progress_bar = tqdm(range(max_iters), disable=self._quiet)
        loss = 0
        for _ in progress_bar:
            X, y = next(data_iter, (None, None))
            if X is None:
                data_iter = iter(dataloader)
                X, y = next(data_iter)
            loss = self.step(X, y, lr=lr, **kwargs)
            progress_bar.set_postfix({
                "loss": round(loss, 4),
                "min_val_acc": round(self._evaluate_min_val_acc(val_dataset), 4)
            })
            self._interval_model.train()
            if loss < loss_obj:
                self._print("Loss objective reached. Stop training.")
                break
        self._print("-" * 10,
                    f" Training with volume parameter constrain {round(self._current_volume, 4)} completed with loss  "
                    f"{round(loss)}",
                    "-" * 10)
        return copy.deepcopy(self._interval_model)

    def _evaluate_min_val_acc(self,
                              val_dataset: torch.utils.data,
                              num_samples: int = 64) -> float:
        X, y = next(iter(torch.utils.data.DataLoader(val_dataset, batch_size=num_samples, shuffle=True)))
        self._interval_model.eval()
        with torch.no_grad():
            X = X.unsqueeze(-1)
            X = X.expand(*X.shape[:-1], 2)
            y_pred = self._interval_model(X)
            min_acc = Safebox.min_acc(y, y_pred)
        return min_acc

    @abstractmethod
    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> float:
        raise NotImplementedError
