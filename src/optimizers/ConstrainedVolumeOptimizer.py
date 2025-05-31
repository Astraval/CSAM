from abc import ABC, abstractmethod

import torch.nn
from tqdm import tqdm

from src.cert import Safebox


class ConstrainedVolumeOptimizer(ABC):
    def __init__(self, model: torch.nn.Sequential):
        self._interval_model = Safebox.modelToBModel(model)

    @abstractmethod
    def set_volume_constrain(self, volume: float):
        raise NotImplementedError

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100, batch_size: int = 64, lr: float = 1e-4, **kwargs):
        self._interval_model.train()
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size == batch_size, shuffle=True)
        data_iter = iter(dataloader)
        progress_bar = tqdm(range(max_iters))
        for i in progress_bar:
            X, y = next(data_iter, (None, None))
            if X is None:
                data_iter = iter(dataloader)
                X, y = next(data_iter)
            loss = self.step(X, y, lr=lr, **kwargs)
            progress_bar.set_postfix({
                "loss": round(loss, 4),
                "min_val_acc": self._evaluate_min_val_acc(val_dataset)
            })

    def _evaluate_min_val_acc(self,
                              val_dataset: torch.utils.data,
                              num_samples: int = 64) -> float:
        X, y = next(iter(torch.utils.data.DataLoader(val_dataset, batch_size=num_samples, shuffle=True)))
        self._interval_model.eval()
        with torch.no_grad():
            X = X.unsqueeze(-1)
            X = X.expand(*X.shape[:-1], 2)
            preds = self._interval_model(X)
            min_acc = Safebox.min_acc(y, preds)
        return min_acc

    @abstractmethod
    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> float:
        raise NotImplementedError
