from abc import ABC, abstractmethod

import torch.nn
from tqdm import tqdm

class Trainer(ABC):
    def __init__(self, quiet: bool = False, device: str = "cpu"):
        self._quiet = quiet
        self._device = device

    def _print(self, *messages, **kwargs):
        if not self._quiet:
            print(*messages, **kwargs)

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100,
              batch_size: int = 64, lr: float = 1e-4, **kwargs) -> torch.nn.Sequential:
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        data_iter = iter(dataloader)
        progress_bar = tqdm(range(max_iters), disable=self._quiet)
        loss = 0
        for _ in progress_bar:
            X, y = next(data_iter, (None, None))
            if X is None:
                data_iter = iter(dataloader)
                X, y = next(data_iter)
            X, y = X.to(self._device), y.to(self._device)
            loss, info_dict = self.step(X, y, lr=lr, **kwargs)
            progress_bar.set_postfix({
                                         "loss": round(loss, 4),
                                     } | info_dict)
            if loss < loss_obj:
                self._print("Loss objective reached. Stop training.")
                break
        self._print("-" * 10,
                    f" Training completed with loss  "
                    f"{round(loss)}",
                    "-" * 10)
        return self.result()

    @abstractmethod
    def result(self) -> torch.nn.Sequential:
        raise NotImplementedError

    @abstractmethod
    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        raise NotImplementedError
