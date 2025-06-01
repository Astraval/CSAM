import copy

import torch

from src.optimizers.Trainer import Trainer
from src.optimizers.sam import SAM
from src.utils.evaluation import evaluate_accuracy


class SAMTrainer(Trainer):
    def __init__(self, model: torch.nn.Sequential, quiet: bool = False, device: str = "cpu"):
        super().__init__(quiet=quiet, device=device)
        self._base_optimizer: torch.optim.Optimizer | None = None
        self._optimizer: SAM | None = None
        self._model = copy.deepcopy(model)
        self._val_dataset : torch.utils.data.Dataset | None = None
    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
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
        return copy.deepcopy(self._model)
    def _evaluate_accuracy(self, num_samples: int = 64) -> float:
        self._model.eval()
        with torch.no_grad():
            return evaluate_accuracy(self._val_dataset, self._model, num_samples=num_samples, device=self._device)


    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100,
              batch_size: int = 64, lr: float = 1e-4,
              rho=0.05, adaptive=False, **kwargs) -> torch.nn.Sequential:
        self._base_optimizer = torch.optim.Adam
        self._optimizer = SAM(self._model.parameters(),self._base_optimizer,rho,adaptive,**kwargs)
        self._val_dataset = val_dataset
        return super().train(
            train_dataset,
            val_dataset,
            loss_obj, max_iters,
            batch_size, lr,
        )