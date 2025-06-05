import copy
import torch
from src.optimizers.Trainer import Trainer
from src.optimizers.sam import SAM
from src.utils.evaluation import evaluate_accuracy
from src.optimizers.PGD_attack import pgd_attack

class PGDTrainer(Trainer):
    def __init__(self, model: torch.nn.Sequential, quiet: bool = False, device: str = "cpu"):
        super().__init__(quiet=quiet, device=device)
        self._val_dataset = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._model = copy.deepcopy(model).to(device)

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100,
              batch_size: int = 64, lr: float = 1e-4, epsilon: float = 0.3, alpha: float = 0.01, num_iters: int = 10, **kwargs) -> torch.nn.Sequential:
        
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._val_dataset = val_dataset
        return super().train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss_obj=loss_obj,
            batch_size=batch_size,
            lr=lr,
            max_iters=max_iters,
            epsilon = epsilon,
            alpha = alpha,
            num_iters = num_iters
            **kwargs
        )
    
    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, epsilon: float = 0.3, alpha: float =0.01, num_iters: int = 10, **kwargs) -> (float, dict[str, float]):
        self._model.train()
        self._model.zero_grad()
        self._optimizer.zero_grad()
        x_adv = pgd_attack(self._model, X, y,)
        y_pred = self._model(x_adv)
        loss = torch.nn.CrossEntropyLoss()(y_pred, y)
        loss.backward()
        self._optimizer.step()

    def result(self) -> torch.nn.Sequential:
        return copy.deepcopy(self._model)
    
    def _evaluate_accuracy(self, num_samples: int = 64) -> float:
        self._model.eval()
        with torch.inference_mode():
            return evaluate_accuracy(self._val_dataset, self._model, num_samples=num_samples, device=self._device)
        

    
        
        
    