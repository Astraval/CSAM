import torch.nn

from src.cert import Safebox
from src.optimizers.ConstrainedVolumeTrainer import ConstrainedVolumeTrainer


class HypercubeTrainer(ConstrainedVolumeTrainer):
    def __init__(self, model: torch.nn.Sequential, device: str = "cpu"):
        super().__init__(model, device=device)
        self._optimizer = None

    def _set_volume_constrain(self, volume: float):
        Safebox.assign_epsilon(self._interval_model, volume)

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100, batch_size: int = 64, lr: float = 1e-4, **kwargs):
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
        super().train(
            train_dataset, val_dataset, loss_obj, max_iters=max_iters, batch_size=batch_size, lr=lr, **kwargs
        )

    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> float:
        self._optimizer.zero_grad()
        self._interval_model.zero_grad()
        self._interval_model.train()
        X = X.unsqueeze(-1)
        X = X.expand(*X.shape[:-1], 2)
        y_pred = self._interval_model(X)
        max_loss = Safebox.max_loss(y, y_pred)
        max_loss.backward()
        self._optimizer.step()
        return max_loss.item()
