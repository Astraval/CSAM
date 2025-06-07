from typing import Callable

import cooper
import torch.nn

from src.cert import Safebox
from src.optimizers.ConstrainedVolumeTrainer import ConstrainedVolumeTrainer
from src.optimizers.volumes import VolumeFunction


class LagrangianTrainer(ConstrainedVolumeTrainer):

    def __init__(self,
                 model: torch.nn.Sequential,
                 volume_function: VolumeFunction,
                 device: str = "cpu", quiet: bool = False):
        super().__init__(model, device=device, quiet=quiet)
        self._optimizer = None
        self._volume_function = volume_function
        self._min_volume: torch.Tensor | None = None

    def set_volume_constrain(self, epsilon: float):
        Safebox.assign_epsilon(self._interval_model, epsilon)
        self._min_volume = self._volume_function.compute_volume(self._interval_model).item()

    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              loss_obj: float, max_iters: int = 100, batch_size: int = 64, lr: float = 1e-4, **kwargs):
        for layer in self._interval_model:
            if isinstance(layer, Safebox.BDense) or isinstance(layer, Safebox.BConv2d):
                layer.W_c.requires_grad = True  # only enable to center updates
                layer.b_c.requires_grad = True
                layer.W_u.requires_grad = True
                layer.b_u.requires_grad = True
                layer.W_l.requires_grad = True
                layer.b_l.requires_grad = True

        primal_optimizer = torch.optim.Adam(
            self._interval_model.parameters(), lr=lr
        )

        cmp = VolumeConstrainedIntervalMinimizer(self._min_volume, self._volume_function, device=self._device)
        dual_optimizer = torch.optim.Adam(cmp.dual_parameters(), lr=lr, maximize=True)

        self._optimizer = cooper.optim.AlternatingDualPrimalOptimizer(
            cmp=cmp, primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer
        )

        self._optimizer.zero_grad()
        return super().train(
            train_dataset, val_dataset, loss_obj, max_iters=max_iters, batch_size=batch_size, lr=lr, **kwargs
        )

    def _optimize_step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        self._interval_model.train()
        self._interval_model.zero_grad()
        self._optimizer.zero_grad()
        X = X.unsqueeze(-1)
        X = X.expand(*X.shape[:-1], 2)
        y_pred = self._interval_model(X)
        max_loss = Safebox.max_loss(y, y_pred)

        compute_cmp_state_kwargs = {"model": self._interval_model, "loss": max_loss}
        self._optimizer.roll(compute_cmp_state_kwargs=compute_cmp_state_kwargs)

        with torch.no_grad():
            # enforcing non-empty intervals
            for layer in self._interval_model:
                if isinstance(layer, Safebox.BDense) or isinstance(layer, Safebox.BConv2d):
                    layer.W_u.data.copy_(torch.relu(layer.W_u).clone().detach())
                    layer.W_l.data.copy_(torch.relu(layer.W_l).clone().detach())
                    layer.b_u.data.copy_(torch.relu(layer.b_u).clone().detach())
                    layer.b_l.data.copy_(torch.relu(layer.b_l).clone().detach())

        min_acc = self._evaluate_min_val_acc(self._current_val_dataset, num_samples=64)
        return max_loss.item(), {"min_val_acc": round(min_acc, 4),
                                 "current_volume": round(
                                     self._volume_function.compute_volume(self._interval_model).item(), 4)}


class VolumeConstrainedIntervalMinimizer(cooper.ConstrainedMinimizationProblem):
    def __init__(self, volume_threshold: torch.Tensor, volume_function: VolumeFunction, device: str):
        super().__init__()
        self.volume_threshold = volume_threshold
        self.volume_function = volume_function
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1, device=device)
        self.volume_constraint = cooper.Constraint(
            multiplier=multiplier,
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.Lagrangian,
        )

    def compute_cmp_state(self, model, loss) -> cooper.CMPState:
        volume: torch.Tensor = self.volume_function.compute_volume(model)
        volume_constraint_state = cooper.ConstraintState(violation=-(volume - self.volume_threshold))
        observed_constraints = {self.volume_constraint: volume_constraint_state}
        return cooper.CMPState(loss=loss, observed_constraints=observed_constraints, )
