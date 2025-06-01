import cooper
import torch.nn

from src.cert import Safebox
from src.optimizers.ConstrainedVolumeTrainer import ConstrainedVolumeTrainer

class LagrangianTrainer(ConstrainedVolumeTrainer):

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
                layer.W_u.requires_grad = True
                layer.b_u.requires_grad = True
                layer.W_l.requires_grad = True
                layer.b_l.requires_grad = True

        primal_optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, self._interval_model.parameters()), lr=lr
        )  # Kept the same code but to be changed (no need for the filter anymore)

        cmp = VolumeConstrainedIntervalMinimizer(self._current_volume, log_volume)
        dual_optimizer = torch.optim.Adam(cmp.dual_parameters(), lr = lr, maximize=True)

        self._optimizer = cooper.optim.AlternatingDualPrimalOptimizer(
            cmp=cmp, primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer
        )

        self._optimizer.zero_grad()
        super().train(
            train_dataset, val_dataset, loss_obj, max_iters=max_iters, batch_size=batch_size, lr=lr, **kwargs
        )

    def _optimize_step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        #self._interval_model.zero_grad()
        self._interval_model.train()
        X = X.unsqueeze(-1)
        X = X.expand(*X.shape[:-1], 2)
        y_pred = self._interval_model(X)
        max_loss = Safebox.max_loss(y, y_pred)
        print(max_loss)

        compute_cmp_state_kwargs = {"model": self._interval_model, "loss": max_loss}
        self._optimizer.roll(compute_cmp_state_kwargs=compute_cmp_state_kwargs)
        return max_loss.item(), {}

class VolumeConstrainedIntervalMinimizer(cooper.ConstrainedMinimizationProblem):
    def __init__(self, volume_threshold: float, volume_function):
        super().__init__()
        self.volume_threshold = volume_threshold
        self.volume_function = volume_function
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1, device='cpu') # device =
        self.volume_constraint = cooper.Constraint(
            multiplier=multiplier,
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.Lagrangian,
        )

    def compute_cmp_state(self, model, loss) -> cooper.CMPState:
        volume = self.volume_function(model)
        volume_constraint_state = cooper.ConstraintState(violation=-(volume- self.volume_threshold))
        observed_constraints = {self.volume_constraint: volume_constraint_state}
        return cooper.CMPState(loss=loss, observed_constraints=observed_constraints, )

def log_volume(model, epsilon: float = 1e-8):
    volume = torch.tensor(0.0, device='cpu')
    for layer in model:
        if isinstance(layer, Safebox.BDense) or isinstance(layer, Safebox.BConv2d):
            volume += torch.log(layer.W_u + layer.W_l + epsilon).sum() + torch.log(layer.b_u + layer.b_l + epsilon).sum()

    return volume