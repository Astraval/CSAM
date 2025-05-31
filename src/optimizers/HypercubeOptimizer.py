import torch.nn

from src.cert import Safebox
from src.optimizers.ConstrainedVolumeOptimizer import ConstrainedVolumeOptimizer


class HypercubeOptimizer(ConstrainedVolumeOptimizer):
    def __init__(self, model: torch.nn.Sequential):
        super().__init__(model)

    def set_volume_constrain(self, volume: float):
        Safebox.assign_epsilon(self._interval_model, volume)

    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> float:
        pass
