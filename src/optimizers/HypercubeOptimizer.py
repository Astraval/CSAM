import torch.nn

from src.optimizers.ConstrainedVolumeOptimizer import ConstrainedVolumeOptimizer


class HypercubeOptimizer(ConstrainedVolumeOptimizer):
    def __init__(self, model: torch.nn.Sequential):
        super().__init__(model)

    def set_volume_constrain(self, volume: float):
        pass  # TODO: implement

    def train(self, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, loss_obj: float,
              max_iters: int = 100, batch_size: int = 64, lr: float = 1e-4):
        pass  # TODO: implement
