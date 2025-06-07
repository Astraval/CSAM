import copy

import torch

from src.optimizers.SimpleTrainer import SimpleTrainer
from src.cert import Safebox


class CertifiedPostTrainer(SimpleTrainer):
    def __init__(self, bound_model: torch.nn.Sequential, quiet: bool = False, device: str = "cpu",
                 acc_evaluation_steps: int = 60):
        model = Safebox.bmodelToModel(bound_model)
        super().__init__(model, quiet=quiet, device=device,
                         acc_evaluation_steps=acc_evaluation_steps)
        self._bound_model = copy.deepcopy(bound_model)

    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        values = super().step(X, y, lr, **kwargs)
        # project back into certified box
        with torch.no_grad():
            for layerModel, layerBound in zip(self._model, self._bound_model):
                if isinstance(layerBound, Safebox.BDense) or isinstance(layerBound, Safebox.BConv2d):
                    layerModel.weight.data.clamp_(
                        min=(layerBound.W_c - layerBound.W_l), max=(layerBound.W_c + layerBound.W_u)
                    )
                    layerModel.bias.data.clamp_(
                        min=(layerBound.b_c - layerBound.b_l), max=(layerBound.b_c + layerBound.b_u)
                    )
        return values
