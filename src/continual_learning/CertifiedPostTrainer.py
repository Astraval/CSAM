import copy

import torch

from src.optimizers.SimpleTrainer import SimpleTrainer
from src.cert import Safebox

    
class CertifiedPostTrainer(SimpleTrainer):
    """ This class allows for continual learning by projecting the weights back to a certified box after each step. 
        That way, by never leaving that certified region, you guarantee a minimum-accuracy on the first dataset.

    """

    def __init__(self, bound_model: torch.nn.Sequential, quiet: bool = False, device: str = "cpu",
                 acc_evaluation_steps: int = 60):
        """Initializes the CertifiedPostTrainer for Continual Learning

        Args:
            bound_model (torch.nn.Sequential):  The interval model defining the certified region for the weights.
            quiet (bool, optional): If True, suppresses training output. Defaults to False.
            device (str, optional): The device to use for training ('cpu' or 'cuda'). Defaults to "cpu".
            acc_evaluation_steps (int, optional): Number of steps between validation accuracy evaluations. Defaults to 60.
        """
        model = Safebox.bmodelToModel(bound_model) #Convert to a classical neural network so it can train with a standard optimizer
        super().__init__(model, quiet=quiet, device=device,
                         acc_evaluation_steps=acc_evaluation_steps)
        self._bound_model = copy.deepcopy(bound_model)


    def step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-4, **kwargs) -> (float, dict[str, float]):
        """One optimisation step + projection back into the safe certified box.

        Args:
            X (torch.Tensor): batch of inputs
            y (torch.Tensor): corresponding labels
            lr (float): Learning rate (default to 1e-4)
            **kwargs : any extra argument required

        Returns a tuple containing the current batch loss and a dictionnary with the current validation accuracy
        """
        values = super().step(X, y, lr, **kwargs)  # optimisation step
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
