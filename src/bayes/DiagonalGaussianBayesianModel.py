import copy

import torch

from src.bayes.BayesianModel import BayesianModel


class DiagonalGaussianBayesianModel(BayesianModel):
    def __init__(self, mean_model: torch.nn.Module, std_params: list[torch.Tensor]):
        self._mean_model = copy.deepcopy(mean_model).cpu()
        self._std_params = [param.detach().clone().cpu() for param in std_params]

    def sample(self, n_models: int) -> list[torch.nn.Module]:
        models = [copy.deepcopy(self._mean_model) for _ in range(n_models)]
        params_iterators = [iter(model.parameters()) for model in models]
        for mean_p, std_p in zip(self._mean_model.parameters(), self._std_params):
            assert mean_p.shape == std_p.shape
            param_samples = torch.normal(
                mean_p.unsqueeze(0).expand(n_models, *mean_p.shape),
                std_p.unsqueeze(0).expand(n_models, *mean_p.shape),
            )
            for param_sample, param_iterator in zip(param_samples, params_iterators):
                next(param_iterator).data.copy_(param_sample)
        return models
