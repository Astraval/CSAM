import copy

import torch

from src.bayes.BayesianModel import BayesianModel


class UniformBayesianModel(BayesianModel):
    """
    Represents a bayesian model where each parameter is uniformly distributed
    """
    def __init__(self, mean_model: torch.nn.Module, params_bounds: list[torch.Tensor]):
        self._mean_model = copy.deepcopy(mean_model).cpu()
        self._params_bounds = [param.detach().clone().cpu() for param in params_bounds]

    def sample(self, n_models: int) -> list[torch.nn.Module]:
        """
        Samples n_models from the posterior distribution.

        :param n_models: number of models to sample.

        :return: List of sampled models
        """
        models = [copy.deepcopy(self._mean_model) for _ in range(n_models)]
        params_iterators = [iter(model.parameters()) for model in models]
        for mean_p, bound_p in zip(self._mean_model.parameters(), self._params_bounds):
            assert mean_p.shape == bound_p.shape
            mean_p = mean_p.unsqueeze(0).expand(n_models, *mean_p.shape)
            bound_p = bound_p.unsqueeze(0).expand(n_models, *bound_p.shape)
            low = mean_p - bound_p
            high = mean_p + bound_p
            param_samples = torch.rand_like(mean_p)*(high - low) + low
            for param_sample, param_iterator in zip(param_samples, params_iterators):
                next(param_iterator).data.copy_(param_sample)
        return models
