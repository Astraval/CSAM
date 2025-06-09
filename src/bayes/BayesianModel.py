from abc import ABC, abstractmethod

import torch

"""
This abstract class represents a Bayesian Model. 
"""
class BayesianModel(ABC):



    def predict(self, X: torch.tensor, n_samples: int,
                apply_softmax: bool = True, device: str = "cpu") -> (torch.Tensor, torch.Tensor):
        """
        Returns the uncertainty estimate of the bayesian model on the given sample batch X.

        :param X: Sample batch. Tensor needs to have shape (m, ...)
        :param n_samples: Number of models to sample from the posterior distribution for the uncertianty estimate
        :param apply_softmax: If true will apply softmax to the logits of the models
        :param device: device on which the computation is performed

        :return: means and standard deviation of the outputs of the sampled models.
        both tensors have shape (m, n_classes).
        """
        results = [model.to(device)(X.to(device)) for model in self.sample(n_samples)]
        if apply_softmax:
            results = list(map(lambda logit: torch.softmax(logit, dim=1), results))
        results = torch.stack(results, dim=2)
        means = results.mean(dim=2)
        entropies = - (means * means.log()).sum(dim=1)
        return means, results.std(dim=2), entropies

    @abstractmethod
    def sample(self, n_models: int) -> list[torch.nn.Module]:
        """
        Samples n_models from the posterior distribution.

        :param n_models: number of models to sample.

        :return: List of sampled models
        """
        raise NotImplementedError
