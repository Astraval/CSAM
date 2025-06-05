from abc import ABC, abstractmethod

import torch


class BayesianModel(ABC):
    def predict(self, X: torch.tensor, n_samples: int, apply_softmax: bool =True) -> (torch.Tensor, torch.Tensor):
        results = [model(X) for model in self.sample(n_samples)]
        if apply_softmax:
            results = list(map(lambda logit: torch.softmax(logit, dim=1), results))
        results = torch.stack(results, dim=2)
        return results.mean(dim=2), results.std(dim=2)

    @abstractmethod
    def sample(self, n_models: int) -> list[torch.nn.Module]:
        raise NotImplementedError
