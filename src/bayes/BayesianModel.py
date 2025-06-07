from abc import ABC, abstractmethod

import torch


class BayesianModel(ABC):
    def predict(self, X: torch.tensor, n_samples: int,
                apply_softmax: bool = True, device: str = "cpu") -> (torch.Tensor, torch.Tensor):
        results = [model.to(device)(X.to(device)) for model in self.sample(n_samples)]
        if apply_softmax:
            results = list(map(lambda logit: torch.softmax(logit, dim=1), results))
        results = torch.stack(results, dim=2)
        means = results.mean(dim=2)
        entropies = - (means * means.log()).sum(dim=1)
        return means, results.std(dim=2), entropies

    @abstractmethod
    def sample(self, n_models: int) -> list[torch.nn.Module]:
        raise NotImplementedError
