from typing import Callable

import torch


def evaluate_accuracy(dataset: torch.utils.data.Dataset, model: torch.nn.Sequential, num_samples: int = 2000,
                      device="cpu") -> float:
    X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    X, y = X.to(device), y.to(device)
    logit = model(X)
    y_pred = torch.argmax(logit, dim=1)
    return ((y_pred == y).sum() / y.shape[0]).item()


def evaluate_run(run: Callable[[], list[float]], n_runs: int) -> (list[float], list[float]):
    run_statistics = []
    for i in range(n_runs):
        run_statistics.append(run())
    run_statistics = torch.tensor(run_statistics)
    means = run_statistics.mean(dim=0)
    stds = run_statistics.std(dim=0)
    return means.tolist(), stds.tolist()
