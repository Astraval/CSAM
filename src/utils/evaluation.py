from typing import Callable
from tqdm import tqdm
import torch


def evaluate_accuracy(dataset: torch.utils.data.Dataset, model: torch.nn.Sequential, num_samples: int = 2000,
                      device="cpu") -> float:
    X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    X, y = X.to(device), y.to(device)
    logit = model(X)
    y_pred = torch.argmax(logit, dim=1)
    return ((y_pred == y).sum() / y.shape[0]).item()


def evaluate_fgsm_accuracy(dataset: torch.utils.data.Dataset, model: torch.nn.Sequential, num_samples: int = 2000,
                           device="cpu", epsilon: float = 1e-3) -> float:
    X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    X: torch.Tensor = X.to(device)
    y: torch.Tensor = y.to(device)
    X.requires_grad = True
    logit = model(X)
    loss = torch.nn.CrossEntropyLoss()(logit, y)
    loss.backward()
    model.zero_grad()
    X_adv = X + X.grad.sign() * epsilon
    print("coucou")
    X_adv = torch.clamp(X_adv, -1, 1)
    with torch.no_grad():
        logit_adv = model(X_adv)
        y_pred = torch.argmax(logit_adv, dim=1)
    return ((y_pred == y).sum() / y.shape[0]).item()


def evaluate_run(run: Callable[[], list[float]], n_runs: int) -> (list[float], list[float], list[float]):
    run_statistics = []
    progress_bar = tqdm(range(n_runs))
    for i in progress_bar:
        values = run()
        run_statistics.append(values)
        progress_bar.set_postfix({
            "Current Values ": values
        })
    all_runs = run_statistics
    run_statistics = torch.tensor(run_statistics)
    means = run_statistics.mean(dim=0)
    stds = run_statistics.std(dim=0)
    return means.tolist(), stds.tolist(), all_runs
