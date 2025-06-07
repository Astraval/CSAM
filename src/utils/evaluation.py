from typing import Callable
from tqdm import tqdm
import torch

from src.cert import Safebox


def evaluate_accuracy(dataset: torch.utils.data.Dataset, model: torch.nn.Sequential, num_samples: int = 2000,
                      device="cpu") -> float:
    X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    X, y = X.to(device), y.to(device)
    logit = model(X)
    y_pred = torch.argmax(logit, dim=1)
    return ((y_pred == y).sum() / y.shape[0]).item()


def evaluate_fgsm_accuracy(dataset: torch.utils.data.Dataset, model: torch.nn.Sequential, num_samples: int = 2000,
                           device="cpu", epsilon: float = 1e-3, data_domain: tuple[float] = (0.0, 1.0)) -> float:
    X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    X: torch.Tensor = X.to(device)
    y: torch.Tensor = y.to(device)
    X.requires_grad = True
    logit = model(X)
    loss = torch.nn.CrossEntropyLoss()(logit, y)
    loss.backward()
    model.zero_grad()
    X_adv = X + X.grad.sign() * epsilon
    X_adv = torch.clamp(X_adv, *data_domain)
    with torch.no_grad():
        logit_adv = model(X_adv)
        y_pred = torch.argmax(logit_adv, dim=1)
    return ((y_pred == y).sum() / y.shape[0]).item()


def evaluate_certified_adv_accuracy(dataset: torch.utils.data.Dataset, model: torch.nn.Sequential, num_samples,
                                    data_domain: tuple[float],
                                    device="cpu", epsilon: float = 1e-3) -> float:
    X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    X: torch.Tensor = X.to(device)
    y: torch.Tensor = y.to(device)
    model = Safebox.modelToBModel(model)
    Safebox.assign_epsilon(model, 0.0)
    X = torch.stack([X - epsilon, X + epsilon], dim=-1).clamp(data_domain[0], data_domain[1])
    return Safebox.min_acc(y, model(X)).item()



def evaluate_run(run: Callable[[], list[float]], n_runs: int) -> (list[float], list[float], list[float]):
    run_statistics = []
    for i in range(n_runs):
        print(f" ---------- Starting Run #{i} ---------- ")
        values = run()
        run_statistics.append(values)
        print(f"-> Run finished with values : {values}")
    all_runs = run_statistics
    run_statistics = torch.tensor(run_statistics)
    means = run_statistics.mean(dim=0)
    stds = run_statistics.std(dim=0)
    return means.tolist(), stds.tolist(), all_runs
