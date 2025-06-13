from typing import Callable
from tqdm import tqdm
import torch

from src.cert import Safebox

"""
This file provides functions to evaluate neural networks models on standard, adversarial, and certified adversarial accuracy.
"""

def evaluate_accuracy(dataset: torch.utils.data.Dataset, model: torch.nn.Sequential, num_samples: int = 2000,
                      device="cpu") -> float:
    """
    Computes the standard accuracy of a model on a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to evaluate on.
        model (torch.nn.Sequential): The model to evaluate.
        num_samples (int, optional): Number of samples to use for evaluation. Defaults to 2000.
        device (str, optional): The device to use for computation. Defaults to "cpu".

    Returns:
        float: The accuracy of the model on the dataset.
    """
    X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    X, y = X.to(device), y.to(device)
    logit = model(X)
    y_pred = torch.argmax(logit, dim=1)
    return ((y_pred == y).sum() / y.shape[0]).item()


def evaluate_fgsm_accuracy(dataset: torch.utils.data.Dataset, model: torch.nn.Sequential, num_samples: int = 2000,
                           device="cpu", epsilon: float = 1e-3, data_domain: tuple[float] = (0.0, 1.0)) -> float:
    """
    Computes the accuracy of a model on adversarial examples generated using the FGSM attack.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to evaluate on.
        model (torch.nn.Sequential): The model to evaluate.
        num_samples (int, optional): Number of samples to use for evaluation. Defaults to 2000.
        device (str, optional): The device to use for computation. Defaults to "cpu".
        epsilon (float, optional): Magnitude of the FGSM perturbation. Defaults to 1e-3.
        data_domain (tuple[float], optional): Minimum and maximum values for input data. Defaults to (0.0, 1.0).

    Returns:
        float: The accuracy of the model on FGSM adversarial examples.
    """
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
    """Computes the certified adversarial accuracy of a model using interval bounds.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to evaluate on.
        model (torch.nn.Sequential): The model to evaluate.
        num_samples (int): Number of samples to use for evaluation.
        data_domain (tuple[float]): Minimum and maximum values for input data.
        device (str, optional): The device to use for computation. Defaults to "cpu".
        epsilon (float, optional): Size of the certified adversarial perturbation. Defaults to 1e-3.

    Returns:
        float: The certified adversarial accuracy of the model.
    """
    X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    X: torch.Tensor = X.to(device)
    y: torch.Tensor = y.to(device)
    model = Safebox.modelToBModel(model)
    Safebox.assign_epsilon(model, 0.0)
    X = torch.stack([X - epsilon, X + epsilon], dim=-1).clamp(data_domain[0], data_domain[1])
    return Safebox.min_acc(y, model(X)).item()



def evaluate_run(run: Callable[[], list[float]], n_runs: int) -> (list[float], list[float], list[float]):
    """
    Runs an experiment multiple times and aggregate the results.

    Args:
        run (Callable[[], list[float]]): A function that runs a single experiment and returns a list of results.
        n_runs (int): Number of times to repeat the experiment.

    Returns:
        tuple: (means, stds, all_runs)
            means (list[float]): Mean of each result across runs.
            stds (list[float]): Standard deviation of each result across runs.
            all_runs (list[float]): Raw results from all runs.
    """
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
