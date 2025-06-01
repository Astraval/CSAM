import torch


def evaluate_accuracy(dataset: torch.utils.data.Dataset, model: torch.nn.Sequential, num_samples:int = 2000, device="cpu") -> float:
    X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)))
    X, y = X.to(device), y.to(device)
    logit = model(X)
    y_pred = torch.argmax(logit, dim=1)
    return ((y_pred == y).sum() / y.shape[0]).item()
