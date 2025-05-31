import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BFlatten(nn.Module):
    def __init__(self):
        super(BFlatten, self).__init__()

    def forward(self, X):
        X_l, X_u = torch.unbind(X, dim=-1)
        batch_size = 1
        if X_l.shape[0] is not None:
            batch_size = X_l.shape[0]
        prod = 1
        for i in range(1, len(X_l.shape)):
            prod *= X_l.shape[i]

        return torch.stack(
            (X_l.view(batch_size, prod), X_u.view(batch_size, prod)), dim=2
        )

    def deviation_sum(self):
        return 0


class BDense(nn.Module):
    def __init__(self, in_features, out_features):
        super(BDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_c = nn.Parameter(torch.zeros(in_features, out_features))
        self.W_u = nn.Parameter(torch.zeros(in_features, out_features))
        self.W_l = nn.Parameter(torch.zeros(in_features, out_features))

        self.b_c = nn.Parameter(torch.zeros(out_features))
        self.b_u = nn.Parameter(torch.zeros(in_features, out_features))
        self.b_l = nn.Parameter(torch.zeros(in_features, out_features))

    def deviation_sum(self):
        return torch.sum((self.W_u + self.W_l) / 2) + torch.sum(
            (self.b_u + self.b_l) / 2
        )

    def task_safe_set_vol(self):
        return torch.prod(self.W_u + self.W_l) * torch.prod(self.b_u + self.b_l)

    def n_params(self):
        return self.W_c.shape[0] * self.W_c.shape[1] + self.b_c.shape[0]

    def forward(self, X):
        X_l, X_u = torch.unbind(X, dim=-1)
        X_mu = (X_u + X_l) / 2
        X_r = (X_u - X_l) / 2
        W_mu = ((self.W_c + self.W_u) + (self.W_c - self.W_l)) / 2
        W_r = ((self.W_c + self.W_u) - (self.W_c - self.W_l)) / 2
        M_z = torch.mm(X_r, torch.abs(self.W_c).T)
        M_w = torch.mm(torch.abs(X_mu), W_r.T)
        Q = torch.mm(torch.abs(X_r), torch.abs(W_r).T)

        lower = torch.mm(X_mu, W_mu.T) - M_w - M_z - Q + self.b_c - self.b_l

        upper = torch.mm(X_mu, W_mu.T) + M_w + M_z + Q + self.b_c + self.b_u
        return torch.stack((lower, upper), dim=2)

    def __repr__(self):
        return (
            f"BDense(in_features={self.in_features}, out_features={self.out_features}"
        )


class BConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W_c = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.W_u = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.W_l = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))

        self.b_c = nn.Parameter(torch.zeros(out_channels))
        self.b_u = nn.Parameter(torch.zeros(out_channels))
        self.b_l = nn.Parameter(torch.zeros(out_channels))

    def deviation_sum(self):
        return torch.sum((self.W_u + self.W_l) / 2) + torch.sum(
            (self.b_u + self.b_l) / 2
        )

    def n_params(self):
        return self.W_c.shape[0] * self.W_c.shape[1] * self.W_c.shape[2] * self.W_c.shape[3]

    def forward(self, X):
        X_l, X_u = torch.unbind(X, dim=-1)
        X_mu = (X_u + X_l) / 2
        X_r = (X_u - X_l) / 2

        bias_upper = self.b_c + self.b_u
        bias_lower = self.b_c - self.b_l
        h_mu = F.conv2d(X_mu, self.W_c, 0, self.stride, self.padding)
        x_rad = F.conv2d(X_r, torch.abs(self.W_c), 0, self.stride, self.padding)
        W_upper = F.conv2d(torch.abs(X_mu), self.W_u, 0, self.stride, self.padding)
        W_lower = F.conv2d(torch.abs(X_mu), self.W_l, 0, self.stride, self.padding)
        Quad_upper = F.conv2d(torch.abs(X_r), torch.abs(self.W_u), 0, self.stride, self.padding)
        Quad_lower = F.conv2d(torch.abs(X_r), torch.abs(self.W_l), 0, self.stride, self.padding)
        h_u = torch.add(torch.add(torch.add(torch.add(h_mu, x_rad), W_upper), Quad_upper), bias_upper)
        h_l = torch.add(torch.subtract(torch.subtract(torch.subtract(h_mu, x_rad), W_lower), Quad_lower), bias_lower)

        return torch.stack((h_l, h_u), dim=2)


def replace_linear_with_dense(model):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            new_layer = BDense(child.in_features, child.out_features)
            new_layer.W_c = nn.Parameter(child.weight.clone().detach())
            new_layer.b_c = nn.Parameter(child.bias.clone().detach())
            new_layer.W_l = nn.Parameter(torch.zeros_like(child.weight))
            new_layer.W_u = nn.Parameter(torch.zeros_like(child.weight))
            new_layer.b_l = nn.Parameter(torch.zeros_like(child.bias))
            new_layer.b_u = nn.Parameter(torch.zeros_like(child.bias))
            # Replace with Dense, preserving parameters
            setattr(model, name, new_layer)
        elif isinstance(child, nn.Conv2d):
            print("Not implemented")
        elif isinstance(child, nn.Flatten):
            continue
        elif isinstance(child, nn.ReLU):
            continue
        elif isinstance(child, nn.Softmax):
            continue
        elif isinstance(child, nn.ModuleList):
            # Recursively apply to child modules
            replace_linear_with_dense(child)


def modelToBModel(model):
    model = copy.deepcopy(model)
    replace_linear_with_dense(model)
    return model


def replace_dense_with_linear(model):
    for name, child in model.named_children():
        if isinstance(child, BDense):
            new_layer = nn.Linear(child.in_features, child.out_features)
            new_layer.weight = nn.Parameter(child.W_c.clone().detach())
            new_layer.bias = nn.Parameter(child.b_c.clone().detach())
            setattr(model, name, new_layer)
        elif isinstance(child, nn.Flatten):
            continue
        elif isinstance(child, nn.ReLU):
            continue
        elif isinstance(child, nn.Softmax):
            continue
        elif isinstance(child, nn.ModuleList):
            # Recursively apply to child modules
            replace_dense_with_linear(child)


def bmodelToModel(bmodel):
    bmodel = copy.deepcopy(bmodel)
    replace_dense_with_linear(bmodel)
    return bmodel


def assign_epsilon(bmodel, epsilon: float):
    for layer in bmodel:
        if isinstance(layer, BDense):
            layer.W_u = nn.Parameter(torch.ones_like(layer.W_u) * epsilon)
            layer.W_l = nn.Parameter(torch.ones_like(layer.W_l) * epsilon)
            layer.b_u = nn.Parameter(torch.ones_like(layer.b_u) * epsilon)
            layer.b_l = nn.Parameter(torch.ones_like(layer.b_l) * epsilon)


def max_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    min_log = y_true_one_hot * pred_l  # get lower bound prediction for target
    min_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_u  # get upper bound prediction for non targets
    return F.cross_entropy(min_log, y_true)


def min_acc(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    min_log = y_true_one_hot * pred_l  # get lower bound prediction for target
    min_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_u  # get upper bound prediction for non targets
    return torch.sum(torch.argmax(min_log, dim=1) == y_true) / y_pred.shape[0]


def soft_min_acc(y_true: torch.Tensor, y_pred: torch.Tensor, T=10):
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    min_log = y_true_one_hot * pred_l  # get lower bound prediction for target
    min_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_u  # get upper bound prediction for non targets

    probs = F.softmax(min_log * T, dim=1)
    correct_probs = probs[torch.arange(probs.size(0)), y_true]
    return correct_probs.mean()


def min_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    max_log = y_true_one_hot * pred_u  # get upper bound pred for target
    max_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_l  # get lower bound pred for non target
    return F.cross_entropy(max_log, y_true)


def max_acc(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    max_log = y_true_one_hot * pred_u  # get upper bound pred for target
    max_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_l  # get lower bound pred for non target
    return torch.sum(torch.argmax(max_log, dim=1) == y_true) / y_pred.shape[0]


def soft_max_acc(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    max_log = y_true_one_hot * pred_u  # get upper bound pred for target
    max_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_l  # get lower bound pred for non target

    T = 1000
    probs = F.softmax(max_log * T, dim=1)
    correct_probs = probs[torch.arange(probs.size(0)), y_true]
    return correct_probs.mean()


def get_num_model_params(bmodel):
    num_params = 0
    for layerB in bmodel:
        if hasattr(layerB, "W_c"):
            num_params += layerB.W_c.numel()
            num_params += layerB.b_c.numel()

    return num_params


def get_task_safe_set_size(bmodel):
    devsum = 0
    for layerB in bmodel:
        if hasattr(layerB, "deviation_sum"):
            devsum += layerB.deviation_sum()
    return devsum


def get_task_safe_set_volume(bmodel):
    vol = 1
    for layerB in bmodel:
        if hasattr(layerB, "task_safe_set_vol"):
            vol *= layerB.task_safe_set_vol()
    return vol


def copy_bounds(bmodel, target_model):
    for target_layer, layerB in zip(bmodel, target_model):
        if isinstance(layerB, BDense):
            layerB.W_u.data.copy_(target_layer.W_u.clone().detach())
            layerB.W_l.data.copy_(target_layer.W_l.clone().detach())
            layerB.b_u.data.copy_(target_layer.b_u.clone().detach())
            layerB.b_l.data.copy_(target_layer.b_l.clone().detach())


def get_intersection(model: nn.Sequential, model2: nn.Sequential) -> nn.Sequential:
    out = copy.deepcopy(model2)
    for target_layer, layer in zip(out, model):
        if isinstance(target_layer, BDense):
            target_layer.W_l.data.copy_(torch.min(layer.W_l, target_layer.W_l))
            target_layer.W_u.data.copy_(torch.min(layer.W_u, target_layer.W_u))
            target_layer.b_l.data.copy_(torch.min(layer.b_l, target_layer.b_l))
            target_layer.b_u.data.copy_(torch.min(layer.b_u, target_layer.b_u))

    return out