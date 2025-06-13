import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BFlatten(nn.Module):
    def __init__(self):
        super(BFlatten, self).__init__()

    def forward(self, X):
        """ Flatten a bounded tensor while keeping its intervals together

        Args:
            X : Input bounds.

        Returns:
            Bounds after flattening.
        """
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
        """

        Returns:
            product of interval widths
        """
        return torch.prod(self.W_u + self.W_l) * torch.prod(self.b_u + self.b_l)

    def n_params(self):
        """

        Returns:
            number of center parameters
        """
        return self.W_c.shape[0] * self.W_c.shape[1] + self.b_c.shape[0]

    def forward(self, X):
        """Propagate input bounds through the dense layer

        Args:
            X (torch.Tensor): Input bounds 

        Returns:
            torch,Tensor : Output bounds  
        """
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
    """ 
    This is a 2-D convolutional nn with interval weights and biases

    
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BConv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W_c = nn.Parameter(torch.zeros(out_channels, in_channels, *kernel_size))
        self.W_u = nn.Parameter(torch.zeros(out_channels, in_channels, *kernel_size))
        self.W_l = nn.Parameter(torch.zeros(out_channels, in_channels, *kernel_size))

        self.b_c = nn.Parameter(torch.zeros(out_channels))
        self.b_u = nn.Parameter(torch.zeros(out_channels))
        self.b_l = nn.Parameter(torch.zeros(out_channels))

    def deviation_sum(self):
        return torch.sum((self.W_u + self.W_l) / 2) + torch.sum(
            (self.b_u + self.b_l) / 2
        )

    def n_params(self):
        """
        Returns:
            Number of center parameters
        """
        return self.W_c.shape[0] * self.W_c.shape[1] * self.W_c.shape[2] * self.W_c.shape[3]

    def forward(self, X):
        
        """
        Borrowed directly from
        https://github.com/matthewwicker/RobustExplanationConstraintsForNeuralNetworks/blob/master/GradCertModule.py
        with explicit permission of the author. All rights reserved.
        """
        X_l, X_u = torch.unbind(X, dim=-1)
        x_mu = (X_u + X_l) / 2
        x_r = (X_u - X_l) / 2

        # W = torch.Tensor(W)
        W_mu = ((self.W_c+self.W_u) + (self.W_c-self.W_l))/2
        W_r = ((self.W_c + self.W_u) - (self.W_c - self.W_l)) / 2

        # https://discuss.pytorch.org/t/adding-bias-to-convolution-output/82684/5
        b_size = self.b_c.shape[0]
        b_u = torch.reshape(self.b_c + self.b_u, (1, b_size, 1, 1))
        b_l = torch.reshape(self.b_c - self.b_l, (1, b_size, 1, 1))
        h_mu = torch.nn.functional.conv2d(x_mu, W_mu, stride=self.stride, padding=self.padding)
        x_rad = torch.nn.functional.conv2d(x_r, torch.abs(W_mu), stride=self.stride, padding=self.padding)
        # assert((x_rad >= 0).all())
        W_rad = torch.nn.functional.conv2d(torch.abs(x_mu), W_r, stride=self.stride, padding=self.padding)
        # assert((W_rad >= 0).all())
        Quad = torch.nn.functional.conv2d(torch.abs(x_r), torch.abs(W_r), stride=self.stride, padding=self.padding)
        # assert((Quad >= 0).all())
        h_u = torch.add(torch.add(torch.add(torch.add(h_mu, x_rad), W_rad), Quad), b_u)
        h_l = torch.add(torch.subtract(torch.subtract(torch.subtract(h_mu, x_rad), W_rad), Quad), b_l)
        assert ((h_u >= h_l).all())

        return torch.stack((h_l, h_u), dim=-1)


def _replace_layers_with_blayers(model):
    """
    Replace linear, convolutional and flatten layers with bounded versions

    """
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
            new_layer = BConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding
            )
            new_layer.W_c = nn.Parameter(child.weight.clone().detach())
            new_layer.b_c = nn.Parameter(child.bias.clone().detach())
            new_layer.W_l = nn.Parameter(torch.zeros_like(child.weight))
            new_layer.W_u = nn.Parameter(torch.zeros_like(child.weight))
            new_layer.b_l = nn.Parameter(torch.zeros_like(child.bias))
            new_layer.b_u = nn.Parameter(torch.zeros_like(child.bias))
            setattr(model, name, new_layer)
        elif isinstance(child, nn.Flatten):
            new_layer = BFlatten()
            setattr(model, name, new_layer)
        elif isinstance(child, nn.ReLU):
            continue
        elif isinstance(child, nn.Softmax):
            raise NotImplementedError
        elif isinstance(child, nn.ModuleList):
            # Recursively apply to child modules
            _replace_layers_with_blayers(child)


def modelToBModel(model):
    """
    Returns a deep copy model with bounded layers inserted
        
    """
    model = copy.deepcopy(model)
    _replace_layers_with_blayers(model)
    return model


def _replate_blayers_with_layers(model):
    """
    Replace bounded layers with classical not bounded layers
    """
    for name, child in model.named_children():
        if isinstance(child, BDense):
            new_layer = nn.Linear(child.in_features, child.out_features)
            new_layer.weight.data.copy_(child.W_c.clone().detach())
            new_layer.bias.data.copy_(child.b_c.clone().detach())
            setattr(model, name, new_layer)
        elif isinstance(child, BFlatten):
            new_layer = nn.Flatten(start_dim=1, end_dim=-1)
            setattr(model, name, new_layer)
        elif isinstance(child, nn.ReLU):
            continue
        elif isinstance(child, nn.Softmax):
            continue
        elif isinstance(child, nn.ModuleList):
            # Recursively apply to child modules
            _replate_blayers_with_layers(child)
        elif isinstance(child, BConv2d):
            new_layer = nn.Conv2d(child.in_channels, child.out_channels, child.kernel_size, child.stride, child.padding)
            new_layer.weight.data.copy_(child.W_c.clone().detach())
            new_layer.bias.data.copy_(child.b_c.clone().detach())
            setattr(model, name, new_layer)


def bmodelToModel(bmodel):
    """
    Returns a deep copy of the model with all the bounded layers replaced with classical layers
    """
    bmodel = copy.deepcopy(bmodel)
    _replate_blayers_with_layers(bmodel)
    return bmodel


def assign_epsilon(bmodel, epsilon: float):
    for layer in bmodel:
        if isinstance(layer, BDense) or isinstance(layer, BConv2d):
            layer.W_u.data.copy_(torch.ones_like(layer.W_u) * epsilon)
            layer.W_l.data.copy_(torch.ones_like(layer.W_l) * epsilon)
            layer.b_u.data.copy_(torch.ones_like(layer.b_u) * epsilon)
            layer.b_l.data.copy_(torch.ones_like(layer.b_l) * epsilon)


def max_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Compute the worst-case cross entropy los, considering the interval predictions

    Args:
        y_true (torch.Tensor): truth labels
        y_pred (torch.Tensor): predicted bounds raw scores

    Returns:
        torch.Tensor: loss value
    """
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    min_log = y_true_one_hot * pred_l  # get lower bound prediction for target
    min_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_u  # get upper bound prediction for non targets
    return F.cross_entropy(min_log, y_true)


def min_acc(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Computes the minimum possible accuracy (in the worst case)

    Args:
        y_true (torch.Tensor): truth labels
        y_pred (torch.Tensor): predicted bounds raw scores

    """
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    min_log = y_true_one_hot * pred_l  # get lower bound prediction for target
    min_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_u  # get upper bound prediction for non targets
    return torch.sum(torch.argmax(min_log, dim=1) == y_true) / y_pred.shape[0]


def soft_min_acc(y_true: torch.Tensor, y_pred: torch.Tensor, T=10):
    """version of min_acc using softmax with temperature scaling

    Args:
        y_true (torch.Tensor): truth labels
        y_pred (torch.Tensor): predicted bounds raw scores
        T (int, optional): Temperature scaling factor. Defaults to 10.

    """
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
    """Compute the best case cross-entropy loss

    Args:
        y_true (torch.Tensor): truth labels
        y_pred (torch.Tensor): predicted bounds raw scores

    Returns:
        torch.Tensor: Loss value
    """
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    max_log = y_true_one_hot * pred_u  # get upper bound pred for target
    max_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_l  # get lower bound pred for non target
    return F.cross_entropy(max_log, y_true)


def max_acc(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Compute the maximum possible accuracy (in the best case)

    Args:
        y_true (torch.Tensor): truth labels
        y_pred (torch.Tensor): predictor bounds raw scores

    """
    y_true = y_true.squeeze(dim=-1)
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    pred_l, pred_u = torch.unbind(y_pred, dim=-1)
    max_log = y_true_one_hot * pred_u  # get upper bound pred for target
    max_log += (
                       torch.ones_like(y_true_one_hot) - y_true_one_hot
               ) * pred_l  # get lower bound pred for non target
    return torch.sum(torch.argmax(max_log, dim=1) == y_true) / y_pred.shape[0]


def soft_max_acc(y_true: torch.Tensor, y_pred: torch.Tensor):
    """version of max_acc using softmax with temperature scaling

    Args:
        y_true (torch.Tensor): truth labels
        y_pred (torch.Tensor): predicted bounds raw scores

    """
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
    """
    Return the total number of parameters in a bounded interval model
    """
    num_params = 0
    for layerB in bmodel:
        if hasattr(layerB, "W_c"):
            num_params += layerB.W_c.numel()
            num_params += layerB.b_c.numel()

    return num_params


def get_task_safe_set_size(bmodel):
    """
    Returns the sum of all deviations in the bounded model (how uncertain the bounded model is)
    """
    devsum = 0
    for layerB in bmodel:
        if hasattr(layerB, "deviation_sum"):
            devsum += layerB.deviation_sum()
    return devsum


def get_task_safe_set_volume(bmodel):
    """
    Return the product of all deviations (the volume of the uncertainty)
    """
    vol = 1
    for layerB in bmodel:
        if hasattr(layerB, "task_safe_set_vol"):
            vol *= layerB.task_safe_set_vol()
    return vol


def copy_bounds(bmodel, target_model):
    """Copies the interval bounds from a model of interest to a target model

    Args:
        bmodel : model of interest
        target_model : target model that will receive the copied interval bounds values
    """
    for target_layer, layerB in zip(bmodel, target_model):
        if isinstance(layerB, BDense):
            layerB.W_u.data.copy_(target_layer.W_u.clone().detach())
            layerB.W_l.data.copy_(target_layer.W_l.clone().detach())
            layerB.b_u.data.copy_(target_layer.b_u.clone().detach())
            layerB.b_l.data.copy_(target_layer.b_l.clone().detach())


def get_intersection(model: nn.Sequential, model2: nn.Sequential) -> nn.Sequential:
    """
    Takes 2 bounded interval models and return a new model whose intervals are the intersections of the intervals from both models
    """
    out = copy.deepcopy(model2)
    for target_layer, layer in zip(out, model):
        if isinstance(target_layer, BDense):
            target_layer.W_l.data.copy_(torch.min(layer.W_l, target_layer.W_l))
            target_layer.W_u.data.copy_(torch.min(layer.W_u, target_layer.W_u))
            target_layer.b_l.data.copy_(torch.min(layer.b_l, target_layer.b_l))
            target_layer.b_u.data.copy_(torch.min(layer.b_u, target_layer.b_u))

    return out
