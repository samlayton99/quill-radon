"""Inner-layer parameterizations for single-hidden-layer tanh MLPs.

Three layer types, each representing the pre-activation gamma * (x - center):

GammaLinear:     Raw parameterization. Stores gamma and centers directly.
                 Forward: gamma * (x - centers).
                 Natural match to QI construction but gradients vanish at large gamma.

GammaExpLinear:  Log-gamma reparameterization. Stores log_gamma and centers.
                 Forward: exp(log_gamma) / h * (x - centers).
                 Gradient d(phi)/d(log_gamma) = O(1) instead of O(1/N).

StandardLinear:  Standard weight/bias parameterization (nn.Linear wrapper).
                 Forward: x @ weight + bias.
                 Baseline for the reparameterization experiments (expD03).

All layers: input [batch, 1], output [batch, width], dtype float64.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GammaLinear(nn.Module):
    """Linear layer parameterized as gamma * (x - center).

    Parameters:
        gamma:   nn.Parameter, shape [1, width].
        centers: nn.Parameter, shape [1, width].

    QI construction sets:
        gamma = lambda_star / h  (scalar, broadcast to all neurons)
        centers = [-1 + k*h for k in range(N)]  (uniform grid)
    """

    def __init__(self, width: int, gamma_init=None, center_init=None):
        super().__init__()
        if gamma_init is not None:
            self.gamma = nn.Parameter(gamma_init.clone().detach().reshape(1, width))
        else:
            self.gamma = nn.Parameter(torch.ones(1, width, dtype=torch.float64))
        if center_init is not None:
            self.centers = nn.Parameter(center_init.clone().detach().reshape(1, width))
        else:
            self.centers = nn.Parameter(torch.zeros(1, width, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """gamma * (x - centers). x: [batch, 1] -> [batch, width]."""
        return self.gamma * (x - self.centers)


class GammaExpLinear(nn.Module):
    """Linear layer with gamma = exp(log_gamma) / h.

    Reparameterization ensuring gradient d(loss)/d(log_gamma) = O(1).

    Parameters:
        log_gamma: nn.Parameter, shape [1, width].
        centers:   nn.Parameter, shape [1, width].
        h:         float (not a parameter). Grid spacing 2/N.

    For QI init: log_gamma = log(lambda_star), h = 2/N,
    so gamma = lambda_star / h = lambda_star * N / 2.
    """

    def __init__(self, width: int, h: float = 1.0, log_gamma_init=None, center_init=None):
        super().__init__()
        if log_gamma_init is not None:
            self.log_gamma = nn.Parameter(log_gamma_init.clone().detach().reshape(1, width))
        else:
            self.log_gamma = nn.Parameter(torch.zeros(1, width, dtype=torch.float64))
        if center_init is not None:
            self.centers = nn.Parameter(center_init.clone().detach().reshape(1, width))
        else:
            self.centers = nn.Parameter(torch.zeros(1, width, dtype=torch.float64))
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """exp(log_gamma)/h * (x - centers). x: [batch, 1] -> [batch, width]."""
        gamma = torch.exp(self.log_gamma) / self.h
        return gamma * (x - self.centers)

    def get_gamma(self) -> torch.Tensor:
        """Return effective gamma = exp(log_gamma) / h."""
        return torch.exp(self.log_gamma) / self.h


class StandardLinear(nn.Module):
    """Standard weight/bias linear layer. Baseline for the reparameterization experiments (expD03).

    Wraps nn.Linear(1, width). Does not separate gamma and center.
    """

    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(1, width, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x @ weight + bias. x: [batch, 1] -> [batch, width]."""
        return self.linear(x)


def get_layer(layer_type: str, width: int, **kwargs) -> nn.Module:
    """Factory for inner layer types.

    Args:
        layer_type: "gamma_linear" | "gamma_exp" | "standard"
        width: Number of hidden neurons.
        **kwargs: Layer-specific kwargs (h for GammaExpLinear, inits).

    Returns:
        Instantiated layer module.
    """
    if layer_type == "gamma_linear":
        return GammaLinear(width, **kwargs)
    elif layer_type == "gamma_exp":
        return GammaExpLinear(width, **kwargs)
    elif layer_type == "standard":
        return StandardLinear(width)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
