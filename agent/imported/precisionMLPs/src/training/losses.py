"""Loss functions for training.

All follow signature: (model, x, y) -> scalar tensor.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn as nn


def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return ((model(x) - y) ** 2).mean()


def lp_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, p: float = 6.0) -> torch.Tensor:
    """Lp loss: mean(|f(x) - y|^p). Approximates L_inf as p -> inf."""
    return (torch.abs(model(x) - y) ** p).mean()


def hybrid_boundary_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                         boundary_weight: float = 5.0, boundary_fraction: float = 0.1) -> torch.Tensor:
    """MSE with extra weight on points near domain boundaries."""
    resid2 = (model(x) - y) ** 2
    # Determine boundary mask based on x position: near min/max of x range.
    x_flat = x.detach().reshape(-1)
    x_min = x_flat.min()
    x_max = x_flat.max()
    span = x_max - x_min
    threshold = boundary_fraction * span
    near_boundary = (x_flat - x_min < threshold) | (x_max - x_flat < threshold)
    weights = torch.where(near_boundary, boundary_weight, 1.0).to(resid2.dtype).reshape(resid2.shape)
    return (weights * resid2).mean()


def get_loss_fn(name: str, **kwargs) -> Callable:
    """Look up loss function by name. Returns (model, x, y) -> scalar.

    Usage:
        loss_fn = get_loss_fn("mse")
        loss_fn = get_loss_fn("lp", p=6.0)
    """
    if name == "mse":
        return mse
    elif name == "lp":
        return partial(lp_loss, p=kwargs.get("p", 6.0))
    elif name == "hybrid_boundary":
        return partial(hybrid_boundary_loss, **kwargs)
    raise ValueError(f"Unknown loss: {name}")
