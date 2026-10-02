"""Parameter freezing via requires_grad.

PyTorch freezing is simple: param.requires_grad_(False) excludes a parameter
from gradient computation. No custom variable types needed.

Usage:
    model = QIMlp(config)
    initialize_from_construction(model, qi_result)
    freeze_inner_layer(model)       # gamma + centers frozen
    # Training only updates readout weights
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn


def freeze_params(model: nn.Module, filter_fn: Callable[[str, nn.Parameter], bool]) -> None:
    """Freeze parameters where filter_fn returns True."""
    for name, param in model.named_parameters():
        if filter_fn(name, param):
            param.requires_grad_(False)


def unfreeze_params(model: nn.Module, filter_fn: Callable[[str, nn.Parameter], bool]) -> None:
    """Unfreeze parameters where filter_fn returns True."""
    for name, param in model.named_parameters():
        if filter_fn(name, param):
            param.requires_grad_(True)


def freeze_inner_layer(model: nn.Module) -> None:
    """Freeze all inner layer parameters (gamma/log_gamma and centers)."""
    freeze_params(model, lambda name, _p: name.startswith("inner_layer"))


def freeze_gamma(model: nn.Module) -> None:
    """Freeze only gamma. Keep centers trainable.

    The "gamma" suffix also matches GammaExpLinear's "log_gamma" parameter.
    """
    freeze_params(model, lambda name, _p: name.endswith("gamma"))


def freeze_centers(model: nn.Module) -> None:
    """Freeze only centers."""
    freeze_params(model, lambda name, _p: name.endswith("centers"))


def freeze_readout(model: nn.Module) -> None:
    """Freeze readout weights and bias."""
    freeze_params(model, lambda name, _p: name.startswith("readout"))


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad_(True)


def get_trainable_param_count(model: nn.Module) -> int:
    """Count parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
