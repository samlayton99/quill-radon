"""Project QI construction results into model parameters.

Three patterns:
1. Full construction init: copy gamma, centers, readout from QIResult.
2. Partial construction: copy only some parameter groups.
3. Readout solve: given current inner layer, solve readout via numpy lstsq.

All parameter writes use torch.no_grad().
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from src.construction.qi_mpmath import QIResult
from src.construction.readout import build_phi, solve_readout_with_bias
from src.config.schema import FreezeConfig, InitConfig
from src.models.layers import GammaLinear, GammaExpLinear


def initialize_from_construction(
    model: nn.Module,
    qi_result: QIResult,
    *,
    construct_gamma: bool = True,
    construct_centers: bool = True,
    construct_readout: bool = True,
) -> None:
    """Copy QIResult values into model parameters in-place.

    Uses the interior (non-halo) centers and coefficients to match model width.
    For models with halo nodes, use the full centers/a_coeffs directly.
    """
    with torch.no_grad():
        inner = model.inner_layer
        width = model.config.width

        # Determine which centers/coeffs to use based on model width
        if width == len(qi_result.centers):
            centers = qi_result.centers
            a_coeffs = qi_result.a_coeffs
        elif width == len(qi_result.interior_centers):
            centers = qi_result.interior_centers
            a_coeffs = qi_result.interior_a_coeffs
        else:
            raise ValueError(
                f"Model width {width} doesn't match construction: "
                f"full={len(qi_result.centers)}, interior={len(qi_result.interior_centers)}"
            )

        if construct_gamma:
            if isinstance(inner, GammaLinear):
                inner.gamma.data.fill_(qi_result.gamma)
            elif isinstance(inner, GammaExpLinear):
                # GammaExpLinear computes effective gamma = exp(log_gamma) / h.
                # To reproduce the construction's gamma = lambda* / h (which grows
                # O(N)), the layer's h MUST equal the construction grid spacing.
                # QIMlp does not forward h to the inner layer, so set it here --
                # otherwise the default h=1.0 collapses gamma to lambda* (O(1)),
                # which would artificially manufacture violation #1.
                inner.h = qi_result.h
                inner.log_gamma.data.fill_(math.log(qi_result.lambda_val))

        if construct_centers:
            if isinstance(inner, (GammaLinear, GammaExpLinear)):
                inner.centers.data.copy_(
                    torch.tensor(centers, dtype=torch.float64).reshape(1, width)
                )

        if construct_readout:
            model.readout.weight.data.copy_(
                torch.tensor(a_coeffs, dtype=torch.float64).reshape(1, width)
            )
            model.readout.bias.data.fill_(qi_result.c0)


def initialize_and_freeze(
    model: nn.Module,
    qi_result: QIResult,
    freeze_config: FreezeConfig,
    init_config: InitConfig,
) -> None:
    """Initialize from construction, then freeze per config."""
    initialize_from_construction(
        model, qi_result,
        construct_gamma=init_config.construct_gamma,
        construct_centers=init_config.construct_centers,
        construct_readout=init_config.construct_readout,
    )

    inner = model.inner_layer
    if freeze_config.gamma:
        if isinstance(inner, GammaLinear):
            inner.gamma.requires_grad_(False)
        elif isinstance(inner, GammaExpLinear):
            inner.log_gamma.requires_grad_(False)

    if freeze_config.centers:
        if isinstance(inner, (GammaLinear, GammaExpLinear)):
            inner.centers.requires_grad_(False)

    if freeze_config.readout:
        model.readout.weight.requires_grad_(False)
        model.readout.bias.requires_grad_(False)


def initialize_with_readout_solve(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    method: str = "lstsq",
    ridge_alpha: float = 0.0,
) -> dict:
    """Solve readout weights via numpy lstsq given current inner layer.

    Detaches Phi to numpy, solves, copies weights back with torch.no_grad().
    Returns info dict with solve diagnostics.
    """
    with torch.no_grad():
        Phi = model.features(x_train).detach().cpu().numpy()
        y_np = y_train.detach().cpu().numpy().ravel()

        v, bias, info = solve_readout_with_bias(Phi, y_np, method=method, ridge_alpha=ridge_alpha)

        model.readout.weight.data.copy_(
            torch.tensor(v, dtype=torch.float64).reshape(1, -1)
        )
        model.readout.bias.data.fill_(bias)

    return info
