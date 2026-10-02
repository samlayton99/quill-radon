"""Target function registry for approximation experiments.

9 functions spanning the 6 target categories of the success criterion:
low_freq (sine, cosine), high_freq (sine_8pi), boundary_layer (runge,
tanh_steep), mixed_scale (sine_mixture), polynomial (exp, poly5), rough
(abs_cubed).

The QI construction needs the analytic derivative g' (the kernel is
sech^2 = tanh'), so every target supplies one. Each TargetFn provides fn/deriv
(torch) for training/evaluation, plus fn_numpy/deriv_numpy for construction.

Usage:
    target = get_target("sine")
    y = target.fn(x)          # x is torch.Tensor
    y_np = target.fn_numpy(x_np)  # x_np is numpy array
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import numpy as np


@dataclass(frozen=True)
class TargetFn:
    """A target function with analytic derivative.

    Fields:
        fn:          f(x) -> y, operates on torch tensors.
        deriv:       f'(x) -> y', analytic derivative (torch).
        fn_numpy:    f(x) -> y, operates on numpy arrays (for construction).
        deriv_numpy: f'(x) -> y', numpy version (for construction).
        name:        Registry key.
        category:    One of the 6 target categories.
    """
    fn: Callable[[torch.Tensor], torch.Tensor]
    deriv: Callable[[torch.Tensor], torch.Tensor]
    fn_numpy: Callable[[np.ndarray], np.ndarray]
    deriv_numpy: Callable[[np.ndarray], np.ndarray]
    name: str
    category: str


TARGET_REGISTRY: dict[str, TargetFn] = {}


def _register(name: str, category: str,
              fn_torch, deriv_torch, fn_np, deriv_np):
    TARGET_REGISTRY[name] = TargetFn(
        fn=fn_torch, deriv=deriv_torch,
        fn_numpy=fn_np, deriv_numpy=deriv_np,
        name=name, category=category,
    )


# --- Low-frequency analytic ---

_register(
    "sine", "low_freq",
    fn_torch=lambda x: torch.sin(2 * math.pi * x),
    deriv_torch=lambda x: 2 * math.pi * torch.cos(2 * math.pi * x),
    fn_np=lambda x: np.sin(2 * np.pi * x),
    deriv_np=lambda x: 2 * np.pi * np.cos(2 * np.pi * x),
)

_register(
    "cosine", "low_freq",
    fn_torch=lambda x: torch.cos(2 * math.pi * x),
    deriv_torch=lambda x: -2 * math.pi * torch.sin(2 * math.pi * x),
    fn_np=lambda x: np.cos(2 * np.pi * x),
    deriv_np=lambda x: -2 * np.pi * np.sin(2 * np.pi * x),
)

# --- High-frequency analytic ---

_register(
    "sine_8pi", "high_freq",
    fn_torch=lambda x: torch.sin(8 * math.pi * x),
    deriv_torch=lambda x: 8 * math.pi * torch.cos(8 * math.pi * x),
    fn_np=lambda x: np.sin(8 * np.pi * x),
    deriv_np=lambda x: 8 * np.pi * np.cos(8 * np.pi * x),
)

# --- Boundary-layer / steep transition ---

_register(
    "runge", "boundary_layer",
    fn_torch=lambda x: 1.0 / (1.0 + 25.0 * x ** 2),
    deriv_torch=lambda x: -50.0 * x / (1.0 + 25.0 * x ** 2) ** 2,
    fn_np=lambda x: 1.0 / (1.0 + 25.0 * x ** 2),
    deriv_np=lambda x: -50.0 * x / (1.0 + 25.0 * x ** 2) ** 2,
)

_register(
    "tanh_steep", "boundary_layer",
    fn_torch=lambda x: torch.tanh(20.0 * x),
    deriv_torch=lambda x: 20.0 / torch.cosh(20.0 * x) ** 2,
    fn_np=lambda x: np.tanh(20.0 * x),
    deriv_np=lambda x: 20.0 / np.cosh(20.0 * x) ** 2,
)

# --- Mixed-scale analytic ---

_register(
    "sine_mixture", "mixed_scale",
    fn_torch=lambda x: torch.sin(2 * math.pi * x) + 0.5 * torch.sin(6 * math.pi * x) + 0.25 * torch.sin(14 * math.pi * x),
    deriv_torch=lambda x: 2 * math.pi * torch.cos(2 * math.pi * x) + 3 * math.pi * torch.cos(6 * math.pi * x) + 3.5 * math.pi * torch.cos(14 * math.pi * x),
    fn_np=lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x) + 0.25 * np.sin(14 * np.pi * x),
    deriv_np=lambda x: 2 * np.pi * np.cos(2 * np.pi * x) + 3 * np.pi * np.cos(6 * np.pi * x) + 3.5 * np.pi * np.cos(14 * np.pi * x),
)

# --- Polynomial / entire function ---

_register(
    "exp", "polynomial",
    fn_torch=lambda x: torch.exp(x),
    deriv_torch=lambda x: torch.exp(x),
    fn_np=lambda x: np.exp(x),
    deriv_np=lambda x: np.exp(x),
)

_register(
    "poly5", "polynomial",
    fn_torch=lambda x: x ** 5 - 3 * x ** 3 + x,
    deriv_torch=lambda x: 5 * x ** 4 - 9 * x ** 2 + 1,
    fn_np=lambda x: x ** 5 - 3 * x ** 3 + x,
    deriv_np=lambda x: 5 * x ** 4 - 9 * x ** 2 + 1,
)

# --- Slightly rough ---

_register(
    "abs_cubed", "rough",
    fn_torch=lambda x: torch.abs(x) ** 3,
    deriv_torch=lambda x: 3 * x * torch.abs(x),
    fn_np=lambda x: np.abs(x) ** 3,
    deriv_np=lambda x: 3 * x * np.abs(x),
)


def get_target(name: str) -> TargetFn:
    """Look up target function by name."""
    if name not in TARGET_REGISTRY:
        raise KeyError(f"Unknown target '{name}'. Available: {list(TARGET_REGISTRY.keys())}")
    return TARGET_REGISTRY[name]


def get_all_targets() -> list[TargetFn]:
    """Return all registered targets."""
    return list(TARGET_REGISTRY.values())
