"""Smooth analytic targets R^2 -> R for the 2D ridge/Radon regime.

Lightweight registry: name + fn_numpy only (no derivatives -- there is no QI
construction in 2D; we build Phi and solve the readout by least squares).
Flavors mirror the 1D set: low/high frequency, radial, mixed-scale, plus a
single plane-wave control that should be representable to ~machine epsilon.

All functions take X of shape [n_points, 2] and return [n_points].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Target2D:
    fn_numpy: Callable[[np.ndarray], np.ndarray]
    name: str
    category: str


TARGET2D_REGISTRY: dict[str, Target2D] = {}


def _register(name: str, category: str, fn: Callable[[np.ndarray], np.ndarray]) -> None:
    TARGET2D_REGISTRY[name] = Target2D(fn_numpy=fn, name=name, category=category)


def _xy(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    return X[:, 0], X[:, 1]


def _sine2d(X):
    x, y = _xy(X)
    return np.sin(np.pi * x) * np.cos(np.pi * y)


def _sine2d_hi(X):
    x, y = _xy(X)
    return np.sin(4.0 * np.pi * x) * np.cos(4.0 * np.pi * y)


def _gauss_bump(X):
    x, y = _xy(X)
    return np.exp(-(x ** 2 + y ** 2))


def _runge2d(X):
    x, y = _xy(X)
    return 1.0 / (1.0 + 25.0 * (x ** 2 + y ** 2))


def _mixed2d(X):
    x, y = _xy(X)
    return np.exp(np.sin(np.pi * x) + np.cos(np.pi * y))


def _planewave(X):
    # A single ridge aligned with (0.6, 0.8): exactly one plane wave.
    x, y = _xy(X)
    return np.tanh(3.0 * (0.6 * x + 0.8 * y))


_register("sine2d", "low_freq", _sine2d)
_register("sine2d_hi", "high_freq", _sine2d_hi)
_register("gauss_bump", "radial", _gauss_bump)
_register("runge2d", "boundary_layer", _runge2d)
_register("mixed2d", "mixed_scale", _mixed2d)
_register("planewave", "ridge_control", _planewave)


def get_target_2d(name: str) -> Target2D:
    if name not in TARGET2D_REGISTRY:
        raise KeyError(
            f"Unknown 2D target '{name}'. Available: {list(TARGET2D_REGISTRY)}"
        )
    return TARGET2D_REGISTRY[name]
