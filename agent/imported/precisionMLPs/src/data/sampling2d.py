"""Sampling on a 2D disk for the ridge/Radon regime.

Targets are fit and evaluated on the unit disk (radius 1); the neurons extend
into the collar (radius 1.4). All samplers return float64 arrays of shape
[n_points, 2]. The grid sampler is deterministic for reproducible eval L_inf
(grid L_inf is a sup lower bound -- the same convention as the 1D experiments).
"""

from __future__ import annotations

import numpy as np


def disk_uniform(n_points: int, radius: float = 1.0, seed: int = 0) -> np.ndarray:
    """Area-uniform random points in the disk: r = R*sqrt(U), theta = 2*pi*U."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n_points))
    theta = 2.0 * np.pi * rng.random(n_points)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1).astype(np.float64)


def disk_grid(n_side: int, radius: float = 1.0) -> np.ndarray:
    """Cartesian grid over [-R, R]^2 clipped to the disk. Deterministic.

    n_side controls resolution; the returned count is whatever falls inside the
    disk (roughly pi/4 of n_side^2).
    """
    g = np.linspace(-radius, radius, n_side)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= radius ** 2
    return pts[inside].astype(np.float64)
