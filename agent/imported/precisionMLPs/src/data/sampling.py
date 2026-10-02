"""Sampling functions for training and evaluation points.

All return [n_points, 1] float64 torch tensors, sorted.
"""

from __future__ import annotations

import torch


def equispaced(n_points: int, domain: tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """linspace(a, b, n) including endpoints."""
    return torch.linspace(domain[0], domain[1], n_points, dtype=torch.float64).unsqueeze(1)


def uniform_random(n_points: int, domain: tuple[float, float] = (-1.0, 1.0),
                   seed: int = 0) -> torch.Tensor:
    """Uniform random in [a, b] with endpoints always included, sorted."""
    gen = torch.Generator().manual_seed(seed)
    interior = torch.rand(n_points - 2, generator=gen, dtype=torch.float64) * (domain[1] - domain[0]) + domain[0]
    pts = torch.cat([
        torch.tensor([domain[0]], dtype=torch.float64),
        interior,
        torch.tensor([domain[1]], dtype=torch.float64),
    ])
    pts, _ = pts.sort()
    return pts.unsqueeze(1)


def chebyshev(n_points: int, domain: tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """Chebyshev nodes cos(pi*k/(n-1)) mapped to [a, b], sorted."""
    import math
    k = torch.arange(n_points, dtype=torch.float64)
    nodes = torch.cos(math.pi * k / (n_points - 1))
    # Map from [-1, 1] to [a, b]
    a, b = domain
    mapped = 0.5 * (a + b) + 0.5 * (b - a) * nodes
    mapped, _ = mapped.sort()
    return mapped.unsqueeze(1)


def qi_grid(N: int, domain: tuple[float, float] = (-1.0, 1.0),
            halo: int = 0) -> torch.Tensor:
    """QI grid x_k = a + k*h for k in range(-halo, N + halo + 1)."""
    a, b = domain
    h = (b - a) / N
    k = torch.arange(-halo, N + halo + 1, dtype=torch.float64)
    pts = a + k * h
    return pts.unsqueeze(1)


def get_sampling_fn(name: str):
    """Look up sampling function by name: "equispaced" | "uniform" | "chebyshev" | "qi_grid"."""
    fns = {"equispaced": equispaced, "uniform": uniform_random,
           "chebyshev": chebyshev, "qi_grid": qi_grid}
    return fns[name]
