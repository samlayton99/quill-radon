"""Ethier-Steinman 3D Beltrami flow -- exact unsteady solution of the unforced
incompressible Navier-Stokes equations (Ethier & Steinman, IJNMF 1994). The
standard exact 3D NS verifier (used e.g. by the original PINN papers).

    u = -a [e^{ax} sin(ay+dz) + e^{az} cos(ax+dy)] e^{-nu d^2 t}
    v = -a [e^{ay} sin(az+dx) + e^{ax} cos(ay+dz)] e^{-nu d^2 t}
    w = -a [e^{az} sin(ax+dy) + e^{ay} cos(az+dx)] e^{-nu d^2 t}
    p = -(a^2/2) [e^{2ax} + e^{2ay} + e^{2az}
        + 2 sin(ax+dy) cos(az+dx) e^{a(y+z)}
        + 2 sin(ay+dz) cos(ax+dy) e^{a(z+x)}
        + 2 sin(az+dx) cos(ay+dz) e^{a(x+y)}] e^{-2 nu d^2 t}

Domain used here: (x,y,z) in [-1,1]^3, t in [0,1], a = pi/4, d = pi/2.
Correctness is asserted by tests/test_expF12_tensor_ns3d.py, which pushes these
formulas through torch autograd and checks the NS residual vanishes.
"""
from __future__ import annotations

import numpy as np
import torch

A0 = np.pi / 4.0
D0 = np.pi / 2.0
NU = 1.0
DOMAINS = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]  # x, y, z, t


def fields(X, nu=NU, a=A0, d=D0):
    """X [n,4] (numpy or torch) -> [n,4] stacked (u, v, w, p)."""
    lib = torch if torch.is_tensor(X) else np
    x, y, z, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    e = lib.exp(-nu * d * d * t)
    u = -a * (lib.exp(a * x) * lib.sin(a * y + d * z)
              + lib.exp(a * z) * lib.cos(a * x + d * y)) * e
    v = -a * (lib.exp(a * y) * lib.sin(a * z + d * x)
              + lib.exp(a * x) * lib.cos(a * y + d * z)) * e
    w = -a * (lib.exp(a * z) * lib.sin(a * x + d * y)
              + lib.exp(a * y) * lib.cos(a * z + d * x)) * e
    p = -0.5 * a * a * (
        lib.exp(2 * a * x) + lib.exp(2 * a * y) + lib.exp(2 * a * z)
        + 2 * lib.sin(a * x + d * y) * lib.cos(a * z + d * x) * lib.exp(a * (y + z))
        + 2 * lib.sin(a * y + d * z) * lib.cos(a * x + d * y) * lib.exp(a * (z + x))
        + 2 * lib.sin(a * z + d * x) * lib.cos(a * y + d * z) * lib.exp(a * (x + y))
    ) * e * e
    if lib is torch:
        return torch.stack([u, v, w, p], dim=1)
    return np.stack([u, v, w, p], axis=1)


def grid_targets(grids, nu=NU):
    """Four 1-D axis arrays -> targets [Mx,My,Mz,Mt,4] on the tensor grid."""
    P = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 4)
    Y = fields(P, nu=nu)
    return Y.reshape(*(len(g) for g in grids), 4)
