"""The fixed direction basis and the normalized coordinates every target is written in.

The suite deliberately avoids axis-aligned toy problems by fixing a *deterministic*
orthogonal basis: the DCT-II matrix

    Q[j, 0] = 1/sqrt(d),
    Q[j, k] = sqrt(2/d) * cos(pi * (j + 1/2) * k / d)   for k > 0.

Its columns are the directions ``u_1, ..., u_d`` (1-indexed in the spec, so ``u_k`` is
column ``k - 1``). For ``d = 1`` this degenerates to ``Q = [[1]]``.

Every target is written in the normalized coordinates

    z_k(x) = (u_k . x) / ||u_k||_1,

which run over exactly ``[-1, 1]`` on the cube (both endpoints attained at a corner),
so a length scale written in ``z`` means the same thing in every dimension. The raw
projection ``y_v(x) = v . x`` is kept for the baseline geometry and for the
center-density calculation, which both work in the raw coordinate.

Chain rule, used by every target's analytic gradient: ``dz_k/dx_j = Q[j,k]/||u_k||_1``,
so a gradient in ``z`` becomes a gradient in ``x`` as ``(grad_z / scales) @ Q^T``.
"""

from __future__ import annotations

import functools

import numpy as np

__all__ = ["dct_basis", "u", "y_coord", "z_coord", "l1_scale", "l1_scales",
           "z_of", "grad_z_to_grad_x"]


@functools.lru_cache(maxsize=None)
def dct_basis(d: int) -> np.ndarray:
    """The DCT-II matrix ``Q_d`` (orthogonal, float64). Columns are ``u_1..u_d``."""
    if d < 1:
        raise ValueError("d must be >= 1")
    j = np.arange(d, dtype=np.float64)[:, None]
    k = np.arange(d, dtype=np.float64)[None, :]
    Q = np.sqrt(2.0 / d) * np.cos(np.pi * (j + 0.5) * k / d)
    Q[:, 0] = 1.0 / np.sqrt(d)
    Q.setflags(write=False)
    return Q


def u(d: int, k: int) -> np.ndarray:
    """Direction ``u_k`` (1-indexed) in ``R^d``: column ``k-1`` of ``Q_d``."""
    if not 1 <= k <= d:
        raise ValueError(f"u_{k} does not exist in dimension {d}")
    return np.array(dct_basis(d)[:, k - 1], dtype=np.float64)


def l1_scale(v: np.ndarray) -> float:
    """``||v||_1`` -- the half-range of ``v . x`` over the cube ``[-1,1]^d``."""
    return float(np.abs(np.asarray(v, dtype=np.float64)).sum())


@functools.lru_cache(maxsize=None)
def l1_scales(d: int) -> np.ndarray:
    """``(||u_1||_1, ..., ||u_d||_1)`` -- the divisors that turn ``y`` into ``z``."""
    s = np.abs(dct_basis(d)).sum(axis=0)
    s.setflags(write=False)
    return s


def y_coord(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Raw projection ``v . x`` for ``X`` of shape ``[n, d]``."""
    return np.asarray(X, dtype=np.float64) @ np.asarray(v, dtype=np.float64)


def z_coord(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Normalized coordinate ``(v . x) / ||v||_1``, in ``[-1, 1]`` on the cube."""
    v = np.asarray(v, dtype=np.float64)
    return y_coord(X, v) / l1_scale(v)


def z_of(X: np.ndarray, d: int) -> np.ndarray:
    """All ``d`` normalized coordinates at once: ``z[:, k] = (u_k . x)/||u_k||_1``."""
    return (np.asarray(X, dtype=np.float64) @ dct_basis(d)) / l1_scales(d)


def grad_z_to_grad_x(grad_z: np.ndarray, d: int) -> np.ndarray:
    """Convert a gradient taken in ``z`` into a gradient in ``x`` (both ``[n, d]``)."""
    return (np.asarray(grad_z, dtype=np.float64) / l1_scales(d)) @ dct_basis(d).T
