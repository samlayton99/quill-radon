"""2D hexagonal tangent-line geometry for the ridge / Radon regime.

Extends the 1D tanh-MLP construction to R^2 -> R. Each neuron is a ridge
(plane wave): tanh(gamma * signed-distance(x, line_m)), where line_m is the
line tangent to a circle at a hexagonally-packed grid point. Concretely, for a
grid point c_m with radius r_m = ||c_m|| and outward unit normal n_m = c_m/r_m:

    W_m   = gamma * n_m          (first-layer row; ||W_m|| = gamma)
    beta_m = -gamma * r_m        (bias)
    preactivation = gamma * (n_m . x - r_m) = gamma * dist(x, tangent line).

This is the 2D analog of the 1D center x_m: a signed scalar position becomes a
(direction, offset) pair (n_m, r_m) -- a point in the Radon/line domain.

Bandwidth is dual. With hex spacing d = d(N), the dimensionless bandwidth is
lambda = gamma * d (the 1D analog of lambda = gamma * h). Specify EITHER
``lambda_star`` (default usage) -> gamma = lambda/d, OR ``gamma`` -> lambda =
gamma*d; the other is backed out from the spacing.

All arithmetic is float64. The readout is solved separately (see
``src/construction/readout.py``); this module only builds the frozen geometry
and the feature matrix.
"""

from __future__ import annotations

import numpy as np

# Axial hex directions, ordered for ring traversal (redblobgames convention).
# Nearest-neighbour steps on the triangular lattice in axial (q, r) coords.
_HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def centered_hex_number(K: int) -> int:
    """Total points in K complete centered-hexagonal rings: H_K = 1+3K(K+1).

    H_0=1, H_1=7, H_2=19, H_3=37, H_4=61, ...
    """
    return 1 + 3 * K * (K + 1)


def _largest_full_rings(N: int) -> int:
    """Largest K with centered_hex_number(K) <= N."""
    K = 0
    while centered_hex_number(K + 1) <= N:
        K += 1
    return K


def _axial_to_cart(q: int, r: int) -> tuple[float, float]:
    """Axial (q, r) -> cartesian on a unit-spacing triangular lattice."""
    return (q + r / 2.0, r * np.sqrt(3.0) / 2.0)


def _ring_axial(k: int) -> list[tuple[int, int]]:
    """The 6k axial coordinates of centered-hex ring k (k >= 1)."""
    if k == 0:
        return [(0, 0)]
    pts = []
    # Start at a corner k steps along direction[4], then walk the 6 edges.
    q, r = _HEX_DIRS[4][0] * k, _HEX_DIRS[4][1] * k
    for i in range(6):
        dq, dr = _HEX_DIRS[i]
        for _ in range(k):
            pts.append((q, r))
            q, r = q + dq, r + dr
    return pts


def hex_pack(N: int, radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Pack N points into a disk of the given radius, hexagonally, centered.

    Fills the largest number of complete centered-hex rings that fit within N,
    then distributes any remaining points uniformly in angle on a partial outer
    ring. The spacing is set so the outermost ring (full or partial) lands at
    ``radius``, so the packing grows denser/more compact as N increases.

    Returns (centers[N, 2], radii[N], spacing_d) where spacing_d is the
    nearest-neighbour lattice distance (used for lambda = gamma * spacing_d).
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    if N == 1:
        centers = np.zeros((1, 2), dtype=np.float64)
        return centers, np.zeros(1, dtype=np.float64), float(radius)

    K = _largest_full_rings(N)             # complete rings 0..K
    p = N - centered_hex_number(K)         # leftover for the partial ring K+1

    # Spacing: outermost occupied ring index reaches `radius`.
    outer_ring = K if p == 0 else K + 1
    d = float(radius) / outer_ring

    pts = [(0.0, 0.0)]                      # ring 0 (center)
    for k in range(1, K + 1):
        for (q, r) in _ring_axial(k):
            cx, cy = _axial_to_cart(q, r)
            pts.append((d * cx, d * cy))

    if p > 0:
        # Partial outer ring: p points uniform in angle at radius = `radius`.
        for j in range(p):
            theta = 2.0 * np.pi * j / p
            pts.append((radius * np.cos(theta), radius * np.sin(theta)))

    centers = np.asarray(pts, dtype=np.float64)
    radii = np.linalg.norm(centers, axis=1)
    assert centers.shape == (N, 2), (centers.shape, N)
    return centers, radii, d


def tangent_geometry(
    centers: np.ndarray,
    radii: np.ndarray,
    spacing_d: float,
    *,
    lambda_star: float | None = None,
    gamma: float | None = None,
    center_dir: tuple[float, float] = (1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Map packed grid points to first-layer weights and biases.

    Each neuron's transition line is tangent to the circle of radius r_m at
    c_m: W_m = gamma * (c_m / r_m), beta_m = -gamma * r_m.

    Specify exactly one of ``lambda_star`` or ``gamma``; the other is backed out
    via lambda = gamma * spacing_d. The r=0 center point (undefined normal) gets
    the arbitrary direction ``center_dir`` and beta = 0 (a ridge through 0).

    Returns (W[N, 2], beta[N], gamma, lambda_val).
    """
    if (lambda_star is None) == (gamma is None):
        raise ValueError("specify exactly one of lambda_star or gamma")
    if lambda_star is not None:
        gamma = float(lambda_star) / spacing_d
        lambda_val = float(lambda_star)
    else:
        gamma = float(gamma)
        lambda_val = gamma * spacing_d

    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    N = centers.shape[0]

    nhat = np.empty((N, 2), dtype=np.float64)
    interior = radii > 0.0
    nhat[interior] = centers[interior] / radii[interior, None]
    cd = np.asarray(center_dir, dtype=np.float64)
    nhat[~interior] = cd / np.linalg.norm(cd)

    W = gamma * nhat                        # rows have norm exactly gamma
    beta = -gamma * radii                   # center (r=0) -> beta=0
    return W, beta, gamma, lambda_val


def build_hex_geometry(
    N: int,
    *,
    radius: float = 1.4,
    lambda_star: float | None = None,
    gamma: float | None = None,
) -> dict:
    """Convenience wrapper: pack N points and build the tangent-line geometry.

    Returns a dict with W, beta, centers, radii, spacing_d, gamma, lambda_val.
    """
    centers, radii, d = hex_pack(N, radius)
    W, beta, gamma, lambda_val = tangent_geometry(
        centers, radii, d, lambda_star=lambda_star, gamma=gamma
    )
    return {
        "W": W, "beta": beta, "centers": centers, "radii": radii,
        "spacing_d": d, "gamma": gamma, "lambda_val": lambda_val,
    }


def build_phi_2d(X: np.ndarray, W: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Feature matrix Phi[i, m] = tanh(W_m . x_i + beta_m).

    X: [n_points, 2], W: [N, 2], beta: [N]. Returns [n_points, N].
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64).ravel()
    return np.tanh(X @ W.T + beta[None, :])
