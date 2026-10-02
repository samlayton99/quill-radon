"""Spikes: non-uniform (graded) ridge blocks whose offset density follows the residual.

A spike is one direction ``v`` carrying ``n`` offsets placed by the inverse CDF of a
density built from the residual's energy profile along ``t = v.z``: dense where the
residual lives, tapering off where it is quiet, with the expH02/expH04 safety rules --
the density is smoothed so the resulting mesh has no content below ~12 gaps in
wavelength, neighbor gaps change gradually, a floor keeps the density strictly positive
on the band, and every offset gets its local width ``gamma_j = 0.25 / h_j``.

This is the ``expH04`` monitor->density->mesh pipeline reduced to one block, aimed by the
current residual instead of a target monitor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import Geometry, band_half_width, solve_augmented, RCOND, LAMBDA, direction_pool

GRADE = 0.15                 # cap on the relative change of adjacent gaps
MESH_WAVELENGTH = 12.0       # no mesh content below this many gaps (expH04 limit)
FLOOR_S = 0.5                # density = (1-s) * even + s * residual-profile


@dataclass
class NUBlock:
    """A block with per-offset spacing: offsets t_j and widths gamma_j = 0.25 / h_j."""
    v: np.ndarray
    offsets: np.ndarray
    gammas: np.ndarray
    T: float
    kind: str = "spike"

    @property
    def n(self) -> int:
        return len(self.offsets)


def local_gaps(t: np.ndarray) -> np.ndarray:
    """Per-offset local spacing: mean of the two adjacent gaps (one-sided at the ends)."""
    d = np.diff(t)
    h = np.empty_like(t)
    h[0], h[-1] = d[0], d[-1]
    h[1:-1] = 0.5 * (d[:-1] + d[1:])
    return h


def graded_density(t_grid, energy, n, floor_s=FLOOR_S):
    """Density on the band: even floor plus the (smoothed, normalized) energy profile.

    Smoothing: Gaussian with sigma chosen so, at the target count ``n``, the density has
    no content below MESH_WAVELENGTH gaps (sigma in t-units = wavelength * mean-gap *
    sqrt(ln 100 / (2 pi^2)), the expH04 derivation)."""
    T = t_grid[-1]
    e = np.maximum(np.asarray(energy, dtype=np.float64), 0.0)
    mean_gap = 2.0 * T / n
    sigma = MESH_WAVELENGTH * mean_gap * np.sqrt(np.log(100.0) / (2 * np.pi ** 2))
    dx = t_grid[1] - t_grid[0]
    k = int(np.ceil(4 * sigma / dx))
    if k > 0:
        w = np.exp(-0.5 * (np.arange(-k, k + 1) * dx / sigma) ** 2)
        pad = np.pad(e, k, mode="edge")
        e = np.convolve(pad, w / w.sum(), mode="valid")
    if e.sum() * dx <= 0:
        e = np.ones_like(e)
    rho = (1.0 - floor_s) / (2 * T) + floor_s * e / (e.sum() * dx)
    return rho


def grade(h_of_t, t_grid, cap=GRADE):
    """Cap |dh/dt| at ``cap`` by two monotone passes (the expH04 grading)."""
    h = h_of_t.copy()
    dx = t_grid[1] - t_grid[0]
    for _ in range(2):
        for i in range(1, len(h)):
            h[i] = min(h[i], h[i - 1] + cap * dx)
        for i in range(len(h) - 2, -1, -1):
            h[i] = min(h[i], h[i + 1] + cap * dx)
    return h


def place_offsets(t_grid, rho, n):
    """``n`` offsets by the inverse CDF of the graded density."""
    h = grade(1.0 / np.maximum(rho, 1e-12), t_grid)
    rho_g = 1.0 / h
    cdf = np.cumsum(rho_g)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    q = (np.arange(n) + 0.5) / n
    return np.interp(q, cdf, t_grid)


def make_spike(v, Z, resid, n, margin=1.25, n_bins=201, floor_s=FLOOR_S):
    """One graded block along ``v``: offset density from the residual energy profile."""
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    T = band_half_width(v, Z, margin)
    t = Z @ v
    bins = np.linspace(-T, T, n_bins + 1)
    idx = np.clip(np.digitize(t, bins) - 1, 0, n_bins - 1)
    energy = np.bincount(idx, weights=resid * resid, minlength=n_bins)
    counts = np.maximum(np.bincount(idx, minlength=n_bins), 1)
    t_grid = 0.5 * (bins[:-1] + bins[1:])
    rho = graded_density(t_grid, energy / counts, n, floor_s)
    offsets = place_offsets(t_grid, rho, n)
    gammas = LAMBDA / local_gaps(offsets)
    return NUBlock(v=v, offsets=offsets, gammas=gammas, T=T)


def spike_score(block: NUBlock, Z, resid, rcond=RCOND):
    """Fraction of the residual left after fitting this one block (plus a bias)."""
    th = np.tanh(block.gammas[None, :] * ((Z @ block.v)[:, None] - block.offsets[None, :]))
    A = np.hstack([th, np.ones((len(Z), 1))])
    fit = solve_augmented(A, resid, rcond=rcond)
    return float(np.linalg.norm(resid - A @ fit.coef) / np.linalg.norm(resid))


def best_spikes(Z, resid, n, k=1, pool_size=None, rcond=RCOND, seed=0, floor_s=FLOOR_S,
                min_sep_cos=0.95):
    """Scan a spread pool of directions with graded blocks; return the ``k`` best spikes
    with pairwise |cos| below ``min_sep_cos`` (so a batch is not one direction k times)."""
    d = Z.shape[1]
    if pool_size is None:
        pool_size = {2: 180, 3: 400, 4: 1000}.get(d, 2000)
    P = direction_pool(d, pool_size, seed=seed)
    sub = slice(None) if len(Z) <= 6000 else np.random.default_rng(seed).choice(len(Z), 6000, replace=False)
    Zs, rs = Z[sub], resid[sub]
    scores = []
    for v in P:
        blk = make_spike(v, Zs, rs, min(n, 24), floor_s=floor_s)
        scores.append(spike_score(blk, Zs, rs, rcond))
    picked = []
    for j in np.argsort(scores):
        v = P[j]
        if all(abs(float(v @ u)) < min_sep_cos for u in picked):
            picked.append(v)
        if len(picked) == k:
            break
    return [make_spike(v, Z, resid, n, floor_s=floor_s) for v in picked], [float(min(scores))]


def best_spike(Z, resid, n, **kw):
    blks, scs = best_spikes(Z, resid, n, k=1, **kw)
    return blks[0], scs[0]
