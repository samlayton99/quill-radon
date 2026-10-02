"""Geometry, solver, nested directions and targets shared by every expH06 mode.

Coordinates: everything is centered on the data's origin ``x0`` (``z = x - x0``). A block
is one direction ``v`` (unit) with ``n`` evenly spaced offsets ``t_j`` on the projected
data's band ``[-T, T]`` (``T = margin * max_i |v.z_i|``, the 25% collar) and the width
``gamma = lam / h`` from the spacing ``h = 2T/n`` -- the expH02 rule ``gamma h = 0.25``.
Feature: ``tanh(gamma (v.z - t_j))``. Even spacing within a block means the mesh-map
smoothness constraint (no content below ~12 gaps) is satisfied trivially; refinement
of a block replaces its grid by a finer even grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LAMBDA = 0.25
MARGIN = 1.25
RCOND = 1e-13


# ---------------------------------------------------------------------------
# blocks and features
# ---------------------------------------------------------------------------

@dataclass
class Block:
    v: np.ndarray          # unit direction, shape (d,)
    n: int                 # number of offsets
    T: float               # half band width along v (about the origin)
    kind: str = "bg"       # "bg" (nested background) or "atom" (found from the residual)

    @property
    def h(self) -> float:
        return 2.0 * self.T / self.n

    @property
    def gamma(self) -> float:
        return LAMBDA / self.h

    @property
    def offsets(self) -> np.ndarray:
        return -self.T + (np.arange(self.n) + 0.5) * self.h


def band_half_width(v: np.ndarray, Z: np.ndarray, margin: float = MARGIN) -> float:
    return float(margin * np.max(np.abs(Z @ v)))


def make_block(v, Z, n, kind="bg", margin=MARGIN) -> Block:
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    return Block(v=v, n=int(n), T=band_half_width(v, Z, margin), kind=kind)


@dataclass
class Geometry:
    blocks: list[Block] = field(default_factory=list)

    @property
    def units(self) -> int:
        return int(sum(b.n for b in self.blocks))

    @property
    def n_dir(self) -> int:
        return len(self.blocks)

    def copy(self) -> "Geometry":
        import copy as _copy
        return Geometry([_copy.deepcopy(b) for b in self.blocks])

    def flat(self):
        """Per-unit arrays: directions (units, d), offsets (units,), gammas (units,)."""
        D = np.vstack([np.repeat(b.v[None, :], b.n, axis=0) for b in self.blocks])
        t = np.concatenate([np.asarray(b.offsets) for b in self.blocks])
        g = np.concatenate([(np.asarray(b.gammas) if hasattr(b, "gammas") else np.full(b.n, b.gamma))
                            for b in self.blocks])
        return D, t, g

    def block_slices(self):
        out, i0 = [], 0
        for b in self.blocks:
            out.append(slice(i0, i0 + b.n))
            i0 += b.n
        return out

    def augmented(self, Z: np.ndarray, block_rows: int = 8192) -> np.ndarray:
        """``[Phi, 1]`` for centered points ``Z``, built row-block by row-block."""
        D, t, g = self.flat()
        m = len(t)
        A = np.empty((len(Z), m + 1), dtype=np.float64)
        Dt, tt, gg = D.T, t[None, :], g[None, :]
        for i0 in range(0, len(Z), block_rows):
            sl = slice(i0, min(i0 + block_rows, len(Z)))
            A[sl, :m] = np.tanh(gg * (Z[sl] @ Dt - tt))
        A[:, m] = 1.0
        return A

    def describe(self) -> dict:
        return {"n_dir": self.n_dir, "units": self.units,
                "n_per": [int(b.n) for b in self.blocks],
                "kinds": [b.kind for b in self.blocks],
                "n_atoms": int(sum(b.kind == "atom" for b in self.blocks))}


# ---------------------------------------------------------------------------
# the solve
# ---------------------------------------------------------------------------

@dataclass
class Fit:
    coef: np.ndarray       # (units + 1, k): readout weights, bias last
    U: np.ndarray          # left singular vectors kept (n, rank) -- for projections
    rank: int
    n_cols: int

    def predict(self, geom: Geometry, Z: np.ndarray) -> np.ndarray:
        return geom.augmented(Z) @ self.coef


QR_ABOVE_COLS = 6000     # columns beyond which the solve goes through a Householder QR first


def solve_augmented(A: np.ndarray, Y: np.ndarray, rcond: float = RCOND, keep_U: bool = False,
                    via_qr: bool | None = None, overwrite_a: bool = False) -> Fit:
    """Truncated-SVD least squares (bias column last); several right-hand sides at once.

    Two routes with the same truncation and the same numbers to rounding: the direct SVD of
    ``A``, or (for wide matrices) a Householder QR of ``A`` followed by the SVD of the small
    ``R``: ``A = Q R``, ``R = U_r s V^T``, ``A^+ = V s^+ U_r^T Q^T``. The QR route holds one
    ``n x m`` array plus ``m x m`` factors instead of ``A``, ``U`` and the SVD workspace.
    """
    Y = np.asarray(Y, dtype=np.float64)
    squeeze = Y.ndim == 1
    if squeeze:
        Y = Y[:, None]
    if via_qr is None:
        via_qr = A.shape[1] > QR_ABOVE_COLS
    if via_qr:
        import scipy.linalg as sla
        Q, Rm = sla.qr(A, mode="economic", overwrite_a=overwrite_a, check_finite=False)
        Ur, s, Vt = np.linalg.svd(Rm, full_matrices=False)
        del Rm
        keep = s > rcond * s[0]
        s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
        coef = Vt.T @ (s_inv[:, None] * (Ur.T @ (Q.T @ Y)))
        if keep_U:
            U = Q @ Ur[:, keep]
            del Q
        else:
            U = None
    else:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        keep = s > rcond * s[0]
        s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
        coef = Vt.T @ (s_inv[:, None] * (U.T @ Y))
        U = U[:, keep] if keep_U else None
    if squeeze:
        coef = coef[:, 0]
    return Fit(coef=coef, U=U, rank=int(keep.sum()), n_cols=A.shape[1])


def fit_geometry(geom: Geometry, Z: np.ndarray, Y: np.ndarray, rcond: float = RCOND, keep_U: bool = False) -> Fit:
    A = geom.augmented(Z)
    return solve_augmented(A, Y, rcond=rcond, keep_U=keep_U)


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def max_abs(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(np.abs(pred - truth)))


# ---------------------------------------------------------------------------
# nested directions: a projective farthest-point sequence from a spread pool
# ---------------------------------------------------------------------------

def canonical_sign(V: np.ndarray) -> np.ndarray:
    """Flip each row so its last non-zero coordinate is positive (v and -v identified)."""
    V = np.array(V, dtype=np.float64)
    for i, v in enumerate(V):
        nz = np.nonzero(np.abs(v) > 1e-14)[0]
        if len(nz) and v[nz[-1]] < 0:
            V[i] = -v
    return V


def direction_pool(d: int, n: int, seed: int = 0) -> np.ndarray:
    """A spread pool of candidate directions on the half sphere: spherical Fibonacci in
    d = 3, angles in d = 2, normalized Gaussians otherwise."""
    if d == 2:
        th = (np.arange(n) + 0.5) * np.pi / n
        return np.stack([np.cos(th), np.sin(th)], axis=1)
    if d == 3:
        i = np.arange(n, dtype=np.float64) + 0.5
        z = i / n
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        return canonical_sign(np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1))
    rng = np.random.default_rng(seed + 7919 * d)
    G = rng.normal(size=(n, d))
    return canonical_sign(G / np.linalg.norm(G, axis=1, keepdims=True))


def nested_directions(d: int, n_max: int, pool_size: int | None = None, seed: int = 0) -> np.ndarray:
    """The first ``n_max`` directions of a projective farthest-point sequence: every
    prefix is an evenly spread set, and every prefix is contained in the next one.

    Projective distance ``1 - |u.v|``; the sequence starts at the pool point closest to
    the first coordinate axis, then repeatedly adds the pool point farthest from all
    chosen ones.
    """
    if pool_size is None:
        pool_size = max(4096, 64 * n_max)
    P = direction_pool(d, pool_size, seed=seed)
    start = int(np.argmax(np.abs(P[:, 0])))
    chosen = [start]
    dist = 1.0 - np.abs(P @ P[start])
    for _ in range(1, n_max):
        j = int(np.argmax(dist))
        chosen.append(j)
        dist = np.minimum(dist, 1.0 - np.abs(P @ P[j]))
    return P[chosen]


def fibonacci_directions(d: int, n: int) -> np.ndarray:
    """The expH01 even directions (spherical Fibonacci in d = 3) -- the reference set."""
    return direction_pool(d, n)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def ball(n: int, d: int, r: float, rng: np.random.Generator) -> np.ndarray:
    """``n`` points uniform in the ball of radius ``r`` about the origin (centered coordinates)."""
    g = rng.normal(size=(n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    u = rng.uniform(size=(n, 1)) ** (1.0 / d)
    return r * u * g


X0 = {2: np.array([0.35, -0.25]),
      3: np.array([0.35, -0.25, 0.2]),
      4: np.array([0.35, -0.25, 0.2, -0.1]),
      5: np.array([0.35, -0.25, 0.2, -0.1, 0.15])}


def origin(d: int) -> np.ndarray:
    return X0[d].copy()
