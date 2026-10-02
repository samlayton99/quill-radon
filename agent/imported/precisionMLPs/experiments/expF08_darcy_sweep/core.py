"""Frozen-ridge + collocation-lstsq PDE solver on [-1,1]^2.

Vendored from precision-mlps expF01 (experiments/expF01_linear_de_zoo/run.py
@ e4806ce). Model: u(p) = sum_m c_m tanh(gamma (w_m . p - t_m)) + poly_deg<=3(p),
Radon tensor-ridge geometry (sqrt(W) directions x sqrt(W) offsets), all
coefficients from ONE min-norm lstsq over stacked PDE + BC collocation rows.
No training anywhere. Every tanh derivative is a closed-form polynomial in tanh.

An operator is a list of terms [((ax, ay), coeff)] meaning
L = sum coeff * d^ax_x d^ay_y, with coeff a float or callable(P[n,2]) -> [n].
"""
from __future__ import annotations

import numpy as np

RCOND = 1e-13
COLLAR_SQUARE = 1.6

MONO_2D = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
           (3, 0), (2, 1), (1, 2), (0, 3)]


def psi(order, t):
    """d^order/dz^order tanh(z) as a polynomial in t = tanh(z)."""
    if order == 0:
        return t
    if order == 1:
        return 1.0 - t * t
    if order == 2:
        return -2.0 * t * (1.0 - t * t)
    if order == 3:
        s = 1.0 - t * t
        return -2.0 * s * (1.0 - 3.0 * t * t)
    raise ValueError(order)


def _ffact(d, o):
    """Falling factorial d*(d-1)*...*(d-o+1); derivative of x^d is ffact*x^(d-o)."""
    out = 1
    for k in range(o):
        out *= (d - k)
    return out


def _coeff_col(coeff, pts):
    """Evaluate a term coefficient at points -> column vector or scalar."""
    if callable(coeff):
        return np.asarray(coeff(pts), dtype=np.float64).reshape(-1, 1)
    return float(coeff)


def pi_thetas(J):
    return np.pi * np.arange(J) / J


def radon_geometry(W, lam, collar=COLLAR_SQUARE):
    J = int(round(np.sqrt(W)))
    M = W // J
    thetas = pi_thetas(J)
    ts = np.linspace(-collar, collar, M)
    dirs = np.repeat(np.stack([np.cos(thetas), np.sin(thetas)], axis=1), M, axis=0)
    offs = np.tile(ts, J)
    h_ref = 2.8 / np.sqrt(J * M)
    return dirs, offs, lam / h_ref


def rows_2d(P, dirs, offs, gamma, terms):
    """[L Phi | L poly] rows at points P for terms [((ax, ay), coeff)]."""
    t = np.tanh(gamma * (P @ dirs.T - offs[None, :]))
    A = np.zeros_like(t)
    polys = np.zeros((len(P), len(MONO_2D)))
    x, y = P[:, 0], P[:, 1]
    for (ax, ay), coeff in terms:
        o = ax + ay
        cc = _coeff_col(coeff, P)
        dir_fac = (dirs[:, 0] ** ax * dirs[:, 1] ** ay)[None, :]
        A += cc * (gamma ** o) * dir_fac * psi(o, t)
        ccr = cc.ravel() if np.ndim(cc) else cc
        for k, (px, py) in enumerate(MONO_2D):
            if ax <= px and ay <= py:
                mono = (_ffact(px, ax) * _ffact(py, ay)
                        * x ** (px - ax) * y ** (py - ay))
                polys[:, k] += ccr * mono
    return np.hstack([A, polys])


def boundary_points_square(n_per_edge=480):
    s = np.linspace(-1.0, 1.0, n_per_edge)
    edges = [np.stack([s, np.full_like(s, -1.0)], axis=1),
             np.stack([s, np.full_like(s, 1.0)], axis=1),
             np.stack([np.full_like(s, -1.0), s], axis=1),
             np.stack([np.full_like(s, 1.0), s], axis=1)]
    return np.concatenate(edges, axis=0)


def interior_points_square(W, rng):
    n = max(5 * W, 2000)
    return rng.uniform(-1.0, 1.0, (n, 2))


def solve_square(terms, forcing, bc_blocks, W, lam, seed=42):
    """One stacked min-norm lstsq: PDE rows (scaled to O(1) by their max entry)
    + BC blocks (each weighted sqrt(n_pde/n_block)).

    bc_blocks: [dict(points=[n,2], terms=[...], values=[n])].
    """
    rng = np.random.default_rng(seed)
    dirs, offs, gamma = radon_geometry(W, lam)
    P = interior_points_square(W, rng)
    A_pde = rows_2d(P, dirs, offs, gamma, terms)
    y_pde = forcing(P) if callable(forcing) else np.full(len(P), float(forcing))
    s = np.abs(A_pde).max()
    rows, vals = [A_pde / s], [y_pde / s]
    for blk in bc_blocks:
        Pb = np.asarray(blk["points"], dtype=np.float64)
        r = rows_2d(Pb, dirs, offs, gamma, blk["terms"])
        w = np.sqrt(len(P) / len(Pb))
        rows.append(w * r)
        vals.append(w * np.asarray(blk["values"], dtype=np.float64))
    A = np.vstack(rows)
    y = np.concatenate(vals)
    sol = np.linalg.lstsq(A, y, rcond=RCOND)[0]
    return dict(dirs=dirs, offs=offs, gamma=gamma, sol=sol, W=len(offs))


def eval_model(model, P, terms=(((0, 0), 1.0),), chunk=4096):
    """Evaluate L[u_hat] at P; default L = identity (u_hat itself)."""
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        R = rows_2d(P[i:i + chunk], model["dirs"], model["offs"],
                    model["gamma"], list(terms))
        out[i:i + chunk] = R @ model["sol"]
    return out


def rel_l2(u_hat, u_true):
    return float(np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true))


def linf(u_hat, u_true):
    return float(np.max(np.abs(u_hat - u_true)))
