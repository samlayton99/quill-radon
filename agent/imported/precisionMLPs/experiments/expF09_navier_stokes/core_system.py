"""Multi-field frozen-ridge + collocation-lstsq PDE solver on [-1,1]^2 (expF09).

Scalar primitives vendored verbatim from expF08 core.py (itself expF01's core).
Model per field: u(p) = sum_m c_m tanh(gamma (w_m . p - t_m)) + poly_deg<=3(p),
Radon tensor-ridge geometry. `solve_system` couples several fields (e.g. u,v,p)
through equation blocks in ONE stacked min-norm lstsq. No training anywhere.

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


def rel_l2(u_hat, u_true):
    return float(np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true))


def linf(u_hat, u_true):
    return float(np.max(np.abs(u_hat - u_true)))


# --- multi-field system solve (new for expF09) --------------------------------

def solve_system(fields, equations, W, lam, seed=42, interior_ref=None):
    """Coupled multi-field min-norm collocation solve.

    fields:    list of field names, e.g. ["u", "v", "p"]. Every field shares the
               same frozen Radon geometry + degree-3 poly tail; the unknown is the
               concatenation over fields (length n_fields * (W + len(MONO_2D))).
    equations: list of dicts, each
               {points: [n,2],
                blocks: {field: [((ax,ay), coeff), ...]},   # operator on that field
                rhs: callable(P)->[n] or scalar,
                weight: float = 1.0}
               meaning sum_field blocks[field] applied to field = rhs at points.
               Each equation's rows are scaled to O(1) by their max entry (so the
               gamma^order magnitude spread across momentum/continuity/BC rows does
               not dominate the least squares), then multiplied by weight.
    interior_ref: unused by the solve; accepted so callers can pass their sampling
               for reproducibility/inspection.
    """
    dirs, offs, gamma = radon_geometry(W, lam)
    block_w = len(offs) + len(MONO_2D)
    fidx = {name: k for k, name in enumerate(fields)}
    F = len(fields)
    rows, vals = [], []
    for eq in equations:
        P = np.asarray(eq["points"], dtype=np.float64)
        n = len(P)
        A = np.zeros((n, F * block_w))
        for fname, terms in eq["blocks"].items():
            k = fidx[fname]
            A[:, k * block_w:(k + 1) * block_w] = rows_2d(P, dirs, offs, gamma, terms)
        rhs = eq["rhs"]
        y = rhs(P) if callable(rhs) else np.full(n, float(rhs))
        s = np.abs(A).max()
        s = s if s > 0 else 1.0
        w = float(eq.get("weight", 1.0))
        rows.append(w * A / s)
        vals.append(w * np.asarray(y, dtype=np.float64) / s)
    Amat = np.vstack(rows)
    yvec = np.concatenate(vals)
    sol = np.linalg.lstsq(Amat, yvec, rcond=RCOND)[0]
    return dict(dirs=dirs, offs=offs, gamma=gamma, block_w=block_w,
                fields=list(fields), fidx=fidx, sol=sol, W=len(offs))


def eval_field(model, field, P, terms=(((0, 0), 1.0),), chunk=4096):
    """Evaluate an operator `terms` applied to one field of a system model."""
    k = model["fidx"][field]
    bw = model["block_w"]
    sub = model["sol"][k * bw:(k + 1) * bw]
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        R = rows_2d(P[i:i + chunk], model["dirs"], model["offs"],
                    model["gamma"], list(terms))
        out[i:i + chunk] = R @ sub
    return out
