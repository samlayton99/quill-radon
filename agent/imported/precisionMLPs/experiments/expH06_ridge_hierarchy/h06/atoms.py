"""Finding directions from data: projection pursuit on the residual, then a Gauss-Newton
(variable-projection) polish of chosen directions with the readout eliminated.

Projection pursuit: for a candidate direction ``v`` fit one even QI block along ``v`` to
the residual ``r`` and score the fraction of ``||r||`` it removes; scan a spread pool of
directions, keep the best few, polish each by Gauss-Newton on the direction alone, keep
the winner. For a target that is a sum of ridges this is Friedman-Stuetzle projection
pursuit with an exact 1-D approximator.

Gauss-Newton polish (variable projection, Kaufman's Jacobian): parameters are the
tangent-space coordinates ``delta`` of each polished direction, ``v = normalize(v0 + U delta)``
with ``U`` an orthonormal basis of ``v0``'s orthogonal complement. Residual
``r(delta) = y - A(delta) c(delta)`` with ``c`` the truncated-SVD readout; the Jacobian is
``J_k = -P_perp (dA/ddelta_k) c`` where ``dA/ddelta_k = gamma sech^2(.) (Z U_k)`` on the
block's columns and ``P_perp`` projects off the kept left singular vectors. Damped steps
(halve until the residual drops). Converges quadratically near a zero-residual optimum,
which is what a ridge target with exact directions is.
"""

from __future__ import annotations

import numpy as np

from .core import Geometry, Block, make_block, direction_pool, solve_augmented, band_half_width, RCOND


def tangent_basis(v: np.ndarray) -> np.ndarray:
    """Orthonormal basis (d, d-1) of the orthogonal complement of the unit vector ``v``."""
    U, _, _ = np.linalg.svd(v[:, None], full_matrices=True)
    return U[:, 1:]


def block_columns(b: Block, Z: np.ndarray):
    """``(tanh(.), sech^2(.))`` for one block on the points ``Z``."""
    arg = b.gamma * (Z @ b.v)[:, None] - b.gamma * b.offsets[None, :]
    th = np.tanh(arg)
    return th, 1.0 - th * th


# ---------------------------------------------------------------------------
# one-direction score and projection pursuit
# ---------------------------------------------------------------------------

def _one_dir_fit(v, Z, r, n_off, rcond=RCOND):
    b = make_block(v, Z, n_off, kind="atom")
    th, _ = block_columns(b, Z)
    A = np.hstack([th, np.ones((len(Z), 1))])
    fit = solve_augmented(A, r, rcond=rcond)
    res = r - A @ fit.coef
    return b, float(np.linalg.norm(res) / np.linalg.norm(r))


def projection_pursuit(Z, r, n_off=32, pool_size=None, n_keep=3, gn_iters=12, rcond=RCOND, seed=0):
    """Best single ridge direction for the residual ``r`` on the centered points ``Z``.

    Returns ``(v, score_coarse, score_polished)`` with score = remaining fraction of ||r||.
    """
    d = Z.shape[1]
    if pool_size is None:
        pool_size = {2: 180, 3: 600, 4: 1500}.get(d, 3000)
    P = direction_pool(d, pool_size, seed=seed)
    scores = np.array([_one_dir_fit(v, Z, r, n_off, rcond)[1] for v in P])
    order = np.argsort(scores)[:n_keep]
    best = (None, np.inf, np.inf)
    for j in order:
        g = Geometry([make_block(P[j], Z, n_off, kind="atom")])
        g, hist = varpro_polish(g, Z, r, which=[0], iters=gn_iters, rcond=rcond)
        s_pol = hist[-1]["rel_residual"]
        if s_pol < best[2]:
            best = (g.blocks[0].v.copy(), float(scores[j]), float(s_pol))
    return best


# ---------------------------------------------------------------------------
# Gauss-Newton polish of directions (variable projection)
# ---------------------------------------------------------------------------

def _rebuild(geom: Geometry, Z: np.ndarray, which, new_vs) -> Geometry:
    g = geom.copy()
    for bi, v in zip(which, new_vs):
        b = g.blocks[bi]
        v = v / np.linalg.norm(v)
        g.blocks[bi] = Block(v=v, n=b.n, T=band_half_width(v, Z), kind=b.kind)
    return g


def _residual(geom, Z, y, rcond, keep_U):
    """Fit and residual; the feature matrix is not kept (the polish never needs it again)."""
    A = geom.augmented(Z)
    if A.shape[1] > 6000:                   # wide: factorize in place, keep the residual only
        r_pred = None
        fit = solve_augmented(A, y, rcond=rcond, keep_U=keep_U, overwrite_a=True)
        A = geom.augmented(Z)               # rebuilt once for the residual (cheap vs the solve)
        r = y - A @ fit.coef
        del A
        return None, fit, r
    fit = solve_augmented(A, y, rcond=rcond, keep_U=keep_U)
    r = y - A @ fit.coef
    del A
    return None, fit, r


def varpro_polish(geom: Geometry, Z: np.ndarray, y: np.ndarray, which, iters=10, rcond=RCOND,
                  tol=1e-3, max_halvings=6, verbose=False):
    """Gauss-Newton on the directions of the blocks listed in ``which``; the readout is
    re-solved at every trial. Returns ``(geometry, history)``; history rows carry the
    relative training residual and the step size taken."""
    y = np.asarray(y, dtype=np.float64).ravel()
    ynorm = np.linalg.norm(y)
    A, fit, r = _residual(geom, Z, y, rcond, keep_U=True)
    hist = [{"iter": 0, "rel_residual": float(np.linalg.norm(r) / ynorm), "step": 0.0}]
    slices = geom.block_slices()
    for it in range(1, iters + 1):
        cols = []
        bases = []
        for bi in which:
            b = geom.blocks[bi]
            Ub = tangent_basis(b.v)
            bases.append(Ub)
            _, sech2 = block_columns(b, Z)
            cb = fit.coef[slices[bi]]
            w = b.gamma * (sech2 @ cb)                 # (n,) = sum_j c_j gamma sech^2_j
            ZU = Z @ Ub                                # (n, d-1)
            cols.append(-w[:, None] * ZU)
        J = np.hstack(cols)
        J = J - fit.U @ (fit.U.T @ J)                  # Kaufman: project off range(A)
        step = -np.linalg.lstsq(J, r, rcond=None)[0]
        base = float(np.linalg.norm(r))
        scale, accepted = 1.0, False
        for _ in range(max_halvings + 1):
            new_vs, k0 = [], 0
            for bi, Ub in zip(which, bases):
                p = Ub.shape[1]
                new_vs.append(geom.blocks[bi].v + Ub @ (scale * step[k0:k0 + p]))
                k0 += p
            trial = _rebuild(geom, Z, which, new_vs)
            A2, fit2, r2 = _residual(trial, Z, y, rcond, keep_U=True)
            if np.linalg.norm(r2) < base:
                geom, A, fit, r = trial, A2, fit2, r2
                accepted = True
                break
            scale *= 0.5
        rel = float(np.linalg.norm(r) / ynorm)
        hist.append({"iter": it, "rel_residual": rel, "step": float(scale * np.linalg.norm(step)) if accepted else 0.0})
        if verbose:
            print(f"    GN {it:2d} rel_res={rel:.2e} step={hist[-1]['step']:.2e} accepted={accepted}", flush=True)
        if not accepted or (base - np.linalg.norm(r)) < tol * base:
            break
    return geom, hist
