"""expF17 1-D solve engine (dysts arm). Ports the expF14 pattern -- per-component
RMS scaling sigma, weighted IC rows, block Gauss-Newton over components with a
damped backtracking line search, warm init from a cheap fp64 RK solve (an
initializer, never an oracle) -- onto the W-exact poly-free 1-D dictionaries.
rcond = 1e-15 (SPEC 13.7; expF14's own default was 1e-13, noted)."""
from __future__ import annotations

import time

import numpy as np

from . import dicts1d as f1
from .solve import RCOND, OVERSAMPLE
from .tasks1d import dysts_task

MAX_NEWTON = 30
WARM_RTOL, WARM_ATOL = 1e-6, 1e-9


def lstsq(A, y):
    return np.linalg.lstsq(A, y, rcond=RCOND)[0]


def _sample_nodes(n_nodes, n_want, rng, chebyshev):
    """Indices into a uniform node grid. Polynomial-spectral methods draw with
    Chebyshev (arcsine) density -- their literature-standard placement; uniform
    density at high degree is the Platte-Trefethen-Kuijlaars unstable regime
    and measures sampling incompatibility, not the method. Others draw uniform."""
    if not chebyshev:
        return rng.choice(n_nodes, size=min(n_want, n_nodes), replace=False)
    su = np.cos(np.pi * rng.uniform(0.0, 1.0, 2 * n_want))    # arcsine on [-1,1]
    idx = np.unique(np.round((su + 1.0) * 0.5 * (n_nodes - 1)).astype(int))
    rng.shuffle(idx)
    idx = idx[:n_want]
    if len(idx) < n_want:
        rest = np.setdiff1d(np.arange(n_nodes), idx)
        idx = np.concatenate([idx, rng.choice(rest, n_want - len(idx),
                                              replace=False)])
    return idx


CHEB_SAMPLED = ("spectral", "bwler")


def oracle_fit_1d(task, d, seed):
    """Fit every component of the reference trajectory: one factorization,
    d right-hand sides. Data = 4W of the 6001 reference nodes (the truth is
    only known there), density per the method's standard placement."""
    ts, Yref = task["ts"], task["Yref"]
    rng = np.random.default_rng(90_000 + seed)
    idx = _sample_nodes(len(ts), OVERSAMPLE * d.cols, rng,
                        d.meta["method"] in CHEB_SAMPLED)
    s = 2.0 * ts / task["T"] - 1.0
    sigma = np.maximum(np.sqrt(np.mean(Yref ** 2, axis=0)), 1e-12)
    t0 = time.time()
    A = lstsq(d.rows(s[idx], 0), Yref[idx] / sigma[None, :]).T  # (d, W)
    return A, sigma, dict(n_data=len(idx), t_solve=round(time.time() - t0, 2))


def _assemble_1d(sys_obj, A, D0, D1, sigma, T, need_jac=True):
    d, p = A.shape
    n = D0.shape[0]
    U = (D0 @ A.T) * sigma[None, :]
    Fv = sys_obj.F(U)
    R = (2.0 / T) * (D1 @ A.T) - Fv / sigma[None, :]
    if not need_jac:
        return R, None
    Jf = sys_obj.J(U)
    J = np.zeros((d * n, d * p))
    for c in range(d):
        rs = slice(c * n, (c + 1) * n)
        J[rs, c * p:(c + 1) * p] = (2.0 / T) * D1
        for k in range(d):
            coef = -(sigma[k] / sigma[c]) * Jf[:, c, k]
            if np.any(coef):
                J[rs, k * p:(k + 1) * p] += coef[:, None] * D0
    return R, J


def _stacked_1d(sys_obj, A, D0, D1, sigma, T, B, g, scale):
    R, _ = _assemble_1d(sys_obj, A, D0, D1, sigma, T, need_jac=False)
    r = np.concatenate([R.T.ravel() / scale, B @ A.ravel() - g])
    return r, float(np.linalg.norm(r)), float(np.max(np.abs(R)))


def gauss_newton_1d(sys_obj, A0, D0, D1, sigma, T, B, g, max_it=MAX_NEWTON):
    A = A0.copy()
    d, p = A.shape
    hist, no_progress, diverged = [], 0, False
    for _ in range(max_it):
        R, J = _assemble_1d(sys_obj, A, D0, D1, sigma, T)
        if not np.all(np.isfinite(J)):
            diverged = True
            break
        scale = max(np.abs(J).max(), 1e-300)
        Jst = np.vstack([J / scale, B])
        r, rn, rmax = _stacked_1d(sys_obj, A, D0, D1, sigma, T, B, g, scale)
        hist.append(rmax)
        if not np.isfinite(rn):
            diverged = True
            break
        try:
            step = np.linalg.lstsq(Jst, -r, rcond=RCOND)[0].reshape(d, p)
        except np.linalg.LinAlgError:
            from scipy.linalg import lstsq as slstsq
            try:
                step = slstsq(Jst, -r, cond=RCOND,
                              lapack_driver="gelsy")[0].reshape(d, p)
            except Exception:
                diverged = True
                break
        alpha = 1.0
        for _ in range(10):
            _, new_n, _ = _stacked_1d(sys_obj, A + alpha * step, D0, D1, sigma,
                                      T, B, g, scale)
            if new_n <= rn * (1.0 - 1e-4 * alpha) or new_n < 1e-14:
                break
            alpha *= 0.5
        A = A + alpha * step
        small = alpha * np.linalg.norm(step) < 1e-14 * max(1.0, np.linalg.norm(A))
        no_progress = (no_progress + 1) if (len(hist) >= 2
                                            and hist[-1] > 0.5 * hist[-2]) else 0
        if small or no_progress >= 2:
            break
    return A, hist, diverged


def dynamic_solve_1d(task, method, W, config, seed, w_mult=1.0):
    """Whole-trajectory collocation solve of the ODE (expF14 framing)."""
    from .dicts import dict_rng
    from .tasks1d import f14_reference
    sys_obj, T = task["system"], task["T"]
    t0 = time.time()
    d = f1.build_dictionary_1d(method, W, config, dict_rng(method, W, seed))
    n_col = OVERSAMPLE * d.cols
    if method in CHEB_SAMPLED:  # Chebyshev-density collocation (their standard)
        s_col = np.cos(np.pi * (np.arange(n_col) + 0.5) / n_col)[::-1].copy()
    else:
        s_col = np.linspace(-1.0, 1.0, n_col)
    t_col = T * (s_col + 1.0) / 2.0
    D0, D1 = d.rows(s_col, 0), d.rows(s_col, 1)

    Yrk, nfev = f14_reference.rk_trajectory(sys_obj, T, t_col, WARM_RTOL, WARM_ATOL)
    sigma = np.maximum(np.sqrt(np.mean(Yrk ** 2, axis=0)), 1e-12)

    dd, p = sys_obj.d, d.cols
    w_ic = w_mult * np.sqrt(n_col)
    ic_row = d.rows(np.array([-1.0]), 0)[0]
    B = np.zeros((dd, dd * p))
    g = np.empty(dd)
    for c in range(dd):
        B[c, c * p:(c + 1) * p] = w_ic * ic_row
        g[c] = w_ic * sys_obj.ic[c] / sigma[c]

    # warm init: IC-weighted fit of the RK trajectory (expF14's dominant knob)
    M = np.vstack([D0, w_ic * ic_row[None, :]])
    rhs = np.vstack([Yrk / sigma[None, :], w_ic * (sys_obj.ic / sigma)[None, :]])
    A0 = lstsq(M, rhs).T
    A, hist, diverged = gauss_newton_1d(sys_obj, A0, D0, D1, sigma, T, B, g)
    info = dict(iters=len(hist), init="warm", diverged=bool(diverged),
                res=float(hist[-1]) if hist else None,
                t_solve=round(time.time() - t0, 2))
    return d, A, sigma, info


def score_1d(task, d, A, sigma):
    """Aggregate rel L2 over all components on the full 6001-node grid, plus
    the worst single component and absolute Linf (expF14's errors())."""
    s = 2.0 * task["ts"] / task["T"] - 1.0
    Yhat = (d.rows(s, 0) @ A.T) * sigma[None, :]
    E = Yhat - task["Yref"]
    rel = float(np.linalg.norm(E) / np.linalg.norm(task["Yref"]))
    per = (np.linalg.norm(E, axis=0)
           / np.maximum(np.linalg.norm(task["Yref"], axis=0), 1e-300))
    return dict(rel_l2=rel, rel_l2_worst_comp=float(np.max(per)),
                linf=float(np.max(np.abs(E))))
