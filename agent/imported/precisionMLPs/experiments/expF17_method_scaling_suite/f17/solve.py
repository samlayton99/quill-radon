"""expF17 solve engine. One path for every method, both regimes.

Oracle regime: truncated lstsq fit of the oracle on 4C shuffled uniform interior
points plus the perimeter set (rcond = 1e-15, gelsd; the normal equations are
never formed -- expA01 measured them at ~7 decades).

Dynamic regime: strong-form collocation. Interior rows = the task's lin_terms
(+ Gauss-Newton on nl), BC blocks as weighted rows (weight sqrt(n_pde/n_blk),
the expF02/expF13 recipe). Linear tasks are one lstsq; nonlinear tasks run
damped GN with backtracking from a cascade init (converge at ~quarter budget,
refit, polish -- expF03's winning recipe); Burgers additionally runs the expF13
vanishing-viscosity continuation ladder (nu: 0.5 -> 0.01/pi, warm-started).
"""
from __future__ import annotations

import time

import numpy as np

from . import dicts as fd
from .tasks import TASKS, f13

RCOND = 1e-15
MAX_NEWTON = 30
OVERSAMPLE = 4          # data rows per column (SPEC 13.7)
N_EDGE = 480
N_PERIODIC = 320
NU_LADDER = [0.5, 0.1, 0.05, 0.02, f13.NU_BURGERS]


def lstsq(A, y):
    v, res, rank, sv = np.linalg.lstsq(A, y, rcond=RCOND)
    return v, int(rank)


# ---------------------------------------------------------------------------
# point sets
# ---------------------------------------------------------------------------

CHEB_SAMPLED = ("spectral", "bwler")


def _draw(n, rng, cheb_x, cheb_y):
    """Interior draw with method-owned density per axis: arcsine (Chebyshev)
    on polynomial axes, uniform otherwise. High-degree polynomial regression on
    uniform-density samples is the Platte-Trefethen-Kuijlaars unstable regime,
    so the spectral/BWLer arms get their literature-standard placement; every
    other method draws uniform. Counts are identical for all methods."""
    x = (np.cos(np.pi * rng.uniform(0, 1, n)) if cheb_x
         else rng.uniform(-1.0, 1.0, n))
    y = (np.cos(np.pi * rng.uniform(0, 1, n)) if cheb_y
         else rng.uniform(-1.0, 1.0, n))
    return np.stack([x, y], axis=1)


def interior_points(task, n, rng, method=None):
    cheb = method in CHEB_SAMPLED
    cheb_x = cheb and not task.get("periodic_fourier", False)
    cheb_y = cheb
    mask = task["mask"]
    P = _draw(n, rng, cheb_x, cheb_y)
    if mask is not None:
        P = P[mask(P)]
        while len(P) < n:
            Q = _draw(n, rng, cheb_x, cheb_y)
            P = np.vstack([P, Q[mask(Q)]])
        P = P[:n]
    return P


def _edge(where, n=N_EDGE):
    s = np.linspace(-1.0, 1.0, n)
    if where == "ic":
        return np.stack([s, np.full(n, -1.0)], axis=1)
    if where == "left":
        return np.stack([np.full(n, -1.0), s], axis=1)
    if where == "right":
        return np.stack([np.full(n, 1.0), s], axis=1)
    raise ValueError(where)


def _square(n_per_edge=160):
    s = np.linspace(-1.0, 1.0, n_per_edge, endpoint=False)
    return np.concatenate([
        np.stack([s, np.full_like(s, -1.0)], axis=1),
        np.stack([np.full_like(s, 1.0), s], axis=1),
        np.stack([-s, np.full_like(s, 1.0)], axis=1),
        np.stack([np.full_like(s, -1.0), -s], axis=1)])


def _holes(n_per_hole=120):
    ang = np.linspace(0, 2 * np.pi, n_per_hole, endpoint=False)
    ring = f13.HOLE_RADIUS * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return np.concatenate([ring + c for c in f13.HOLE_CENTERS])


def build_bcs(task, d, n_pde, w_mult=1.0):
    """[(rows, values, weight)] per bc block, incl. periodic difference rows.
    w_mult scales every block weight -- the dynamic arm's declared solver knob
    (expF03 measured the block weighting as load-bearing)."""
    bcs = []
    for blk in task["bc_blocks"]:
        where = blk["where"]
        if where == "periodic_x":
            eta = np.linspace(-1.0, 1.0, N_PERIODIC)
            PL = np.stack([np.full_like(eta, -1.0), eta], axis=1)
            PR = np.stack([np.full_like(eta, 1.0), eta], axis=1)
            B = d.rows(PL, blk["terms"]) - d.rows(PR, blk["terms"])
            g = np.zeros(len(eta))
        else:
            Pb = (_square() if where == "square"
                  else _holes() if where == "holes" else _edge(where))
            B = d.rows(Pb, blk["terms"])
            val = blk["value"]
            g = val(Pb) if callable(val) else np.full(len(Pb), float(val))
        bcs.append((B, g, w_mult * np.sqrt(n_pde / len(g))))
    return bcs


# ---------------------------------------------------------------------------
# oracle regime
# ---------------------------------------------------------------------------

def perimeter_points(task):
    """The closed square's boundary (and the hole rings for poisson_man): the
    same point sets the dynamic arm's BC rows sit on."""
    Pb = _square()
    if task["mask"] is not None:
        Pb = np.vstack([Pb, _holes()])
    return Pb


def oracle_fit(task, d, seed):
    """Data footprint = the CLOSED square (SPEC 18.9): 4C interior points plus
    the perimeter set with exact values. An open-square draw never lands on the
    boundary lines the score grid contains, leaving the halo columns unpinned
    there: measured 7-10x on every method at the floor (the dynamic arm, whose
    BC rows sit on the perimeter, landed BELOW the oracle). With the perimeter
    rows the fit matches a full-grid fit."""
    rng = np.random.default_rng(90_000 + seed)  # collocation RNG (13.9)
    P = interior_points(task, OVERSAMPLE * d.cols, rng,
                        method=d.meta.get("method"))
    P = np.vstack([P, perimeter_points(task)])
    y = task["exact"](P)
    t0 = time.time()
    A = d.rows(P, [((0, 0), 1.0)])
    a, rank = lstsq(A, y)
    return a, dict(n_data=len(P), rank=rank, t_solve=round(time.time() - t0, 2))


def darcy_orig_fit(task, d, seed):
    """Reference-capped: train points are grid nodes (the truth exists only
    there), 4C of them, sampled without replacement; the spectral/BWLer arms
    draw the nodes with arcsine density per axis (see _draw)."""
    P_all, u_all = task["eval_grid"]()
    rng = np.random.default_rng(90_000 + seed)
    n_want = min(OVERSAMPLE * d.cols, len(P_all))
    if d.meta.get("method") in CHEB_SAMPLED:
        m = int(round(np.sqrt(len(P_all))))
        Q = _draw(2 * n_want, rng, True, True)
        ij = np.round((Q + 1.0) * 0.5 * (m - 1)).astype(int)
        idx = np.unique(ij[:, 0] * m + ij[:, 1])
        rng.shuffle(idx)
        idx = idx[:n_want]
        if len(idx) < n_want:
            rest = np.setdiff1d(np.arange(len(P_all)), idx)
            idx = np.concatenate([idx, rng.choice(rest, n_want - len(idx),
                                                  replace=False)])
    else:
        idx = rng.choice(len(P_all), size=n_want, replace=False)
    t0 = time.time()
    A = d.rows(P_all[idx], [((0, 0), 1.0)])
    a, rank = lstsq(A, u_all[idx])
    return a, dict(n_data=len(idx), rank=rank, t_solve=round(time.time() - t0, 2))


# ---------------------------------------------------------------------------
# dynamic regime
# ---------------------------------------------------------------------------

def _needed_indices(task):
    idxs = {i for i, _ in task["lin_terms"]}
    idxs |= set(task["nl"]["fields"])
    return sorted(idxs)


def assemble(task, d, seed, w_mult=1.0):
    rng = np.random.default_rng(90_000 + seed)
    P = interior_points(task, OVERSAMPLE * d.cols, rng,
                        method=d.meta.get("method"))
    D = {i: d.rows(P, [(i, 1.0)]) for i in _needed_indices(task)}
    lin = None
    for idx, c in task["lin_terms"]:
        cc = fd._coeff_col(c, P)
        lin = (0.0 if lin is None else lin) + (cc * D[idx] if np.ndim(cc) else cc * D[idx])
    f = task["forcing"]
    fv = f(P) if callable(f) else np.full(len(P), float(f))
    bcs = build_bcs(task, d, len(P), w_mult=w_mult)
    return dict(P=P, D=D, lin=lin, fv=fv, bcs=bcs)


def _stacked_res(a, asm, nl, s):
    vals = {i: asm["D"][i] @ a for i in nl["fields"]}
    r_pde = ((asm["lin"] @ a if asm["lin"] is not None else 0.0)
             + nl["res"](vals, asm["P"]) - asm["fv"])
    parts = [r_pde / s] + [w * (B @ a - g) for (B, g, w) in asm["bcs"]]
    r = np.concatenate(parts)
    return r, float(np.linalg.norm(r)), float(np.max(np.abs(r_pde)))


def linear_solve(asm):
    s = max(np.abs(asm["lin"]).max(), 1e-300)
    A = np.vstack([asm["lin"] / s] + [w * B for (B, g, w) in asm["bcs"]])
    y = np.concatenate([asm["fv"] / s] + [w * g for (B, g, w) in asm["bcs"]])
    a, rank = lstsq(A, y)
    return a, dict(iters=1, rank=rank, hist=[])


def gauss_newton(task, asm, a0, max_newton=MAX_NEWTON):
    """Damped GN with backtracking on the stacked residual (expF02 semantics)."""
    nl = task["nl"]
    a = a0.copy()
    history = []
    for _ in range(max_newton):
        vals = {i: asm["D"][i] @ a for i in nl["fields"]}
        J_pde = asm["lin"].copy() if asm["lin"] is not None else 0.0
        for idx, coef in nl["jac"](vals, asm["P"]).items():
            J_pde = J_pde + coef[:, None] * asm["D"][idx]
        s = max(np.abs(J_pde).max(), 1e-300)
        r, rnorm, rmax = _stacked_res(a, asm, nl, s)
        history.append(rmax)
        if not (np.isfinite(rnorm) and np.isfinite(np.asarray(J_pde)).all()):
            break
        J = np.vstack([J_pde / s] + [w * B for (B, g, w) in asm["bcs"]])
        step, _ = lstsq(J, -r)
        alpha = 1.0
        for _ in range(8):
            _, new_norm, _ = _stacked_res(a + alpha * step, asm, nl, s)
            if np.isfinite(new_norm) and (new_norm <= rnorm * (1.0 - 1e-4 * alpha)
                                          or new_norm < 1e-14):
                break
            alpha *= 0.5
        a = a + alpha * step
        if (alpha * np.linalg.norm(step) < 1e-13 * max(1.0, np.linalg.norm(a))
                or (len(history) > 2 and history[-1] > 0
                    and abs(history[-2] / history[-1] - 1.0) < 1e-3)):
            break
    return a, history


def _fit_to_function(asm, d, u_vals):
    """Init helper: lstsq fit of the dictionary to given interior values + BCs."""
    Phi = asm["D"].get((0, 0))
    if Phi is None:
        Phi = d.rows(asm["P"], [((0, 0), 1.0)])
    A = np.vstack([Phi] + [w * B for (B, g, w) in asm["bcs"]])
    y = np.concatenate([u_vals] + [w * g for (B, g, w) in asm["bcs"]])
    a, _ = lstsq(A, y)
    return a


def dynamic_solve(task, method, n_ax, config, seed, fourier_x=False, poly=False,
                  cascade=True, max_newton=MAX_NEWTON, w_mult=1.0, _depth=0):
    """The dynamic arm for one cell. Returns (dictionary, coeffs, info)."""
    t0 = time.time()
    d = fd.build_dictionary(method, n_ax, config, fd.dict_rng(method, n_ax, seed),
                            fourier_x=fourier_x, poly=poly)

    if task["key"] == "burgers":
        asm = None
        a = None
        rungs = []
        for nu in NU_LADDER:
            rung_task = dict(task)
            prob = f13.make_burgers(nu)
            rung_task.update(lin_terms=prob["lin_terms"], nl=prob["nl"])
            if asm is None:
                asm = assemble(rung_task, d, seed, w_mult=w_mult)
                a = np.zeros(d.cols)
            else:  # D is nu-independent; only the linear combination changes
                lin = None
                for idx, c in prob["lin_terms"]:
                    lin = (0.0 if lin is None else lin) + float(c) * asm["D"][idx]
                asm["lin"] = lin
            a, hist = gauss_newton(rung_task, asm, a, max_newton=max_newton)
            rungs.append(dict(nu=float(nu), iters=len(hist),
                              res=float(hist[-1]) if hist else None))
        info = dict(iters=int(sum(r["iters"] for r in rungs)), rungs=rungs,
                    t_solve=round(time.time() - t0, 2))
        return d, a, info

    asm = assemble(task, d, seed, w_mult=w_mult)
    if not task["nl"]["fields"]:
        a, info = linear_solve(asm)
        info["t_solve"] = round(time.time() - t0, 2)
        return d, a, info

    # nonlinear: cascade init (expF03) then GN polish
    if cascade and _depth == 0:
        n_coarse = max(9, (n_ax + 1) // 2 - 1)
        cfg_c = dict(config)
        if "halo" in cfg_c:
            cfg_c["halo"] = max(1, min(cfg_c["halo"], (n_coarse - 4) // 2))
        d_c, a_c, sub = dynamic_solve(task, method, n_coarse, cfg_c, seed,
                                      fourier_x=fourier_x, poly=poly,
                                      cascade=False, max_newton=max_newton,
                                      w_mult=w_mult, _depth=1)
        u_c = d_c.rows(asm["P"], [((0, 0), 1.0)]) @ a_c
        a0 = _fit_to_function(asm, d, u_c)
        init_mode = "cascade"
    else:
        a0 = np.zeros(d.cols)
        init_mode = "zero"
    a, hist = gauss_newton(task, asm, a0, max_newton=max_newton)
    info = dict(iters=len(hist), init=init_mode,
                res=float(hist[-1]) if hist else None,
                t_solve=round(time.time() - t0, 2))
    return d, a, info


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

_EVAL_CACHE = {}


def eval_data(task):
    key = task["key"]
    if key not in _EVAL_CACHE:
        P, u = task["eval_grid"]()
        disk = np.hypot(P[:, 0], P[:, 1]) <= 1.0
        _EVAL_CACHE[key] = (P, u, disk)
    return _EVAL_CACHE[key]


def score(task, d, a, chunk=8192):
    P, u, disk = eval_data(task)
    u_hat = np.empty(len(P))
    for i in range(0, len(P), chunk):
        u_hat[i:i + chunk] = d.rows(P[i:i + chunk], [((0, 0), 1.0)]) @ a
    err = u_hat - u
    out = dict(rel_l2=float(np.linalg.norm(err) / np.linalg.norm(u)),
               linf=float(np.max(np.abs(err))),
               rel_l2_disk=float(np.linalg.norm(err[disk]) / np.linalg.norm(u[disk])),
               linf_disk=float(np.max(np.abs(err[disk]))))
    return out
