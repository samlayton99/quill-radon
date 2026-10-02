"""expF03 part 4: baseline dictionaries on the NONLINEAR zoo (expF02).

The nonlinear Gauss-Newton solver in common.py is basis-agnostic: it only needs a
derivative dictionary D[idx] = basis.rows(pts, [(idx, 1)]) plus the problem's
pointwise nonlinearity nl["res"]/nl["jac"]. So the ELM / Kansa-RBF / Chebyshev
dictionaries from baselines.py drop straight into OUR machinery -- same Newton,
same cascade init, same rcond -- and we ask: with our good techniques, can the
other bases reach precision on these harder PDEs?

Methods here are the dictionary bases (ELM, RBF, spectral). Trained PINN lives in
baselines_pinn.py (nonlinear residuals added there); finite differences are not run
on the nonlinear zoo (nonlinear FD = Newton on the stencil grid, and the disk is
excluded) -- it is greyed out in the bar chart as the algebraic-convergence control
already characterized in part 3.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from common import (DEFAULT_CFG, PART4_DIR, eval_grid, gauss_newton, lstsq_solve,
                    load_problem_suite, interior_points_2d, metrics, _bc_points_for,
                    _needed_indices)
from baselines import (ElmBasis, RbfBasis, franke_c, spectral_basis_for,
                       _fresh_points_for)

# smaller scale than part 3: short DOF ladders, few seeds
LADDER_1D = [17, 33, 65, 129]
LADDER_2D = [150, 300, 600, 1200]
STOCH_SEEDS = [0, 1]        # ELM / RBF (random features / centers)
DET_SEEDS = [0]            # spectral (deterministic)
INITS = ["cascade", "bcfit", "zero"]   # our-technique inits; keep best by no-oracle signal

# our good techniques as the solver config for every baseline
BASE_CFG = replace(DEFAULT_CFG, rcond=1e-15, w_mult=1.0, max_newton=40)


# ---------------------------------------------------------------------------
# generic nonlinear assembly + solve on a basis (mirrors common.assemble_nonlinear
# but with basis.rows in place of the QI geometry rows)
# ---------------------------------------------------------------------------

def assemble_nonlinear_basis(prob, basis, cfg, rng):
    cat = prob["category"]
    if cat == "ode1d":
        n_col = cfg.oversample_1d * basis.dof
        pts = basis.colloc_1d(n_col) if hasattr(basis, "colloc_1d") \
            else np.linspace(-1, 1, n_col)
        w_bc = cfg.w_mult * np.sqrt(n_col / max(len(prob["conditions"]), 1))
        bcs = []
        for (x0, o) in prob["conditions"]:
            Bm = basis.rows(np.array([x0]), [(o, 1.0)])
            val = float(np.asarray(prob["exact_derivs"][o](np.array([x0]))).ravel()[0])
            bcs.append((Bm, np.array([val]), w_bc))
    else:
        pts = basis.colloc_2d(max(5 * basis.dof, 2000)) if hasattr(basis, "colloc_2d") \
            else interior_points_2d(cat, basis.dof, cfg, rng)
        suite_f01 = load_problem_suite("f01")
        bcs = []
        for blk in prob["bc_blocks"]:
            Pb = _bc_points_for(blk, prob, suite_f01)
            Bm = basis.rows(Pb, blk["terms"])
            w = cfg.w_mult * np.sqrt(len(pts) / len(Pb))
            bcs.append((Bm, np.asarray(blk["value"](Pb)).ravel(), w))
    row_fn = lambda tm: basis.rows(pts, tm)
    D = {i: row_fn([(i, 1.0)]) for i in _needed_indices(prob)}
    lin_rows = None
    for i, c in prob["lin_terms"]:
        cc = c(pts) if callable(c) else c
        contrib = (np.asarray(cc).reshape(-1, 1) * D[i]) if np.ndim(cc) else (c * D[i])
        lin_rows = contrib if lin_rows is None else lin_rows + contrib
    f = prob["forcing"]
    fv = f(pts) if callable(f) else np.full(len(pts), float(f))
    return dict(D=D, lin_rows=lin_rows, fv=fv, bcs=bcs, pts=pts, row_fn=row_fn,
                dof=basis.dof)


def _presolve(asm, A_rows, rhs, cfg):
    s0 = max(np.abs(A_rows).max(), 1e-300)
    A = np.vstack([A_rows / s0] + [w * B for (B, g, w) in asm["bcs"]])
    y = np.concatenate([rhs / s0] + [w * g for (B, g, w) in asm["bcs"]])
    a, _ = lstsq_solve(A, y, cfg)
    return a


def _init_basis(prob, asm, make_basis, dof, cfg):
    mode = cfg.init
    if mode == "zero":
        return np.zeros(asm["dof"])
    if mode == "bcfit":
        A = np.vstack([w * B for (B, g, w) in asm["bcs"]])
        y = np.concatenate([w * g for (B, g, w) in asm["bcs"]])
        a, _ = lstsq_solve(A, y, cfg)
        return a
    if mode == "default":
        if prob.get("init"):
            init = prob["init"]
            A0 = asm["row_fn"](init["terms"])
            rhs = init["rhs"](asm["pts"]) if callable(init["rhs"]) \
                else np.full(len(asm["pts"]), float(init["rhs"]))
            return _presolve(asm, A0, rhs, cfg)
        return np.zeros(asm["dof"])
    if mode == "cascade":
        coarse = max(16 if prob["category"] == "ode1d" else 40, dof // 4)
        if coarse >= dof:
            return _init_basis(prob, asm, make_basis, dof, replace(cfg, init="default"))
        sub = solve_nonlinear_basis(prob, make_basis, coarse, replace(cfg, init="default"))
        if sub is None:
            return _init_basis(prob, asm, make_basis, dof, replace(cfg, init="default"))
        zero_idx = 0 if prob["category"] == "ode1d" else (0, 0)
        u_c = sub["basis"].rows(asm["pts"], [(zero_idx, 1.0)]) @ sub["a"]
        Phi = asm["D"].get(zero_idx)
        if Phi is None:
            Phi = asm["row_fn"]([(zero_idx, 1.0)])
        A = np.vstack([Phi] + [w * B for (B, g, w) in asm["bcs"]])
        y = np.concatenate([u_c] + [w * g for (B, g, w) in asm["bcs"]])
        a, _ = lstsq_solve(A, y, cfg)
        return a
    raise ValueError(mode)


def _tune_signal(prob, basis, a, asm):
    """No-oracle signal: max(fresh PDE residual, condition-row residual)."""
    fr = _fresh_points_for(prob, DEFAULT_CFG)
    nl = prob["nl"]
    idxs = set(nl["fields"]) | {i for i, _ in prob["lin_terms"]}
    vals = {i: basis.rows(fr, [(i, 1.0)]) @ a for i in idxs}
    n = len(fr)
    r = np.zeros(n)
    for i, c in prob["lin_terms"]:
        cc = c(fr) if callable(c) else c
        r = r + cc * vals[i]
    r = r + nl["res"](vals, fr)
    fv = prob["forcing"](fr) if callable(prob["forcing"]) else float(prob["forcing"])
    r = r - fv
    res_pde = float(np.max(np.abs(r)))
    res_bc = max((float(np.max(np.abs(B @ a - g))) for (B, g, w) in asm["bcs"]), default=0.0)
    return res_pde, res_bc


def solve_nonlinear_basis(prob, make_basis, dof, cfg):
    rng = np.random.default_rng(cfg.seed)
    basis = make_basis(dof)
    t0 = time.time()
    asm = assemble_nonlinear_basis(prob, basis, cfg, rng)
    t_asm = time.time() - t0
    a0 = _init_basis(prob, asm, make_basis, dof, cfg)
    if a0 is None:
        return None
    t0 = time.time()
    a, hist = gauss_newton(prob, asm, a0, cfg)
    t_solve = time.time() - t0
    cat = prob["category"]
    pts_e, _ = eval_grid(cat)
    zero_idx = 0 if cat == "ode1d" else (0, 0)
    u_hat = basis.rows(pts_e, [(zero_idx, 1.0)]) @ a
    if not np.all(np.isfinite(u_hat)):
        rel, linf = np.inf, np.inf
    else:
        rel, linf = metrics(u_hat, prob["exact"](pts_e))
    res_pde, res_bc = _tune_signal(prob, basis, a, asm)
    return dict(rel_l2=float(rel), linf=float(linf), dof=basis.dof, basis=basis, a=a,
                newton_iters=len(hist), newton_final_res=float(hist[-1]) if hist else None,
                tune_signal=float(max(res_pde, res_bc)),
                t_assemble=t_asm, t_solve=t_solve)


# ---------------------------------------------------------------------------
# per-method factories + tuning
# ---------------------------------------------------------------------------

def _dim(prob):
    return 1 if prob["category"] == "ode1d" else 2


def _elm_factory(R, dim, seed):
    return lambda dof: ElmBasis(dof, R, dim, seed)


def _rbf_factory(c, dim, seed, kind):
    return lambda dof: RbfBasis(dof, c, dim, seed, kind)


def _spectral_factory(category):
    return lambda dof: spectral_basis_for(category, dof)


def _best_over_inits(prob, make_basis, dof, seed):
    """Solve at (dof, seed) trying each init; keep the one with the smallest
    no-oracle signal (the deployable choice). Returns (record, result)."""
    best = None
    for init in INITS:
        cfg = replace(BASE_CFG, seed=seed, init=init)
        try:
            res = solve_nonlinear_basis(prob, make_basis, dof, cfg)
        except Exception:
            res = None
        if res is None:
            continue
        if best is None or res["tune_signal"] < best[1]["tune_signal"]:
            best = (init, res)
    return best


def _tune_scalar(prob, factory_of, grid, dim, tune_dof):
    """Pick the scale (R for ELM, c for RBF) minimizing the no-oracle signal at a
    mid ladder width, cascade init, seed 0 -- Dong & Yang's residual-tuning idea."""
    best = None
    for val in grid:
        mk = factory_of(val, 0)
        res = solve_nonlinear_basis(prob, mk, tune_dof, replace(BASE_CFG, seed=0, init="cascade"))
        if res is None:
            continue
        if best is None or res["tune_signal"] < best[1]:
            best = (val, res["tune_signal"])
    return best[0] if best else grid[len(grid) // 2]


def run_method_nonlinear(method):
    f02 = load_problem_suite("f02")
    rows = []
    t_all = time.time()
    for key, prob in f02.PROBLEMS.items():
        dim = _dim(prob)
        widths = LADDER_1D if dim == 1 else LADDER_2D
        seeds = DET_SEEDS if method == "spectral" else STOCH_SEEDS
        tune_dof = widths[-2]
        # per-method scale tuning (no oracle)
        if method == "elm":
            R = _tune_scalar(prob, lambda R, s: _elm_factory(R, dim, s), [1.0, 2.0, 4.0, 8.0],
                             dim, tune_dof)
            make_of = lambda seed, R=R: _elm_factory(R, dim, seed)
            meta = dict(R=R)
        elif method == "rbf":
            fc = franke_c(tune_dof)
            best = None
            for kind in ("mq", "gauss"):
                for mult in (0.5, 1.0, 2.0, 4.0):
                    c = mult * fc
                    res = solve_nonlinear_basis(prob, _rbf_factory(c, dim, 0, kind), tune_dof,
                                                replace(BASE_CFG, seed=0, init="cascade"))
                    if res and (best is None or res["tune_signal"] < best[2]):
                        best = (kind, c, res["tune_signal"])
            kind, c = (best[0], best[1]) if best else ("gauss", fc)
            make_of = lambda seed, c=c, kind=kind: _rbf_factory(c, dim, seed, kind)
            meta = dict(kind=kind, c=c)
        else:  # spectral
            make_of = lambda seed: _spectral_factory(prob["category"])
            meta = {}
        for dof in widths:
            for seed in seeds:
                best = _best_over_inits(prob, make_of(seed), dof, seed)
                if best is None:
                    continue
                init, res = best
                rows.append(dict(method=method, key=key, dof=res["dof"], seed=seed,
                                 init=init, rel_l2=res["rel_l2"], linf=res["linf"],
                                 tune_signal=res["tune_signal"],
                                 newton_iters=res["newton_iters"],
                                 newton_final_res=res["newton_final_res"],
                                 t_solve=res["t_solve"], t_assemble=res["t_assemble"],
                                 **meta))
                print(f"  {method} {key} dof={res['dof']} s{seed} init={init}: "
                      f"relL2={res['rel_l2']:.2e} sig={res['tune_signal']:.1e} "
                      f"({res['newton_iters']} it, {res['t_solve']:.1f}s)", flush=True)
        (PART4_DIR / f"nl_{method}.json").write_text(json.dumps(rows, indent=1))
    print(f"{method}: {len(rows)} solves in {(time.time() - t_all) / 60:.1f} min", flush=True)
    return rows
