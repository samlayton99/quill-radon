"""expF03 shared machinery: one consolidated copy of the expF01/expF02 solvers,
with every design choice lifted into a config so the ablation/selection/baseline
parts can turn knobs without code edits.

Differences from the expF01/expF02 run.py copies (behavior-identical at the
default config -- enforced by reproduction_gate()):
  - SolveConfig collects all knobs (lambda, rcond, block weights, halo/collar,
    oversampling, poly degree, column equilibration, Newton options, init mode).
  - assemble_*() returns UNSCALED blocks so the weighting axis re-stacks without
    re-assembly, and svd_factor()/svd_solve() factor once and re-threshold per
    rcond (the whole rcond axis costs one SVD).
  - Every solve emits a signal dict (rank, singular values, coefficient norms,
    train residual, fresh-point physical residual, Newton history, timings) and
    can cache eval-grid predictions to .npz for nested-width self-consistency.

Problem suites are loaded from expF01/expF02 problems.py via importlib (both
files are named problems.py, so they cannot be plain-imported side by side).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, replace, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF03_ablation_and_baselines"
CACHE_DIR = RESULTS_DIR / "cache"
# Data + figures are grouped by experiment part so the results folder stays parsable.
# cache/ stays at the top level because part-1 solves populate it and part-2 reads it.
PART1_DIR = RESULTS_DIR / "part1_ablations"    # ablate/refine/rescue json + ablation figures
PART2_DIR = RESULTS_DIR / "part2_selection"    # select json/summaries + selection figures
PART3_DIR = RESULTS_DIR / "part3_baselines"    # baselines_*.json + baseline figures
PART4_DIR = RESULTS_DIR / "part4_nonlinear_baselines"  # nonlinear-zoo baseline comparison
for _d in (CACHE_DIR, PART1_DIR, PART2_DIR, PART3_DIR, PART4_DIR):
    _d.mkdir(parents=True, exist_ok=True)

N_EVAL_2D = 241          # matches expF01/expF02 eval grid
N_EVAL_1D = 8193         # fixed 1D eval grid (expF01 used 3*n_col+1, width-dependent;
                         # a fixed grid is needed so cached predictions are comparable)


def load_problem_suite(which):
    """which in {'f01','f02'} -> the problems module of that experiment."""
    name = {"f01": "expF01_linear_de_zoo", "f02": "expF02_nonlinear_de_zoo"}[which]
    path = REPO_ROOT / "experiments" / name / "problems.py"
    modname = f"expf03_problems_{which}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolveConfig:
    lam: float = 0.25
    rcond: float = 1e-13
    w_mult: float = 1.0            # multiplier on the sqrt(n_pde/n_blk) block weight
    halo: str = "max(70,0.4N)"     # 1D halo spec: "0", "20", "70", "0.4N", "0.8N", default
    collar_steady: float = 1.25
    collar_time: float = 1.6
    oversample_1d: int = 4         # collocation rows per neuron (1D)
    oversample_2d: int = 5         # (2D; floor of 2000 points kept)
    poly_degree: int | None = 3    # None = no polynomial block
    equilibrate: bool = False      # normalize stacked columns before lstsq
    seed: int = 42
    # nonlinear options
    max_newton: int = 25
    init: str = "default"          # default|zero|bcfit|linpart|cascade|picard|homotopy
    lm: bool = False               # Levenberg-Marquardt instead of backtracking
    rcond_tighten: bool = False    # per-step rcond ~ residual-scaled
    nl_scale: float = 1.0          # homotopy scaling of the nonlinearity


DEFAULT_CFG = SolveConfig()


def parse_halo(spec, N):
    if spec == "max(70,0.4N)":
        return max(70, int(np.ceil(0.4 * N)))
    if spec.endswith("N"):
        return int(np.ceil(float(spec[:-1]) * N))
    return int(spec)


def collar_for(cfg, category):
    return cfg.collar_steady if category == "pde2d_steady" else cfg.collar_time


# ---------------------------------------------------------------------------
# dictionary: tanh derivative ladder, monomials, geometry, feature rows
# ---------------------------------------------------------------------------

def psi(order, t):
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
    out = 1
    for k in range(o):
        out *= (d - k)
    return out


def mono_1d(degree):
    return [] if degree is None else list(range(degree + 1))


def mono_2d(degree):
    if degree is None:
        return []
    out = []
    for d in range(degree + 1):
        for px in range(d, -1, -1):
            out.append((px, d - px))
    return out


def geometry_1d(N, cfg):
    h = 2.0 / N
    R = parse_halo(cfg.halo, N)
    centers = -1.0 + h * np.arange(-R, N + R + 1)
    return centers, cfg.lam / h


def radon_geometry(W, cfg, category):
    J = int(round(np.sqrt(W)))
    M = W // J
    thetas = np.pi * np.arange(J) / J
    ts = np.linspace(-collar_for(cfg, category), collar_for(cfg, category), M)
    dirs = np.repeat(np.stack([np.cos(thetas), np.sin(thetas)], axis=1), M, axis=0)
    offs = np.tile(ts, J)
    h_ref = 2.8 / np.sqrt(J * M)
    return dirs, offs, cfg.lam / h_ref


def _coeff_col(coeff, pts):
    """Coefficient -> column vector or scalar. Accepts float, callable, or ndarray
    (an ndarray is a per-point frozen coefficient, e.g. a Picard iterate)."""
    if callable(coeff):
        return np.asarray(coeff(pts), dtype=np.float64).reshape(-1, 1)
    if isinstance(coeff, np.ndarray):
        return coeff.reshape(-1, 1)
    return float(coeff)


def rows_1d(x, centers, gamma, terms, monos):
    t = np.tanh(gamma * (x[:, None] - centers[None, :]))
    A = np.zeros_like(t)
    polys = np.zeros((len(x), len(monos)))
    for o, coeff in terms:
        cc = _coeff_col(coeff, x)
        A += cc * (gamma ** o) * psi(o, t)
        for k, d in enumerate(monos):
            mono = _ffact(d, o) * x ** (d - o) if o <= d else np.zeros_like(x)
            polys[:, k] += (cc.ravel() if np.ndim(cc) else cc) * mono
    return np.hstack([A, polys]) if monos else A


def rows_2d(P, dirs, offs, gamma, terms, monos):
    t = np.tanh(gamma * (P @ dirs.T - offs[None, :]))
    A = np.zeros_like(t)
    polys = np.zeros((len(P), len(monos)))
    x, y = P[:, 0], P[:, 1]
    for (ax, ay), coeff in terms:
        o = ax + ay
        cc = _coeff_col(coeff, P)
        dir_fac = (dirs[:, 0] ** ax * dirs[:, 1] ** ay)[None, :]
        A += cc * (gamma ** o) * dir_fac * psi(o, t)
        ccr = cc.ravel() if np.ndim(cc) else cc
        for k, (px, py) in enumerate(monos):
            if ax <= px and ay <= py:
                mono = (_ffact(px, ax) * _ffact(py, ay)
                        * x ** (px - ax) * y ** (py - ay))
                polys[:, k] += ccr * mono
    return np.hstack([A, polys]) if monos else A


def bc_points_2d(where, n=480):
    if where == "circle":
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(ang), np.sin(ang)], axis=1)
    if where == "ic":
        return np.stack([np.linspace(-1, 1, n), np.full(n, -1.0)], axis=1)
    if where == "left":
        return np.stack([np.full(n, -1.0), np.linspace(-1, 1, n)], axis=1)
    if where == "right":
        return np.stack([np.full(n, 1.0), np.linspace(-1, 1, n)], axis=1)
    raise ValueError(where)


def interior_points_2d(category, Wreq, cfg, rng):
    n = max(cfg.oversample_2d * Wreq, 2000)
    if category == "pde2d_steady":
        rr = np.sqrt(rng.random(n))
        th = 2 * np.pi * rng.random(n)
        return np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
    return rng.uniform(-1, 1, (n, 2))


def _bc_points_for(blk, prob, suite_f01=None):
    """Resolve a 2D bc block's points (block-local generator, inflow, or edge)."""
    if "points" in blk:
        return blk["points"]()
    if blk["where"] == "inflow":
        # expF01 B1's inflow arc: b.n < 0 with n the point on the unit circle
        B = bc_points_2d("circle")
        bvec = suite_f01._B
        return B[B @ bvec < 0]
    return bc_points_2d(blk["where"])


# ---------------------------------------------------------------------------
# assembly (unscaled blocks) + stacking + SVD solve
# ---------------------------------------------------------------------------

def assemble_linear(prob, size, cfg):
    """Unscaled PDE block + condition blocks for a linear (expF01-style) problem.

    Returns dict: A_pde, y_pde, blocks=[(B, g)], geometry fields, monos, pts.
    """
    t0 = time.time()
    if prob["category"] == "ode1d":
        centers, gamma = geometry_1d(size, cfg)
        W = len(centers)
        monos = mono_1d(cfg.poly_degree)
        n_col = cfg.oversample_1d * W
        x = np.linspace(-1, 1, n_col)
        A_pde = rows_1d(x, centers, gamma, prob["terms"], monos)
        f = prob["forcing"]
        y_pde = f(x) if callable(f) else np.full(n_col, float(f))
        blocks = []
        for (x0, o) in prob["conditions"]:
            B = rows_1d(np.array([x0]), centers, gamma, [(o, 1.0)], monos)
            val = float(np.asarray(prob["exact_derivs"][o](np.array([x0]))).ravel()[0])
            blocks.append((B, np.array([val])))
        return dict(kind="1d", centers=centers, gamma=gamma, W=W, monos=monos,
                    pts=x, A_pde=A_pde, y_pde=y_pde, blocks=blocks,
                    t_assemble=time.time() - t0)
    rng = np.random.default_rng(cfg.seed)
    cat = prob["category"]
    dirs, offs, gamma = radon_geometry(size, cfg, cat)
    monos = mono_2d(cfg.poly_degree)
    P = interior_points_2d(cat, size, cfg, rng)
    A_pde = rows_2d(P, dirs, offs, gamma, prob["terms"], monos)
    f = prob["forcing"]
    y_pde = f(P) if callable(f) else np.full(len(P), float(f))
    suite_f01 = load_problem_suite("f01")
    blocks = []
    for blk in prob["bc_blocks"]:
        Pb = _bc_points_for(blk, prob, suite_f01)
        B = rows_2d(Pb, dirs, offs, gamma, blk["terms"], monos)
        blocks.append((B, blk["value"](Pb)))
    return dict(kind="2d", dirs=dirs, offs=offs, gamma=gamma, W=len(offs),
                monos=monos, pts=P, A_pde=A_pde, y_pde=y_pde, blocks=blocks,
                t_assemble=time.time() - t0)


def stack(asm, cfg):
    """Scale the PDE block to O(1), weight condition blocks, stack. Returns A, y, s."""
    s = np.abs(asm["A_pde"]).max()
    n_pde = len(asm["A_pde"])
    rows, vals = [asm["A_pde"] / s], [asm["y_pde"] / s]
    if asm["kind"] == "1d":
        # expF01 semantics: every 1D condition row gets sqrt(n_col / n_conditions)
        w = cfg.w_mult * np.sqrt(n_pde / max(len(asm["blocks"]), 1))
        for B, g in asm["blocks"]:
            rows.append(w * B)
            vals.append(w * g)
    else:
        for B, g in asm["blocks"]:
            w = cfg.w_mult * np.sqrt(n_pde / len(B))
            rows.append(w * B)
            vals.append(w * g)
    return np.vstack(rows), np.concatenate(vals), s


def svd_factor(A):
    try:
        U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        # gesdd can fail to converge on ill-conditioned Newton Jacobians;
        # gesvd is slower but robust (same route np.lstsq's gelsd effectively takes)
        from scipy.linalg import svd as scipy_svd
        U, sv, Vt = scipy_svd(A, full_matrices=False, lapack_driver="gesvd")
    return U, sv, Vt


def svd_solve(fac, y, rcond):
    """Min-norm lstsq via a precomputed SVD; matches np.linalg.lstsq(rcond)."""
    U, sv, Vt = fac
    cutoff = rcond * sv[0] if rcond is not None and rcond > 0 else 0.0
    keep = sv > cutoff
    inv = np.zeros_like(sv)
    inv[keep] = 1.0 / sv[keep]
    a = Vt.T @ (inv * (U.T @ y))
    return a, dict(rank=int(keep.sum()), n_cols=len(sv),
                   sigma_max=float(sv[0]), sigma_min_kept=float(sv[keep][-1]) if keep.any() else 0.0)


def lstsq_solve(A, y, cfg):
    """Solve min ||Aa - y|| with optional column equilibration. Returns a, svd_info."""
    if cfg.equilibrate:
        cn = np.linalg.norm(A, axis=0)
        cn[cn == 0] = 1.0
        fac = svd_factor(A / cn)
        a, info = svd_solve(fac, y, cfg.rcond)
        return a / cn, info
    fac = svd_factor(A)
    return svd_solve(fac, y, cfg.rcond)


# ---------------------------------------------------------------------------
# evaluation, metrics, signals, prediction cache
# ---------------------------------------------------------------------------

def eval_model(category, model, pts, chunk=4096):
    if category == "ode1d":
        t = np.tanh(model["gamma"] * (pts[:, None] - model["centers"][None, :]))
        out = t @ model["sol"][:model["W"]]
        for k, d in enumerate(model["monos"]):
            out += model["sol"][model["W"] + k] * pts ** d
        return out
    W = model["W"]
    out = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        Q = pts[i:i + chunk]
        t = np.tanh(model["gamma"] * (Q @ model["dirs"].T - model["offs"][None, :]))
        v = t @ model["sol"][:W]
        for k, (px, py) in enumerate(model["monos"]):
            v += model["sol"][W + k] * Q[:, 0] ** px * Q[:, 1] ** py
        out[i:i + chunk] = v
    return out


def eval_grid(category):
    """Fixed evaluation grid per category (shared across widths for the cache)."""
    if category == "ode1d":
        return np.linspace(-1, 1, N_EVAL_1D), None
    g = np.linspace(-1, 1, N_EVAL_2D)
    GX, GY = np.meshgrid(g, g)
    P = np.stack([GX.ravel(), GY.ravel()], axis=1)
    if category == "pde2d_steady":
        mask = (P ** 2).sum(axis=1) <= 1.0
        return P[mask], (GX, GY, mask.reshape(GX.shape))
    return P, (GX, GY, np.ones_like(GX, dtype=bool))


def metrics(u_hat, u_true):
    err = u_hat - u_true
    return (float(np.linalg.norm(err) / np.linalg.norm(u_true)),
            float(np.max(np.abs(err))))


def evaluate(prob, model):
    """(rel_l2, linf, u_hat on the fixed grid)."""
    pts, _ = eval_grid(prob["category"])
    u_hat = eval_model(prob["category"], model, pts)
    rel, linf = metrics(u_hat, prob["exact"](pts))
    return rel, linf, u_hat


def fresh_points(prob, cfg, n=2000):
    rng = np.random.default_rng(cfg.seed + 90210)
    if prob["category"] == "ode1d":
        return rng.uniform(-1, 1, n)
    if prob["category"] == "pde2d_steady":
        rr = np.sqrt(rng.random(n))
        th = 2 * np.pi * rng.random(n)
        return np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
    return rng.uniform(-1, 1, (n, 2))


def fresh_residual_linear(prob, model, cfg):
    """max |L u_hat - f| on fresh points (PHYSICAL, unscaled)."""
    pts = fresh_points(prob, cfg)
    if prob["category"] == "ode1d":
        A = rows_1d(pts, model["centers"], model["gamma"], prob["terms"], model["monos"])
    else:
        A = rows_2d(pts, model["dirs"], model["offs"], model["gamma"], prob["terms"], model["monos"])
    f = prob["forcing"]
    fv = f(pts) if callable(f) else np.full(len(pts), float(f))
    return float(np.max(np.abs(A @ model["sol"] - fv)))


def fresh_residual_nonlinear(prob, model, cfg):
    pts = fresh_points(prob, cfg)
    row_fn = (lambda tm: rows_1d(pts, model["centers"], model["gamma"], tm, model["monos"])) \
        if prob["category"] == "ode1d" else \
        (lambda tm: rows_2d(pts, model["dirs"], model["offs"], model["gamma"], tm, model["monos"]))
    idxs = sorted({i for i, _ in prob["lin_terms"]} | set(prob["nl"]["fields"]),
                  key=lambda i: (i if isinstance(i, tuple) else (i,)))
    D = {i: row_fn([(i, 1.0)]) for i in idxs}
    r = np.zeros(len(pts))
    for i, c in prob["lin_terms"]:
        cc = c(pts) if callable(c) else c
        r += cc * (D[i] @ model["sol"])
    vals = {i: D[i] @ model["sol"] for i in prob["nl"]["fields"]}
    r += cfg.nl_scale * prob["nl"]["res"](vals, pts)
    f = prob["forcing"]
    fv = f(pts) if callable(f) else np.full(len(pts), float(f))
    return float(np.max(np.abs(r - fv)))


def condition_mismatch(prob, model):
    """max |condition-row residual| (PHYSICAL, unweighted). The PDE-only fresh
    residual is blind to wrong-branch solutions of homogeneous problems (u=0
    solves the logistic / Burgers / KdV PDE exactly); the condition rows are
    where such solutions fail, so tuning signals must include this term."""
    worst = 0.0
    if prob["category"] == "ode1d":
        for (x0, o) in prob["conditions"]:
            xp = np.array([x0])
            r = rows_1d(xp, model["centers"], model["gamma"], [(o, 1.0)], model["monos"])
            val = float(np.asarray(prob["exact_derivs"][o](xp)).ravel()[0])
            worst = max(worst, abs(float((r @ model["sol"])[0]) - val))
        return worst
    for blk in prob["bc_blocks"]:
        Pb = blk["points"]() if "points" in blk else bc_points_2d(blk["where"], n=517)
        B = rows_2d(Pb, model["dirs"], model["offs"], model["gamma"], blk["terms"],
                    model["monos"])
        worst = max(worst, float(np.max(np.abs(B @ model["sol"] - blk["value"](Pb)))))
    return worst


def coeff_norms(model):
    a = model["sol"]
    W = model["W"]
    return dict(norm_a_tanh=float(np.linalg.norm(a[:W])),
                norm_a_poly=float(np.linalg.norm(a[W:])) if len(a) > W else 0.0)


def cache_pred(tag, u_hat):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_DIR / f"{tag}.npz", u=u_hat)


def load_pred(tag):
    return np.load(CACHE_DIR / f"{tag}.npz")["u"]


# ---------------------------------------------------------------------------
# linear solve
# ---------------------------------------------------------------------------

def solve_linear(prob, size, cfg, asm=None, cache_tag=None):
    """One linear collocation solve. Returns dict(model, rel_l2, linf, signals)."""
    asm = asm or assemble_linear(prob, size, cfg)
    A, y, s = stack(asm, cfg)
    t0 = time.time()
    a, info = lstsq_solve(A, y, cfg)
    t_solve = time.time() - t0
    model = _model_from(asm, a)
    rel, linf, u_hat = evaluate(prob, model)
    if cache_tag:
        cache_pred(cache_tag, u_hat)
    sig = dict(info, **coeff_norms(model),
               train_res_scaled=float(np.max(np.abs(A @ a - y))),
               fresh_res_phys=fresh_residual_linear(prob, model, cfg),
               fresh_bc_err=condition_mismatch(prob, model),
               t_assemble=asm["t_assemble"], t_solve=t_solve,
               n_rows=A.shape[0], n_cols_total=A.shape[1], pde_scale=float(s))
    return dict(model=model, rel_l2=rel, linf=linf, signals=sig, W=model["W"])


def _model_from(asm, a):
    if asm["kind"] == "1d":
        return dict(centers=asm["centers"], gamma=asm["gamma"], W=asm["W"],
                    monos=asm["monos"], sol=a)
    return dict(dirs=asm["dirs"], offs=asm["offs"], gamma=asm["gamma"], W=asm["W"],
                monos=asm["monos"], sol=a)


# ---------------------------------------------------------------------------
# nonlinear solve: assembly, Gauss-Newton (backtracking or LM), init ladder
# ---------------------------------------------------------------------------

def _needed_indices(prob):
    idxs = {i for i, _ in prob["lin_terms"]}
    idxs |= set(prob["nl"]["fields"])
    return sorted(idxs, key=lambda i: (i if isinstance(i, tuple) else (i,)))


def assemble_nonlinear(prob, size, cfg, extra_lin=None):
    """Derivative dictionary D, linear rows, forcing, weighted bc blocks, points.

    extra_lin: additional [(index, coeff)] linear terms (e.g. viscous regularization);
    they are folded into lin_rows so the residual/Jacobian see them everywhere.
    """
    t0 = time.time()
    if prob["category"] == "ode1d":
        centers, gamma = geometry_1d(size, cfg)
        W = len(centers)
        monos = mono_1d(cfg.poly_degree)
        n_col = cfg.oversample_1d * W
        pts = np.linspace(-1, 1, n_col)
        row_fn = lambda tm: rows_1d(pts, centers, gamma, tm, monos)
        geom = dict(kind="1d", centers=centers, gamma=gamma, W=W, monos=monos)
        cond_iter = [(np.array([x0]), o) for (x0, o) in prob["conditions"]]
        w_bc = cfg.w_mult * np.sqrt(n_col / max(len(cond_iter), 1))
        bcs = []
        for (xp, o) in cond_iter:
            B = rows_1d(xp, centers, gamma, [(o, 1.0)], monos)
            val = float(np.asarray(prob["exact_derivs"][o](xp)).ravel()[0])
            bcs.append((B, np.array([val]), w_bc))
    else:
        rng = np.random.default_rng(cfg.seed)
        cat = prob["category"]
        dirs, offs, gamma = radon_geometry(size, cfg, cat)
        monos = mono_2d(cfg.poly_degree)
        pts = interior_points_2d(cat, size, cfg, rng)
        row_fn = lambda tm: rows_2d(pts, dirs, offs, gamma, tm, monos)
        geom = dict(kind="2d", dirs=dirs, offs=offs, gamma=gamma, W=len(offs), monos=monos)
        suite_f01 = load_problem_suite("f01")
        bcs = []
        for blk in prob["bc_blocks"]:
            Pb = _bc_points_for(blk, prob, suite_f01)
            B = rows_2d(Pb, dirs, offs, gamma, blk["terms"], monos)
            w = cfg.w_mult * np.sqrt(len(pts) / len(Pb))
            bcs.append((B, blk["value"](Pb), w))
    extra = list(extra_lin or [])
    idxs = sorted(set(_needed_indices(prob)) | {i for i, _ in extra},
                  key=lambda i: (i if isinstance(i, tuple) else (i,)))
    D = {i: row_fn([(i, 1.0)]) for i in idxs}
    lin_rows = None
    for i, c in list(prob["lin_terms"]) + extra:
        cc = _coeff_col(c, pts)
        contrib = (cc * D[i]) if np.ndim(cc) else (c * D[i])
        lin_rows = contrib if lin_rows is None else lin_rows + contrib
    f = prob["forcing"]
    fv = f(pts) if callable(f) else np.full(len(pts), float(f))
    return dict(geom=geom, D=D, lin_rows=lin_rows, fv=fv, bcs=bcs, pts=pts,
                row_fn=row_fn, t_assemble=time.time() - t0)


def _nl_scaled(prob, cfg):
    nl = prob["nl"]
    if cfg.nl_scale == 1.0:
        return nl
    s = cfg.nl_scale
    return dict(fields=nl["fields"],
                res=lambda v, p: s * nl["res"](v, p),
                jac=lambda v, p: {k: s * g for k, g in nl["jac"](v, p).items()})


def _stacked_res(a, lin_rows, D, nl, pts, f, bcs, s):
    vals = {idx: D[idx] @ a for idx in nl["fields"]}
    r_pde = (lin_rows @ a if lin_rows is not None else 0.0) + nl["res"](vals, pts) - f
    parts = [r_pde / s]
    parts += [w * (B @ a - g) for (B, g, w) in bcs]
    r = np.concatenate(parts)
    return r, float(np.linalg.norm(r)), float(np.max(np.abs(r_pde)))


def gauss_newton(prob, asm, a0, cfg):
    """Damped GN (backtracking, expF02 semantics) or LM if cfg.lm. Returns (a, history)."""
    nl = _nl_scaled(prob, cfg)
    D, lin_rows, pts, fv, bcs = asm["D"], asm["lin_rows"], asm["pts"], asm["fv"], asm["bcs"]
    a = a0.copy()
    history = []
    mu = 1e-8  # LM parameter (adaptive)
    for it in range(cfg.max_newton):
        vals = {idx: D[idx] @ a for idx in nl["fields"]}
        J_pde = lin_rows.copy() if lin_rows is not None else 0.0
        jd = nl["jac"](vals, pts)
        for idx, coef in jd.items():
            J_pde = J_pde + coef[:, None] * D[idx]
        s = max(np.abs(J_pde).max(), 1e-300)
        r, rnorm, rmax = _stacked_res(a, lin_rows, D, nl, pts, fv, bcs, s)
        history.append(rmax)
        # divergence guard: a degenerate Jacobian (e.g. Monge-Ampere linearized at
        # u=0 is identically zero) or an overflowed iterate ends the iteration
        # honestly instead of feeding inf/NaN into the SVD
        if not (np.isfinite(rnorm) and np.isfinite(np.asarray(J_pde)).all()):
            break
        rcond_it = cfg.rcond
        if cfg.rcond_tighten:
            rcond_it = float(np.clip(rnorm * 1e-3, 1e-15, cfg.rcond))
        J = np.vstack([J_pde / s] + [w * B for (B, g, w) in bcs])
        if cfg.lm:
            accepted = False
            for _ in range(10):
                Jlm = np.vstack([J, np.sqrt(mu) * np.eye(J.shape[1])])
                rlm = np.concatenate([-r, np.zeros(J.shape[1])])
                step, _ = lstsq_solve(Jlm, rlm, replace(cfg, rcond=rcond_it))
                _, new_norm, _ = _stacked_res(a + step, lin_rows, D, nl, pts, fv, bcs, s)
                if np.isfinite(new_norm) and (new_norm < rnorm or new_norm < 1e-14):
                    a = a + step
                    mu = max(mu / 3.0, 1e-14)
                    accepted = True
                    break
                mu *= 4.0
            if not accepted:
                break
            alpha, step_used = 1.0, step
        else:
            step, _ = lstsq_solve(J, -r, replace(cfg, rcond=rcond_it))
            alpha = 1.0
            for _ in range(8):
                _, new_norm, _ = _stacked_res(a + alpha * step, lin_rows, D, nl,
                                              pts, fv, bcs, s)
                if np.isfinite(new_norm) and (new_norm <= rnorm * (1.0 - 1e-4 * alpha)
                                              or new_norm < 1e-14):
                    break
                alpha *= 0.5
            a = a + alpha * step
            step_used = alpha * step
        if (np.linalg.norm(step_used) < 1e-13 * max(1.0, np.linalg.norm(a))
                or (len(history) > 2 and history[-1] > 0
                    and abs(history[-2] / history[-1] - 1.0) < 1e-3)):
            break
    return a, history


# --- Newton initialization ladder -------------------------------------------

# Picard freeze-first-factor specs per expF02 problem key:
# N(v) == sum_l coef_l(v) * v_l must hold at the fixed point.
PICARD_SPECS = {
    "ode1d_o1": lambda v: {0: 3.0 * v[0]},
    "ode1d_o3": lambda v: {2: 0.5 * v[0]},
    "pde2d_steady_o1": lambda v: {(1, 0): v[(1, 0)], (0, 1): v[(0, 1)]},
    "pde2d_steady_o2": lambda v: {(0, 2): v[(2, 0)], (1, 1): -v[(1, 1)]},
    "pde2d_steady_o3": lambda v: {(1, 0): v[(0, 0)]},
    "pde2d_time_o1": lambda v: {(1, 0): v[(0, 0)]},
    "pde2d_time_o2": lambda v: {(1, 0): v[(0, 0)]},
    "pde2d_time_o3": lambda v: {(1, 0): 6.0 * v[(0, 0)]},
}


def _linear_presolve(asm, A_rows, rhs, cfg):
    """Solve [A_rows; weighted bcs] a = [rhs; bc values] (used by several inits)."""
    s0 = max(np.abs(A_rows).max(), 1e-300)
    A = np.vstack([A_rows / s0] + [w * B for (B, g, w) in asm["bcs"]])
    y = np.concatenate([rhs / s0] + [w * g for (B, g, w) in asm["bcs"]])
    a, _ = lstsq_solve(A, y, cfg)
    return a


def initial_a(prob, asm, cfg, size):
    n_cols = asm["geom"]["W"] + len(asm["geom"]["monos"])
    mode = cfg.init
    if mode == "zero":
        return np.zeros(n_cols)
    if mode == "default":
        if prob.get("init"):
            init = prob["init"]
            A0 = asm["row_fn"](init["terms"])
            rhs = init["rhs"](asm["pts"]) if callable(init["rhs"]) else \
                np.full(len(asm["pts"]), float(init["rhs"]))
            return _linear_presolve(asm, A0, rhs, cfg)
        return np.zeros(n_cols)
    if mode == "bcfit":
        # min-norm fit of the condition rows alone
        A = np.vstack([w * B for (B, g, w) in asm["bcs"]])
        y = np.concatenate([w * g for (B, g, w) in asm["bcs"]])
        a, _ = lstsq_solve(A, y, cfg)
        return a
    if mode == "linpart":
        if asm["lin_rows"] is None:
            return None  # not applicable (no linear part)
        return _linear_presolve(asm, asm["lin_rows"], asm["fv"], cfg)
    if mode == "cascade":
        coarse = max(16, size // 4) if prob["category"] == "ode1d" else max(144, size // 4)
        if coarse >= size:
            # bottom rung: no coarser level exists -- use the classical init
            # (zero would diverge on e.g. the eikonal or Monge-Ampere)
            return initial_a(prob, asm, replace(cfg, init="default"), size)
        sub = solve_nonlinear(prob, coarse, replace(cfg, init="default"))
        u_c = eval_model(prob["category"], sub["model"], asm["pts"])
        zero_idx = 0 if prob["category"] == "ode1d" else (0, 0)
        Phi = asm["D"].get(zero_idx)
        if Phi is None:
            Phi = asm["row_fn"]([(zero_idx, 1.0)])
        A = np.vstack([Phi] + [w * B for (B, g, w) in asm["bcs"]])
        y = np.concatenate([u_c] + [w * g for (B, g, w) in asm["bcs"]])
        a, _ = lstsq_solve(A, y, cfg)
        return a
    if mode == "picard":
        spec = PICARD_SPECS.get(prob["key"])
        if spec is None:
            return None  # no bilinear structure (e.g. Bratu)
        a = np.zeros(n_cols)
        for _ in range(3):
            vals = {idx: asm["D"][idx] @ a for idx in prob["nl"]["fields"]}
            A_pic = asm["lin_rows"].copy() if asm["lin_rows"] is not None else 0.0
            for idx, coef in spec(vals).items():
                A_pic = A_pic + cfg.nl_scale * coef[:, None] * asm["D"][idx]
            if np.ndim(A_pic) == 0:
                return None
            a = _linear_presolve(asm, A_pic, asm["fv"], cfg)
        return a
    raise ValueError(mode)


def solve_nonlinear(prob, size, cfg, extra_lin=None, a0_override=None, cache_tag=None):
    """Newton solve. cfg.init selects the starting point; cfg.init='homotopy' ramps
    the nonlinearity s in {0.25,0.5,0.75,1.0} with warm starts."""
    asm = assemble_nonlinear(prob, size, cfg, extra_lin=extra_lin)
    t0 = time.time()
    if cfg.init == "homotopy" and a0_override is None:
        a = np.zeros(asm["geom"]["W"] + len(asm["geom"]["monos"]))
        hist_all = []
        for sval in (0.25, 0.5, 0.75, 1.0):
            a, h = gauss_newton(prob, asm, a, replace(cfg, nl_scale=sval, init="zero"))
            hist_all.extend(h)
        hist = hist_all
    else:
        a0 = a0_override if a0_override is not None else initial_a(prob, asm, cfg, size)
        if a0 is None:
            return None  # init mode not applicable to this problem
        a, hist = gauss_newton(prob, asm, a0, cfg)
    t_solve = time.time() - t0
    model = _model_from_geom(asm["geom"], a)
    rel, linf, u_hat = evaluate(prob, model)
    if cache_tag:
        cache_pred(cache_tag, u_hat)
    stalled = bool(len(hist) >= 3 and hist[-1] > 1e-10
                   and hist[-1] > 0.5 * hist[max(0, len(hist) - 4)])
    sig = dict(**coeff_norms(model),
               newton_iters=len(hist), newton_hist=[float(h) for h in hist],
               newton_final_res=float(hist[-1]) if hist else None, stalled=stalled,
               fresh_res_phys=fresh_residual_nonlinear(prob, model, cfg),
               fresh_bc_err=condition_mismatch(prob, model),
               t_assemble=asm["t_assemble"], t_solve=t_solve)
    return dict(model=model, rel_l2=rel, linf=linf, signals=sig, W=model["W"])


def _model_from_geom(geom, a):
    m = dict(geom)
    m.pop("kind")
    m["sol"] = a
    return m


# ---------------------------------------------------------------------------
# reproduction gate
# ---------------------------------------------------------------------------

def reproduction_gate(verbose=True):
    """common.py must reproduce known expF01/expF02 cells before any experiment runs.

    2D cells reuse the exact recorded (W, lam, seed) so agreement should be near-exact;
    1D uses a different (fixed, finer) eval grid, hence the 2.5x tolerance.
    """
    import json
    checks = []
    f01 = load_problem_suite("f01")
    d01 = json.loads((REPO_ROOT / "results/checkpoint_F_applications/expF01_linear_de_zoo/data.json").read_text())

    def cell(data, key, **match):
        for c in data[key]["cells"]:
            if all(c[k] == v for k, v in match.items()):
                return c
        raise KeyError((key, match))

    c = cell(d01, "ode1d_o3", N=32)
    r = solve_linear(f01.PROBLEMS["ode1d_o3"], 32, DEFAULT_CFG)
    checks.append(("f01 ode1d_o3 N=32", c["rel_l2"], r["rel_l2"]))

    c = cell(d01, "pde2d_steady_o2", N=2304)
    r = solve_linear(f01.PROBLEMS["pde2d_steady_o2"], 2304, replace(DEFAULT_CFG, lam=c["lam"]))
    checks.append(("f01 steady_o2 W=2304", c["rel_l2"], r["rel_l2"]))

    c = cell(d01, "pde2d_time_o3", N=1024)
    r = solve_linear(f01.PROBLEMS["pde2d_time_o3"], 1024, replace(DEFAULT_CFG, lam=c["lam"]))
    checks.append(("f01 time_o3 W=1024", c["rel_l2"], r["rel_l2"]))

    f02 = load_problem_suite("f02")
    d02 = json.loads((REPO_ROOT / "results/checkpoint_F_applications/expF02_nonlinear_de_zoo/data.json").read_text())
    c = cell(d02, "pde2d_steady_o2", N=2304)
    r = solve_nonlinear(f02.PROBLEMS["pde2d_steady_o2"], 2304, replace(DEFAULT_CFG, lam=c["lam"]))
    checks.append(("f02 Monge-Ampere W=2304", c["rel_l2"], r["rel_l2"]))

    ok = True
    for name, want, got in checks:
        ratio = got / want if want > 0 else np.inf
        good = 0.4 < ratio < 2.5
        ok &= good
        if verbose:
            print(f"  gate {name}: recorded {want:.2e} reproduced {got:.2e} "
                  f"ratio {ratio:.2f} {'ok' if good else 'FAIL'}", flush=True)
    if not ok:
        raise AssertionError("reproduction gate failed -- common.py does not match expF01/expF02")
    return checks
