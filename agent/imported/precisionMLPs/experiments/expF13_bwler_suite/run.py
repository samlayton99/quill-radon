"""Experiment expF13 -- the BWLer PDE suite under the frozen-geometry solve.

Runs the five BWLer benchmarks (convection c=40/80, reaction, wave, Burgers
nu=0.01/pi, Poisson-CG) plus a manufactured Poisson control through the
expF01/expF02 recipe: frozen Radon tensor-ridge geometry on the square,
degree-3 polynomial supplement, one min-norm lstsq per (Gauss-Newton) step.
Nonlinear problems (reaction, Burgers) use damped Gauss-Newton; Burgers
additionally gets the expF06 vanishing-viscosity continuation ladder
(nu: 0.5 -> 0.01/pi, warm-started), since nu = 0.01/pi is convection-dominated
and develops a shock of width ~0.015.

lambda policy: anchored sweep (expF02 style, uses the eval metric) for the
closed-form problems; flat lambda = 0.25 (no-oracle) for Burgers and
Poisson-CG, whose references are external.

Outputs (results/checkpoint_F_applications/expF13_bwler_suite/):
  error_vs_width.png     -- per-problem rel L2 / L_inf vs width, BWLer Table-2
                            reference lines
  function_representations/{time gifs, poisson pngs, burgers png}
  data.json              -- per-cell errors, Newton iters, self-consistency,
                            Burgers continuation table

Usage:
    uv run --extra dev python experiments/expF13_bwler_suite/run.py [--smoke] [--plot]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from problems import (PROBLEMS, NU_BURGERS, HOLE_CENTERS, HOLE_RADIUS,
                      in_poisson_domain, load_burgers_reference,
                      load_poisson_reference, make_burgers, verify_all)

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF13_bwler_suite"
REPR_DIR = RESULTS_DIR / "function_representations"
DATA_PATH = RESULTS_DIR / "data.json"

SMOKE = "--smoke" in sys.argv
PLOT_ONLY = "--plot" in sys.argv

W_GRID = [576, 1024, 2304, 4096] if not SMOKE else [144, 400]
ANCHOR_W = 1024 if not SMOKE else 400
LAM_GRID = [0.12, 0.16, 0.20, 0.25, 0.30]
FLAT_LAM = 0.25
RCOND = 1e-15  # 1e-13 truncates modes these oscillatory targets need (measured 35x on wave)
COLLAR = 1.6
SEED = 42
MAX_NEWTON = 30 if not SMOKE else 12
NU_LADDER = ([0.5, 0.1, 0.05, 0.02, NU_BURGERS]
             if not SMOKE else [0.5, 0.1, NU_BURGERS])

# problems whose lambda is anchored on their own (closed-form) eval metric
ANCHORED = ["convection_c40", "convection_c80", "reaction", "wave", "poisson_man"]
FLAT = ["burgers", "poisson_cg"]

# ---------------------------------------------------------------------------
# feature machinery (verbatim from expF02)
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


MONO_2D = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
           (3, 0), (2, 1), (1, 2), (0, 3)]


def radon_geometry(W, lam, collar=COLLAR):
    J = int(round(np.sqrt(W)))
    M = W // J
    thetas = np.pi * np.arange(J) / J
    ts = np.linspace(-collar, collar, M)
    dirs = np.repeat(np.stack([np.cos(thetas), np.sin(thetas)], axis=1), M, axis=0)
    offs = np.tile(ts, J)
    h_ref = 2.8 / np.sqrt(J * M)
    return dirs, offs, lam / h_ref


def _coeff_col(coeff, pts):
    if callable(coeff):
        return np.asarray(coeff(pts), dtype=np.float64).reshape(-1, 1)
    return float(coeff)


def rows_2d(P, dirs, offs, gamma, terms):
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


# ---------------------------------------------------------------------------
# point sets
# ---------------------------------------------------------------------------

def interior_points(prob, W, rng):
    n = max(5 * W, 2000)
    P = rng.uniform(-1, 1, (n, 2))
    if prob["category"] == "steady":  # square minus holes
        P = P[in_poisson_domain(P)]
        while len(P) < n:
            Q = rng.uniform(-1, 1, (n, 2))
            P = np.vstack([P, Q[in_poisson_domain(Q)]])[:n]
    return P


def edge_points(where, n=480):
    s = np.linspace(-1, 1, n)
    if where == "ic":
        return np.stack([s, np.full(n, -1.0)], axis=1)
    if where == "left":
        return np.stack([np.full(n, -1.0), s], axis=1)
    if where == "right":
        return np.stack([np.full(n, 1.0), s], axis=1)
    raise ValueError(where)


def square_points(n_per_edge=160):
    s = np.linspace(-1, 1, n_per_edge, endpoint=False)
    return np.concatenate([
        np.stack([s, np.full_like(s, -1.0)], axis=1),
        np.stack([np.full_like(s, 1.0), s], axis=1),
        np.stack([-s, np.full_like(s, 1.0)], axis=1),
        np.stack([np.full_like(s, -1.0), -s], axis=1),
    ])


def hole_points(n_per_hole=120):
    ang = np.linspace(0, 2 * np.pi, n_per_hole, endpoint=False)
    ring = HOLE_RADIUS * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return np.concatenate([ring + c for c in HOLE_CENTERS])


def build_bcs(prob, dirs, offs, gamma, n_pde):
    """List of (rows, values, weight) blocks, incl. periodic difference rows."""
    bcs = []
    for blk in prob["bc_blocks"]:
        where = blk["where"]
        if where == "periodic_x":
            eta = np.linspace(-1, 1, 320)
            PL = np.stack([np.full_like(eta, -1.0), eta], axis=1)
            PR = np.stack([np.full_like(eta, 1.0), eta], axis=1)
            B = (rows_2d(PL, dirs, offs, gamma, blk["terms"])
                 - rows_2d(PR, dirs, offs, gamma, blk["terms"]))
            g = np.zeros(len(eta))
            n_b = len(eta)
        else:
            if where == "square":
                Pb = square_points()
            elif where == "holes":
                Pb = hole_points()
            else:
                Pb = edge_points(where)
            B = rows_2d(Pb, dirs, offs, gamma, blk["terms"])
            val = blk["value"]
            g = val(Pb) if callable(val) else np.full(len(Pb), float(val))
            n_b = len(Pb)
        bcs.append((B, g, np.sqrt(n_pde / n_b)))
    return bcs


# ---------------------------------------------------------------------------
# damped Gauss-Newton (verbatim from expF02)
# ---------------------------------------------------------------------------

def _stacked_res_norm(a, lin_rows, D, nl, pts, f, bcs, s):
    vals = {idx: D[idx] @ a for idx in nl["fields"]}
    r_pde = (lin_rows @ a if lin_rows is not None else 0.0) + nl["res"](vals, pts) - f
    parts = [r_pde / s]
    parts += [w * (B @ a - g) for (B, g, w) in bcs]
    r = np.concatenate(parts)
    return r, float(np.linalg.norm(r)), float(np.max(np.abs(r_pde)))


def gauss_newton(prob, lin_rows, D, pts, f, bcs, a0):
    nl = prob["nl"]
    a = a0.copy()
    history = []
    for it in range(MAX_NEWTON):
        vals = {idx: D[idx] @ a for idx in nl["fields"]}
        J_pde = lin_rows.copy() if lin_rows is not None else 0.0
        jd = nl["jac"](vals, pts)
        for idx, coef in jd.items():
            J_pde = J_pde + coef[:, None] * D[idx]
        s = max(np.abs(J_pde).max(), 1e-300)
        r, rnorm, rmax = _stacked_res_norm(a, lin_rows, D, nl, pts, f, bcs, s)
        history.append(rmax)
        J = np.vstack([J_pde / s] + [w * B for (B, g, w) in bcs])
        step = np.linalg.lstsq(J, -r, rcond=RCOND)[0]
        alpha = 1.0
        for _ in range(8):
            _, new_norm, _ = _stacked_res_norm(a + alpha * step, lin_rows, D, nl,
                                               pts, f, bcs, s)
            if new_norm <= rnorm * (1.0 - 1e-4 * alpha) or new_norm < 1e-14:
                break
            alpha *= 0.5
        a = a + alpha * step
        if (alpha * np.linalg.norm(step) < 1e-13 * max(1.0, np.linalg.norm(a))
                or (len(history) > 2 and history[-1] > 0
                    and abs(history[-2] / history[-1] - 1.0) < 1e-3)):
            break
    return a, history


# ---------------------------------------------------------------------------
# solves
# ---------------------------------------------------------------------------

def _needed_indices(prob):
    idxs = {i for i, _ in prob["lin_terms"]}
    idxs |= set(prob["nl"]["fields"])
    return sorted(idxs)


def assemble(prob, Wreq, lam, seed=SEED):
    rng = np.random.default_rng(seed)
    dirs, offs, gamma = radon_geometry(Wreq, lam)
    P = interior_points(prob, Wreq, rng)
    D = {i: rows_2d(P, dirs, offs, gamma, [(i, 1.0)]) for i in _needed_indices(prob)}
    bcs = build_bcs(prob, dirs, offs, gamma, len(P))
    return dirs, offs, gamma, P, D, bcs


def _lin_rows(prob, D):
    lin = None
    for idx, c in prob["lin_terms"]:
        lin = (0.0 if lin is None else lin) + c * D[idx]
    return lin


def solve(prob, Wreq, lam, seed=SEED):
    dirs, offs, gamma, P, D, bcs = assemble(prob, Wreq, lam, seed)
    f = prob["forcing"]
    fv = f(P) if callable(f) else np.full(len(P), float(f))
    a0 = np.zeros(len(offs) + len(MONO_2D))
    a, hist = gauss_newton(prob, _lin_rows(prob, D), D, P, fv, bcs, a0)
    return dict(dirs=dirs, offs=offs, gamma=gamma, sol=a, W=len(offs),
                iters=len(hist), hist=hist)


def solve_burgers(Wreq, lam, seed=SEED):
    """nu-continuation ladder, warm-started; D is nu-independent so it is built once."""
    prob0 = make_burgers(NU_LADDER[0])
    dirs, offs, gamma, P, D, bcs = assemble(prob0, Wreq, lam, seed)
    a = np.zeros(len(offs) + len(MONO_2D))
    rungs = []
    for nu in NU_LADDER:
        prob = make_burgers(nu)
        fv = np.zeros(len(P))
        a, hist = gauss_newton(prob, _lin_rows(prob, D), D, P, fv, bcs, a)
        rungs.append(dict(nu=float(nu), iters=len(hist), res=float(hist[-1])))
        print(f"    burgers W={Wreq} lam={lam:.2f} nu={nu:.5g} it={len(hist):2d} "
              f"res={hist[-1]:.2e}", flush=True)
    return dict(dirs=dirs, offs=offs, gamma=gamma, sol=a, W=len(offs),
                iters=sum(r["iters"] for r in rungs), rungs=rungs)


def eval_model(model, P, chunk=4096):
    W = model["W"]
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        Q = P[i:i + chunk]
        t = np.tanh(model["gamma"] * (Q @ model["dirs"].T - model["offs"][None, :]))
        v = t @ model["sol"][:W]
        for k, (px, py) in enumerate(MONO_2D):
            v += model["sol"][W + k] * Q[:, 0] ** px * Q[:, 1] ** py
        out[i:i + chunk] = v
    return out


# ---------------------------------------------------------------------------
# evaluation sets (fixed per problem; scaled coords)
# ---------------------------------------------------------------------------

def eval_set(prob):
    """(P_eval, u_true). Matches bwler's eval protocol per problem."""
    key = prob["key"]
    if key == "burgers":
        u_ref, t, x = load_burgers_reference()
        TT, XX = np.meshgrid(t, x, indexing="ij")
        P = np.stack([XX.ravel(), 2 * TT.ravel() - 1], axis=1)
        return P, u_ref.ravel()
    if key == "poisson_cg":
        Pn, vn = load_poisson_reference()
        return Pn, vn
    if key == "poisson_man":
        g = np.linspace(-1, 1, 241)
        GX, GY = np.meshgrid(g, g)
        P = np.stack([GX.ravel(), GY.ravel()], axis=1)
        P = P[in_poisson_domain(P)]
        return P, prob["exact"](P)
    # time problems: 200x200 uniform, as bwler
    g = np.linspace(-1, 1, 200)
    GX, GE = np.meshgrid(g, g)
    P = np.stack([GX.ravel(), GE.ravel()], axis=1)
    return P, prob["exact"](P)


def metrics(u_hat, u_true):
    err = u_hat - u_true
    return (float(np.linalg.norm(err) / np.linalg.norm(u_true)),
            float(np.max(np.abs(err))))


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

MODELS = {}  # key -> (cell, model): best model per problem, reused by the figures


def run_sweep():
    data = {}
    for key, prob in PROBLEMS.items():
        t0 = time.time()
        P_eval, u_true = eval_set(prob)
        cells, u_hats = [], []

        def record(model, Wreq, lam):
            u_hat = eval_model(model, P_eval)
            rel, linf = metrics(u_hat, u_true)
            cell = dict(N=Wreq, width=model["W"], lam=lam, rel_l2=rel, linf=linf,
                        iters=model["iters"])
            if "rungs" in model:
                cell["rungs"] = model["rungs"]
            return cell, u_hat

        if key in ANCHORED:
            anchor = []
            for lam in LAM_GRID:
                model = solve(prob, ANCHOR_W, lam)
                cell, _ = record(model, ANCHOR_W, lam)
                anchor.append((lam, cell["rel_l2"]))
                print(f"  {key} anchor W={ANCHOR_W} lam={lam:.2f} it={model['iters']:2d} "
                      f"relL2={cell['rel_l2']:.2e}", flush=True)
            lam0 = min(anchor, key=lambda c: c[1])[0]
            lam_sets = [[lam0]] * len(W_GRID)
        else:
            lam_sets = [[FLAT_LAM]] * len(W_GRID)

        for Wreq, lams in zip(W_GRID, lam_sets):
            best = None
            for lam in lams:
                model = (solve_burgers(Wreq, lam) if key == "burgers"
                         else solve(prob, Wreq, lam))
                cell, u_hat = record(model, Wreq, lam)
                if best is None or cell["rel_l2"] < best[0]["rel_l2"]:
                    best = (cell, u_hat)
                    if (key not in MODELS
                            or cell["rel_l2"] < MODELS[key][0]["rel_l2"]):
                        MODELS[key] = (cell, model)
            cells.append(best[0])
            u_hats.append(best[1])
            print(f"  {key} W={Wreq:5d} best lam={best[0]['lam']:.2f} "
                  f"it={best[0]['iters']:3d} relL2={best[0]['rel_l2']:.2e} "
                  f"Linf={best[0]['linf']:.2e}", flush=True)

        # nested-width self-consistency (the u*-free diagnostic, expF01)
        sc = [float(np.linalg.norm(u_hats[i + 1] - u_hats[i])
                    / np.linalg.norm(u_hats[i + 1]))
              for i in range(len(u_hats) - 1)]
        data[key] = dict(cells=cells, self_consistency=sc,
                         bwler_ref=prob["bwler_ref"],
                         seconds=round(time.time() - t0, 1))
        print(f"{key} done in {data[key]['seconds']}s  self-consistency={sc}\n",
              flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=1))
    return data


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

ORDERED = ["convection_c40", "convection_c80", "reaction", "wave",
           "burgers", "poisson_cg", "poisson_man"]


def plot_grid(data):
    # merge extended-width cells (extend_width.py) if present
    ext_path = RESULTS_DIR / "extend_width.json"
    ext = json.loads(ext_path.read_text()) if ext_path.exists() else {}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for k, key in enumerate(ORDERED):
        ax = axes[k // 4, k % 4]
        prob = PROBLEMS[key]
        cells = data[key]["cells"] + [
            dict(width=e["W"], rel_l2=e["rel_l2"], linf=e["linf"])
            for e in ext.get(key, [])]
        cells.sort(key=lambda c: c["width"])
        wid = [c["width"] for c in cells]
        ax.loglog(wid, [c["rel_l2"] for c in cells], "o-", color="C0",
                  label="rel $L_2$")
        ax.loglog(wid, [c["linf"] for c in cells], "s--", color="C1",
                  label="$L_\\infty$")
        main_wid = [c["width"] for c in data[key]["cells"]]
        if len(main_wid) > 1:
            ax.loglog(main_wid[1:], data[key]["self_consistency"], "^:",
                      color="C2", label="self-consist.")
        if data[key]["bwler_ref"] is not None:
            ax.axhline(data[key]["bwler_ref"], color="C3", lw=1.2, ls="--",
                       label="BWLer best")
        ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
        ax.set_title(prob["title"], fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("width $W$")
        if k % 4 == 0:
            ax.set_ylabel("error (eval set)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2,
                  borderaxespad=0, fontsize=7, frameon=False)
    axes[1, 3].axis("off")
    fig.tight_layout()
    out = RESULTS_DIR / "error_vs_width.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def best_cell(data, key):
    return min(data[key]["cells"], key=lambda c: c["rel_l2"])


def _resolve(key, cell):
    if key in MODELS:  # cached from the sweep; only --plot mode re-solves
        return MODELS[key][1]
    prob = PROBLEMS[key]
    return (solve_burgers(cell["N"], cell["lam"]) if key == "burgers"
            else solve(prob, cell["N"], cell["lam"]))


def repr_time(data):
    outdir = REPR_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    nx, nt = 401, 61
    xg = np.linspace(-1, 1, nx)     # scaled xi
    eg = np.linspace(-1, 1, nt)     # scaled eta
    for key in ["convection_c40", "convection_c80", "reaction", "wave", "burgers"]:
        prob = PROBLEMS[key]
        cell = best_cell(data, key)
        model = _resolve(key, cell)
        EE, XX = np.meshgrid(eg, xg)
        P = np.stack([XX.ravel(), EE.ravel()], axis=1)
        if key == "burgers":
            u_ref, tr, xr = load_burgers_reference()
            from scipy.interpolate import RegularGridInterpolator
            itp = RegularGridInterpolator((tr, xr), u_ref)
            u_true = itp(np.stack([0.5 * (P[:, 1] + 1.0), P[:, 0]], axis=1))
        else:
            u_true = prob["exact"](P)
        u_true = u_true.reshape(nx, nt)
        u_hat = eval_model(model, P).reshape(nx, nt)
        err = np.abs(u_hat - u_true) + 1e-18
        ylo, yhi = min(u_true.min(), u_hat.min()), max(u_true.max(), u_hat.max())
        pad = 0.1 * (yhi - ylo)
        elo, ehi = max(1e-17, err.min() / 10), err.max() * 10

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
        l_true, = ax1.plot(xg, u_true[:, 0], color="k", lw=2.2, label="reference $u^*$")
        l_hat, = ax1.plot(xg, u_hat[:, 0], color="C1", lw=1.1, ls="--",
                          label="solved $\\hat u$")
        ax1.set_xlim(-1, 1); ax1.set_ylim(ylo - pad, yhi + pad)
        ax1.set_xlabel("$\\xi$ (scaled $x$)"); ax1.set_ylabel("$u$")
        ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                   borderaxespad=0, frameon=False)
        l_err, = ax2.semilogy(xg, err[:, 0], color="C3", lw=1.0)
        ax2.set_xlim(-1, 1); ax2.set_ylim(elo, ehi)
        ax2.set_xlabel("$\\xi$"); ax2.set_ylabel("$|\\hat u - u^*|$")
        ax2.grid(True, which="both", alpha=0.25)
        title = fig.suptitle("", y=0.97)
        fig.tight_layout(rect=(0, 0, 1, 0.88))

        def update(i, l_true=l_true, l_hat=l_hat, l_err=l_err, title=title,
                   u_true=u_true, u_hat=u_hat, err=err, cell=cell, prob=prob):
            l_true.set_ydata(u_true[:, i])
            l_hat.set_ydata(u_hat[:, i])
            l_err.set_ydata(err[:, i])
            t_phys = 0.5 * (eg[i] + 1.0)
            title.set_text(f"{prob['title']}   W={cell['width']}, "
                           f"$\\lambda$={cell['lam']:.2f}, rel $L_2$="
                           f"{cell['rel_l2']:.1e}   t={t_phys:.2f}")
            return l_true, l_hat, l_err, title

        anim = FuncAnimation(fig, update, frames=nt, blit=False)
        out = outdir / f"{key}.gif"
        anim.save(out, writer=PillowWriter(fps=12), dpi=90)
        plt.close(fig)
        print(f"wrote {out}", flush=True)


def repr_poisson(data):
    outdir = REPR_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    for key in ["poisson_cg", "poisson_man"]:
        prob = PROBLEMS[key]
        cell = best_cell(data, key)
        model = _resolve(key, cell)
        g = np.linspace(-1, 1, 241)
        GX, GY = np.meshgrid(g, g)
        P = np.stack([GX.ravel(), GY.ravel()], axis=1)
        mask = in_poisson_domain(P)
        u_hat = np.full(len(P), np.nan)
        u_hat[mask] = eval_model(model, P[mask])
        U = u_hat.reshape(GX.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
        pc1 = ax1.pcolormesh(GX, GY, U, shading="auto", cmap="viridis")
        ax1.set_aspect("equal"); ax1.set_title("solved $\\hat u$", fontsize=10)
        fig.colorbar(pc1, ax=ax1, shrink=0.85)
        if key == "poisson_man":
            E = np.full(len(P), np.nan)
            E[mask] = np.log10(np.abs(u_hat[mask] - prob["exact"](P[mask])) + 1e-18)
            pc2 = ax2.pcolormesh(GX, GY, E.reshape(GX.shape), shading="auto",
                                 cmap="magma")
            ax2.set_title("$\\log_{10}|\\hat u - u^*|$", fontsize=10)
        else:
            Pn, vn = load_poisson_reference()
            e = np.abs(eval_model(model, Pn) - vn) + 1e-18
            pc2 = ax2.scatter(Pn[:, 0], Pn[:, 1], c=np.log10(e), s=14, cmap="magma")
            ax2.set_title("$\\log_{10}|\\hat u - u_{\\rm COMSOL}|$ at ref nodes "
                          "(float32 ceiling)", fontsize=10)
        ax2.set_aspect("equal")
        fig.colorbar(pc2, ax=ax2, shrink=0.85)
        fig.suptitle(f"{prob['title']}   W={cell['width']}, "
                     f"$\\lambda$={cell['lam']:.2f}, rel $L_2$={cell['rel_l2']:.1e}",
                     y=0.99)
        fig.tight_layout()
        out = outdir / f"{key}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}", flush=True)


def main():
    print("verifying problem definitions...", flush=True)
    verify_all(verbose=False)
    print("ok\n", flush=True)
    if PLOT_ONLY and DATA_PATH.exists():
        data = json.loads(DATA_PATH.read_text())
    else:
        data = run_sweep()
    plot_grid(data)
    repr_time(data)
    repr_poisson(data)
    print("\nall outputs written to", RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
