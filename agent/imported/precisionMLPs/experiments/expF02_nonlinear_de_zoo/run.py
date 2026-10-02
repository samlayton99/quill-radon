"""Experiment expF02 -- nonlinear differential-equation zoo: Newton, one lstsq per step.

Nine classical nonlinear problems (problems.py): logistic / Bratu / Blasius (1D),
eikonal / Monge-Ampere / third-order Burgers-type stress (2D steady on the disk),
inviscid Burgers / viscous Burgers (Cole-Hopf front) / KdV soliton (space-time).

Same frozen geometry, dictionary, collocation, and sweep design as expF01. The
nonlinear residual r(a) = [lin rows] a + N(fields) - f is solved by damped
Gauss-Newton: each step linearizes the PDE around the current iterate (a
variable-coefficient linear PDE for the correction) and solves it with one
min-norm lstsq; backtracking halves the step if the stacked residual grows.
Newton starts from zero except eikonal (harmonic extension of boundary data)
and Monge-Ampere (classical initializer Delta u = 2 sqrt(f)), each one linear solve.

Outputs (results/checkpoint_F_applications/expF02_nonlinear_de_zoo/): same as
expF01 -- error_vs_width.png (3x3), function_representations/{ode1d,pde2d_steady}
order PNGs + pde2d_time order GIFs, data.json (per-cell errors + Newton iterations).

Usage:
    uv run --extra dev python experiments/expF02_nonlinear_de_zoo/run.py [--smoke] [--plot]
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

from problems import PROBLEMS, CATEGORIES, verify_all

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF02_nonlinear_de_zoo"
REPR_DIR = RESULTS_DIR / "function_representations"
DATA_PATH = RESULTS_DIR / "data.json"

SMOKE = "--smoke" in sys.argv
PLOT_ONLY = "--plot" in sys.argv

LAM_1D = 0.25
N_GRID_1D = [8, 16, 32, 64, 128, 256, 512, 1024] if not SMOKE else [8, 32, 128]
W_GRID_2D = [144, 256, 576, 1024, 2304] if not SMOKE else [144, 400]
ANCHOR_W_2D = 1024 if not SMOKE else 400
LAM_GRID_2D = [0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35]
RCOND = 1e-13
COLLAR = {"pde2d_steady": 1.25, "pde2d_time": 1.6}
N_EVAL_2D = 241
SEED = 42
MAX_NEWTON = 25

# ---------------------------------------------------------------------------
# feature machinery (as in expF01)
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


MONO_1D = [0, 1, 2, 3]
MONO_2D = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
           (3, 0), (2, 1), (1, 2), (0, 3)]


def geometry_1d(N, lam):
    h = 2.0 / N
    R = max(70, int(np.ceil(0.4 * N)))
    centers = -1.0 + h * np.arange(-R, N + R + 1)
    return centers, lam / h


def radon_geometry(W, lam, collar):
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


def rows_1d(x, centers, gamma, terms):
    t = np.tanh(gamma * (x[:, None] - centers[None, :]))
    A = np.zeros_like(t)
    polys = np.zeros((len(x), len(MONO_1D)))
    for o, coeff in terms:
        cc = _coeff_col(coeff, x)
        A += cc * (gamma ** o) * psi(o, t)
        for k, d in enumerate(MONO_1D):
            mono = _ffact(d, o) * x ** (d - o) if o <= d else np.zeros_like(x)
            polys[:, k] += (cc.ravel() if np.ndim(cc) else cc) * mono
    return np.hstack([A, polys])


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


def interior_points_2d(category, W, rng):
    n = max(5 * W, 2000)
    if category == "pde2d_steady":
        rr = np.sqrt(rng.random(n))
        th = 2 * np.pi * rng.random(n)
        return np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
    return rng.uniform(-1, 1, (n, 2))


# ---------------------------------------------------------------------------
# damped Gauss-Newton on the readout
# ---------------------------------------------------------------------------

def _stacked_res_norm(a, lin_rows, D, nl, pts, f, bcs, s):
    vals = {idx: D[idx] @ a for idx in nl["fields"]}
    r_pde = (lin_rows @ a if lin_rows is not None else 0.0) + nl["res"](vals, pts) - f
    parts = [r_pde / s]
    parts += [w * (B @ a - g) for (B, g, w) in bcs]
    r = np.concatenate(parts)
    return r, float(np.linalg.norm(r)), float(np.max(np.abs(r_pde)))


def gauss_newton(prob, lin_rows, D, pts, f, bcs, a0):
    """Damped GN. bcs: list of (rows, values, weight). Returns (a, history)."""
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
        # backtracking line search on the stacked residual norm
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
# per-category solve
# ---------------------------------------------------------------------------

def _needed_indices(prob):
    idxs = {i for i, _ in prob["lin_terms"]}
    idxs |= set(prob["nl"]["fields"])
    return sorted(idxs, key=lambda i: (i if isinstance(i, tuple) else (i,)))


def solve_1d(prob, N, lam=LAM_1D):
    centers, gamma = geometry_1d(N, lam)
    W = len(centers)
    n_col = 4 * W
    x = np.linspace(-1, 1, n_col)
    D = {i: rows_1d(x, centers, gamma, [(i, 1.0)]) for i in _needed_indices(prob)}
    lin_rows = None
    for o, c in prob["lin_terms"]:
        lin_rows = (0.0 if lin_rows is None else lin_rows) + c * D[o]
    f = prob["forcing"]
    fv = f(x) if callable(f) else np.full(n_col, float(f))
    conds = prob["conditions"]
    w_bc = np.sqrt(n_col / len(conds))
    bcs = []
    for (x0, o) in conds:
        B = rows_1d(np.array([x0]), centers, gamma, [(o, 1.0)])
        val = float(np.asarray(prob["exact_derivs"][o](np.array([x0]))).ravel()[0])
        bcs.append((B, np.array([val]), w_bc))
    a0 = np.zeros(W + len(MONO_1D))
    a, hist = gauss_newton(prob, lin_rows, D, x, fv, bcs, a0)
    return dict(centers=centers, gamma=gamma, sol=a, W=W, n_col=n_col, iters=len(hist),
                hist=hist)


def solve_2d(prob, Wreq, lam, seed=SEED):
    rng = np.random.default_rng(seed)
    cat = prob["category"]
    dirs, offs, gamma = radon_geometry(Wreq, lam, COLLAR[cat])
    P = interior_points_2d(cat, Wreq, rng)
    D = {i: rows_2d(P, dirs, offs, gamma, [(i, 1.0)]) for i in _needed_indices(prob)}
    lin_rows = None
    for idx, c in prob["lin_terms"]:
        lin_rows = (0.0 if lin_rows is None else lin_rows) + c * D[idx]
    f = prob["forcing"]
    fv = f(P) if callable(f) else np.full(len(P), float(f))
    bcs = []
    for blk in prob["bc_blocks"]:
        # a block may carry its own point generator (e.g. the eikonal inflow arc)
        Pb = blk["points"]() if "points" in blk else bc_points_2d(blk["where"])
        B = rows_2d(Pb, dirs, offs, gamma, blk["terms"])
        w = np.sqrt(len(P) / len(Pb))
        bcs.append((B, blk["value"](Pb), w))
    # Newton starting point: zero, or one linear pre-solve (init spec)
    a0 = np.zeros(len(offs) + len(MONO_2D))
    if prob.get("init"):
        init = prob["init"]
        A0 = rows_2d(P, dirs, offs, gamma, init["terms"])
        r0 = init["rhs"](P) if callable(init["rhs"]) else np.full(len(P), float(init["rhs"]))
        s0 = max(np.abs(A0).max(), 1e-300)
        Ai = np.vstack([A0 / s0] + [w * B for (B, g, w) in bcs])
        yi = np.concatenate([r0 / s0] + [w * g for (B, g, w) in bcs])
        a0 = np.linalg.lstsq(Ai, yi, rcond=RCOND)[0]
    a, hist = gauss_newton(prob, lin_rows, D, P, fv, bcs, a0)
    return dict(dirs=dirs, offs=offs, gamma=gamma, sol=a, W=len(offs), iters=len(hist),
                hist=hist)


def eval_1d(model, x):
    t = np.tanh(model["gamma"] * (x[:, None] - model["centers"][None, :]))
    W = model["W"]
    out = t @ model["sol"][:W]
    for k, d in enumerate(MONO_1D):
        out += model["sol"][W + k] * x ** d
    return out


def eval_2d(model, P, chunk=4096):
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


def eval_grid_2d(category):
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


def eval_problem(prob, model):
    if prob["category"] == "ode1d":
        x = np.linspace(-1, 1, 3 * model["n_col"] + 1)
        return metrics(eval_1d(model, x), prob["exact"](x))
    P, _ = eval_grid_2d(prob["category"])
    return metrics(eval_2d(model, P), prob["exact"](P))


# ---------------------------------------------------------------------------
# sweep (same anchored-lambda policy as expF01)
# ---------------------------------------------------------------------------

def run_sweep():
    data = {}
    for key, prob in PROBLEMS.items():
        cells = []
        t0 = time.time()
        if prob["category"] == "ode1d":
            for N in N_GRID_1D:
                model = solve_1d(prob, N)
                rel, linf = eval_problem(prob, model)
                cells.append(dict(N=N, width=model["W"], lam=LAM_1D, rel_l2=rel,
                                  linf=linf, iters=model["iters"]))
                print(f"  {key} N={N:5d} W={model['W']:5d} it={model['iters']:2d} "
                      f"relL2={rel:.2e} Linf={linf:.2e}", flush=True)
        else:
            anchor = []
            for lam in LAM_GRID_2D:
                model = solve_2d(prob, ANCHOR_W_2D, lam)
                rel, _ = eval_problem(prob, model)
                anchor.append((lam, rel))
                print(f"  {key} anchor W={ANCHOR_W_2D} lam={lam:.2f} it={model['iters']:2d} "
                      f"relL2={rel:.2e}", flush=True)
            lam0 = min(anchor, key=lambda c: c[1])[0]
            for W in W_GRID_2D:
                best = None
                for lam in sorted({max(0.05, lam0 - 0.04), lam0, lam0 + 0.04}):
                    model = solve_2d(prob, W, lam)
                    rel, linf = eval_problem(prob, model)
                    if best is None or rel < best["rel_l2"]:
                        best = dict(N=W, width=model["W"], lam=lam, rel_l2=rel,
                                    linf=linf, iters=model["iters"])
                cells.append(best)
                print(f"  {key} W={W:5d} best lam={best['lam']:.2f} it={best['iters']:2d} "
                      f"relL2={best['rel_l2']:.2e} Linf={best['linf']:.2e}", flush=True)
        data[key] = dict(cells=cells, seconds=round(time.time() - t0, 1))
        print(f"{key} done in {data[key]['seconds']}s", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=1))
    return data


# ---------------------------------------------------------------------------
# figures (same layout as expF01)
# ---------------------------------------------------------------------------

CAT_LABEL = {"ode1d": "1D nonlinear ODE", "pde2d_steady": "2D steady nonlinear PDE",
             "pde2d_time": "space-time nonlinear PDE"}


def plot_grid(data):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5))
    for col, cat in enumerate(CATEGORIES):
        for row, order in enumerate([1, 2, 3]):
            key = f"{cat}_o{order}"
            prob = PROBLEMS[key]
            cells = data[key]["cells"]
            wid = [c["width"] for c in cells]
            ax = axes[row, col]
            ax.loglog(wid, [c["rel_l2"] for c in cells], "o-", color="C0",
                      label="rel $L_2$")
            ax.loglog(wid, [c["linf"] for c in cells], "s--", color="C1",
                      label="$L_\\infty$")
            ax.set_title(prob["title"], fontsize=10)
            ax.grid(True, which="both", alpha=0.25)
            ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
            if row == 2:
                ax.set_xlabel("width $W$ (neurons)")
            if col == 0:
                ax.set_ylabel(f"order {order}\nerror (test grid)")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=2,
                      borderaxespad=0, fontsize=8, frameon=False)
    for col, cat in enumerate(CATEGORIES):
        axes[0, col].annotate(CAT_LABEL[cat], xy=(0.5, 1.32),
                              xycoords="axes fraction", ha="center",
                              fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = RESULTS_DIR / "error_vs_width.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def best_cell(data, key):
    return min(data[key]["cells"], key=lambda c: c["rel_l2"])


def repr_1d(data):
    outdir = REPR_DIR / "ode1d"
    outdir.mkdir(parents=True, exist_ok=True)
    for order in [1, 2, 3]:
        key = f"ode1d_o{order}"
        prob = PROBLEMS[key]
        cell = best_cell(data, key)
        model = solve_1d(prob, cell["N"])
        x = np.linspace(-1, 1, 3000)
        u_true, u_hat = prob["exact"](x), eval_1d(model, x)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
        ax1.plot(x, u_true, color="k", lw=2.2, label="exact $u^*$")
        ax1.plot(x, u_hat, color="C1", lw=1.1, ls="--", label="solved $\\hat u$")
        ax1.set_xlabel("$x$"); ax1.set_ylabel("$u$")
        ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                   borderaxespad=0, frameon=False)
        ax2.semilogy(x, np.abs(u_hat - u_true) + 1e-18, color="C3", lw=1.0,
                     label="$|\\hat u - u^*|$")
        ax2.set_xlabel("$x$"); ax2.set_ylabel("abs error")
        ax2.grid(True, which="both", alpha=0.25)
        ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
                   borderaxespad=0, frameon=False)
        fig.suptitle(f"{prob['title']}   W={cell['width']}, $\\lambda$={cell['lam']:.2f}, "
                     f"{cell['iters']} Newton steps, rel $L_2$={cell['rel_l2']:.1e}", y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / f"order{order}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {outdir}/order{order}.png", flush=True)


def repr_2d_steady(data):
    outdir = REPR_DIR / "pde2d_steady"
    outdir.mkdir(parents=True, exist_ok=True)
    for order in [1, 2, 3]:
        key = f"pde2d_steady_o{order}"
        prob = PROBLEMS[key]
        cell = best_cell(data, key)
        model = solve_2d(prob, cell["N"], cell["lam"])
        P, (GX, GY, mask) = eval_grid_2d("pde2d_steady")
        u_true, u_hat = prob["exact"](P), eval_2d(model, P)
        U = np.full(GX.shape, np.nan); E = np.full(GX.shape, np.nan)
        U[mask] = u_true
        E[mask] = np.log10(np.abs(u_hat - u_true) + 1e-18)
        fig = plt.figure(figsize=(12.5, 4.6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(GX, GY, U, cmap="viridis", rstride=3, cstride=3,
                         linewidth=0, antialiased=True)
        ax1.set_title("exact $u^*$ (solved overlays at $\\sim$fp64)", fontsize=9)
        ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$")
        ax2 = fig.add_subplot(1, 2, 2)
        pc = ax2.pcolormesh(GX, GY, E, shading="auto", cmap="magma")
        ax2.set_aspect("equal")
        ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$")
        ax2.set_title("$\\log_{10}|\\hat u - u^*|$", fontsize=10)
        fig.colorbar(pc, ax=ax2, shrink=0.9)
        fig.suptitle(f"{prob['title']}   W={cell['width']}, $\\lambda$={cell['lam']:.2f}, "
                     f"{cell['iters']} Newton steps, rel $L_2$={cell['rel_l2']:.1e}", y=1.0)
        fig.tight_layout()
        fig.savefig(outdir / f"order{order}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {outdir}/order{order}.png", flush=True)


def repr_2d_time(data):
    outdir = REPR_DIR / "pde2d_time"
    outdir.mkdir(parents=True, exist_ok=True)
    nx, nt = 401, 61
    xg = np.linspace(-1, 1, nx)
    tg = np.linspace(0, 1, nt)
    for order in [1, 2, 3]:
        key = f"pde2d_time_o{order}"
        prob = PROBLEMS[key]
        cell = best_cell(data, key)
        model = solve_2d(prob, cell["N"], cell["lam"])
        TT, XX = np.meshgrid(tg, xg)
        P = np.stack([XX.ravel(), 2 * TT.ravel() - 1], axis=1)
        u_true = prob["exact"](P).reshape(nx, nt)
        u_hat = eval_2d(model, P).reshape(nx, nt)
        err = np.abs(u_hat - u_true) + 1e-18
        ylo = min(u_true.min(), u_hat.min()); yhi = max(u_true.max(), u_hat.max())
        pad = 0.1 * (yhi - ylo)
        elo = max(1e-17, err.min() / 10); ehi = err.max() * 10

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
        l_true, = ax1.plot(xg, u_true[:, 0], color="k", lw=2.2, label="exact $u^*$")
        l_hat, = ax1.plot(xg, u_hat[:, 0], color="C1", lw=1.1, ls="--",
                          label="solved $\\hat u$")
        ax1.set_xlim(-1, 1); ax1.set_ylim(ylo - pad, yhi + pad)
        ax1.set_xlabel("$x$"); ax1.set_ylabel("$u$")
        ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                   borderaxespad=0, frameon=False)
        l_err, = ax2.semilogy(xg, err[:, 0], color="C3", lw=1.0)
        ax2.set_xlim(-1, 1); ax2.set_ylim(elo, ehi)
        ax2.set_xlabel("$x$"); ax2.set_ylabel("$|\\hat u - u^*|$")
        ax2.grid(True, which="both", alpha=0.25)
        title = fig.suptitle("", y=0.97)
        fig.tight_layout(rect=(0, 0, 1, 0.88))

        def update(i):
            l_true.set_ydata(u_true[:, i])
            l_hat.set_ydata(u_hat[:, i])
            l_err.set_ydata(err[:, i])
            title.set_text(f"{prob['title']}   W={cell['width']}, "
                           f"$\\lambda$={cell['lam']:.2f}, {cell['iters']} Newton steps, "
                           f"rel $L_2$={cell['rel_l2']:.1e}   t={tg[i]:.2f}")
            return l_true, l_hat, l_err, title

        anim = FuncAnimation(fig, update, frames=nt, blit=False)
        out = outdir / f"order{order}.gif"
        anim.save(out, writer=PillowWriter(fps=12), dpi=90)
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
    repr_1d(data)
    repr_2d_steady(data)
    repr_2d_time(data)
    print("\nall outputs written to", RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
