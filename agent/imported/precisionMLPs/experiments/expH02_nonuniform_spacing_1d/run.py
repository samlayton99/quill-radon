"""expH02 -- smoothly non-uniform center spacing in 1-D, keeping lambda = 0.25 everywhere.

Question: if the centers are spaced smoothly non-uniformly, with every neuron's
bandwidth set from its *local* spacing so that lambda = gamma_j * h_j = 0.25 everywhere,
does the fixed-geometry least-squares fit still reach machine precision (about 1e-13
relative error) at the same width as evenly spaced centers?

Geometry
--------
Interior centers are placed so that equal fractions of a chosen "center distribution" q
on [-1, 1] lie between neighbors (c_j = Q^{-1}(j/N), Q^{-1} the inverse of q's cumulative
distribution). A parameter s in [0, 1] moves from evenly spaced centers (s = 0) to that
placement (s = 1) by placing the centers according to the mixture of densities

    q_s(x) = (1 - s) * 1/2  +  s * q(x),        c_j(s) = Q_s^{-1}(j / N),   j = 0..N,

so s = 0 is exactly the standard evenly spaced grid, s = 1 is the most regular set of
centers whose smoothed histogram would look like q, and every in-between row keeps a
uniform floor of (1 - s)/2 in its density. (Interpolating the center *positions* instead
gives a different in-between density; see center_placement.png, which overlays both.) Halo (the extra centers placed
beyond both ends of the interval, as in the standard construction): default_halo(N, 0.25)
of them on each side, continuing the spacing at that end. Local spacing by central
difference,

    h_j = (c_{j+1} - c_{j-1}) / 2      (one-sided at the two outer halo ends),

and gamma_j = lambda* / h_j with lambda* = 0.25. Output weights: least squares by
truncated SVD with a bias column, singular values below 1e-13 of the largest dropped
(the repo standard).

Center distributions
--------------------
  halfgauss  the left half of a Gaussian: density ~ exp(-t^2/2) with t running
             linearly from -1.5 (x = -1) to 0 (x = +1); spacing decreases smoothly
             left to right.
  bimodal    0.10 uniform + 0.90 * [0.5 N(-0.5, 0.2^2) + 0.5 N(+0.5, 0.2^2)] on [-1, 1].
  beta       Beta(2, 5) on x = (t + 1)/2: sharply asymmetric, peaked near x = -0.6.

Data
----
Four training sets per cell: uniform on [-1,1] (n = 2W and n = 16W) and sampled from
the row's own interpolated center distribution (same two sizes), W = N + 2R + 1.
The center-distribution sample is x = Q_s^{-1}(u), u ~ U(0,1), so its distribution is
exactly the one the centers follow.

Metric: relative L2 on a 4001-point uniform grid over [-1, 1] (misaligned with every
training set). The data file also stores the error on a sample from the row's center
distribution (eval="PX"), but only the uniform-grid error is plotted.

Deliverable: exactly three figures, spacing_{halfgauss,bimodal,beta}.png (4x4 each).
diagnostics/centers_and_gammas.png is a check of the geometry (centers, gamma_j, density per row).

Usage:
    uv run --extra dev python experiments/expH02_nonuniform_spacing_1d/run.py          # run + plot
    uv run --extra dev python experiments/expH02_nonuniform_spacing_1d/run.py --plot   # replot
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import default_halo   # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH02_nonuniform_spacing_1d"
DATA_PATH = RESULTS_DIR / "data.json"

LAMBDA_STAR = 0.25
RCOND = 1e-13
WIDTHS = [32, 64, 128, 256, 512]
S_LEVELS = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
S_LABEL = ["evenly spaced (s=0)", "one third of the way (s=1/3)",
           "two thirds of the way (s=2/3)", "fully non-uniform (s=1)"]
N_EVAL = 4001
LOW_MULT, HIGH_MULT = 2, 16
SEED = 0

# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------

def _packet(x):
    z = x
    return 0.55 * np.sin(np.pi * z - 0.3) + np.exp(-((z - 0.45) / 0.16) ** 2) * np.sin(12 * np.pi * (z - 0.45))


TARGETS = {
    "sine":      (lambda x: np.sin(2 * np.pi * x), r"$\sin(2\pi x)$"),
    "runge":     (lambda x: 1.0 / (1.0 + 25.0 * x ** 2), r"$1/(1+25x^2)$"),
    "sine_8pi":  (lambda x: np.sin(8 * np.pi * x), r"$\sin(8\pi x)$"),
    "packet":    (_packet, r"$.55\sin(\pi x-.3)+e^{-((x-.45)/.16)^2}\sin(12\pi(x-.45))$"),
}

# ---------------------------------------------------------------------------
# center distributions and their inverse cumulative distributions on [-1, 1]
# ---------------------------------------------------------------------------

_XG = np.linspace(-1.0, 1.0, 20001)


def _density(name: str, x: np.ndarray) -> np.ndarray:
    if name == "halfgauss":
        t = -1.5 + 1.5 * (x + 1.0) / 2.0          # t in [-1.5, 0]
        return np.exp(-0.5 * t * t)
    if name == "bimodal":
        g = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        return 0.10 * 0.5 + 0.90 * (0.5 * g(-0.5, 0.2) + 0.5 * g(0.5, 0.2))
    if name == "beta":
        u = (x + 1.0) / 2.0
        return np.clip(u, 0, 1) ** 1 * np.clip(1 - u, 0, 1) ** 4      # Beta(2,5) kernel
    raise KeyError(name)


class CenterDist:
    """A center distribution on [-1,1] with a numeric inverse cumulative distribution."""

    def __init__(self, name: str):
        self.name = name
        dens = _density(name, _XG)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(_XG))])
        self.cdf = cdf / cdf[-1]
        self.pdf = dens / np.trapezoid(dens, _XG)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(u, dtype=np.float64), self.cdf, _XG)

    def density(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, _XG, self.pdf)


DISTS = {name: CenterDist(name) for name in ("halfgauss", "bimodal", "beta")}
DIST_TITLE = {"halfgauss": "left half-Gaussian (density rises to the right)",
              "bimodal": "bimodal (two Gaussians at $\\pm0.5$, $\\sigma=0.2$, plus 10% uniform)",
              "beta": "Beta(2,5) (peaked near $x=-0.6$)"}


_MIX_CACHE: dict = {}


def _mixture_cdf(dist: CenterDist, s: float) -> np.ndarray:
    """CDF on _XG of the mixture density (1-s)*uniform + s*q."""
    key = (dist.name, round(float(s), 12))
    if key not in _MIX_CACHE:
        dens = (1.0 - s) * 0.5 + s * dist.pdf
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(_XG))])
        _MIX_CACHE[key] = cdf / cdf[-1]
    return _MIX_CACHE[key]


def interp_quantile(dist: CenterDist, u: np.ndarray, s: float) -> np.ndarray:
    """Inverse cumulative distribution of the in-between density (1-s)*uniform + s*q.

    So the row's centers follow the plain mixture of densities: at s=0 uniform, at s=1
    exactly q, and in between a density that keeps the uniform floor (1-s)/2 everywhere.
    """
    u = np.asarray(u, dtype=np.float64)
    if s <= 0.0:
        return 2.0 * u - 1.0
    return np.interp(u, _mixture_cdf(dist, s), _XG)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def nonuniform_geometry(dist: CenterDist, s: float, N: int):
    """Centers (with halo), local spacings h_j, and gamma_j = lambda* / h_j."""
    R = default_halo(N, lambda_star=LAMBDA_STAR)
    j = np.arange(N + 1, dtype=np.float64)
    interior = interp_quantile(dist, j / N, s)             # c_0 = -1, c_N = +1
    hl = interior[1] - interior[0]
    hr = interior[-1] - interior[-2]
    left = interior[0] - hl * np.arange(R, 0, -1)
    right = interior[-1] + hr * np.arange(1, R + 1)
    c = np.concatenate([left, interior, right])
    h = np.empty_like(c)
    h[1:-1] = 0.5 * (c[2:] - c[:-2])                        # central difference
    h[0] = c[1] - c[0]
    h[-1] = c[-1] - c[-2]
    gamma = LAMBDA_STAR / h
    return c, h, gamma, R


def solve(Phi, y):
    A = np.hstack([Phi, np.ones((len(Phi), 1))])
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    keep = sv > RCOND * sv[0]
    inv = np.where(keep, 1.0 / np.where(keep, sv, 1.0), 0.0)
    sol = Vt.T @ (inv * (U.T @ y))
    return sol[:-1], sol[-1], int(keep.sum())


def features(x, c, gamma):
    return np.tanh(gamma[None, :] * (x[:, None] - c[None, :]))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_eval = np.linspace(-1.0, 1.0, N_EVAL)
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for dname, dist in DISTS.items():
        for si, s in enumerate(S_LEVELS):
            x_eval_p = interp_quantile(dist, rng.random(N_EVAL), s)     # dense sample from P_X
            for N in WIDTHS:
                c, h, gamma, R = nonuniform_geometry(dist, s, N)
                W = len(c)
                Phi_eval = features(x_eval, c, gamma)
                Phi_eval_p = features(x_eval_p, c, gamma)
                data_sets = {}
                for mult, tag in ((LOW_MULT, "low"), (HIGH_MULT, "high")):
                    n = mult * W
                    data_sets[f"uniform_{tag}"] = rng.uniform(-1.0, 1.0, n)
                    data_sets[f"center_{tag}"] = interp_quantile(dist, rng.random(n), s)
                for dkey, x_tr in data_sets.items():
                    Phi_tr = features(x_tr, c, gamma)
                    for tname, (f, _) in TARGETS.items():
                        y = f(x_tr)
                        v, b, rank = solve(Phi_tr, y)
                        for ekey, (Pe, xe) in {"U": (Phi_eval, x_eval), "PX": (Phi_eval_p, x_eval_p)}.items():
                            ft = f(xe)
                            err = Pe @ v + b - ft
                            rows.append({"dist": dname, "s": s, "s_idx": si, "N": N, "W": W, "R": R,
                                         "data": dkey, "n_train": len(x_tr), "target": tname,
                                         "eval": ekey,
                                         "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(ft)),
                                         "linf": float(np.abs(err).max()),
                                         "rank": rank, "w_norm": float(np.linalg.norm(v)),
                                         "h_min": float(h.min()), "h_max": float(h.max()),
                                         "h_ratio_max": float(np.max(h[1:] / h[:-1]).item()
                                                              if len(h) > 1 else 1.0)})
                print(f"{dname:9s} s={s:.2f} N={N:4d} W={W:4d} | " +
                      " ".join(f"{r['target'][:5]}/{r['data'][:3]}{r['data'][-1]}={r['rel_l2']:.1e}"
                               for r in rows[-32:] if r["eval"] == "U" and r["target"] in ("sine", "packet")
                               and r["data"] in ("uniform_high", "center_high"))
                      + f" | {time.time()-t0:.0f}s", flush=True)
    with open(DATA_PATH, "w") as f:
        json.dump({"widths": WIDTHS, "s_levels": S_LEVELS, "rows": rows}, f)
    print(f"saved {DATA_PATH} ({time.time()-t0:.0f}s)")
    return rows


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

LINE = {"uniform_low": ("#1f77b4", "--", "uniform data, n=2W"),
        "uniform_high": ("#1f77b4", "-", "uniform data, n=16W"),
        "center_low": ("#d62728", "--", "center-distribution data, n=2W"),
        "center_high": ("#d62728", "-", "center-distribution data, n=16W")}


def plot_grid(rows, dname, eval_key="U"):
    fig, axes = plt.subplots(4, 4, figsize=(15, 13), sharex=True, sharey=True)
    for si, s in enumerate(S_LEVELS):
        for ti, (tname, (_, tlabel)) in enumerate(TARGETS.items()):
            ax = axes[si][ti]
            for dkey, (color, ls, label) in LINE.items():
                ys = [next(r["rel_l2"] for r in rows if r["dist"] == dname and r["s_idx"] == si
                           and r["N"] == N and r["data"] == dkey and r["target"] == tname
                           and r["eval"] == eval_key) for N in WIDTHS]
                ax.loglog(WIDTHS, ys, ls, marker="o", ms=3.5, color=color, lw=1.4, label=label)
            ax.set_ylim(1e-15, 1e1)
            ax.set_xticks(WIDTHS)
            ax.set_xticklabels([str(w) for w in WIDTHS])
            ax.grid(alpha=0.3, which="both")
            if si == 0:
                ax.set_title(tlabel, fontsize=9)
            if ti == 0:
                ax.set_ylabel(f"{S_LABEL[si]}\nrelative $L_2$ error", fontsize=9)
            if si == 3:
                ax.set_xlabel("width $N$ (interior centers)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4, fontsize=10,
               frameon=False)
    fig.suptitle(f"expH02: non-uniform center spacing, {DIST_TITLE[dname]}; $\\lambda=\\gamma_j h_j=0.25$, "
                 f"$h_j=(c_{{j+1}}-c_{{j-1}})/2$, standard halo; relative $L_2$ error on a uniform grid over $[-1,1]$",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = RESULTS_DIR / f"spacing_{dname}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_geometry():
    """Companion figure: density, centers, and gamma_j per row for each distribution (N=128)."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.5))
    N = 128
    for di, (dname, dist) in enumerate(DISTS.items()):
        for si, s in enumerate(S_LEVELS):
            ax = axes[di][si]
            c, h, gamma, R = nonuniform_geometry(dist, s, N)
            inner = slice(R, R + N + 1)
            ax.plot(c[inner], gamma[inner], "-", color="#1f77b4", lw=1.4, label=r"$\gamma_j=\lambda^*/h_j$")
            ax.set_yscale("log")
            ax.set_ylim(1, 1e4)
            ax2 = ax.twinx()
            xg = np.linspace(-1, 1, 801)
            u = np.linspace(0, 1, 801)
            xq = interp_quantile(dist, u, s)
            dens = np.gradient(u, xq)
            ax2.plot(xq, dens, color="#d62728", lw=1.2, label="row center density")
            ax2.set_ylim(0, None)
            ax2.set_yticks([])
            ax.plot(c[inner], np.full(N + 1, 1.3), "|", color="k", ms=8, label="centers")
            ax.set_xlim(-1.05, 1.05)
            ax.grid(alpha=0.3)
            if di == 0:
                ax.set_title(S_LABEL[si], fontsize=10)
            if si == 0:
                ax.set_ylabel(f"{dname}\n$\\gamma_j$ (log)", fontsize=9)
            if di == 0 and si == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3,
                           frameon=False, fontsize=10)
    fig.suptitle(f"expH02 geometry at $N={N}$: interior centers (ticks), local $\\gamma_j$ (blue, log), "
                 f"and the row's center density (red)", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    (RESULTS_DIR / "diagnostics").mkdir(exist_ok=True)
    out = RESULTS_DIR / "diagnostics" / "centers_and_gammas.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    if "--plot" in sys.argv:
        rows = json.load(open(DATA_PATH))["rows"]
    else:
        rows = run()
    for dname in DISTS:
        print("saved", plot_grid(rows, dname))
    print("saved", plot_geometry())


# ---------------------------------------------------------------------------
# fourth figure: do the placed centers really follow the intended density,
# and is the spacing regular?
# ---------------------------------------------------------------------------

def plot_center_placement(N: int = 128):
    """4 rows (s levels) x 3 columns (center distributions).

    Each panel: the interior centers as ticks; a density-normalized histogram of
    those centers (grey); the intended density (1-s)*uniform + s*q that the centers are
    placed by (red, solid); and, for the record, the density that interpolating the
    center positions would have produced instead (black, dashed). The two agree at
    s = 0 and s = 1 and differ in between. The panel text gives the largest ratio of
    neighboring spacings and the L1 distance between the histogram and the red curve.
    """
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True)
    xg = np.linspace(-1, 1, 801)
    for ci, (dname, dist) in enumerate(DISTS.items()):
        for si, s in enumerate(S_LEVELS):
            ax = axes[si][ci]
            c, h, gamma, R = nonuniform_geometry(dist, s, N)
            inner = c[R:R + N + 1]
            # the mixture density the centers are placed by
            u = np.linspace(0, 1, 4001)
            xq = xg
            dens_built = (1 - s) * 0.5 + s * dist.density(xg)
            # for the record: the density that interpolating the *positions* would give
            xpos = (1.0 - s) * (2.0 * u - 1.0) + s * dist.quantile(u)
            dens_mix = np.interp(xg, xpos, np.gradient(u, xpos))
            ax.hist(inner, bins=32, range=(-1, 1), density=True, color="#bbbbbb",
                    label="histogram of placed centers")
            kde = gaussian_kde(inner, bw_method=0.08)(xg)
            ax.plot(xq, dens_built, color="#d62728", lw=1.6, label=r"intended density $(1-s)\,\mathrm{unif}+s\,q$")
            ax.plot(xg, dens_mix, color="k", lw=1.2, ls="--", label="if positions were interpolated instead")
            ax.plot(inner, np.full_like(inner, -0.06 * max(dens_built.max(), 1)), "|", color="#1f77b4",
                    ms=9, label="centers")
            l1 = float(np.trapezoid(np.abs(kde - np.interp(xg, xq, dens_built)), xg))
            ratio = float(np.max(h[R + 1:R + N + 1] / h[R:R + N]))
            ax.text(0.02, 0.97, f"max $h_{{j+1}}/h_j$ = {ratio:.2f}\nL1(KDE, red) = {l1:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8.5)
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-0.1 * max(dens_built.max(), 1), 1.15 * max(dens_built.max(), dens_mix.max()))
            ax.grid(alpha=0.3)
            if si == 0:
                ax.set_title(DIST_TITLE[dname], fontsize=9.5)
            if ci == 0:
                ax.set_ylabel(f"{S_LABEL[si]}\ndensity", fontsize=9)
            if si == 3:
                ax.set_xlabel("$x$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4,
               fontsize=10, frameon=False)
    fig.suptitle(f"expH02: where the centers are placed ($N={N}$) and the density they follow",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = RESULTS_DIR / "center_placement.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("saved", plot_center_placement())


# ---------------------------------------------------------------------------
# residual figures at a fixed width: signed error vs x, one 4x4 grid per distribution
# ---------------------------------------------------------------------------

def plot_residuals(N: int = 128, seed: int = SEED):
    """For each center distribution, a 4x4 grid (rows = s, columns = functions) of the
    signed residual fit(x) - f(x) on a symmetric-log axis, one line per training set.
    The centers of that row are drawn as a rug along the bottom of each panel, so the
    sparse and dense regions can be read directly against the error."""
    x = np.linspace(-1.0, 1.0, 2001)
    rng = np.random.default_rng(seed + 1)
    outs = []
    for dname, dist in DISTS.items():
        fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharex=True, sharey=True)
        for si, s in enumerate(S_LEVELS):
            c, h, gamma, R = nonuniform_geometry(dist, s, N)
            W = len(c)
            Phi_x = features(x, c, gamma)
            fits = {}
            for mult, tag in ((LOW_MULT, "low"), (HIGH_MULT, "high")):
                n = mult * W
                fits[f"uniform_{tag}"] = rng.uniform(-1.0, 1.0, n)
                fits[f"center_{tag}"] = interp_quantile(dist, rng.random(n), s)
            for ti, (tname, (f, tlabel)) in enumerate(TARGETS.items()):
                ax = axes[si][ti]
                for dkey, (color, ls, label) in LINE.items():
                    x_tr = fits[dkey]
                    v, b, _ = solve(features(x_tr, c, gamma), f(x_tr))
                    r = Phi_x @ v + b - f(x)
                    ax.plot(x, r, ls, color=color, lw=0.9, alpha=0.85, label=label)
                ax.set_yscale("symlog", linthresh=1e-15, linscale=0.4)
                ax.set_ylim(-1.0, 1.0)
                ax.set_yticks([-1e-3, -1e-6, -1e-9, -1e-12, 0, 1e-12, 1e-9, 1e-6, 1e-3])
                ax.axhline(0.0, color="k", lw=0.6)
                inner = c[R:R + N + 1]
                ax.plot(inner, np.full_like(inner, -0.6), "|", color="#555555", ms=7, alpha=0.7)
                ax.set_xlim(-1.02, 1.02)
                ax.grid(alpha=0.25, which="major")
                if si == 0:
                    ax.set_title(tlabel, fontsize=9)
                if ti == 0:
                    ax.set_ylabel(f"{S_LABEL[si]}\nfit $-$ true", fontsize=9)
                if si == 3:
                    ax.set_xlabel("$x$")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4,
                   fontsize=10, frameon=False)
        fig.suptitle(f"expH02: signed residual at $N={N}$, {DIST_TITLE[dname]}; "
                     f"$\\lambda=\\gamma_j h_j=0.25$; grey ticks at $y=-0.6$ mark the row's centers",
                     fontsize=11, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = RESULTS_DIR / f"residual_N{N}_{dname}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        outs.append(out)
    return outs


if __name__ == "__main__":
    for o in plot_residuals():
        print("saved", o)
