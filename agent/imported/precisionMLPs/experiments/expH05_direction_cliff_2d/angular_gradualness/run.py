"""expH05 follow-up: how gradual does the *angular* spacing have to be?

expH02 asked this for center spacing in 1-D: place the centers by the inverse CDF of a
mixture density ``q_s = (1-s) * uniform + s * q`` and see what survives. A smoothly
varying density cost nothing; a density that vanishes at an endpoint left one gap that
stayed ~4x its neighbour at every width, and the error stalled.

This asks the same question one level up, about the *directions* of a 2-D ridge network.
The angle of a direction ``v = (cos th, sin th)`` lives on the circle ``[0, pi)`` --
``th`` and ``th + pi`` are the same spoke -- so every density here is periodic with
period ``pi`` and every "gap" is measured with the wrap-around gap included.

Everything else is expH05's r = 0.4 setting, copied verbatim (see the block marked
"copied from expH05/run.py"): data uniform in the ball of radius r = 0.4 about
x0 = (0.35, -0.25); the ridge system recentered on x0; ``n_per = 128`` offsets on every
spoke at every M, ``t`` evenly spaced (cell-centered) over ``[-1.25 r, 1.25 r]``;
``gamma = 0.25 / h`` with ``h = 2 * 1.25 r / 128``; ``n_train = 8 * units`` points with
``units = 128 M``, seed 0; one truncated-SVD least squares with a bias column at
``rcond = 1e-13``; error = relative L2 on 20000 points uniform in the ball of radius
0.9 r. Only the *set of angles* changes.

The functions used from expH05 are copied rather than imported so this subfolder is
self-contained and cannot break when expH05/run.py is edited.

Angle placements
----------------
Figure 1 (``angle_spacing_vs_M.png``), four placements against M:
  even      th_i = (i + 1/2) pi / M                          (expH05's reference)
  smooth    th_i = Q^{-1}((i + 1/2)/M) for q(th) propto 1 + 0.8 cos(2(th - pi/4))
            -- smooth, 9x between its max and its min, never zero
  jitter    the even angles each shifted by N(0, (0.25 pi/M)^2), wrapped mod pi
  random    M angles uniform on [0, pi)
The two random placements use 3 seeds; the plot shows the median with a min/max band.

Figure 2 (``angle_spacing_h02_style.png``), expH02's mixture ladder: rows are
``s in {0, 1/3, 2/3, 1}`` with the angle density ``q_s = (1-s)/pi + s q``, angles at
``Q_s^{-1}((i + 1/2)/M)``; the three shapes q are
  lobe1  propto exp(1.5 cos(2(th - pi/4)))    one smooth lobe at 45 deg
  lobe2  propto exp(1.5 cos(4(th - pi/4)))    two smooth lobes at 45 and 135 deg
  zero   propto sin(th)^2                     zero at th = 0 (= pi), the analogue of
                                              expH02's Beta(2,5) endpoint
``s = 0`` is exactly the even reference for all three shapes, so that row is computed
once and reused.

Usage:
    OMP_NUM_THREADS=6 uv run --extra dev python \
        experiments/expH05_direction_cliff_2d/angular_gradualness/run.py
    uv run --extra dev python \
        experiments/expH05_direction_cliff_2d/angular_gradualness/run.py --plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "experiments" / "expH01_highdim_suite"))

from h01suite.metrics import error_metrics                       # noqa: E402

RESULTS_DIR = (REPO_ROOT / "results" / "checkpoint_H_highdim" /
               "expH05_direction_cliff_2d" / "angular_gradualness")
FIG_DIR = RESULTS_DIR / "figures"
DATA_JSON = RESULTS_DIR / "data.json"

# --- copied from expH05/run.py (unchanged) ---------------------------------
X0 = np.array([0.35, -0.25])          # center of the data ball
N_PER = 128                           # offsets per direction, every direction, every M
MARGIN = 1.25                         # collar beyond the data's projection band
LAMBDA = 0.25
RCOND = 1e-13
N_TRAIN_PER_UNIT = 8
N_TEST = 20000
TEST_SHRINK = 0.9                     # score only the inner 90% of the data ball
SEED_TRAIN, SEED_TEST = 0, 1
A_BUMP = np.array([0.2, 0.1])
A_RAD = np.array([0.3, -0.2])


def _rho(X, a):
    """Scaled radial distance ||x - a|| / sqrt(2), the suite's convention in d = 2."""
    dif = np.asarray(X, dtype=np.float64) - a[None, :]
    return np.sqrt(np.einsum("nk,nk->n", dif, dif) / 2.0)


def f_fast_waves(X):
    return np.cos(6 * np.pi * _rho(X, A_RAD))


def f_composition(X):
    return np.exp(np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]))


def f_radial_runge(X):
    return 1.0 / (1.0 + 16.0 * _rho(X, A_RAD) ** 2)


def ball(n, r, x0, rng):
    """``n`` points uniform in the ball of radius ``r`` about ``x0`` (d = 2)."""
    g = rng.normal(size=(n, 2))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    u = rng.uniform(size=(n, 1)) ** 0.5
    return x0 + r * u * g


def solve_many(Phi, Y, rcond=RCOND):
    """``_solve_svd`` for several right-hand sides at once (one SVD, many solves)."""
    A = np.hstack([Phi, np.ones((len(Phi), 1), dtype=np.float64)])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > rcond * s[0]
    s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    sol = Vt.T @ (s_inv[:, None] * (U.T @ np.asarray(Y, dtype=np.float64)))
    return sol[:-1], sol[-1], {"rank": int(keep.sum()), "n_cols": A.shape[1]}
# --- end of the copied block -----------------------------------------------


R_DATA = 0.4                          # the one data radius used everywhere here
N_DIRS = [4, 6, 8, 12, 16, 24, 32]
S_LEVELS = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
S_LABEL = ["even ($s=0$)", "one third of the way ($s=1/3$)",
           "two thirds of the way ($s=2/3$)", "fully non-uniform ($s=1$)"]
SEEDS = [0, 1, 2]
JITTER_FRAC = 0.25                    # jitter sd, as a fraction of the even step pi/M

TARGETS = [
    ("fast_waves", "fast concentric waves", r"$\cos(6\pi\rho)$", f_fast_waves),
    ("composition", "composition", r"$\exp(\sin(\pi x)\cos(\pi y))$", f_composition),
    ("radial_runge", "radial Runge", r"$1/(1+16\rho^2)$", f_radial_runge),
]


# ---------------------------------------------------------------------------
# angle densities on the circle [0, pi) and their inverse CDFs
# ---------------------------------------------------------------------------

_TH = np.linspace(0.0, np.pi, 20001)          # one full period


def _shape_density(name, th):
    """Unnormalized angle density, periodic with period pi."""
    if name == "smooth":                       # figure 1's gradual placement
        return 1.0 + 0.8 * np.cos(2.0 * (th - np.pi / 4))
    if name == "lobe1":                        # one smooth lobe at 45 deg
        return np.exp(1.5 * np.cos(2.0 * (th - np.pi / 4)))
    if name == "lobe2":                        # two smooth lobes at 45 and 135 deg
        return np.exp(1.5 * np.cos(4.0 * (th - np.pi / 4)))
    if name == "zero":                         # vanishes at th = 0 (= pi)
        return np.sin(th) ** 2
    raise KeyError(name)


SHAPE_TITLE = {
    "smooth": r"smooth: $q \propto 1 + 0.8\cos 2(\theta - 45^\circ)$",
    "lobe1": r"one lobe: $q \propto e^{1.5\cos 2(\theta-45^\circ)}$",
    "lobe2": r"two lobes: $q \propto e^{1.5\cos 4(\theta-45^\circ)}$",
    "zero": r"vanishing: $q \propto \sin^2\theta$  (zero at $0^\circ$)",
}
SHAPES_FIG2 = ["lobe1", "lobe2", "zero"]


class AngleDist:
    """A pi-periodic angle density with a numeric inverse CDF on [0, pi)."""

    def __init__(self, name):
        self.name = name
        dens = _shape_density(name, _TH)
        self.pdf = dens / np.trapezoid(dens, _TH)

    def mixture_pdf(self, s):
        return (1.0 - s) / np.pi + s * self.pdf

    def quantile(self, u, s):
        """Inverse CDF of ``q_s = (1-s)/pi + s q``.  s = 0 gives exactly ``pi u``."""
        u = np.asarray(u, dtype=np.float64)
        if s <= 0.0:
            return np.pi * u
        dens = self.mixture_pdf(s)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(_TH))])
        cdf /= cdf[-1]
        return np.interp(u, cdf, _TH)


DISTS = {name: AngleDist(name) for name in ("smooth", "lobe1", "lobe2", "zero")}


# ---------------------------------------------------------------------------
# angle placements
# ---------------------------------------------------------------------------

def even_angles(M):
    return (np.arange(M) + 0.5) * np.pi / M


def place_angles(kind, M, s=1.0, seed=0):
    """The M angles for one placement.  ``kind`` is 'even', 'jitter', 'random',
    or the name of an angle density (used with the mixture level ``s``)."""
    if kind == "even":
        return even_angles(M)
    if kind == "jitter":
        rng = np.random.default_rng(1000 * M + seed)
        th = even_angles(M) + rng.normal(0.0, JITTER_FRAC * np.pi / M, M)
        return np.sort(np.mod(th, np.pi))
    if kind == "random":
        rng = np.random.default_rng(5000 * M + seed)
        return np.sort(rng.uniform(0.0, np.pi, M))
    return np.sort(DISTS[kind].quantile((np.arange(M) + 0.5) / M, s))


def gap_stats(angles):
    """Gaps between neighbouring spokes on the circle (the wrap gap included).

    Returns the largest ratio between two *neighbouring* gaps (always >= 1, the
    analogue of expH02's max h_{j+1}/h_j) and the widest/narrowest gap ratio.
    """
    th = np.sort(np.asarray(angles, dtype=np.float64))
    g = np.diff(np.concatenate([th, [th[0] + np.pi]]))     # M gaps, circular
    nxt = np.roll(g, -1)
    ratio = float(np.max(np.maximum(g / nxt, nxt / g)))
    return ratio, float(g.max() / g.min()), g


# ---------------------------------------------------------------------------
# the model: expH05's RecenteredRidge, but taking the direction set directly
# ---------------------------------------------------------------------------

class RecenteredRidge:
    """``n_per`` offsets on each of the given unit directions, spanning the data's
    projection band about ``v.x0`` plus a 25% collar; width from the spacing."""

    def __init__(self, V, n_per, x0, r, margin=MARGIN, lam=LAMBDA):
        dirs, cens, gams = [], [], []
        for v in V:
            T = margin * r * float(np.linalg.norm(v))
            h = 2.0 * T / n_per
            t = -T + (np.arange(n_per) + 0.5) * h
            dirs.append(np.repeat(v[None, :], n_per, axis=0))
            cens.append(float(v @ x0) + t)
            gams.append(np.full(n_per, lam / h))
        self.directions = np.vstack(dirs)
        self.centers = np.concatenate(cens)
        self.gammas = np.concatenate(gams)

    def features(self, X):
        return np.tanh(self.gammas[None, :] * (X @ self.directions.T - self.centers[None, :]))


def fit(angles, Xtr, Ytr, Xte, Yte):
    """One geometry, one SVD, all three targets."""
    V = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    model = RecenteredRidge(V, N_PER, X0, R_DATA)
    W, b, info = solve_many(model.features(Xtr), Ytr)
    pred = model.features(Xte) @ W + b[None, :]
    out = []
    for k, (key, name, _, _) in enumerate(TARGETS):
        m = error_metrics(pred[:, k], Yte[:, k])
        out.append({"function": key, "function_name": name,
                    "rel_l2": m["rel_l2"], "max_abs": m["max_abs"],
                    "rank": info["rank"], "n_cols": info["n_cols"],
                    "readout_norm": float(np.linalg.norm(W[:, k]))})
    return out


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def run():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Xte = ball(N_TEST, TEST_SHRINK * R_DATA, X0, np.random.default_rng(SEED_TEST))
    Yte = np.stack([fn(Xte) for _, _, _, fn in TARGETS], axis=1)

    # what to fit, per M: (tag, kind, s, seed)
    def jobs():
        js = [("even", "even", 0.0, -1), ("smooth", "smooth", 1.0, -1)]
        js += [("jitter", "jitter", 0.0, sd) for sd in SEEDS]
        js += [("random", "random", 0.0, sd) for sd in SEEDS]
        for shape in SHAPES_FIG2:
            for s in S_LEVELS[1:]:                      # s = 0 reuses "even"
                js.append((shape, shape, s, -1))
        return js

    rows, t_start = [], time.time()
    for M in N_DIRS:
        units = M * N_PER
        n_train = N_TRAIN_PER_UNIT * units
        Xtr = ball(n_train, R_DATA, X0, np.random.default_rng(SEED_TRAIN))
        Ytr = np.stack([fn(Xtr) for _, _, _, fn in TARGETS], axis=1)
        for tag, kind, s, seed in jobs():
            angles = place_angles(kind, M, s=s, seed=max(seed, 0))
            nb_ratio, wide_ratio, _ = gap_stats(angles)
            t0 = time.time()
            recs = fit(angles, Xtr, Ytr, Xte, Yte)
            for rec in recs:
                rec.update({"tag": tag, "kind": kind, "s": float(s), "seed": seed,
                            "M": M, "units": units, "n_train": n_train,
                            "neighbour_gap_ratio": nb_ratio, "gap_spread": wide_ratio})
                rows.append(rec)
            with open(DATA_JSON, "w") as f:
                json.dump(rows, f)
            print(f"M={M:3d} {tag:8s} s={s:.2f} seed={seed:2d} "
                  f"gapratio={nb_ratio:5.2f} rank={recs[0]['rank']:5d}/{recs[0]['n_cols']:5d} "
                  f"[{time.time()-t0:5.1f}s]  "
                  + " ".join(f"{r['function'][:5]}={r['rel_l2']:.0e}" for r in recs),
                  flush=True)
        del Xtr, Ytr
    print(f"total {time.time() - t_start:.0f}s -> {DATA_JSON}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

FLOOR = 1e-16

PLACEMENT_STYLE = {
    "even": ("#000000", "-", "o", "even (reference)"),
    "smooth": ("#1f77b4", "-", "s", r"smooth gradual density ($1+0.8\cos2(\theta-45^\circ)$)"),
    "jitter": ("#2ca02c", "--", "^", r"even + jitter $N(0,(0.25\pi/M)^2)$"),
    "random": ("#d62728", "--", "v", r"uniform random on $[0,\pi)$"),
}
SHAPE_STYLE = {
    "lobe1": ("#1f77b4", "-", "s", "one lobe at $45^\\circ$"),
    "lobe2": ("#ff7f0e", "-", "^", "two lobes at $45^\\circ,135^\\circ$"),
    "zero": ("#d62728", "-", "v", r"vanishing at $0^\circ$ ($\sin^2\theta$)"),
}


def _pick(rows, key, **cond):
    out = [r for r in rows if r["function"] == key
           and all(abs(r[k] - v) < 1e-9 if isinstance(v, float) else r[k] == v
                   for k, v in cond.items())]
    return out


def _band(rows, key, tag):
    """Median / min / max over seeds, per M."""
    med, lo, hi = [], [], []
    for M in N_DIRS:
        es = [r["rel_l2"] for r in _pick(rows, key, tag=tag, M=M)]
        med.append(np.median(es)); lo.append(min(es)); hi.append(max(es))
    return (np.maximum(med, FLOOR), np.maximum(lo, FLOOR), np.maximum(hi, FLOOR))


def plot_fig1(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    handles = None
    for ax, (key, name, formula, _) in zip(axes, TARGETS):
        for tag, (color, ls, mk, label) in PLACEMENT_STYLE.items():
            med, lo, hi = _band(rows, key, tag)
            if tag in ("jitter", "random"):
                ax.fill_between(N_DIRS, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(N_DIRS, med, ls, marker=mk, ms=5, lw=1.6, color=color, label=label)
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_DIRS)
        ax.set_xticklabels([str(m) for m in N_DIRS])
        ax.minorticks_off()
        ax.set_ylim(1e-15, 1e1)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("number of directions $M$")
        ax.set_title(f"{name}\n{formula}", fontsize=10)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("relative $L_2$ inside the ball")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=4, frameon=False, fontsize=10)
    fig.suptitle("how the spokes are spread over $[0,\\pi)$: even, a smooth gradual "
                 f"density, jitter, and random ($r$ = {R_DATA:g}, {N_PER} offsets per "
                 "spoke; bands are min/max over 3 seeds, line is the median)",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIG_DIR / "angle_spacing_vs_M.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out, flush=True)


def plot_fig2(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(14.5, 13), sharex=True, sharey=True)
    even = {key: np.maximum([_pick(rows, key, tag="even", M=M)[0]["rel_l2"]
                             for M in N_DIRS], FLOOR) for key, _, _, _ in TARGETS}
    handles = None
    for si, s in enumerate(S_LEVELS):
        for ti, (key, name, formula, _) in enumerate(TARGETS):
            ax = axes[si][ti]
            ax.plot(N_DIRS, even[key], "-", color="0.55", lw=3.0, alpha=0.7, zorder=0,
                    label="even reference ($s=0$)")
            for shape, (color, ls, mk, label) in SHAPE_STYLE.items():
                if si == 0:
                    ys = even[key]                      # s = 0 is the even set exactly
                else:
                    ys = np.maximum([_pick(rows, key, tag=shape, M=M, s=s)[0]["rel_l2"]
                                     for M in N_DIRS], FLOOR)
                ax.plot(N_DIRS, ys, ls, marker=mk, ms=5, lw=1.5, color=color, label=label)
            ax.set_yscale("log")
            ax.set_xscale("log", base=2)
            ax.set_xticks(N_DIRS)
            ax.set_xticklabels([str(m) for m in N_DIRS])
            ax.minorticks_off()
            ax.set_ylim(1e-15, 1e1)
            ax.grid(alpha=0.3, which="both")
            if si == 0:
                ax.set_title(f"{name}\n{formula}", fontsize=10)
                if handles is None:
                    handles, labels = ax.get_legend_handles_labels()
            if ti == 0:
                ax.set_ylabel(f"{S_LABEL[si]}\nrelative $L_2$", fontsize=9.5)
            if si == 3:
                ax.set_xlabel("number of directions $M$")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.958),
               ncol=4, frameon=False, fontsize=10)
    fig.suptitle("angle density $q_s = (1-s)/\\pi + s\\,q$, spokes at "
                 "$\\theta_i = Q_s^{-1}((i+1/2)/M)$: three shapes $q$, four levels of "
                 f"non-uniformity ($r$ = {R_DATA:g}, {N_PER} offsets per spoke)",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    out = FIG_DIR / "angle_spacing_h02_style.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out, flush=True)


def plot_placement(M=16):
    """Where the spokes land, for every placement used, at M = 16."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deg = np.degrees(_TH)
    panels = [("even", "even", 0.0, -1, "even (reference)"),
              ("smooth", "smooth", 1.0, -1, SHAPE_TITLE["smooth"]),
              ("jitter", "jitter", 0.0, 0, r"even + jitter (seed 0)"),
              ("random", "random", 0.0, 0, r"uniform random (seed 0)")]
    for shape in SHAPES_FIG2:
        for si, s in enumerate(S_LEVELS):
            panels.append((shape, shape, s, -1,
                           f"{SHAPE_TITLE[shape]}\n{S_LABEL[si]}"))

    fig, axes = plt.subplots(4, 4, figsize=(16, 11.5), sharex=True)
    for ax, (tag, kind, s, seed, title) in zip(axes.ravel(), panels):
        angles = place_angles(kind, M, s=s, seed=max(seed, 0))
        ratio, spread, _ = gap_stats(angles)
        if kind in DISTS:
            dens = DISTS[kind].mixture_pdf(s) * np.pi          # per unit of (th/pi)
            lbl = "angle density $q_s$ (scaled)"
        else:
            dens = np.ones_like(_TH)
            lbl = "uniform density (reference)"
        ax.plot(deg, dens, color="#d62728", lw=1.6, label=lbl)
        ax.plot(np.degrees(angles), np.full(M, -0.12), "|", color="#1f77b4", ms=14,
                mew=1.6, label="placed spokes")
        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.text(0.02, 0.97,
                f"max neighbour gap ratio = {ratio:.2f}\nwidest/narrowest = {spread:.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5)
        ax.set_xlim(-3, 183)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_ylim(-0.3, 3.2)
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(r"angle $\theta$ (degrees); $0^\circ$ and $180^\circ$ are the same spoke")
    for ax in axes[:, 0]:
        ax.set_ylabel("density", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"where the spokes land at $M$ = {M}, for every placement used",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    out = FIG_DIR / "angle_placement.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out, flush=True)


def summarize(rows):
    """A few numbers for the writeup."""
    print("\n-- neighbour-gap ratios (max over the circle) --")
    for M in N_DIRS:
        line = [f"M={M:2d}"]
        for tag, kind, s in [("even", "even", 0.0), ("smooth", "smooth", 1.0),
                             ("jitter", "jitter", 0.0), ("random", "random", 0.0)]:
            if kind in ("jitter", "random"):
                rr = [gap_stats(place_angles(kind, M, seed=sd))[0] for sd in SEEDS]
                line.append(f"{tag}={np.median(rr):.2f}[{min(rr):.2f},{max(rr):.2f}]")
            else:
                line.append(f"{tag}={gap_stats(place_angles(kind, M, s=s))[0]:.2f}")
        for shape in SHAPES_FIG2:
            for s in S_LEVELS[1:]:
                line.append(f"{shape}/s={s:.2f}:{gap_stats(place_angles(shape, M, s=s))[0]:.2f}")
        print("  " + "  ".join(line))
    print("\n-- relative L2 by M --")
    for key, name, _, _ in TARGETS:
        print(f"  {name}")
        for tag in ["even", "smooth", "jitter", "random"]:
            med, lo, hi = _band(rows, key, tag)
            print(f"    {tag:8s} " + " ".join(f"{m:.1e}" for m in med))
        for shape in SHAPES_FIG2:
            for s in S_LEVELS[1:]:
                ys = [_pick(rows, key, tag=shape, M=M, s=s)[0]["rel_l2"] for M in N_DIRS]
                print(f"    {shape:6s} s={s:.2f} " + " ".join(f"{y:.1e}" for y in ys))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="replot from the saved data.json")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.plot:
        with open(DATA_JSON) as f:
            rows = json.load(f)
    else:
        rows = run()
    plot_fig1(rows)
    plot_fig2(rows)
    plot_placement()
    summarize(rows)
