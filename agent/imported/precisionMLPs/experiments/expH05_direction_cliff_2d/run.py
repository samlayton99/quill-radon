"""The direction cliff in 2-D, on nine functions.

expH04's ``directions_vs_radius.py`` showed, for a single target, that a 2-D ridge
network fitted by one truncated-SVD least squares on the readout does essentially
nothing until the number of directions ``M`` crosses a threshold, then falls five or six
orders of magnitude in two or three steps of ``M``. That was one function. This asks
whether the same shape shows up across functions of very different character, and how
the threshold moves with the radius of the region we are asking for precision on.

Setup (fixed everywhere):

* domain ``[-1, 1]^2``; data uniform in a ball of radius ``r`` about the off-center
  point ``x0 = (0.35, -0.25)``, ``r`` in ``{0.1, 0.2, 0.4, 0.8}``;
* the ridge system is recentered on ``x0``: ``M`` evenly spaced angles on ``[0, pi)``
  offset by half a step, and along each direction ``v`` the offsets are
  ``c = v.x0 + t`` with ``t`` evenly spaced over ``[-T, T]``, ``T = 1.25 r`` (the 25%
  collar expH01/expE01 use), ``n_per = 128`` offsets on every line at every ``M``, so
  the along-direction resolution is never the binding constraint;
* width ``gamma = lambda / h``, ``lambda = 0.25``, ``h = 2T/n_per``; feature
  ``tanh(gamma (v.x - c))``;
* the readout is one truncated-SVD least squares with a bias column at
  ``rcond = 1e-13`` (``h01suite.baseline._solve_svd``);
* ``n_train = 8 * units`` points, ``units = 128 M``; error is measured on 20000 points
  uniform in the ball of radius ``0.9 r`` about ``x0``, so the collar is never scored.

The feature matrix depends only on ``(r, M)``, so the SVD is taken once per ``(r, M)``
and the nine right-hand sides are solved against it. That is arithmetically the same as
calling ``_solve_svd`` nine times; ``--check`` verifies it.

Two follow-ups live in the same file. ``--split`` / ``--split-exact`` sweep the way a
fixed unit budget ``MN`` is split between directions and offsets. ``--tradeoff`` measures
the two floors that split is trading against -- ``e_M(M)`` at fixed ``N`` and ``e_N(N)`` at
fixed ``M`` -- tests whether the whole surface is their maximum, and fits the exponent in
``M* ~ B^alpha`` on a fine budget ladder.

Usage:
    OMP_NUM_THREADS=6 uv run --extra dev python experiments/expH05_direction_cliff_2d/run.py
    uv run --extra dev python experiments/expH05_direction_cliff_2d/run.py --plot
    OMP_NUM_THREADS=6 uv run --extra dev python .../run.py --tradeoff
    uv run --extra dev python .../run.py --tradeoff --plot      # re-derives + replots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "experiments" / "expH01_highdim_suite"))

from h01suite.baseline import even_directions, _solve_svd, LAMBDA, RCOND  # noqa: E402
from h01suite.metrics import error_metrics                                # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH05_direction_cliff_2d"
FIG_DIR = RESULTS_DIR / "figures"
DATA_JSON = RESULTS_DIR / "data.json"

X0 = np.array([0.35, -0.25])          # center of the data ball
RADII = [0.1, 0.2, 0.4, 0.8]
N_DIRS = [1, 2, 3, 4, 6, 8, 12, 16]
N_PER = 128                           # offsets per direction, every direction, every M
MARGIN = 1.25                         # collar beyond the data's projection band
N_TRAIN_PER_UNIT = 8
N_TEST = 20000
TEST_SHRINK = 0.9                     # score only the inner 90% of the data ball
SEED_TRAIN, SEED_TEST = 0, 1

# --- the offsets-vs-directions split experiment ---------------------------
SPLIT_R = 0.4
SPLIT_M = [2, 3, 4, 6, 8, 12, 16, 24, 32]
SPLIT_N = [8, 12, 16, 24, 32, 48, 64, 96, 128]
SPLIT_KEYS = ["fast_waves", "radial_runge", "composition", "spatial_packet"]
SPLIT_BUDGETS = [256, 1024, 4096]
SPLIT_JSON_NAME = "split_heatmap_2d.json"

# powers of two, so every budget M*N is an exact anti-diagonal of grid cells
EXACT_M = [2, 4, 8, 16, 32, 64]
EXACT_N = [8, 16, 32, 64, 128]
EXACT_MAX_UNITS = 4096
EXACT_TIE = 2.0                       # "as good as the best" means within this factor
EXACT_JSON_NAME = "split_exact_2d.json"

# --- the fine tradeoff curve between M and N -------------------------------
TRADE_EM_N = 128                      # offsets held fixed while M moves
TRADE_EM_M = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48]
TRADE_EN_M = 48                       # directions held fixed while N moves
TRADE_EN_N = [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128]
TRADE_BUDGETS = [64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096]
TRADE_M_CAP = 48                      # never more directions than the e_M sweep covers
TRADE_N_MIN = 8                       # never fewer offsets than this on a budget ladder
TRADE_N_PER_BUDGET = 10               # about this many M values per budget
TRADE_FLOOR = 1e-13                   # points at or below this are floor, not signal
TRADE_VALLEY_TIE = 2.0                # "within 2x of the minimum" defines the valley width
TRADE_JSON_NAME = "tradeoff_2d.json"

# anchors used by the targets
A_BUMP = np.array([0.2, 0.1])         # wide/gauss bump anchor
A_RAD = np.array([0.3, -0.2])         # radial-family anchor (inside the r = 0.1 ball)
A_PACKET = X0.copy()                  # the packet sits on the data ball's center


# ---------------------------------------------------------------------------
# targets -- plain numpy, no rotated coordinates, all O(1) on the cube
# ---------------------------------------------------------------------------

def _rho(X, a):
    """Scaled radial distance ||x - a|| / sqrt(2), the suite's convention in d = 2."""
    dif = np.asarray(X, dtype=np.float64) - a[None, :]
    return np.sqrt(np.einsum("nk,nk->n", dif, dif) / 2.0)


def f_gauss_bump(X):
    dif = X - A_BUMP[None, :]
    return np.exp(-np.einsum("nk,nk->n", dif, dif) / 0.5 ** 2)


def f_product_sines(X):
    return np.sin(2 * np.pi * X[:, 0]) * np.sin(2 * np.pi * X[:, 1])


def f_composition(X):
    return np.exp(np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]))


def f_polynomial(X):
    x, y = X[:, 0], X[:, 1]
    return x * x * y - x * y ** 3 + x * y


def f_slow_waves(X):
    return np.cos(np.pi * 1.0 * _rho(X, A_RAD))


def f_radial_runge(X):
    return 1.0 / (1.0 + 16.0 * _rho(X, A_RAD) ** 2)


def f_fast_waves(X):
    return np.cos(6 * np.pi * _rho(X, A_RAD))


def f_narrow_runge(X):
    return 1.0 / (1.0 + 144.0 * _rho(X, A_RAD) ** 2)


def f_spatial_packet(X):
    rp = _rho(X, A_PACKET)
    packet = 0.8 * np.exp(-(rp / 0.18) ** 2) * np.cos(10 * np.pi * rp)
    bump = np.exp(-_rho(X, A_BUMP) ** 2 / (2 * 0.5 ** 2))
    return packet + bump


TARGETS = [
    ("gauss_bump", "gauss bump",
     r"$\exp(-\|x-a_1\|^2/0.5^2)$", f_gauss_bump),
    ("product_sines", "product sines",
     r"$\sin(2\pi x)\,\sin(2\pi y)$", f_product_sines),
    ("composition", "composition",
     r"$\exp(\sin(\pi x)\cos(\pi y))$", f_composition),
    ("polynomial", "polynomial",
     r"$x^2y - xy^3 + xy$", f_polynomial),
    ("slow_waves", "slow concentric waves",
     r"$\cos(\pi\rho_2)$", f_slow_waves),
    ("radial_runge", "radial Runge",
     r"$1/(1+16\rho_2^2)$", f_radial_runge),
    ("fast_waves", "fast concentric waves",
     r"$\cos(6\pi\rho_2)$", f_fast_waves),
    ("narrow_runge", "narrow radial Runge",
     r"$1/(1+144\rho_2^2)$", f_narrow_runge),
    ("spatial_packet", "spatial packet",
     r"$0.8e^{-(\rho_0/0.18)^2}\cos(10\pi\rho_0) + e^{-\rho_1^2/2\cdot0.5^2}$",
     f_spatial_packet),
]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def ball(n, r, x0, rng):
    """``n`` points uniform in the ball of radius ``r`` about ``x0`` (d = 2)."""
    g = rng.normal(size=(n, 2))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    u = rng.uniform(size=(n, 1)) ** 0.5
    return x0 + r * u * g


class RecenteredRidge:
    """``M`` even directions; ``n_per`` offsets on each, spanning the data's projection
    band about ``v.x0`` plus a 25% collar; width from the spacing."""

    def __init__(self, n_dir, n_per, x0, r, margin=MARGIN, lam=LAMBDA):
        V = even_directions(2, n_dir)
        dirs, cens, gams = [], [], []
        for v in V:
            T = margin * r * float(np.linalg.norm(v))
            h = 2.0 * T / n_per
            t = -T + (np.arange(n_per) + 0.5) * h
            dirs.append(np.repeat(v[None, :], n_per, axis=0))
            cens.append(float(v @ x0) + t)
            gams.append(np.full(n_per, lam / h))
        self.unique_directions = V
        self.directions = np.vstack(dirs)
        self.centers = np.concatenate(cens)
        self.gammas = np.concatenate(gams)

    def features(self, X):
        return np.tanh(self.gammas[None, :] * (X @ self.directions.T - self.centers[None, :]))


def solve_many(Phi, Y, rcond=RCOND):
    """``_solve_svd`` for several right-hand sides at once (one SVD, many solves)."""
    A = np.hstack([Phi, np.ones((len(Phi), 1), dtype=np.float64)])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > rcond * s[0]
    s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    sol = Vt.T @ (s_inv[:, None] * (U.T @ np.asarray(Y, dtype=np.float64)))
    return sol[:-1], sol[-1], {"rank": int(keep.sum()), "n_cols": A.shape[1]}


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="replot from the saved data.json")
    ap.add_argument("--check", action="store_true",
                    help="verify solve_many against _solve_svd on a small case and exit")
    ap.add_argument("--control", action="store_true",
                    help="at the largest radius and M = 16, does doubling n_per help?")
    ap.add_argument("--split", action="store_true",
                    help="the directions-vs-offsets tradeoff heat map at r = 0.4")
    ap.add_argument("--split-exact", action="store_true", dest="split_exact",
                    help="the power-of-two split grid: exact iso-budget diagonals "
                         "plus the steepest-descent path")
    ap.add_argument("--tradeoff", action="store_true",
                    help="the two floor curves e_M(M), e_N(N) and the iso-budget "
                         "valleys on a fine budget ladder")
    args = ap.parse_args()

    if args.tradeoff:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / TRADE_JSON_NAME
        if args.plot:
            with open(path) as f:
                data = json.load(f)
            plot_tradeoff(data)          # plot_tradeoff re-derives the analysis
            with open(path, "w") as f:
                json.dump(data, f)
        else:
            plot_tradeoff(tradeoff())
        return

    if args.split_exact:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / EXACT_JSON_NAME
        if args.plot:
            with open(path) as f:
                plot_split_exact(json.load(f))
        else:
            plot_split_exact(split_exact())
        return

    if args.plot and not args.split:
        with open(DATA_JSON) as f:
            rows = json.load(f)
        plot(rows)
        plot_threshold(rows)
        return

    if args.check:
        check()
        return

    if args.control:
        control()
        return

    if args.split:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / SPLIT_JSON_NAME
        if args.plot:
            with open(path) as f:
                plot_split(json.load(f))
        else:
            plot_split(split())
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    check()
    rows = []
    t_start = time.time()
    for r in RADII:
        Xtest = ball(N_TEST, TEST_SHRINK * r, X0, np.random.default_rng(SEED_TEST))
        Ytest = np.stack([fn(Xtest) for _, _, _, fn in TARGETS], axis=1)
        for M in N_DIRS:
            units = M * N_PER
            n_train = N_TRAIN_PER_UNIT * units
            Xtr = ball(n_train, r, X0, np.random.default_rng(SEED_TRAIN))
            Ytr = np.stack([fn(Xtr) for _, _, _, fn in TARGETS], axis=1)
            t0 = time.time()
            model = RecenteredRidge(M, N_PER, X0, r)
            W, b, info = solve_many(model.features(Xtr), Ytr)
            pred = model.features(Xtest) @ W + b[None, :]
            line = []
            for k, (key, name, _, _) in enumerate(TARGETS):
                m = error_metrics(pred[:, k], Ytest[:, k])
                rows.append({"function": key, "function_name": name, "r": r, "M": M,
                             "units": units, "n_train": n_train,
                             "rel_l2": m["rel_l2"], "max_abs": m["max_abs"],
                             "rank": info["rank"], "n_cols": info["n_cols"],
                             "readout_norm": float(np.linalg.norm(W[:, k])),
                             "y_norm": float(np.linalg.norm(Ytest[:, k]))})
                line.append(f"{key[:5]}={m['rel_l2']:.0e}")
            with open(DATA_JSON, "w") as f:
                json.dump(rows, f)
            print(f"r={r:<4g} M={M:3d} units={units:5d} n={n_train:6d} "
                  f"rank={info['rank']:5d}/{info['n_cols']:5d} [{time.time()-t0:5.1f}s]  "
                  + " ".join(line), flush=True)
    print(f"total {time.time() - t_start:.0f}s -> {DATA_JSON}", flush=True)
    plot(rows)
    plot_threshold(rows)


def check():
    """One small case: solve_many must agree with _solve_svd column by column.

    The comparison is on the coefficients, relative to their own norm. The predictions
    of the two can differ by roughly ``||w|| * eps`` -- large cancelling readout weights
    are inherent to this geometry, and that product is what sets the error floor.
    """
    r, M = 0.2, 3
    Xtr = ball(N_TRAIN_PER_UNIT * M * 16, r, X0, np.random.default_rng(SEED_TRAIN))
    model = RecenteredRidge(M, 16, X0, r)
    Phi = model.features(Xtr)
    Y = np.stack([fn(Xtr) for _, _, _, fn in TARGETS], axis=1)
    W, b, info = solve_many(Phi, Y)
    worst_rel, worst_pred = 0.0, 0.0
    for k in range(Y.shape[1]):
        w1, b1, i1 = _solve_svd(Phi, Y[:, k], RCOND)
        assert i1["rank"] == info["rank"]
        worst_rel = max(worst_rel,
                        float(np.linalg.norm(W[:, k] - w1) / np.linalg.norm(w1)))
        worst_pred = max(worst_pred,
                         float(np.max(np.abs(Phi @ (W[:, k] - w1) + (b[k] - b1)))))
    print(f"check: solve_many vs _solve_svd -- worst relative coefficient difference "
          f"{worst_rel:.2e}, worst prediction difference {worst_pred:.2e}", flush=True)
    assert worst_rel < 1e-6, worst_rel


def control():
    """Is 128 offsets per direction the binding constraint at r = 0.8, M = 16?

    Same fit, same scoring, only ``n_per`` changes. If the error at the largest radius
    is limited by the along-direction resolution rather than by the direction count,
    doubling ``n_per`` moves it; if it is limited by ``M``, it does not.
    """
    r, M = RADII[-1], N_DIRS[-1]
    Xtest = ball(N_TEST, TEST_SHRINK * r, X0, np.random.default_rng(SEED_TEST))
    Ytest = np.stack([fn(Xtest) for _, _, _, fn in TARGETS], axis=1)
    out = []
    for n_per in (N_PER, 2 * N_PER):
        units = M * n_per
        Xtr = ball(N_TRAIN_PER_UNIT * units, r, X0, np.random.default_rng(SEED_TRAIN))
        Ytr = np.stack([fn(Xtr) for _, _, _, fn in TARGETS], axis=1)
        t0 = time.time()
        model = RecenteredRidge(M, n_per, X0, r)
        W, b, info = solve_many(model.features(Xtr), Ytr)
        pred = model.features(Xtest) @ W + b[None, :]
        errs = {key: error_metrics(pred[:, k], Ytest[:, k])["rel_l2"]
                for k, (key, _, _, _) in enumerate(TARGETS)}
        out.append({"r": r, "M": M, "n_per": n_per, "units": units,
                    "n_train": len(Xtr), "rank": info["rank"], "rel_l2": errs})
        print(f"r={r} M={M} n_per={n_per:4d} units={units:5d} rank={info['rank']:5d} "
              f"[{time.time()-t0:5.1f}s]  "
              + " ".join(f"{k[:5]}={v:.0e}" for k, v in errs.items()), flush=True)
    with open(RESULTS_DIR / "control_n_per.json", "w") as f:
        json.dump(out, f)
    plot_control(out)


def plot_control(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [t[0] for t in TARGETS]
    names = [t[1] for t in TARGETS]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xs = np.arange(len(keys))
    styles = [("o", "tab:blue", 13), ("s", "tab:orange", 7)]
    for rec, (mk, col, ms) in zip(out, styles):
        ax.semilogy(xs, [rec["rel_l2"][k] for k in keys], mk, ms=ms, color=col,
                    label=f"{rec['n_per']} offsets per direction "
                          f"({rec['units']} units)")
    for j, k in enumerate(keys):
        ax.plot([j, j], [out[0]["rel_l2"][k], out[1]["rel_l2"][k]], "-", color="0.6",
                lw=1, zorder=0)
    ax.axhspan(1e-14, 1e-13, color="0.9", zorder=0)
    ax.text(len(keys) - 0.4, 3e-14, "floor band", fontsize=8, color="0.4", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(1e-15, 1e1)
    ax.set_ylabel("relative $L_2$ inside the ball")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.set_title(f"control at the hardest setting: $r$ = {out[0]['r']:g}, "
                 f"$M$ = {out[0]['M']} -- does the along-direction resolution bind?",
                 fontsize=11, pad=26)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False,
              fontsize=9)
    fig.tight_layout()
    o = FIG_DIR / "control_n_per.png"
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print("saved", o, flush=True)


# ---------------------------------------------------------------------------
# the directions-vs-offsets split
# ---------------------------------------------------------------------------

def split():
    """Same ball, same solve: sweep the two ways of spending the unit budget.

    ``units = M * N`` with ``M`` directions and ``N`` offsets on each. The feature
    matrix depends only on ``(M, N)``, so one SVD per cell serves all four targets.
    """
    r = SPLIT_R
    targets = [t for t in TARGETS if t[0] in SPLIT_KEYS]
    targets.sort(key=lambda t: SPLIT_KEYS.index(t[0]))
    Xtest = ball(N_TEST, TEST_SHRINK * r, X0, np.random.default_rng(SEED_TEST))
    Ytest = np.stack([fn(Xtest) for _, _, _, fn in targets], axis=1)
    rows, t_start = [], time.time()
    for M in SPLIT_M:
        for n_per in SPLIT_N:
            units = M * n_per
            n_train = N_TRAIN_PER_UNIT * units
            Xtr = ball(n_train, r, X0, np.random.default_rng(SEED_TRAIN))
            Ytr = np.stack([fn(Xtr) for _, _, _, fn in targets], axis=1)
            t0 = time.time()
            model = RecenteredRidge(M, n_per, X0, r)
            W, b, info = solve_many(model.features(Xtr), Ytr)
            pred = model.features(Xtest) @ W + b[None, :]
            line = []
            for k, (key, name, _, _) in enumerate(targets):
                m = error_metrics(pred[:, k], Ytest[:, k])
                rows.append({"function": key, "function_name": name, "r": r,
                             "M": M, "N": n_per, "units": units, "n_train": n_train,
                             "rel_l2": m["rel_l2"], "max_abs": m["max_abs"],
                             "rank": info["rank"], "n_cols": info["n_cols"],
                             "weight_norm": float(np.linalg.norm(W[:, k]))})
                line.append(f"{key[:5]}={m['rel_l2']:.0e}")
            with open(RESULTS_DIR / SPLIT_JSON_NAME, "w") as f:
                json.dump(rows, f)
            print(f"M={M:3d} N={n_per:4d} units={units:5d} n={n_train:6d} "
                  f"rank={info['rank']:5d}/{info['n_cols']:5d} [{time.time()-t0:5.1f}s]  "
                  + " ".join(line), flush=True)
    print(f"total {time.time() - t_start:.0f}s -> {RESULTS_DIR / SPLIT_JSON_NAME}",
          flush=True)
    return rows


def _split_grid(rows, key):
    """``log10`` of the relative L2 on the (N, M) grid, plus the raw errors."""
    d = {(x["M"], x["N"]): x["rel_l2"] for x in rows if x["function"] == key}
    E = np.array([[d[(M, N)] for M in SPLIT_M] for N in SPLIT_N])
    return np.log10(np.maximum(E, 1e-16)), E


def split_best(rows, key, budget=None):
    """Best cell overall, or best among the cells with exactly ``units == budget``."""
    cand = [x for x in rows if x["function"] == key
            and (budget is None or x["units"] == budget)]
    if not cand:
        return None
    return min(cand, key=lambda x: x["rel_l2"])


def plot_split(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {t[0]: t[1] for t in TARGETS}
    formulas = {t[0]: t[2] for t in TARGETS}
    logM, logN = np.log(SPLIT_M), np.log(SPLIT_N)

    def x_of(M):
        return np.interp(np.log(M), logM, np.arange(len(SPLIT_M)))

    def y_of(N):
        return np.interp(np.log(N), logN, np.arange(len(SPLIT_N)))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.2))
    im = None
    for ax, key in zip(axes.ravel(), SPLIT_KEYS):
        L, _ = _split_grid(rows, key)
        im = ax.imshow(L, origin="lower", aspect="auto", cmap="viridis",
                       vmin=-14, vmax=0)
        # iso-budget lines
        for B in SPLIT_BUDGETS:
            Mline = np.exp(np.linspace(logM[0], logM[-1], 400))
            Nline = B / Mline
            ok = (Nline >= SPLIT_N[0]) & (Nline <= SPLIT_N[-1])
            xs, ys = x_of(Mline[ok]), y_of(Nline[ok])
            if ok.sum() > 4:
                ax.plot(xs, ys, "--", color="white", lw=1.4)
                h = len(xs) // 2
                ang = np.degrees(np.arctan2(ys[h + 8] - ys[h - 8], xs[h + 8] - xs[h - 8]))
                ax.text(xs[h], ys[h] + 0.18, f"$MN$ = {B}", color="white",
                        fontsize=8.5, rotation=ang, rotation_mode="anchor",
                        va="bottom", ha="center")
            elif ok.sum() >= 1:      # the budget only touches one corner of the grid
                ax.text(xs[0] - 0.15, ys[0] - 0.30, f"$MN$ = {B}", color="white",
                        fontsize=8.5, va="top", ha="right")
            best = split_best(rows, key, B)
            if best is not None:
                ax.plot(x_of(best["M"]), y_of(best["N"]), "o", ms=13, mfc="none",
                        mec="white", mew=2.0)
        gbest = split_best(rows, key)
        ax.plot(x_of(gbest["M"]), y_of(gbest["N"]), "*", ms=18, color="white",
                mec="black", mew=0.8)
        ax.set_xticks(range(len(SPLIT_M)))
        ax.set_xticklabels([str(m) for m in SPLIT_M])
        ax.set_yticks(range(len(SPLIT_N)))
        ax.set_yticklabels([str(n) for n in SPLIT_N])
        ax.set_xlabel("directions $M$")
        ax.set_ylabel("offsets per direction $N$")
        ax.set_title(f"{names[key]}\n{formulas[key]}", fontsize=10)

    star = plt.Line2D([], [], ls="none", marker="*", ms=14, color="0.2",
                      label="best cell overall")
    circ = plt.Line2D([], [], ls="none", marker="o", ms=9, mfc="none", mec="0.2",
                      mew=2.0, label="best cell on an iso-budget line")
    dash = plt.Line2D([], [], ls="--", color="0.2", lw=1.4,
                      label="iso-budget $MN$ = 256, 1024, 4096")
    fig.legend(handles=[star, circ, dash], loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("spending the unit budget: more directions or more offsets per "
                 f"direction (data ball $r$ = {SPLIT_R:g}, error on its inner 90%)",
                 y=0.985, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.90, 0.935))
    cax = fig.add_axes([0.92, 0.08, 0.02, 0.80])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}$ relative $L_2$ inside the ball (dark = accurate)")
    out = FIG_DIR / "split_heatmap_2d.png"
    fig.savefig(out, dpi=140)
    print("saved", out, flush=True)


def exact_cells():
    """The power-of-two grid, dropping anything past the unit cap."""
    return [(M, N) for M in EXACT_M for N in EXACT_N if M * N <= EXACT_MAX_UNITS]


def split_exact():
    """The split sweep on the power-of-two grid; one SVD per cell, four targets."""
    r = SPLIT_R
    targets = [t for t in TARGETS if t[0] in SPLIT_KEYS]
    targets.sort(key=lambda t: SPLIT_KEYS.index(t[0]))
    Xtest = ball(N_TEST, TEST_SHRINK * r, X0, np.random.default_rng(SEED_TEST))
    Ytest = np.stack([fn(Xtest) for _, _, _, fn in targets], axis=1)
    rows, t_start = [], time.time()
    for M, n_per in exact_cells():
        units = M * n_per
        n_train = N_TRAIN_PER_UNIT * units
        Xtr = ball(n_train, r, X0, np.random.default_rng(SEED_TRAIN))
        Ytr = np.stack([fn(Xtr) for _, _, _, fn in targets], axis=1)
        t0 = time.time()
        model = RecenteredRidge(M, n_per, X0, r)
        W, b, info = solve_many(model.features(Xtr), Ytr)
        pred = model.features(Xtest) @ W + b[None, :]
        line = []
        for k, (key, name, _, _) in enumerate(targets):
            m = error_metrics(pred[:, k], Ytest[:, k])
            rows.append({"function": key, "function_name": name, "r": r,
                         "M": M, "N": n_per, "units": units, "n_train": n_train,
                         "rel_l2": m["rel_l2"], "max_abs": m["max_abs"],
                         "rank": info["rank"], "n_cols": info["n_cols"],
                         "weight_norm": float(np.linalg.norm(W[:, k]))})
            line.append(f"{key[:5]}={m['rel_l2']:.0e}")
        with open(RESULTS_DIR / EXACT_JSON_NAME, "w") as f:
            json.dump(rows, f)
        print(f"M={M:3d} N={n_per:4d} units={units:5d} n={n_train:6d} "
              f"rank={info['rank']:5d}/{info['n_cols']:5d} [{time.time()-t0:5.1f}s]  "
              + " ".join(line), flush=True)
    print(f"total {time.time() - t_start:.0f}s -> {RESULTS_DIR / EXACT_JSON_NAME}",
          flush=True)
    return rows


def exact_path(rows, key):
    """For every exact budget: the best cell on that anti-diagonal, and its ties.

    A tie is another cell on the same diagonal whose error is within ``EXACT_TIE`` of
    the best -- a flat-bottomed diagonal, where the split hardly matters.
    """
    d = {}
    for x in rows:
        if x["function"] == key:
            d.setdefault(x["units"], []).append(x)
    out = []
    for B in sorted(d):
        cells = sorted(d[B], key=lambda x: x["M"])
        best = min(cells, key=lambda x: x["rel_l2"])
        ties = [c for c in cells
                if c is not best and c["rel_l2"] <= EXACT_TIE * best["rel_l2"]]
        out.append({"budget": B, "n_cells": len(cells), "M": best["M"], "N": best["N"],
                    "rel_l2": best["rel_l2"], "max_abs": best["max_abs"],
                    "ties": [{"M": c["M"], "N": c["N"], "rel_l2": c["rel_l2"]}
                             for c in ties]})
    return out


def plot_split_exact(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {t[0]: t[1] for t in TARGETS}
    formulas = {t[0]: t[2] for t in TARGETS}
    budgets = sorted({M * N for M, N in exact_cells()})

    fig, axes = plt.subplots(2, 4, figsize=(19.5, 9.6))
    im = None
    for col, key in enumerate(SPLIT_KEYS):
        # ---- top: the heat map, nothing drawn on top of it -------------------
        ax = axes[0, col]
        d = {(x["M"], x["N"]): x["rel_l2"] for x in rows if x["function"] == key}
        L = np.full((len(EXACT_N), len(EXACT_M)), np.nan)
        for i, N in enumerate(EXACT_N):
            for j, M in enumerate(EXACT_M):
                if (M, N) in d:
                    L[i, j] = np.log10(max(d[(M, N)], 1e-16))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        im = ax.imshow(np.ma.masked_invalid(L), origin="lower", aspect="auto",
                       cmap=cmap, vmin=-14, vmax=0)
        ax.set_xticks(range(len(EXACT_M)))
        ax.set_xticklabels([str(m) for m in EXACT_M])
        ax.set_yticks(range(len(EXACT_N)))
        ax.set_yticklabels([str(n) for n in EXACT_N])
        ax.set_xlabel("directions $M$")
        if col == 0:
            ax.set_ylabel("offsets per direction $N$")
        ax.set_title(f"{names[key]}\n{formulas[key]}", fontsize=10)

        # ---- bottom: the steepest-descent path, in the same (M, N) axes -------
        ax = axes[1, col]
        path = exact_path(rows, key)
        mi = {m: j for j, m in enumerate(EXACT_M)}
        ni = {n: i for i, n in enumerate(EXACT_N)}
        # faint exact iso-budget diagonals (straight lines on the log2 index grid)
        for b in budgets:
            pts = sorted([(mi[M], ni[N]) for M, N in exact_cells() if M * N == b])
            if len(pts) >= 2:
                ax.plot([q[0] for q in pts], [q[1] for q in pts], "-", color="0.85",
                        lw=1.0, zorder=1)
            ax.annotate(str(b), xy=pts[0], xytext=(-4, 9), textcoords="offset points",
                        fontsize=7, color="0.45", ha="center")
        # ties: hollow markers at their own (M, N)
        for p in path:
            for t in p["ties"]:
                ax.plot(mi[t["M"]], ni[t["N"]], "o", ms=13, mfc="none", mec="0.35",
                        mew=1.2, zorder=3)
        # the path: optimal cell per budget, connected in budget order, colored by error
        xs = [mi[p["M"]] for p in path]
        ys = [ni[p["N"]] for p in path]
        cs = [np.log10(max(p["rel_l2"], 1e-16)) for p in path]
        ax.plot(xs, ys, "-", color="0.2", lw=1.5, zorder=2)
        ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=-14, vmax=0, s=170,
                   edgecolors="k", linewidths=0.8, zorder=4)
        ax.set_xlim(-0.5, len(EXACT_M) - 0.5)
        ax.set_ylim(-0.5, len(EXACT_N) - 0.5)
        ax.set_xticks(range(len(EXACT_M)))
        ax.set_xticklabels([str(m) for m in EXACT_M])
        ax.set_yticks(range(len(EXACT_N)))
        ax.set_yticklabels([str(n) for n in EXACT_N])
        ax.set_xlabel("directions $M$")
        if col == 0:
            ax.set_ylabel("offsets per direction $N$")
        if col == 0:
            ax.set_title("best cell on each exact budget, connected in budget order\n"
                         f"(node color = its error on the scale at right; hollow = tie within {EXACT_TIE:g}x; "
                         "grey lines = exact iso-budgets)", fontsize=8.5, loc="left")

    fig.suptitle("the budget split on a power-of-two grid: every budget $B = MN$ is an "
                 "exact diagonal of cells "
                 f"(data ball $r$ = {SPLIT_R:g}, error on its inner 90%; blank cells "
                 f"exceed the {EXACT_MAX_UNITS}-unit cap)", y=0.985, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.925, 0.955))
    cax = fig.add_axes([0.945, 0.2, 0.013, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}$ relative $L_2$ (dark = accurate)", fontsize=9)
    out = FIG_DIR / "split_exact_2d.png"
    fig.savefig(out, dpi=140)
    print("saved", out, flush=True)


# ---------------------------------------------------------------------------
# the M-vs-N tradeoff: two floor curves, then iso-budget valleys
# ---------------------------------------------------------------------------

def _augmented(model, X, block=8192):
    """``[Phi, 1]`` built row-block by row-block, so the big matrix exists once.

    Same numbers as ``np.hstack([model.features(X), ones])``; the largest solve here is
    49152 x 6145 and holding two copies of that is 5 GB of avoidable memory.
    """
    m = len(model.centers)
    A = np.empty((len(X), m + 1), dtype=np.float64)
    D, c, g = model.directions.T, model.centers[None, :], model.gammas[None, :]
    for i0 in range(0, len(X), block):
        sl = slice(i0, min(i0 + block, len(X)))
        A[sl, :m] = np.tanh(g * (X[sl] @ D - c))
    A[:, m] = 1.0
    return A


def _solve_augmented(A, Y, rcond=RCOND):
    """Truncated-SVD least squares on an already-augmented ``A`` (bias column last)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > rcond * s[0]
    s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    sol = Vt.T @ (s_inv[:, None] * (U.T @ np.asarray(Y, dtype=np.float64)))
    return sol, {"rank": int(keep.sum()), "n_cols": A.shape[1]}


class _CellCache:
    """One fit per ``(M, N)``, reused everywhere it is asked for."""

    def __init__(self, targets, r=SPLIT_R):
        self.r = r
        self.targets = targets
        self.keys = [t[0] for t in targets]
        self.Xtest = ball(N_TEST, TEST_SHRINK * r, X0, np.random.default_rng(SEED_TEST))
        self.Ytest = np.stack([fn(self.Xtest) for _, _, _, fn in targets], axis=1)
        self.cells = {}

    def get(self, M, N):
        key = (int(M), int(N))
        if key in self.cells:
            return self.cells[key]
        M, N = key
        units = M * N
        n_train = N_TRAIN_PER_UNIT * units
        Xtr = ball(n_train, self.r, X0, np.random.default_rng(SEED_TRAIN))
        Ytr = np.stack([fn(Xtr) for _, _, _, fn in self.targets], axis=1)
        t0 = time.time()
        model = RecenteredRidge(M, N, X0, self.r)
        A = _augmented(model, Xtr)
        sol, info = _solve_augmented(A, Ytr)
        del A
        pred = _augmented(model, self.Xtest) @ sol
        rec = {"M": M, "N": N, "units": units, "n_train": n_train,
               "rank": info["rank"], "n_cols": info["n_cols"],
               "rel_l2": {}, "max_abs": {},
               "weight_norm": {}, "seconds": None}
        for k, name in enumerate(self.keys):
            m = error_metrics(pred[:, k], self.Ytest[:, k])
            rec["rel_l2"][name] = m["rel_l2"]
            rec["max_abs"][name] = m["max_abs"]
            rec["weight_norm"][name] = float(np.linalg.norm(sol[:-1, k]))
        rec["seconds"] = round(time.time() - t0, 2)
        self.cells[key] = rec
        print(f"  M={M:3d} N={N:4d} units={units:5d} n={n_train:6d} "
              f"rank={info['rank']:5d}/{info['n_cols']:5d} [{rec['seconds']:6.1f}s]  "
              + " ".join(f"{k[:5]}={rec['rel_l2'][k]:.1e}" for k in self.keys),
              flush=True)
        return rec


# --- fitting the two floor curves ------------------------------------------

def _fit_forms(xs, es, floor=TRADE_FLOOR):
    """Fit ``e(x)`` above the floor to three decay laws; residuals in log10 error.

    ``exponential`` e = A exp(-a x); ``power`` e = A x^-p; ``stretched``
    e = A exp(-a x^q). Everything is fitted in ``ln e``, so the residual is a relative
    error in the error. The reported residual is the RMS of ``log10(fit) - log10(meas)``.

    "Above the floor" means the leading contiguous run: once a curve has touched the
    floor it bounces around ``10^-14`` and individual later points can read back above
    ``10^-13``. Including those would fit the noise, not the decay.
    """
    n = 0
    for e in es:
        if e <= floor:
            break
        n += 1
    x = np.asarray(xs[:n], dtype=np.float64)
    y = np.log(np.asarray(es[:n], dtype=np.float64))
    out = {"n_points": int(len(x)), "x": x.tolist(),
           "e": np.exp(y).tolist(), "forms": {}, "best": None}
    if len(x) < 3:
        return out

    def resid(pred_ln):
        return float(np.sqrt(np.mean(((pred_ln - y) / np.log(10.0)) ** 2)))

    # exponential: ln e = lnA - a x
    ca = np.polyfit(x, y, 1)
    out["forms"]["exponential"] = {"a": float(-ca[0]), "logA": float(ca[1]),
                                   "resid_log10": resid(np.polyval(ca, x))}
    # power law: ln e = lnA - p ln x
    cp = np.polyfit(np.log(x), y, 1)
    out["forms"]["power"] = {"p": float(-cp[0]), "logA": float(cp[1]),
                             "resid_log10": resid(np.polyval(cp, np.log(x)))}
    # stretched exponential: ln e = lnA - a x^q, q found by 1-D search
    best = None
    for q in np.concatenate([np.linspace(0.05, 3.0, 200), np.linspace(3.0, 6.0, 40)]):
        Xd = np.stack([np.ones_like(x), -x ** q], axis=1)
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        if coef[1] <= 0:
            continue
        rr = resid(Xd @ coef)
        if best is None or rr < best[0]:
            best = (rr, float(q), float(coef[0]), float(coef[1]))
    if best is not None:
        # local refine on q
        lo, hi = max(0.02, best[1] * 0.85), best[1] * 1.15
        for q in np.linspace(lo, hi, 200):
            Xd = np.stack([np.ones_like(x), -x ** q], axis=1)
            coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            if coef[1] <= 0:
                continue
            rr = resid(Xd @ coef)
            if rr < best[0]:
                best = (rr, float(q), float(coef[0]), float(coef[1]))
        out["forms"]["stretched"] = {"a": best[3], "q": best[1], "logA": best[2],
                                     "resid_log10": best[0]}
    out["best"] = min(out["forms"], key=lambda k: out["forms"][k]["resid_log10"])
    return out


def _fit_eval(fit, x):
    """Evaluate the best-fitting form of ``fit`` at ``x``."""
    x = np.asarray(x, dtype=np.float64)
    f = fit["forms"][fit["best"]]
    if fit["best"] == "exponential":
        return np.exp(f["logA"] - f["a"] * x)
    if fit["best"] == "power":
        return np.exp(f["logA"] - f["p"] * np.log(x))
    return np.exp(f["logA"] - f["a"] * x ** f["q"])


def _fit_label(fit):
    f = fit["forms"][fit["best"]]
    if fit["best"] == "exponential":
        return rf"$\exp(-{f['a']:.3g}\,x)$"
    if fit["best"] == "power":
        return rf"$x^{{-{f['p']:.3g}}}$"
    return rf"$\exp(-{f['a']:.3g}\,x^{{{f['q']:.3g}}})$"


def _loglog_interp(xs, es):
    """Piecewise-linear interpolation of ``log e`` against ``log x``, clamped at both
    ends (both curves are flat at their ends, so clamping is the honest extrapolation)."""
    lx = np.log(np.asarray(xs, dtype=np.float64))
    ly = np.log(np.maximum(np.asarray(es, dtype=np.float64), 1e-16))
    order = np.argsort(lx)
    lx, ly = lx[order], ly[order]

    def f(x):
        return np.exp(np.interp(np.log(np.asarray(x, dtype=np.float64)), lx, ly))
    return f


# --- part B: the iso-budget valleys ----------------------------------------

def budget_M_values(B):
    """About ten log-spaced integer direction counts on the budget ``B``."""
    lo = max(2.0, B / 128.0)
    hi = min(float(TRADE_M_CAP), B / float(TRADE_N_MIN))
    if hi < lo:
        return []
    grid = np.geomspace(lo, hi, TRADE_N_PER_BUDGET)
    Ms = sorted({int(round(v)) for v in grid if round(v) >= 2})
    return [M for M in Ms if int(round(B / M)) >= TRADE_N_MIN and M <= TRADE_M_CAP]


def tradeoff():
    """Part A (the two floor curves) then part B (the valleys), one cache throughout."""
    targets = [t for t in TARGETS if t[0] in SPLIT_KEYS]
    targets.sort(key=lambda t: SPLIT_KEYS.index(t[0]))
    keys = [t[0] for t in targets]
    cache = _CellCache(targets)
    path = RESULTS_DIR / TRADE_JSON_NAME
    t_start = time.time()

    def dump(payload):
        with open(path, "w") as f:
            json.dump(payload, f)

    data = {"meta": {"r": SPLIT_R, "x0": X0.tolist(), "lam": LAMBDA, "rcond": RCOND,
                     "margin": MARGIN, "n_train_per_unit": N_TRAIN_PER_UNIT,
                     "n_test": N_TEST, "test_shrink": TEST_SHRINK,
                     "floor": TRADE_FLOOR, "valley_tie": TRADE_VALLEY_TIE,
                     "functions": keys,
                     "e_M_fixed_N": TRADE_EM_N, "e_N_fixed_M": TRADE_EN_M,
                     "budgets": TRADE_BUDGETS}}

    # ---- part A ----------------------------------------------------------
    print(f"part A: e_M(M) at N = {TRADE_EM_N}", flush=True)
    for M in TRADE_EM_M:
        cache.get(M, TRADE_EM_N)
    print(f"part A: e_N(N) at M = {TRADE_EN_M}", flush=True)
    for N in TRADE_EN_N:
        cache.get(TRADE_EN_M, N)
    data["cells"] = sorted(cache.cells.values(), key=lambda c: (c["M"], c["N"]))
    data["e_M"] = {"N": TRADE_EM_N, "M": list(TRADE_EM_M),
                   "rel_l2": {k: [cache.get(M, TRADE_EM_N)["rel_l2"][k]
                                  for M in TRADE_EM_M] for k in keys}}
    data["e_N"] = {"M": TRADE_EN_M, "N": list(TRADE_EN_N),
                   "rel_l2": {k: [cache.get(TRADE_EN_M, N)["rel_l2"][k]
                                  for N in TRADE_EN_N] for k in keys}}
    data["fits"] = {k: {"e_M": _fit_forms(TRADE_EM_M, data["e_M"]["rel_l2"][k]),
                        "e_N": _fit_forms(TRADE_EN_N, data["e_N"]["rel_l2"][k])}
                    for k in keys}
    data["model_test"] = model_test(data)
    dump(data)

    # ---- part B ----------------------------------------------------------
    print("part B: iso-budget valleys", flush=True)
    budgets = []
    for B in TRADE_BUDGETS:
        Ms = budget_M_values(B)
        print(f" B={B:5d}  M in {Ms}", flush=True)
        pts = []
        for M in Ms:
            N = int(round(B / M))
            rec = cache.get(M, N)
            pts.append({"M": M, "N": N, "units": rec["units"],
                        "rel_l2": dict(rec["rel_l2"])})
        budgets.append({"B": B, "points": pts})
        data["budgets"] = budgets
        data["cells"] = sorted(cache.cells.values(), key=lambda c: (c["M"], c["N"]))
        dump(data)

    derive(data)
    data["seconds"] = round(time.time() - t_start, 1)
    dump(data)
    print(f"total {data['seconds']:.0f}s, {len(data['cells'])} distinct (M, N) "
          f"-> {path}", flush=True)
    return data


def derive(data):
    """Everything downstream of the raw cells: the fits, the model test, the valleys and
    the exponents. Recomputed from the saved data on every replot, so the analysis can be
    changed without re-solving anything."""
    keys = data["meta"]["functions"]
    data["fits"] = {k: {"e_M": _fit_forms(data["e_M"]["M"], data["e_M"]["rel_l2"][k]),
                        "e_N": _fit_forms(data["e_N"]["N"], data["e_N"]["rel_l2"][k])}
                    for k in keys}
    data["model_test"] = model_test(data)
    data["valleys"] = valleys(data["budgets"], keys)
    data["alpha"] = alpha_fits(data, keys)
    return data


def model_test(data, exact_path=None):
    """Predict every cell of the exact split grid by ``max(e_M(M), e_N(N))``."""
    exact_path = exact_path or (RESULTS_DIR / EXACT_JSON_NAME)
    if not Path(exact_path).exists():
        return {}
    with open(exact_path) as f:
        rows = json.load(f)
    out = {}
    for k in data["meta"]["functions"]:
        fM = _loglog_interp(data["e_M"]["M"], data["e_M"]["rel_l2"][k])
        fN = _loglog_interp(data["e_N"]["N"], data["e_N"]["rel_l2"][k])
        cells = []
        for x in rows:
            if x["function"] != k:
                continue
            pred = float(max(fM(x["M"]), fN(x["N"])))
            cells.append({"M": x["M"], "N": x["N"], "measured": x["rel_l2"],
                          "predicted": pred,
                          "e_M": float(fM(x["M"])), "e_N": float(fN(x["N"])),
                          "ratio": pred / x["rel_l2"],
                          # a cell sitting on the e_M curve itself is not a prediction
                          "on_curve": x["N"] == data["e_M"]["N"]})
        rat = np.array([c["ratio"] for c in cells])
        fac = np.maximum(rat, 1.0 / rat)
        above = np.array([c["measured"] > 10 * TRADE_FLOOR for c in cells])
        novel = np.array([(not c["on_curve"]) and c["measured"] > 10 * TRADE_FLOOR
                          for c in cells])

        def block(mask):
            if not mask.any():
                return {"n": 0}
            return {"n": int(mask.sum()),
                    "median_ratio": float(np.median(rat[mask])),
                    "min_ratio": float(rat[mask].min()),
                    "max_ratio": float(rat[mask].max()),
                    "worst_factor": float(fac[mask].max())}

        out[k] = {"n_cells": len(cells), "cells": cells,
                  "all": block(np.ones(len(cells), bool)),
                  "above_floor": block(above),
                  "above_floor_off_curve": block(novel)}
    return out


def valleys(budgets, keys):
    """Per (function, budget): the minimizer, the best error, and the valley width."""
    out = {k: [] for k in keys}
    for rec in budgets:
        for k in keys:
            pts = sorted(rec["points"], key=lambda p: p["M"])
            es = np.array([p["rel_l2"][k] for p in pts])
            i = int(np.argmin(es))
            ok = es <= TRADE_VALLEY_TIE * es[i]
            Ms = np.array([p["M"] for p in pts])
            out[k].append({"B": rec["B"], "M_star": int(pts[i]["M"]),
                           "N_star": int(pts[i]["N"]),
                           "units": int(pts[i]["units"]),
                           "best": float(es[i]),
                           "worst": float(es.max()),
                           "n_points": len(pts),
                           "width_M_lo": int(Ms[ok].min()),
                           "width_M_hi": int(Ms[ok].max()),
                           "width_ratio": float(Ms[ok].max() / Ms[ok].min())})
    return out


def _ols(x, y):
    """Slope, intercept and the slope's standard error for ``y = b + m x``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return None
    m, b = np.polyfit(x, y, 1)
    r = y - (m * x + b)
    s2 = float(r @ r) / (n - 2)
    sxx = float(((x - x.mean()) ** 2).sum())
    return {"slope": float(m), "intercept": float(b),
            "se": float(np.sqrt(s2 / sxx)) if sxx > 0 else float("nan"), "n": n}


def _predicted_minimizer(data, k, B):
    """``argmin_M max(e_M(M), e_N(B/M))`` from the part-A fitted laws."""
    lo = max(2.0, B / 128.0)
    hi = min(float(TRADE_M_CAP), B / float(TRADE_N_MIN))
    if hi <= lo:
        return None
    Ms = np.geomspace(lo, hi, 600)
    eM = _fit_eval(data["fits"][k]["e_M"], Ms)
    eN = _fit_eval(data["fits"][k]["e_N"], B / Ms)
    return float(Ms[int(np.argmax(-np.maximum(eM, eN)))])


def alpha_fits(data, keys):
    """``M* ~ B^alpha`` measured, and the same exponent from the part-A laws."""
    out = {}
    for k in keys:
        rows = [v for v in data["valleys"][k] if v["best"] > TRADE_FLOOR]
        meas = _ols(np.log(([v["B"] for v in rows])), np.log([v["M_star"] for v in rows]))
        pred_pts = [(v["B"], _predicted_minimizer(data, k, v["B"])) for v in rows]
        pred_pts = [(b, m) for b, m in pred_pts if m is not None]
        pred = _ols(np.log([b for b, _ in pred_pts]), np.log([m for _, m in pred_pts])) \
            if len(pred_pts) >= 3 else None
        out[k] = {"budgets_used": [v["B"] for v in rows],
                  "alpha": meas["slope"] if meas else None,
                  "alpha_se": meas["se"] if meas else None,
                  "alpha_intercept": meas["intercept"] if meas else None,
                  "n_budgets": meas["n"] if meas else 0,
                  "alpha_pred": pred["slope"] if pred else None,
                  "alpha_pred_se": pred["se"] if pred else None,
                  "alpha_pred_intercept": pred["intercept"] if pred else None,
                  "pred_points": [{"B": b, "M": m} for b, m in pred_pts]}
    return out


def plot_tradeoff(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    derive(data)

    keys = data["meta"]["functions"]
    names = {t[0]: t[1] for t in TARGETS}
    formulas = {t[0]: t[2] for t in TARGETS}
    Bs = [b["B"] for b in data["budgets"]]
    norm = LogNorm(vmin=min(Bs), vmax=max(Bs))
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(3, 4, figsize=(21.5, 15.5))

    for col, k in enumerate(keys):
        # ---- row 1: the two floor curves ---------------------------------
        ax = axes[0, col]
        Ms, eM = data["e_M"]["M"], data["e_M"]["rel_l2"][k]
        Ns, eN = data["e_N"]["N"], data["e_N"]["rel_l2"][k]
        ax.plot(Ms, np.maximum(eM, 1e-16), "-o", ms=5, color="tab:blue",
                label=rf"$e_M(M)$ at $N$ = {data['e_M']['N']}")
        ax.plot(Ns, np.maximum(eN, 1e-16), "-s", ms=5, color="tab:red",
                label=rf"$e_N(N)$ at $M$ = {data['e_N']['M']}")
        fitM, fitN = data["fits"][k]["e_M"], data["fits"][k]["e_N"]
        lab = []
        for fit, xs, col_, nm in ((fitM, Ms, "tab:blue", "e_M"),
                                  (fitN, Ns, "tab:red", "e_N")):
            if fit.get("best") is None:
                lab.append(f"{nm}: no fit")
                continue
            xf = np.geomspace(min(fit["x"]), max(fit["x"]), 200)
            ax.plot(xf, _fit_eval(fit, xf), "--", lw=1.6, color=col_,
                    label=f"{fit['best']} fit, {_fit_label(fit)}")
            lab.append(f"${nm[0]}_{nm[-1]}$")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 128])
        ax.set_xticklabels(["2", "4", "8", "16", "32", "128"])
        ax.minorticks_off()
        ax.set_ylim(1e-15, 1e1)
        ax.set_xlim(1.7, 150)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("count ($M$ for the blue curve, $N$ for the red)")
        if col == 0:
            ax.set_ylabel(r"relative $L_2$ inside the ball")
        ax.set_title(f"{names[k]}\n{formulas[k]}", fontsize=10, pad=44)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                  frameon=False, fontsize=7.5, handlelength=1.6,
                  columnspacing=1.0)

        # ---- row 2: the iso-budget valleys -------------------------------
        ax = axes[1, col]
        for rec in data["budgets"]:
            pts = sorted(rec["points"], key=lambda p: p["M"])
            xs = [p["M"] for p in pts]
            es = np.maximum([p["rel_l2"][k] for p in pts], 1e-16)
            c = cmap(norm(rec["B"]))
            ax.plot(xs, es, "-o", ms=3.5, lw=1.2, color=c)
            i = int(np.argmin(es))
            ax.plot(xs[i], es[i], "*", ms=15, color=c, mec="k", mew=0.7, zorder=5)
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 48])
        ax.set_xticklabels(["2", "4", "8", "16", "32", "48"])
        ax.minorticks_off()
        ax.set_ylim(1e-15, 1e1)
        ax.set_xlim(1.7, 60)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("directions $M$ (with $N \\approx B/M$)")
        if col == 0:
            ax.set_ylabel(r"relative $L_2$ inside the ball")
        star = plt.Line2D([], [], ls="none", marker="*", ms=12, color="0.35",
                          mec="k", mew=0.7, label="minimizer $M^*(B)$")
        ax.legend(handles=[star], loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=1, frameon=False, fontsize=8.5)

        # ---- row 3: M* and N* against the budget -------------------------
        ax = axes[2, col]
        vs = data["valleys"][k]
        used = set(data["alpha"][k]["budgets_used"])
        bb = np.array([v["B"] for v in vs], float)
        ms = np.array([v["M_star"] for v in vs], float)
        ns = np.array([v["N_star"] for v in vs], float)
        inuse = np.array([v["B"] in used for v in vs])
        ax.plot(bb[inuse], ms[inuse], "o", ms=7, color="tab:blue", label=r"$M^*(B)$")
        ax.plot(bb[inuse], ns[inuse], "s", ms=7, color="tab:red", label=r"$N^*(B)$")
        if (~inuse).any():
            ax.plot(bb[~inuse], ms[~inuse], "o", ms=7, mfc="none", mec="tab:blue",
                    label=r"$M^*$, $N^*$ at the floor (excluded from the fit)")
            ax.plot(bb[~inuse], ns[~inuse], "s", ms=7, mfc="none", mec="tab:red")
        a = data["alpha"][k]
        bl = np.geomspace(bb.min(), bb.max(), 100)
        if a["alpha"] is not None:
            ax.plot(bl, np.exp(a["alpha_intercept"]) * bl ** a["alpha"], "-",
                    color="tab:blue", lw=1.4,
                    label=rf"$M^*\propto B^{{{a['alpha']:.2f}\pm{a['alpha_se']:.2f}}}$")
            ax.plot(bl, bl / (np.exp(a["alpha_intercept"]) * bl ** a["alpha"]), "-",
                    color="tab:red", lw=1.4,
                    label=rf"$N^*\propto B^{{{1 - a['alpha']:.2f}}}$")
        if a["alpha_pred"] is not None:
            ax.plot(bl, np.exp(a["alpha_pred_intercept"]) * bl ** a["alpha_pred"], "--",
                    color="0.35", lw=1.6,
                    label=rf"part-A prediction, $B^{{{a['alpha_pred']:.2f}}}$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(Bs)
        ax.set_xticklabels([str(b) for b in Bs], rotation=60, fontsize=7.5)
        ax.set_yticks([2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_yticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
        ax.minorticks_off()
        ax.set_ylim(1.5, 320)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("unit budget $B = MN$")
        if col == 0:
            ax.set_ylabel("optimal $M^*$, $N^*$")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                  frameon=False, fontsize=7.5, handlelength=1.6, columnspacing=1.0)

    fig.suptitle("the $M$-vs-$N$ tradeoff at $r$ = "
                 f"{data['meta']['r']:g}: two independent floors (top), the iso-budget "
                 "valleys they produce (middle), and how the optimal split moves with "
                 "the budget (bottom)", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.955, 0.965))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    p = axes[1, -1].get_position()
    cax = fig.add_axes([0.965, p.y0, 0.011, p.height])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("unit budget $B$", fontsize=9)
    out = FIG_DIR / "tradeoff_2d.png"
    fig.savefig(out, dpi=140)
    print("saved", out, flush=True)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _series(rows, key, r):
    pts = sorted((x["M"], x["rel_l2"]) for x in rows if x["function"] == key and x["r"] == r)
    return [p[0] for p in pts], [p[1] for p in pts]


def _first_below(rows, key, r, tol=1e-10):
    Ms, es = _series(rows, key, r)
    hits = [m for m, e in zip(Ms, es) if e < tol]
    return hits[0] if hits else None


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radii = sorted({x["r"] for x in rows})
    Ms_all = sorted({x["M"] for x in rows})
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5), sharex=True, sharey=True)
    handles = None
    for ax, (key, name, formula, _) in zip(axes.ravel(), TARGETS):
        for i, r in enumerate(radii):
            Ms, es = _series(rows, key, r)
            ax.semilogy(Ms, np.maximum(es, 1e-16), "-o", ms=5,
                        color=cmap(i / max(1, len(radii) - 1)),
                        label=f"data radius $r$ = {r:g}")
        ax.axhline(1e-10, color="0.5", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ms_all)
        ax.set_xticklabels([str(m) for m in Ms_all])
        ax.minorticks_off()
        ax.set_ylim(1e-15, 1e1)
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"{name}\n{formula}", fontsize=10)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    for ax in axes[-1]:
        ax.set_xlabel("number of directions $M$")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative $L_2$ inside the ball")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=4, frameon=False, fontsize=11)
    fig.suptitle("2-D direction cliff: error on the inner 90% of the data ball vs the "
                 f"number of directions ({N_PER} offsets per direction, "
                 "25% collar, one truncated-SVD readout solve)",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = FIG_DIR / "direction_cliff_2d.png"
    fig.savefig(out, dpi=140)
    print("saved", out, flush=True)


def plot_threshold(rows):
    """Two summaries: where the cliff sits, and what the readout weights do across it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radii = sorted({x["r"] for x in rows})
    keys = [t[0] for t in TARGETS]
    names = [t[1] for t in TARGETS]
    grid = np.full((len(keys), len(radii)), np.nan)
    for i, key in enumerate(keys):
        for j, r in enumerate(radii):
            m = _first_below(rows, key, r)
            if m is not None:
                grid[i, j] = m

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.6))

    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")
    im = ax.imshow(masked, cmap=cmap, aspect="auto",
                   norm=matplotlib.colors.LogNorm(vmin=1, vmax=16))
    for i in range(len(keys)):
        for j in range(len(radii)):
            v = grid[i, j]
            txt = "--" if not np.isfinite(v) else f"{int(v)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="0.25" if not np.isfinite(v) else
                    ("white" if v <= 4 else "black"))
    ax.set_xticks(range(len(radii)))
    ax.set_xticklabels([f"r = {r:g}" for r in radii])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(names)
    ax.set_title("smallest $M$ reaching relative $L_2 < 10^{-10}$ inside the ball\n"
                 "(-- : never, up to $M$ = 16)", fontsize=11)
    fig.colorbar(im, ax=ax, label="directions $M$")

    cmapr = plt.get_cmap("viridis")
    for i, r in enumerate(radii):
        e = [x["rel_l2"] for x in rows if x["r"] == r]
        w = [x["readout_norm"] for x in rows if x["r"] == r]
        ax2.loglog(np.maximum(e, 1e-16), np.maximum(w, 1e-3), "o", ms=4, alpha=0.7,
                   color=cmapr(i / max(1, len(radii) - 1)), label=f"data radius $r$ = {r:g}")
    ax2.axvline(1e-10, color="0.5", ls=":", lw=1)
    ax2.set_xlim(1e-16, 1e1)
    ax2.set_ylim(1e-3, 1e11)
    ax2.set_xlabel("relative $L_2$ inside the ball")
    ax2.set_ylabel(r"readout weight norm $\|w\|_2$")
    ax2.grid(alpha=0.3, which="both")
    ax2.set_title("every fit: error against the size of the readout it needed",
                  fontsize=11)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
               fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "cliff_summary_2d.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
