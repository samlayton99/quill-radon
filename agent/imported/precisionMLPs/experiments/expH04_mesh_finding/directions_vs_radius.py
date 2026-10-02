"""How many directions does precision ON THE DATA need, and what sets that number?

Theory (plane-wave expansion of the ridge integral): at a point x, the integrand of the
direction quadrature has angular bandwidth k |x - x0|, with k the target's bandwidth and
x0 the origin the offsets are measured from. So for precision on data within radius r of
x0 the direction count needed is M ~ (k r)^(d-1), not (k R)^(d-1) for the whole domain.

Test: fixed target (fast concentric waves, the suite's task 2.3 / 3.3 function), data
uniform in a ball of radius r around an off-center point x0; the ridge system is
recentered on x0 with offsets confined to the data's projection band (c = v.x0 + t,
|t| <= 1.25 r, evenly spaced, gamma = 0.25/h); sweep the number of directions M at a
fixed, generous number of offsets per direction. Error measured on the data ball
("on data") and on the whole cube ("everywhere"). Prediction: the M at which the on-data
error reaches the floor grows like r^(d-1).

Usage:
    uv run --extra dev python experiments/expH04_mesh_finding/directions_vs_radius.py --dim 2
    uv run --extra dev python experiments/expH04_mesh_finding/directions_vs_radius.py --dim 3
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

from h01suite.baseline import even_directions, _solve_svd, LAMBDA, RCOND   # noqa: E402
from h01suite.metrics import error_metrics                                  # noqa: E402
from h01suite.tasks import get_task                                         # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH04_mesh_finding"
X0 = {2: np.array([0.35, -0.25]), 3: np.array([0.35, -0.25, 0.2])}
RADII = {2: [0.1, 0.2, 0.4, 0.8], 3: [0.15, 0.3, 0.6]}
N_DIRS = {2: [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96], 3: [8, 16, 32, 64, 128, 256]}
N_PER = {2: 48, 3: 32}
N_TRAIN_PER_UNIT = 6


def ball(n, d, r, x0, rng):
    g = rng.normal(size=(n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    u = rng.uniform(size=(n, 1)) ** (1.0 / d)
    return x0 + r * u * g


class RecenteredRidge:
    """Even directions; along each, n_per offsets evenly spaced over the data's projection
    band around v.x0 (collar 25%), width from the spacing."""

    def __init__(self, d, n_dir, n_per, x0, r, margin=1.25):
        V = even_directions(d, n_dir)
        dirs, cens, gams = [], [], []
        for v in V:
            T = margin * r * float(np.linalg.norm(v))
            h = 2.0 * T / n_per
            t = -T + (np.arange(n_per) + 0.5) * h
            dirs.append(np.repeat(v[None, :], n_per, axis=0))
            cens.append(float(v @ x0) + t)
            gams.append(np.full(n_per, LAMBDA / h))
        self.directions, self.centers = np.vstack(dirs), np.concatenate(cens)
        self.gammas = np.concatenate(gams)

    def features(self, X):
        return np.tanh(self.gammas[None, :] * (X @ self.directions.T - self.centers[None, :]))

    def fit(self, X, y):
        self.weights, self.bias, self.info = _solve_svd(self.features(X), y, RCOND)
        return self

    def predict(self, X):
        return self.features(X) @ self.weights + self.bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--plot", action="store_true", help="replot from the saved data")
    args = ap.parse_args()
    d = args.dim
    if args.plot:
        with open(RESULTS_DIR / f"directions_vs_radius_d{d}.json") as f:
            plot(json.load(f), d)
        return
    task = get_task("2.3" if d == 2 else "3.3")
    x0, n_per = X0[d], N_PER[d]
    rng = np.random.default_rng(0)
    rows = []
    Xcube = rng.uniform(-1, 1, size=(20000, d))
    ycube = task.F(Xcube)
    for r in RADII[d]:
        Xtest = ball(20000, d, r, x0, np.random.default_rng(1))
        ytest = task.F(Xtest)
        for M in N_DIRS[d]:
            B = M * n_per
            X = ball(N_TRAIN_PER_UNIT * B, d, r, x0, np.random.default_rng(2))
            y = task.F(X)
            t0 = time.time()
            m = RecenteredRidge(d, M, n_per, x0, r).fit(X, y)
            e_on = error_metrics(m.predict(Xtest), ytest)["rel_l2"]
            e_cube = error_metrics(m.predict(Xcube), ycube)["rel_l2"]
            rows.append({"d": d, "r": r, "n_dir": M, "n_per": n_per, "budget": B,
                         "on_data": e_on, "everywhere": e_cube, "rank": m.info["rank"],
                         "seconds": time.time() - t0})
            print(f"d={d} r={r:<4g} M={M:4d} B={B:6d}  on data={e_on:.1e}  "
                  f"everywhere={e_cube:.1e}  {time.time()-t0:.0f}s", flush=True)
            with open(RESULTS_DIR / f"directions_vs_radius_d{d}.json", "w") as f:
                json.dump(rows, f)
    plot(rows, d)


def plot(rows, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    radii = sorted({r["r"] for r in rows})
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, key in zip(axes[:2], ["on_data", "everywhere"]):
        for k, r in enumerate(radii):
            pts = sorted([(x["n_dir"], x[key]) for x in rows if x["r"] == r])
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], "-o", ms=4,
                      color=cmap(k / max(1, len(radii) - 1)), label=f"data radius r = {r:g}")
        ax.set_xlabel("number of directions M")
        ax.set_ylabel("relative $L_2$ " + ("on the data ball" if key == "on_data"
                                            else "over the whole cube"))
        ax.set_ylim(1e-15, 1e1 if key == "on_data" else 1e10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9,
                   frameon=False)
    # M needed to reach 1e-10 on the data, vs r
    ax = axes[2]
    need = []
    for r in radii:
        pts = sorted([(x["n_dir"], x["on_data"]) for x in rows if x["r"] == r])
        hit = [p[0] for p in pts if p[1] < 1e-10]
        need.append((r, hit[0] if hit else np.nan))
    ax.loglog([p[0] for p in need], [p[1] for p in need], "ko-", label="measured $M(10^{-10})$")
    rr = np.array(radii)
    fin = [p for p in need if np.isfinite(p[1])]
    if fin:
        r0, M0 = fin[-1]          # anchor the asymptote at the largest radius measured
        ax.loglog(rr, M0 * (rr / r0) ** (d - 1), "r--",
                  label=f"$\\propto r^{{{d-1}}}$ (large-$kr$ theory)")
        from math import comb
        ax.axhline(comb(12 + d - 1, d - 1), color="tab:blue", ls=":", lw=1,
                   label=f"polynomial floor $\\binom{{12+d-1}}{{d-1}}$ = {comb(12 + d - 1, d - 1)}")
    ax.set_xlabel("data radius r")
    ax.set_ylabel("directions needed for $10^{-10}$ on the data")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"d = {d}: ridge system recentered on the data; offsets confined to the "
                 f"data's projection band; {N_PER[d]} offsets per direction", y=1.0, fontsize=10)
    fig.tight_layout()
    out = RESULTS_DIR / "figures" / f"directions_vs_radius_d{d}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
