"""Why do adapted meshes sit 10-100x above the even floor on already-resolved targets?

expH02 is the clue: smooth non-uniform spacing (half-Gaussian, bimodal) reached the same
2e-14 floor as even spacing. So the price must come from something this pipeline does
that expH02 did not. Candidates, each isolated here:

  A  roughness of the spacing profile at the scale of one gap: the monitor is a
     histogram smoothed at only 1.5 gaps, and the grading step is a Lipschitz envelope
     that leaves kinks. Knobs: smoothing bandwidth (bw_mult) and grading cap (grade).
  B  the widths: non-uniform gamma_j -> larger readout weights -> more fp64 cancellation.
     Diagnostic: max |w| and the kept rank, against the even mesh.
  C  the pipeline itself (interpolation, cumulative-sum placement): push an analytic
     smooth density (expH02's half-Gaussian shape) through it. If that pays, the
     pipeline is at fault; if not, the monitor's roughness is.
  D  irregularity alone: the even mesh with a small random jitter of the positions
     (neighbor ratio ~1.02..1.2), widths from the jittered local gaps. No monitor at all.

All on targets the even mesh resolves at the budget: 1.11, 1.13, 1.14 (d = 1) at
B = 128, 256, 512, and 2.11 (d = 2) at B = 1024. Dense-region relative L2, plus a
smoothness metric of the placed mesh: max |log(h_{j+1}/h_j)| and the max second
difference of log h_j.

Usage:
    uv run --extra dev python experiments/expH04_mesh_finding/floor_price.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "experiments" / "expH01_highdim_suite"))
sys.path.append(str(Path(__file__).resolve().parent))

from h01suite.baseline import EvenGeometry, _solve_svd, LAMBDA, EDGE_MARGIN, RCOND  # noqa
from h01suite.metrics import error_metrics                                          # noqa
from h01suite.tasks import get_task                                                 # noqa
from mesh import AdaptiveGeometry, Monitors, surrogate_derivatives, place_by_density  # noqa

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH04_mesh_finding"
CELLS = [("1.11", 128), ("1.11", 256), ("1.13", 256), ("1.13", 512), ("1.14", 128),
         ("1.14", 256), ("2.11", 1024)]
BW_MULTS = [1.5, 3.0, 6.0, 12.0]
GRADES = [0.15, 0.05, 0.02]
JITTERS = [0.0, 0.02, 0.05, 0.1, 0.2]


def smoothness(centers_by_dir):
    """max |log(h_{j+1}/h_j)| and max |second difference of log h| over all directions."""
    r1, r2 = 0.0, 0.0
    for c in centers_by_dir:
        h = np.diff(c)
        lh = np.log(h)
        r1 = max(r1, float(np.max(np.abs(np.diff(lh)))))
        if len(lh) > 2:
            r2 = max(r2, float(np.max(np.abs(np.diff(lh, 2)))))
    return r1, r2


def centers_by_direction(model):
    out, k = [], 0
    for n in model.per_direction if hasattr(model, "per_direction") else \
            [model.n_per_direction] * model.n_directions:
        out.append(np.sort(model.centers[k:k + int(n)]))
        k += int(n)
    return out


class JitteredEven(EvenGeometry):
    """Even mesh whose positions are perturbed by N(0, (jitter*h)^2), widths from the
    jittered local gaps (so gamma_j h_j = lambda still holds)."""

    def __init__(self, d, budget, jitter, seed=0):
        super().__init__(d=d, budget=budget)
        rng = np.random.default_rng(seed)
        k = 0
        n = self.n_per_direction
        for i in range(self.n_directions):
            c = self.centers[k:k + n].copy()
            h0 = c[1] - c[0]
            c = np.sort(c + rng.normal(0.0, jitter * h0, size=n))
            h = np.empty(n)
            h[1:-1] = 0.5 * (c[2:] - c[:-2]); h[0] = c[1] - c[0]; h[-1] = c[-1] - c[-2]
            self.centers[k:k + n] = c
            self.gammas[k:k + n] = self.lam / h
            k += n
        self.per_direction = np.full(self.n_directions, n)


class SmoothDensityMesh(AdaptiveGeometry):
    """expH02's half-Gaussian shape as an analytic monitor: density ~ exp(-u^2/2) with
    u running from -1.5 at -T to 0 at +T. Pushed through the same pipeline."""

    def build_smooth(self, s_level):
        V = self.unique_directions
        dirs, cens, gams = [], [], []
        for i, v in enumerate(V):
            n = int(self.per_direction[i])
            T = self.margin * float(np.abs(v).sum())
            grid = np.linspace(-T, T, 2001)
            u = -1.5 + 1.5 * (grid + T) / (2 * T)
            m = np.exp(-0.5 * u * u)
            c, hj, info = place_by_density(grid, m, n, s_level, self.grade)
            dirs.append(np.repeat(v[None, :], n, axis=0)); cens.append(c)
            gams.append(self.lam / hj)
        self.directions = np.vstack(dirs); self.centers = np.concatenate(cens)
        self.gammas = np.concatenate(gams)
        return self


def record(tag, model, task, X, y, sets, y_true, extra):
    t0 = time.time()
    model.fit(X, y)
    err = {k: error_metrics(model.predict(sets[k]), y_true[k]) for k in ("dense_region", "uniform")}
    r1, r2 = smoothness(centers_by_direction(model))
    rec = {"tag": tag, "task": task.id, "d": task.d, "budget": len(model.centers),
           "dense": err["dense_region"]["rel_l2"], "uniform": err["uniform"]["rel_l2"],
           "max_abs_w": float(np.max(np.abs(model.weights))),
           "w_norm2": float(np.linalg.norm(model.weights)),
           "rank": model.info["rank"], "n_cols": model.info["n_cols"],
           "sigma_max": model.info["largest_singular_value"],
           "sigma_min": model.info["smallest_singular_value"],
           "gamma_min": float(model.gammas.min()), "gamma_max": float(model.gammas.max()),
           "log_ratio_max": r1, "log_h_second_diff_max": r2,
           "seconds": time.time() - t0, **extra}
    print(f"  {task.id:5s} B={rec['budget']:5d} {tag:28s} dense={rec['dense']:.1e} "
          f"unif={rec['uniform']:.1e} |w|max={rec['max_abs_w']:.1e} rank={rec['rank']}/"
          f"{rec['n_cols']} ratio={np.exp(r1):.3f} d2logh={r2:.3f}", flush=True)
    return rec


def main():
    rows = []
    for tid, B in CELLS:
        task = get_task(tid)
        X, y = task.train_set(8 * B, seed=0)
        sets = task.test_sets(seed=10_000)
        y_true = {k: task.F(sets[k]) for k in ("dense_region", "uniform")}
        even = EvenGeometry(d=task.d, budget=B)
        rows.append(record("even", even, task, X, y, sets, y_true, {"kind": "even"}))
        V = AdaptiveGeometry(d=task.d, budget=B).unique_directions
        D = surrogate_derivatives(even, X, V, 1)
        # A: smoothing and grading knobs on the estimated-slope monitor
        for bw in BW_MULTS:
            for g in GRADES:
                geo = AdaptiveGeometry(d=task.d, budget=B, bw_mult=bw, grade=g)
                geo.build(X, Monitors("roughness", r=1, deriv=D))
                rows.append(record(f"slope bw={bw:g} g={g:g}", geo, task, X, y, sets, y_true,
                                   {"kind": "slope", "bw_mult": bw, "grade": g}))
        # C: analytic smooth density through the pipeline
        for s_level in (1.0 / 3.0, 2.0 / 3.0, 1.0):
            geo = SmoothDensityMesh(d=task.d, budget=B, s=s_level).build_smooth(s_level)
            rows.append(record(f"halfgauss s={s_level:.2f}", geo, task, X, y, sets, y_true,
                               {"kind": "halfgauss", "s": s_level}))
        # D: jitter only
        for j in JITTERS[1:]:
            geo = JitteredEven(task.d, B, j)
            rows.append(record(f"jitter {j:g}", geo, task, X, y, sets, y_true,
                               {"kind": "jitter", "jitter": j}))
        with open(RESULTS_DIR / "floor_price.json", "w") as f:
            json.dump(rows, f)
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = sorted({(r["task"], r["budget"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cmap = plt.get_cmap("tab10")
    # panel 1: dense error vs the smoothness metric, all rungs, colored by kind
    ax = axes[0]
    kinds = {"even": ("k", "o"), "slope": ("tab:green", "^"), "halfgauss": ("tab:red", "s"),
             "jitter": ("tab:blue", "D")}
    for kind, (c, m) in kinds.items():
        sub = [r for r in rows if r["kind"] == kind]
        ax.loglog([max(r["log_h_second_diff_max"], 1e-4) for r in sub], [r["dense"] for r in sub],
                  m, color=c, ms=5, alpha=0.7, label=kind)
    ax.set_xlabel("roughness of the mesh: max |second difference of log h_j|")
    ax.set_ylabel("relative $L_2$, dense region")
    ax.set_ylim(1e-15, 1e-8)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=9, frameon=False)
    # panel 2: dense error vs max |w|
    ax = axes[1]
    for kind, (c, m) in kinds.items():
        sub = [r for r in rows if r["kind"] == kind]
        ax.loglog([r["max_abs_w"] for r in sub], [r["dense"] for r in sub], m, color=c, ms=5,
                  alpha=0.7, label=kind)
    ax.set_xlabel("largest readout weight |w|")
    ax.set_ylim(1e-15, 1e-8)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=9, frameon=False)
    # panel 3: slope monitor, error vs smoothing bandwidth, one line per grade, per cell
    ax = axes[2]
    for k, (tid, B) in enumerate(cells):
        ev = next(r["dense"] for r in rows if r["kind"] == "even" and r["task"] == tid
                  and r["budget"] == B)
        for g, ls in zip(GRADES, ["-", "--", ":"]):
            pts = sorted([(r["bw_mult"], r["dense"] / ev) for r in rows if r["kind"] == "slope"
                          and r["task"] == tid and r["budget"] == B and r["grade"] == g])
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], ls, color=cmap(k), marker="o",
                      ms=3, label=f"{tid} B={B}, g={g}" if g == GRADES[0] else None)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("monitor smoothing (in even gaps); solid g=0.15, dashed 0.05, dotted 0.02")
    ax.set_ylabel("error / even error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    out = RESULTS_DIR / "figures" / "floor_price.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
