"""Experiment expE01 -- 2D ridge-geometry zoo: which geometry reaches precision?

Generalizes the original 2D hex experiment to six geometries (see geometries.py) compared head-to-head on
a function suite, all fp64 least squares, lambda finetuned per (geometry, width).

Each neuron is a ridge tanh(gamma*(w_m . x - t_m)); a geometry supplies the
(direction w_m, offset t_m) pairs. The bandwidth is set by a common dimensionless
lambda: gamma = lambda / h_ref with h_ref = 2R/sqrt(N) (the same reference spacing
for every geometry at a given width), swept per cell and the best taken.

Targets are fit on the unit disk; neurons live in the disk of radius R = 1.4.
Error is eval L_inf and relative L2 on a dense grid over the unit disk.

Figure (the deliverable): 4 rows (targets: 2 radially-symmetric, 2 asymmetric) x
2 cols (relative L2, L_inf); x = width N (log), y = best-over-lambda error (log);
one line per geometry.

Usage:
    python experiments/expE01_geometry_zoo_2d/run.py            # collect + plot
    python experiments/expE01_geometry_zoo_2d/run.py --plot     # plot from saved data
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometries import GEOMETRIES
from src.data.sampling2d import disk_grid, disk_uniform
from src.data.targets2d import get_target_2d

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_E_2d" / "expE01_geometry_zoo_2d"
DATA_PATH = RESULTS_DIR / "data.json"

# --- experiment grid ---
# Per-geometry halo (option B, node-count style): tangent/area geometries are best
# with a modest collar (R=1.6, empirically optimal); the Radon geometries are
# 1D-per-direction and take a node-count linear halo (R=2.5), sized so the
# INTERIOR offset resolution matches the tangent geometries (~0.63*sqrt(N)).
TANGENT_R = 1.6
RADON_R = 2.5
GEOM_RADIUS = {
    "hex": TANGENT_R, "mode_radial": TANGENT_R, "sixn_rings": TANGENT_R,
    "random_ridges": TANGENT_R, "radon_tensor": RADON_R, "radon_interlaced": RADON_R,
}
WIDTHS = [64, 144, 256, 576, 1024, 2048, 4096]
# 2 radially-symmetric targets, 2 asymmetric (gauss_bump required).
TARGETS = ["gauss_bump", "runge2d", "sine2d", "mixed2d"]
SYMMETRIC = {"gauss_bump", "runge2d"}
LAMBDA_GRID = [0.05, 0.08, 0.12, 0.18, 0.26, 0.38, 0.55, 0.80, 1.20]
N_TRAIN = 8000                                # > 2x max width (overdetermined)
EVAL_SIDE = 120                               # ~11k eval points on the unit disk
RANDOM_SEED = 0                               # for random_ridges (deterministic otherwise)
RCOND = 1e-13
WEIGHTS_DIR = RESULTS_DIR / "weights"         # saved readout + geometry per cell

GEOM_ORDER = list(GEOMETRIES.keys())
GEOM_COLORS = {
    "hex": "#1f77b4", "mode_radial": "#2ca02c", "sixn_rings": "#9467bd",
    "random_ridges": "#8c564b", "radon_tensor": "#d62728",
    "radon_interlaced": "#ff7f0e",
}
GEOM_LABELS = {
    "hex": "1 hex", "mode_radial": "2 mode-radial", "sixn_rings": "4 6k-rings",
    "random_ridges": "5 random", "radon_tensor": "6a radon-tensor",
    "radon_interlaced": "6b radon-interlaced",
}


def collect_data():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_tr = disk_uniform(N_TRAIN, radius=1.0, seed=0)
    X_ev = disk_grid(EVAL_SIDE, radius=1.0)
    ys_tr = {t: get_target_2d(t).fn_numpy(X_tr) for t in TARGETS}
    ys_ev = {t: get_target_2d(t).fn_numpy(X_ev) for t in TARGETS}
    y_norms = {t: float(np.linalg.norm(ys_ev[t])) for t in TARGETS}

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    total = len(GEOM_ORDER) * len(WIDTHS)
    done = 0
    for geom in GEOM_ORDER:
        radius = GEOM_RADIUS[geom]
        for N in WIDTHS:
            dirs, offs = GEOMETRIES[geom](N, radius, seed=RANDOM_SEED)
            n = len(offs)
            # Common interior-spacing reference so lambda is comparable across
            # geometries (interior resolution is matched by construction).
            h_ref = 2.8 / np.sqrt(N)
            DXt = X_tr @ dirs.T                       # [n_train, n]
            DXe = X_ev @ dirs.T                        # [n_eval, n]
            best = {t: {"linf": np.inf, "rel_l2": np.inf} for t in TARGETS}
            for lam in LAMBDA_GRID:
                gamma = lam / h_ref
                Phi_tr = np.tanh(gamma * (DXt - offs[None, :]))
                Phi_ev = np.tanh(gamma * (DXe - offs[None, :]))
                Aug = np.hstack([Phi_tr, np.ones((N_TRAIN, 1))])
                U, s, Vt = np.linalg.svd(Aug, full_matrices=False)
                s_inv = np.where(s > RCOND * s[0], 1.0 / s, 0.0)
                for t in TARGETS:
                    sol = Vt.T @ (s_inv * (U.T @ ys_tr[t]))
                    resid = Phi_ev @ sol[:-1] + sol[-1] - ys_ev[t]
                    linf = float(np.abs(resid).max())
                    rel_l2 = float(np.linalg.norm(resid) / y_norms[t])
                    if linf < best[t]["linf"]:
                        best[t] = {"linf": linf, "rel_l2": rel_l2, "lambda": lam,
                                   "gamma": gamma, "v": sol[:-1].copy(),
                                   "bias": float(sol[-1])}
            for t in TARGETS:
                b = best[t]
                # Save the exact solved MLP: f(x) = bias + sum v*tanh(gamma*(dir.x - off)).
                np.savez(WEIGHTS_DIR / f"{geom}_N{N}_{t}.npz",
                         dirs=dirs, offsets=offs, gamma=b["gamma"], v=b["v"],
                         bias=b["bias"], radius=radius, lam=b["lambda"],
                         linf=b["linf"], rel_l2=b["rel_l2"])
                rows.append({"geom": geom, "N": N, "n_ridges": n, "target": t,
                             "linf": b["linf"], "rel_l2": b["rel_l2"],
                             "lambda": b["lambda"], "gamma": b["gamma"],
                             "weights_file": f"weights/{geom}_N{N}_{t}.npz"})
            done += 1
            tag = " ".join(f"{t[:5]}={best[t]['linf']:.1e}" for t in TARGETS)
            print(f"[{done}/{total}] {geom:17s} N={N:5d} (n={n}) | {tag} | {time.time()-t0:.0f}s")

    out = {
        "config": {
            "geom_radius": GEOM_RADIUS, "widths": WIDTHS, "targets": TARGETS,
            "symmetric": sorted(SYMMETRIC), "lambda_grid": LAMBDA_GRID,
            "n_train": N_TRAIN, "eval_side": EVAL_SIDE, "random_seed": RANDOM_SEED,
            "geom_order": GEOM_ORDER,
        },
        "rows": rows,
    }
    with open(DATA_PATH, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {DATA_PATH} ({time.time()-t0:.0f}s total)")
    return out


def plot_suite(data):
    cfg = data["config"]
    rows = data["rows"]
    targets, widths = cfg["targets"], cfg["widths"]
    metrics = [("rel_l2", r"Relative $L_2$"), ("linf", r"$L_\infty$")]
    fig, axes = plt.subplots(len(targets), 2, figsize=(13, 3.4 * len(targets)), sharex=True)
    fig.suptitle("ExpE01: 2D ridge geometry comparison (best over $\\lambda$, lstsq, fp64)",
                 fontsize=14, y=0.997)
    for ri, target in enumerate(targets):
        sym = "symmetric" if target in cfg["symmetric"] else "asymmetric"
        for ci, (mkey, mlabel) in enumerate(metrics):
            ax = axes[ri][ci]
            for geom in cfg["geom_order"]:
                ys = []
                for N in widths:
                    r = next(x for x in rows if x["geom"] == geom and x["N"] == N
                             and x["target"] == target)
                    ys.append(r[mkey])
                ax.loglog(widths, ys, "-o", color=GEOM_COLORS[geom],
                          markersize=4, label=GEOM_LABELS[geom])
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylabel(f"{target} ({sym})\n{mlabel}" if ci == 0 else mlabel)
            if ri == 0:
                ax.set_title(mlabel)
            if ri == len(targets) - 1:
                ax.set_xlabel("width $N$")
            ax.set_xticks(widths)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 0.985])
    out = RESULTS_DIR / "geometry_suite.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        if not DATA_PATH.exists():
            print(f"No data at {DATA_PATH}. Run without --plot first.")
            sys.exit(1)
        with open(DATA_PATH) as f:
            data = json.load(f)
    else:
        data = collect_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_suite(data)
