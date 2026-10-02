"""Add-ons to expE01 without redoing the whole suite:

  --lambda    : recompute a fine lambda sweep at one representative width and
                plot error vs lambda (the U-shape). Fast.
  --add8192   : compute the N=8192 cells (n_train raised for overdetermination),
                append to data.json + weights, and regenerate geometry_suite.png.
                Slow (8192-column SVDs); run in the background.

Usage:
    python experiments/expE01_geometry_zoo_2d/extra_analysis.py --lambda
    python experiments/expE01_geometry_zoo_2d/extra_analysis.py --add8192
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import run as R                                    # the expE01 driver module
from geometries import GEOMETRIES
from src.data.sampling2d import disk_grid, disk_uniform
from src.data.targets2d import get_target_2d

LAMBDA_SWEEP_N = 1024                              # representative width for the U-shape
LAMBDA_FINE = [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25,
               0.32, 0.42, 0.55, 0.75, 1.00, 1.40]
N8192 = 8192
N8192_NTRAIN = 12000                               # > 8192 (overdetermined)
N8192_LAMBDA = [0.10, 0.15, 0.22, 0.32, 0.45]      # focused (optima cluster here)


def _solve_eval(dirs, offs, gamma, X_tr, X_ev, ys_tr, ys_ev, y_norms):
    Phi_tr = np.tanh(gamma * (X_tr @ dirs.T - offs[None, :]))
    Phi_ev = np.tanh(gamma * (X_ev @ dirs.T - offs[None, :]))
    Aug = np.hstack([Phi_tr, np.ones((X_tr.shape[0], 1))])
    U, s, Vt = np.linalg.svd(Aug, full_matrices=False)
    s_inv = np.where(s > R.RCOND * s[0], 1.0 / s, 0.0)
    out = {}
    for t in ys_tr:
        sol = Vt.T @ (s_inv * (U.T @ ys_tr[t]))
        resid = Phi_ev @ sol[:-1] + sol[-1] - ys_ev[t]
        out[t] = (float(np.abs(resid).max()),
                  float(np.linalg.norm(resid) / y_norms[t]), sol)
    return out


def _data():
    X_tr = disk_uniform(R.N_TRAIN, radius=1.0, seed=0)
    X_ev = disk_grid(R.EVAL_SIDE, radius=1.0)
    ys_tr = {t: get_target_2d(t).fn_numpy(X_tr) for t in R.TARGETS}
    ys_ev = {t: get_target_2d(t).fn_numpy(X_ev) for t in R.TARGETS}
    yn = {t: float(np.linalg.norm(ys_ev[t])) for t in R.TARGETS}
    return X_tr, X_ev, ys_tr, ys_ev, yn


def lambda_sweep():
    X_tr, X_ev, ys_tr, ys_ev, yn = _data()
    h_ref = 2.8 / np.sqrt(LAMBDA_SWEEP_N)
    sweep = {g: {t: {"lam": [], "linf": [], "rel_l2": []} for t in R.TARGETS}
             for g in R.GEOM_ORDER}
    t0 = time.time()
    for g in R.GEOM_ORDER:
        dirs, offs = GEOMETRIES[g](LAMBDA_SWEEP_N, R.GEOM_RADIUS[g], seed=0)
        for lam in LAMBDA_FINE:
            res = _solve_eval(dirs, offs, lam / h_ref, X_tr, X_ev, ys_tr, ys_ev, yn)
            for t in R.TARGETS:
                sweep[g][t]["lam"].append(lam)
                sweep[g][t]["linf"].append(res[t][0])
                sweep[g][t]["rel_l2"].append(res[t][1])
        print(f"lambda sweep {g} done ({time.time()-t0:.0f}s)")
    data = json.load(open(R.DATA_PATH))
    data["lambda_sweep"] = {"N": LAMBDA_SWEEP_N, "lambdas": LAMBDA_FINE, "sweep": sweep}
    json.dump(data, open(R.DATA_PATH, "w"))
    plot_lambda(data)


def plot_lambda(data):
    sw = data["lambda_sweep"]
    N, sweep = sw["N"], sw["sweep"]
    metrics = [("rel_l2", 0, r"Relative $L_2$"), ("linf", 1, r"$L_\infty$")]
    fig, axes = plt.subplots(len(R.TARGETS), 2, figsize=(13, 3.4 * len(R.TARGETS)),
                             sharex=True)
    fig.suptitle(f"ExpE01: error vs $\\lambda$ (U-shape) at N={N}, lstsq fp64",
                 fontsize=14, y=0.997)
    for ri, t in enumerate(R.TARGETS):
        sym = "symmetric" if t in data["config"]["symmetric"] else "asymmetric"
        for mkey, ci, ml in metrics:
            ax = axes[ri][ci]
            for g in R.GEOM_ORDER:
                ax.loglog(sweep[g][t]["lam"], sweep[g][t][mkey], "-o",
                          color=R.GEOM_COLORS[g], markersize=3, label=R.GEOM_LABELS[g])
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylabel(f"{t} ({sym})\n{ml}" if ci == 0 else ml)
            if ri == 0:
                ax.set_title(ml)
            if ri == len(R.TARGETS) - 1:
                ax.set_xlabel(r"$\lambda = \gamma\,h_{\mathrm{ref}}$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 0.985])
    out = R.RESULTS_DIR / "error_vs_lambda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def add_8192():
    X_tr = disk_uniform(N8192_NTRAIN, radius=1.0, seed=0)
    X_ev = disk_grid(R.EVAL_SIDE, radius=1.0)
    ys_tr = {t: get_target_2d(t).fn_numpy(X_tr) for t in R.TARGETS}
    ys_ev = {t: get_target_2d(t).fn_numpy(X_ev) for t in R.TARGETS}
    yn = {t: float(np.linalg.norm(ys_ev[t])) for t in R.TARGETS}
    h_ref = 2.8 / np.sqrt(N8192)
    data = json.load(open(R.DATA_PATH))
    # drop any prior 8192 rows so re-runs are idempotent
    data["rows"] = [r for r in data["rows"] if r["N"] != N8192]
    t0 = time.time()
    for g in R.GEOM_ORDER:
        dirs, offs = GEOMETRIES[g](N8192, R.GEOM_RADIUS[g], seed=0)
        n = len(offs)
        best = {t: {"linf": np.inf} for t in R.TARGETS}
        for lam in N8192_LAMBDA:
            res = _solve_eval(dirs, offs, lam / h_ref, X_tr, X_ev, ys_tr, ys_ev, yn)
            for t in R.TARGETS:
                if res[t][0] < best[t]["linf"]:
                    sol = res[t][2]
                    best[t] = {"linf": res[t][0], "rel_l2": res[t][1], "lambda": lam,
                               "gamma": lam / h_ref, "v": sol[:-1], "bias": float(sol[-1])}
        for t in R.TARGETS:
            b = best[t]
            np.savez(R.WEIGHTS_DIR / f"{g}_N{N8192}_{t}.npz",
                     dirs=dirs, offsets=offs, gamma=b["gamma"], v=b["v"],
                     bias=b["bias"], radius=R.GEOM_RADIUS[g], lam=b["lambda"],
                     linf=b["linf"], rel_l2=b["rel_l2"])
            data["rows"].append({"geom": g, "N": N8192, "n_ridges": n, "target": t,
                                 "linf": b["linf"], "rel_l2": b["rel_l2"],
                                 "lambda": b["lambda"], "gamma": b["gamma"],
                                 "weights_file": f"weights/{g}_N{N8192}_{t}.npz"})
        tag = " ".join(f"{t[:5]}={best[t]['linf']:.1e}" for t in R.TARGETS)
        print(f"8192 {g:17s} (n={n}) | {tag} | {time.time()-t0:.0f}s")
    if N8192 not in data["config"]["widths"]:
        data["config"]["widths"] = sorted(set(data["config"]["widths"] + [N8192]))
    json.dump(data, open(R.DATA_PATH, "w"))
    R.plot_suite(data)
    print("regenerated geometry_suite.png with N=8192")


ANIM_WIDTHS = [64, 144, 256, 576, 1024, 2048, 4096]
ANIM_NTRAIN = 8000


def animate():
    """Compute the lambda sweep at every width and build a looping GIF that
    sweeps N from small to large (each frame is the error-vs-lambda U-shape)."""
    X_tr = disk_uniform(ANIM_NTRAIN, radius=1.0, seed=0)
    X_ev = disk_grid(R.EVAL_SIDE, radius=1.0)
    ys_tr = {t: get_target_2d(t).fn_numpy(X_tr) for t in R.TARGETS}
    ys_ev = {t: get_target_2d(t).fn_numpy(X_ev) for t in R.TARGETS}
    yn = {t: float(np.linalg.norm(ys_ev[t])) for t in R.TARGETS}

    anim = {}
    t0 = time.time()
    for N in ANIM_WIDTHS:
        h_ref = 2.8 / np.sqrt(N)
        sweep = {g: {t: {"linf": [], "rel_l2": []} for t in R.TARGETS}
                 for g in R.GEOM_ORDER}
        for g in R.GEOM_ORDER:
            dirs, offs = GEOMETRIES[g](N, R.GEOM_RADIUS[g], seed=0)
            for lam in LAMBDA_FINE:
                res = _solve_eval(dirs, offs, lam / h_ref, X_tr, X_ev, ys_tr, ys_ev, yn)
                for t in R.TARGETS:
                    sweep[g][t]["linf"].append(res[t][0])
                    sweep[g][t]["rel_l2"].append(res[t][1])
        anim[str(N)] = sweep
        print(f"anim sweep N={N} done ({time.time()-t0:.0f}s)")

    # Saved to its OWN file (not data.json) so concurrent jobs can't clobber it.
    symmetric = json.load(open(R.DATA_PATH))["config"]["symmetric"]
    payload = {"widths": ANIM_WIDTHS, "lambdas": LAMBDA_FINE, "data": anim,
               "symmetric": symmetric}
    json.dump(payload, open(R.RESULTS_DIR / "lambda_anim.json", "w"))
    build_anim(payload)


def build_anim(ad):
    """Build a PAUSABLE animation: an interactive HTML player (play/pause/step/
    loop controls) plus an mp4, from the saved per-width lambda sweep."""
    from matplotlib.animation import FuncAnimation
    widths, lams, D = ad["widths"], ad["lambdas"], ad["data"]
    symmetric = ad["symmetric"]
    metrics = [("rel_l2", 0, r"Relative $L_2$"), ("linf", 1, r"$L_\infty$")]

    # Fixed per-panel y-limits across all N so the descent is visible.
    ylim = {}
    for t in R.TARGETS:
        for mkey, ci, _ in metrics:
            vals = [v for N in widths for g in R.GEOM_ORDER
                    for v in D[str(N)][g][t][mkey] if v > 0]
            lo, hi = min(vals), max(vals)
            ylim[(t, ci)] = (lo * 0.3, hi * 3)

    fig, axes = plt.subplots(len(R.TARGETS), 2, figsize=(13, 3.4 * len(R.TARGETS)),
                             sharex=True)
    lines = {}
    for ri, t in enumerate(R.TARGETS):
        sym = "symmetric" if t in symmetric else "asymmetric"
        for mkey, ci, ml in metrics:
            ax = axes[ri][ci]
            for g in R.GEOM_ORDER:
                (ln,) = ax.loglog(lams, D[str(widths[0])][g][t][mkey], "-o",
                                  color=R.GEOM_COLORS[g], markersize=3,
                                  label=R.GEOM_LABELS[g])
                lines[(t, ci, g)] = ln
            ax.set_xlim(min(lams), max(lams)); ax.set_ylim(*ylim[(t, ci)])
            ax.grid(True, alpha=0.3, which="both")
            ax.set_ylabel(f"{t} ({sym})\n{ml}" if ci == 0 else ml)
            if ri == 0:
                ax.set_title(ml)
            if ri == len(R.TARGETS) - 1:
                ax.set_xlabel(r"$\lambda = \gamma\,h_{\mathrm{ref}}$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Legend INSIDE the figure (animation saves clip anything outside the bbox).
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 0.955))
    suptitle = fig.suptitle("", fontsize=15, y=0.995)

    def update(frame):
        N = widths[frame]
        for ri, t in enumerate(R.TARGETS):
            for mkey, ci, _ in metrics:
                for g in R.GEOM_ORDER:
                    lines[(t, ci, g)].set_ydata(D[str(N)][g][t][mkey])
        suptitle.set_text(f"ExpE01: error vs $\\lambda$ as width grows  --  N = {N}  "
                          f"(lstsq fp64)")
        return list(lines.values()) + [suptitle]

    # Reserve top space for suptitle+legend via subplots_adjust (not tight_layout,
    # which doesn't compose with animation frame saving).
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.08, right=0.98, hspace=0.28, wspace=0.22)
    anim = FuncAnimation(fig, update, frames=len(widths), interval=1100, blit=False)

    # 1) Interactive HTML player -- play/PAUSE, step frame-by-frame, loop toggle,
    #    speed slider. Open in any browser. This is the pausable deliverable.
    html = anim.to_jshtml(default_mode="once")
    html_out = R.RESULTS_DIR / "error_vs_lambda_anim.html"
    html_out.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>ExpE01 error-vs-lambda sweep over N</title>"
        "<body style='font-family:sans-serif'>" + html + "</body>")
    print(f"Saved {html_out}")

    # 2) mp4 (pausable in any video player).
    try:
        from matplotlib.animation import FFMpegWriter
        mp4_out = R.RESULTS_DIR / "error_vs_lambda_anim.mp4"
        anim.save(mp4_out, writer=FFMpegWriter(fps=1, bitrate=2400))
        print(f"Saved {mp4_out}")
    except Exception as e:
        print(f"mp4 skipped ({e})")

    # 3) GIF too (loops; convenient quick-look).
    from matplotlib.animation import PillowWriter
    gif_out = R.RESULTS_DIR / "error_vs_lambda_anim.gif"
    anim.save(gif_out, writer=PillowWriter(fps=0.9))
    print(f"Saved {gif_out}")
    plt.close(fig)


if __name__ == "__main__":
    if "--lambda" in sys.argv:
        lambda_sweep()
    if "--add8192" in sys.argv:
        add_8192()
    if "--animate" in sys.argv:
        animate()
    if "--rebuild-anim" in sys.argv:                 # rebuild players from saved sweep
        build_anim(json.load(open(R.RESULTS_DIR / "lambda_anim.json")))
