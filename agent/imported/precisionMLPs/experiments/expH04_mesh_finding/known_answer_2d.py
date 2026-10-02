"""A known-answer check for the direction monitor (not a suite task).

The suite deliberately contains no target with a preferred direction, so on it the
direction monitor is a null test. Here the answer is known: the target is a ridge with
a fast profile along one direction ``u`` (angle 37 degrees) plus a slow, isotropic
background,

    F(x) = sin(6 pi (u.x)/||u||_1) + 0.5 / (1 + 16 ||x - a||^2 / 2),

on uniform data in [-1,1]^2. The right mesh spends most of its directions near ``u``
(the ridge needs fine resolution only there) and only a few on the background. The
check: does placing the angles by A(theta) = mean |dF/dv_theta|^2 (true, and estimated
from a first even fit) reach the floor at a smaller budget than even angles?

It does not, and the reason is instructive: a ridge is a delta in angle, and
A(theta) = cos^2(theta - 37 deg) * const is broad, so no angle density can single the
ridge direction out. What does single it out is the gradient covariance
E[grad F grad F^T] (the active subspace), whose top eigenvector is ``u`` exactly. The
``active`` rung puts that direction in with the active share of the budget and spends
the rest on an even mesh; it is run here from true and from estimated gradients.

Usage:
    uv run --extra dev python experiments/expH04_mesh_finding/known_answer_2d.py
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

from h01suite.baseline import EvenGeometry                      # noqa: E402
from h01suite.metrics import error_metrics                       # noqa: E402
from mesh import (AdaptiveGeometry, Monitors, surrogate_derivatives,  # noqa: E402
                  gradient_covariance, active_dimension, active_subspace_geometry)

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH04_mesh_finding"
THETA = np.deg2rad(37.0)
U = np.array([np.cos(THETA), np.sin(THETA)])
A_PT = np.array([-0.3, 0.2])
BUDGETS = [256, 512, 1024, 2048, 4096]
ALPHAS = [1.0 / 3.0, 1.0]


def F(X):
    t = X @ U / np.abs(U).sum()
    r2 = ((X - A_PT) ** 2).sum(axis=1) / 2.0
    return np.sin(6 * np.pi * t) + 0.5 / (1 + 16 * r2)


def grad_F(X):
    t = X @ U / np.abs(U).sum()
    r2 = ((X - A_PT) ** 2).sum(axis=1) / 2.0
    g_ridge = (6 * np.pi * np.cos(6 * np.pi * t) / np.abs(U).sum())[:, None] * U[None, :]
    g_bg = (-0.5 * 16 / (1 + 16 * r2) ** 2)[:, None] * (X - A_PT)
    return g_ridge + g_bg


def dir_monitor_true(Xs):
    def A(Vg):
        D = (grad_F(Xs) @ Vg.T).T
        return (D * D).mean(axis=1)
    return A


def dir_monitor_est(model, Xs):
    def A(Vg):
        D = surrogate_derivatives(model, Xs, Vg, 1)
        return (D * D).mean(axis=1)
    return A


def main():
    rng = np.random.default_rng(0)
    Xtest = rng.uniform(-1, 1, size=(20000, 2))
    ytest = F(Xtest)
    rows, angles = [], {}
    for B in BUDGETS:
        X = np.random.default_rng(1).uniform(-1, 1, size=(8 * B, 2))
        y = F(X)
        Xs = X[:4096]
        t0 = time.time()
        even = EvenGeometry(d=2, budget=B).fit(X, y)
        rows.append({"budget": B, "rung": "even", "alpha": None,
                     "errors": error_metrics(even.predict(Xtest), ytest)})
        print(f"B={B:5d} even        {rows[-1]['errors']['rel_l2']:.1e}  {time.time()-t0:.0f}s",
              flush=True)
        for kind in ("true", "est"):
            G = grad_F(X) if kind == "true" else surrogate_derivatives(even, X, np.eye(2), 1).T
            evals, W = gradient_covariance(G)
            geo = active_subspace_geometry(2, B, W, 1, name=f"active_{kind}")
            D = (surrogate_derivatives(even, X, geo.unique_directions, 1) if kind == "est"
                 else (grad_F(X) @ geo.unique_directions.T).T)
            geo.build(X, Monitors("roughness", r=1, deriv=D)).fit(X, y)
            e = error_metrics(geo.predict(Xtest), ytest)
            rows.append({"budget": B, "rung": f"active_{kind}", "alpha": None, "errors": e,
                         "angle_deg": float(np.rad2deg(np.arctan2(W[1, 0], W[0, 0])) % 180)})
            print(f"B={B:5d} active_{kind:4s} {e['rel_l2']:.1e}  top eigenvector at "
                  f"{rows[-1]['angle_deg']:.3f} deg", flush=True)
            if kind == "est":
                # iterate, re-reading the direction from the active units alone (the even
                # units carry the background, whose gradient biases the covariance)
                model = geo
                for it in range(3):
                    n_units = int(model.per_direction[0])
                    part = type("P", (), {})()
                    part.directions, part.centers = model.directions[:n_units], model.centers[:n_units]
                    part.gammas, part.weights = model.gammas[:n_units], model.weights[:n_units]
                    G = surrogate_derivatives(part, X, np.eye(2), 1).T
                    evals, W = gradient_covariance(G)
                    geo2 = active_subspace_geometry(2, B, W, 1, name="active_iter")
                    D = surrogate_derivatives(model, X, geo2.unique_directions, 1)
                    model = geo2.build(X, Monitors("roughness", r=1, deriv=D)).fit(X, y)
                e = error_metrics(model.predict(Xtest), ytest)
                rows.append({"budget": B, "rung": "active_iter", "alpha": None, "errors": e,
                             "angle_deg": float(np.rad2deg(np.arctan2(W[1, 0], W[0, 0])) % 180)})
                print(f"B={B:5d} active_iter {e['rel_l2']:.1e}  direction at "
                      f"{rows[-1]['angle_deg']:.5f} deg", flush=True)
        for alpha in ALPHAS:
            for kind in ("true", "est"):
                geo = AdaptiveGeometry(d=2, budget=B, name=f"dir_{kind}")
                mon = dir_monitor_true(Xs) if kind == "true" else dir_monitor_est(even, Xs)
                geo.set_directions(X, mon, alpha=alpha).build(X, Monitors("even")).fit(X, y)
                e = error_metrics(geo.predict(Xtest), ytest)
                rows.append({"budget": B, "rung": f"dir_{kind}", "alpha": alpha, "errors": e})
                angles[f"{kind}_{alpha:.2f}_{B}"] = geo.mesh_info["direction_monitor"]
                print(f"B={B:5d} dir_{kind:4s} alpha={alpha:.2f} {e['rel_l2']:.1e}", flush=True)
    with open(RESULTS_DIR / "known_answer_2d.json", "w") as f:
        json.dump({"rows": rows,
                   "angles": {k: {"angles": v["angles"].tolist()} for k, v in angles.items()}},
                  f)
    plot(rows, angles)


def plot(rows, angles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    styles = {("even", None): ("k", "-", "even angles"),
              ("dir_true", 1 / 3): ("tab:red", "--", r"angles from true $A(\theta)^{1/3}$"),
              ("dir_est", 1 / 3): ("tab:purple", "-", r"angles from estimated $A(\theta)^{1/3}$"),
              ("dir_true", 1.0): ("tab:red", ":", r"angles from true $A(\theta)$"),
              ("dir_est", 1.0): ("tab:purple", "-.", r"angles from estimated $A(\theta)$"),
              ("active_true", None): ("tab:pink", "--", "active subspace, true gradients"),
              ("active_est", None): ("tab:pink", "-", "active subspace, estimated")}
    # active_iter is in the data but not drawn: for m = 1 the iteration is a no-op by
    # construction (every active unit shares the one direction, so the covariance of
    # their gradients returns it unchanged) and its refit diverged.
    for (rung, alpha), (c, ls, lab) in styles.items():
        pts = sorted([(r["budget"], r["errors"]["rel_l2"]) for r in rows
                      if r["rung"] == rung and (alpha is None or abs(r["alpha"] - alpha) < 1e-9)])
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], ls, color=c, marker="o", ms=4,
                  label=lab)
    ax.set_ylim(1e-15, 1e1)
    ax.set_xlabel("budget $B$")
    ax.set_ylabel("relative $L_2$, uniform test")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, frameon=False)
    ax = axes[1]
    key = f"est_{1/3:.2f}_1024"
    if key in angles:
        info = angles[key]
        ax.plot(np.rad2deg(info["theta"]), info["A"] / info["A"].max(), "k-", lw=1,
                label=r"estimated $A(\theta)$ / max, $B = 1024$")
        ax.plot(np.rad2deg(info["angles"]), np.full(len(info["angles"]), -0.08), "|",
                color="tab:purple", ms=10, label="placed angles")
    ax.axvline(37.0, color="tab:red", ls="--", lw=1, label="ridge direction (37 deg)")
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("directional energy")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    out = RESULTS_DIR / "figures" / "known_answer_2d.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
