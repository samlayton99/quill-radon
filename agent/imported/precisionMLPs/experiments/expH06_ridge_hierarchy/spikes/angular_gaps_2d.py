"""Does the error of a ridge mesh on a data ball sit in the angular gaps between spokes?

2-D, data ball of radius r = 0.4 about x0 = (0.35, -0.25) (the expH05 setting), even angles
M in {6, 8, 12, 16}, N = 64 offsets per direction (generous, so directions bind), three
targets. The fitted error |f_hat - f| is evaluated on a polar grid (radius x angle) inside
the ball; the figure shows |error| against the angle at several radii, with the spoke angles
marked, and a disk map of log10|error| with the spokes drawn through the center.

Usage: uv run --extra dev python experiments/expH06_ridge_hierarchy/spikes/angular_gaps_2d.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from h06.core import Geometry, make_block, fit_geometry, ball, direction_pool   # noqa: E402
from h06.targets import get_target                                            # noqa: E402

RESULTS = HERE.parents[2] / "results" / "checkpoint_H_highdim" / "expH06_ridge_hierarchy" / "spikes"
FIG = RESULTS / "figures"
X0 = np.array([0.35, -0.25])
R_BALL = 0.4
MS = [6, 8, 12, 16]
N_PER = 64
TARGETS = ["radial_runge", "fast_waves", "gauss_bump"]
N_TRAIN = 12000
RADII = [0.12, 0.24, 0.34]
N_ANG = 720


def main():
    rng = np.random.default_rng(0)
    Z = ball(N_TRAIN, 2, R_BALL, rng)
    th = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
    out = {"r": R_BALL, "Ms": MS, "N": N_PER, "radii": RADII, "targets": TARGETS, "rows": []}
    fig, axes = plt.subplots(len(TARGETS), len(MS), figsize=(4.2 * len(MS), 3.4 * len(TARGETS)), squeeze=False)
    fig2, axes2 = plt.subplots(len(TARGETS), len(MS), figsize=(4.0 * len(MS), 3.8 * len(TARGETS)), squeeze=False)
    for i, k in enumerate(TARGETS):
        f = get_target(k, 2)
        y = f(X0[None, :] + Z)
        for j, M in enumerate(MS):
            V = direction_pool(2, M)              # angles (i + 1/2) pi / M
            g = Geometry([make_block(v, Z, N_PER) for v in V])
            fit = fit_geometry(g, Z, y)
            ax = axes[i][j]
            spoke_angles = np.arctan2(V[:, 1], V[:, 0])
            rec = {"target": k, "M": M, "by_radius": {}}
            for rr, col in zip(RADII, ["tab:blue", "tab:orange", "tab:red"]):
                P = rr * np.stack([np.cos(th), np.sin(th)], axis=1)
                err = np.abs(fit.predict(g, P) - f(X0[None, :] + P))
                ax.plot(np.degrees(th), err, color=col, lw=1, label=f"radius {rr}")
                # error at the spoke angles vs midway between spokes (both signs of each spoke)
                sp = np.concatenate([spoke_angles, spoke_angles + np.pi]) % (2 * np.pi)
                mid = (sp + 0.5 * np.pi / M) % (2 * np.pi)
                e_sp = np.interp(sp, th, err, period=2 * np.pi)
                e_mid = np.interp(mid, th, err, period=2 * np.pi)
                rec["by_radius"][str(rr)] = {"median_on_spokes": float(np.median(e_sp)),
                                             "median_between": float(np.median(e_mid)),
                                             "max": float(err.max()), "mean": float(err.mean())}
            for a in spoke_angles:
                for s in (0, np.pi):
                    ax.axvline(np.degrees((a + s) % (2 * np.pi)), color="0.75", lw=0.6, zorder=0)
            ax.set_yscale("log"); ax.set_xlim(0, 360); ax.set_ylim(1e-15, 1)
            ax.set_title(f"{k}, M = {M}", fontsize=9); ax.grid(alpha=0.2)
            if i == len(TARGETS) - 1:
                ax.set_xlabel("angle (deg); grey lines = spoke directions")
            if j == 0:
                ax.set_ylabel("|error| on the circle")
            if i == 0 and j == 0:
                ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8, frameon=False)
            # disk map
            gr = np.linspace(-0.9 * R_BALL, 0.9 * R_BALL, 161)
            GX, GY = np.meshgrid(gr, gr)
            P = np.stack([GX.ravel(), GY.ravel()], axis=1)
            inside = np.linalg.norm(P, axis=1) <= 0.9 * R_BALL
            E = np.full(len(P), np.nan)
            E[inside] = np.log10(np.abs(fit.predict(g, P[inside]) - f(X0[None, :] + P[inside])) + 1e-16)
            ax2 = axes2[i][j]
            im = ax2.imshow(E.reshape(GX.shape), origin="lower", extent=[gr[0], gr[-1], gr[0], gr[-1]],
                            vmin=-15, vmax=0, cmap="viridis")
            for a in spoke_angles:
                ax2.plot([-R_BALL * np.cos(a), R_BALL * np.cos(a)], [-R_BALL * np.sin(a), R_BALL * np.sin(a)],
                         color="w", lw=0.5, alpha=0.7)
            ax2.set_xlim(-R_BALL, R_BALL); ax2.set_ylim(-R_BALL, R_BALL); ax2.set_aspect("equal")
            ax2.set_title(f"{k}, M = {M}: log10|error|", fontsize=9); ax2.set_xticks([]); ax2.set_yticks([])
            out["rows"].append(rec)
            rl = rec["by_radius"][str(RADII[-1])]
            print(f"{k:13s} M={M:2d} radius {RADII[-1]}: median |err| on spokes {rl['median_on_spokes']:.1e}, "
                  f"between {rl['median_between']:.1e}, max {rl['max']:.1e}", flush=True)
    fig.suptitle(f"2-D, ball r = {R_BALL}, N = {N_PER} offsets per spoke: |error| around circles of three radii, spokes marked", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig2.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.6, label="log10 |error|")
    fig2.suptitle("log10 |error| over the inner disk, spokes drawn through the center (white)", y=0.995)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "error_vs_angle.png", dpi=150, bbox_inches="tight")
    fig2.savefig(FIG / "error_disk_maps.png", dpi=150, bbox_inches="tight")
    (RESULTS / "angular_gaps_2d.json").write_text(json.dumps(out, indent=1))
    print("saved", FIG / "error_vs_angle.png", FIG / "error_disk_maps.png")


if __name__ == "__main__":
    main()
