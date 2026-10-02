"""At what wavelength does a perturbation of the mesh map stop hurting the floor?

Every mesh here is c_j = Phi(j h0) for a map Phi from the uniform grid. In the coordinate
xi = Phi^{-1}(x) the network is the uniform construction applied to f o Phi, so Phi must
be as smooth, per gap, as f. This script measures that directly: the even mesh with a
single sinusoidal perturbation of the positions,

    c_j = j h0 + a h0 sin(2 pi j h0 / (L h0) + phase),    widths from the local gaps,

for amplitudes a (fraction of a gap) and wavelengths L (in gaps). If the mesh-map view
is right, the floor error is a function of L that falls spectrally, and the wavelength
where it reaches the floor is the scale any monitor must be smoothed to -- a property of
the construction, independent of the task and of the data.

Usage:
    uv run --extra dev python experiments/expH04_mesh_finding/mesh_map_scale.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "experiments" / "expH01_highdim_suite"))

from h01suite.baseline import EvenGeometry        # noqa: E402
from h01suite.metrics import error_metrics         # noqa: E402
from h01suite.tasks import get_task                # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH04_mesh_finding"
CELLS = [("1.14", 256), ("1.11", 256), ("1.1", 512)]
AMPS = [0.005, 0.02, 0.05, 0.2]
WAVELENGTHS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
PHASES = 3


class WavyEven(EvenGeometry):
    def __init__(self, d, budget, amp, wavelength, phase):
        super().__init__(d=d, budget=budget)
        n = self.n_per_direction
        c = self.centers.copy()
        h0 = c[1] - c[0]
        j = np.arange(n)
        c = c + amp * h0 * np.sin(2 * np.pi * j / wavelength + phase)
        h = np.empty(n)
        h[1:-1] = 0.5 * (c[2:] - c[:-2]); h[0] = c[1] - c[0]; h[-1] = c[-1] - c[-2]
        self.centers, self.gammas = c, self.lam / h


def main():
    rows = []
    for tid, B in CELLS:
        task = get_task(tid)
        X, y = task.train_set(8 * B, seed=0)
        sets = task.test_sets(seed=10_000)
        yd = task.F(sets["dense_region"])
        ev = EvenGeometry(d=1, budget=B).fit(X, y)
        e0 = error_metrics(ev.predict(sets["dense_region"]), yd)["rel_l2"]
        rows.append({"task": tid, "budget": B, "amp": 0.0, "wavelength": None, "dense": e0})
        print(f"{tid} B={B} even {e0:.1e}", flush=True)
        for a in AMPS:
            for L in WAVELENGTHS:
                errs = []
                for k in range(PHASES):
                    m = WavyEven(1, B, a, L, 2 * np.pi * k / PHASES).fit(X, y)
                    errs.append(error_metrics(m.predict(sets["dense_region"]), yd)["rel_l2"])
                rows.append({"task": tid, "budget": B, "amp": a, "wavelength": L,
                             "dense": float(np.median(errs)), "dense_all": errs})
                print(f"  a={a:<6g} L={L:3d} gaps  dense={np.median(errs):.1e}", flush=True)
    with open(RESULTS_DIR / "mesh_map_scale.json", "w") as f:
        json.dump(rows, f)
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = sorted({(r["task"], r["budget"]) for r in rows})
    fig, axes = plt.subplots(1, len(cells), figsize=(5.3 * len(cells), 4.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, (tid, B) in zip(np.atleast_1d(axes), cells):
        e0 = next(r["dense"] for r in rows if r["task"] == tid and r["budget"] == B
                  and r["amp"] == 0.0)
        for k, a in enumerate(AMPS):
            pts = sorted([(r["wavelength"], r["dense"]) for r in rows
                          if r["task"] == tid and r["budget"] == B and r["amp"] == a])
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], "-o", ms=4,
                      color=cmap(k / (len(AMPS) - 1)), label=f"amplitude {a:g} gaps")
        ax.axhline(e0, color="k", ls="--", lw=1, label="even mesh")
        ax.set_xlabel("wavelength of the mesh-map perturbation (gaps)")
        ax.set_title(f"{tid}, B = {B}", fontsize=10)
        ax.set_ylim(1e-15, 1e-7)
        ax.grid(alpha=0.3, which="both")
    np.atleast_1d(axes)[0].set_ylabel("relative $L_2$, dense region")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = RESULTS_DIR / "figures" / "mesh_map_scale.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
