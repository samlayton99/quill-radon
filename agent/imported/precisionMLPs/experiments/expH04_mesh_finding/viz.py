"""Figures for expH04. Legends sit above the axes, never inside them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1] / "expH01_highdim_suite"))
sys.path.append(str(Path(__file__).resolve().parent))

STYLE = {   # rung: (color, linestyle, marker)
    "even":       ("black",     "-",  "o"),
    "data_p13":   ("tab:blue",  "--", "s"),
    "data_p1":    ("tab:blue",  "-",  "s"),
    "oracle_r1":  ("tab:red",   "--", "^"),
    "oracle_r2":  ("tab:red",   ":",  "v"),
    "surr_r1":    ("tab:green", "-",  "^"),
    "surr_r2":    ("tab:green", ":",  "v"),
    "residual":   ("tab:orange", "-", "D"),
    "surr_r1_x3": ("tab:green", "-.", "x"),
    "freq_oracle": ("tab:cyan", "--", "o"),
    "freq":       ("tab:cyan", "-",  "o"),
    "active_oracle": ("tab:pink", "--", "*"),
    "active":     ("tab:pink", "-",  "*"),
    "active_x3":  ("tab:pink", "-.", "*"),
    "dir_oracle": ("tab:purple", "--", "^"),
    "dir_surr":   ("tab:purple", "-",  "^"),
    "both_surr":  ("tab:brown", "-",  "P"),
    "joint_surr": ("tab:brown", "-.", "X"),
}
LABEL = {
    "even": "even (reference)",
    "data_p13": r"data density$^{1/3}$",
    "data_p1": "data density",
    "oracle_r1": r"true slope: $(p\,R_1)^{1/3}$",
    "oracle_r2": r"true curvature: $(p\,R_2)^{1/5}$",
    "surr_r1": r"estimated slope (2 solves)",
    "surr_r2": r"estimated curvature (2 solves)",
    "residual": "residual of first fit (2 solves)",
    "surr_r1_x3": "estimated slope, iterated (3 solves)",
    "freq_oracle": r"true local frequency $\sqrt{R_2/R_1}$",
    "freq": "estimated local frequency (2 solves)",
    "active_oracle": "active subspace from true gradients",
    "active": "active subspace from first fit (2 solves)",
    "active_x3": "active subspace, iterated (4 solves)",
    "dir_oracle": "directions from true slope, even centers",
    "dir_surr": "directions from estimated slope, even centers",
    "both_surr": "directions + centers estimated",
    "joint_surr": "directions + centers + counts estimated",
}
SET_LABEL = {"same_as_train": "test drawn like the training data",
             "uniform": "test uniform over the cube",
             "dense_region": "test on the densest region"}
CENTER_GROUP = ["even", "data_p13", "data_p1", "oracle_r1", "oracle_r2", "surr_r1",
                "surr_r2", "residual", "surr_r1_x3", "freq_oracle", "freq"]
DIRECTION_GROUP = ["even", "surr_r1", "dir_oracle", "dir_surr", "both_surr", "joint_surr",
                   "active_oracle", "active", "active_x3"]


def _key(s):
    return (int(s.split(".")[0]), int(s.split(".")[1]))


def _short(name: str) -> str:
    return name.split("-", 1)[1].replace("-", " ")


def _tiny(name: str) -> str:
    """Task name with the data-geometry prefix removed, for narrow tick labels."""
    s = _short(name)
    for pre in ("uniform data ", "hotspot data ", "even grid ", "curved sheet noisy ",
                "curved sheet "):
        if s.startswith(pre):
            return s[len(pre):]
    return s


def ladder_figure(rows, d, test_set, rungs, path, title):
    ids = sorted({r["task"] for r in rows if r["d"] == d}, key=_key)
    if not ids:
        return None
    ncol = 3 if d == 1 else 5
    nrow = int(np.ceil(len(ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow + 1.2),
                             squeeze=False, sharex=True, sharey=True)
    handles = {}
    for k, tid in enumerate(ids):
        ax = axes[k // ncol][k % ncol]
        name = next(r["name"] for r in rows if r["task"] == tid)
        for rung in rungs:
            pts = sorted([(r["budget"], r["errors"][test_set]["rel_l2"])
                          for r in rows if r["task"] == tid and r["rung"] == rung])
            if not pts:
                continue
            c, ls, m = STYLE[rung]
            (h,) = ax.loglog([p[0] for p in pts], [p[1] for p in pts], ls, color=c,
                             marker=m, ms=4, lw=1.3, label=LABEL[rung])
            handles[rung] = h
        ax.set_title(f"{tid}  {_short(name)}", fontsize=9)
        ax.set_ylim(1e-15, 1e2)
        Bs = sorted({r["budget"] for r in rows if r["task"] == tid})
        ax.set_xticks(Bs)
        ax.set_xticklabels([str(b) for b in Bs], fontsize=8)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=0.3, which="both")
        if k % ncol == 0:
            ax.set_ylabel("relative $L_2$ error")
        if k // ncol == nrow - 1:
            ax.set_xlabel("budget $B$ (number of tanh units)")
    for k in range(len(ids), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{title} -- {SET_LABEL[test_set]}", y=0.995, fontsize=11)
    fig.legend([handles[r] for r in rungs if r in handles],
               [LABEL[r] for r in rungs if r in handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90 if d == 1 else 0.88))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def gain_figure(rows, path):
    """log10(error / even error) at the largest budget, per task, for every rung."""
    tasks = sorted({r["task"] for r in rows}, key=_key)
    rungs = [r for r in STYLE if any(x["rung"] == r for x in rows) and r != "even"]
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 0.9 * len(tasks) + 3), 8), sharex=True)
    for ax, key in zip(axes, ["dense_region", "uniform"]):
        width = 0.8 / len(rungs)
        for j, rung in enumerate(rungs):
            vals = []
            for tid in tasks:
                sub = [r for r in rows if r["task"] == tid]
                Bmax = max(r["budget"] for r in sub)
                ev = next((r for r in sub if r["rung"] == "even" and r["budget"] == Bmax), None)
                me = next((r for r in sub if r["rung"] == rung and r["budget"] == Bmax), None)
                vals.append(np.nan if ev is None or me is None else
                            np.log10(me["errors"][key]["rel_l2"] / ev["errors"][key]["rel_l2"]))
            c, _, _ = STYLE[rung]
            ax.bar(np.arange(len(tasks)) + (j - len(rungs) / 2 + 0.5) * width, vals, width,
                   color=c, alpha=0.5 + 0.5 * ("surr" in rung or "resid" in rung),
                   label=LABEL[rung], edgecolor="k", lw=0.3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(f"log10(error / even error)\n{SET_LABEL[key]}", fontsize=9)
        ax.set_ylim(-8, 3)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=8,
                   frameon=False)
    axes[1].set_xticks(np.arange(len(tasks)))
    axes[1].set_xticklabels(tasks)
    axes[1].set_xlabel("task (largest budget run); below 0 = better than the even mesh")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def floor_figure(rows, path):
    tasks = sorted({r["task"] for r in rows}, key=_key)
    levels = sorted({r["s"] for r in rows})
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.2 * len(tasks), 7.5), squeeze=False,
                             sharex=True, sharey=True)
    for j, tid in enumerate(tasks):
        name = next(r["name"] for r in rows if r["task"] == tid)
        for i, rung in enumerate(["surr_r1", "data_p1"]):
            ax = axes[i][j]
            for k, s in enumerate(levels):
                pts = sorted([(r["budget"], r["errors"]["dense_region"]["rel_l2"]) for r in rows
                              if r["task"] == tid and r["rung"] == rung and r["s"] == s])
                if pts:
                    ax.loglog([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3,
                              color=cmap(k / max(1, len(levels) - 1)), label=f"s = {s:.2f}")
            pts = sorted([(r["budget"], r["errors"]["dense_region"]["rel_l2"]) for r in rows
                          if r["task"] == tid and r["rung"] == "even" and r["s"] == levels[0]])
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], "k--", lw=1, label="even")
            ax.set_ylim(1e-15, 1e1)
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{tid} {_short(name)}\n{LABEL[rung]}", fontsize=9)
            if j == 0:
                ax.set_ylabel("relative $L_2$, densest region")
            if i == 1:
                ax.set_xlabel("budget $B$")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(levels) + 1,
               fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def mesh_examples_1d(path, budget=128, task_ids=("1.13", "1.14", "1.16", "1.15"),
                     rungs=("even", "data_p1", "oracle_r1", "surr_r1", "residual")):
    """Data density, monitors and the placed centers, one row per task."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "h04run", str(Path(__file__).resolve().parent / "run.py"))
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    from h01suite.tasks import get_task

    fig, axes = plt.subplots(len(task_ids), 2, figsize=(14, 3.1 * len(task_ids) + 1),
                             squeeze=False)
    xs = np.linspace(-1, 1, 2001)[:, None]
    for i, tid in enumerate(task_ids):
        task = get_task(tid)
        rows, geoms, X, y = H.run_cell(task, budget, list(rungs), keep_geometry=True)
        axL, axR = axes[i]
        axL.plot(xs[:, 0], task.F(xs), "k-", lw=1)
        axL.hist(X[:, 0], bins=80, density=True, color="0.8", alpha=0.8, label="training data")
        axL.set_title(f"{tid} {_short(task.name)}: target (black) and data histogram",
                      fontsize=9)
        for k, rung in enumerate(rungs):
            if rung not in geoms:
                continue
            g = geoms[rung]
            c, ls, m = STYLE[rung]
            info = getattr(g, "mesh_info", {}).get("per_direction")
            if info:
                grid, rho = info[0]["grid"], info[0]["density"]
                axR.plot(grid, rho / rho.mean(), ls, color=c, lw=1.2, label=LABEL[rung])
            else:
                axR.axhline(1.0, color=c, ls=ls, lw=1.2, label=LABEL[rung])
            cen = g.centers
            axR.plot(cen, np.full(len(cen), -0.3 - 0.35 * k), "|", color=c, ms=7)
        axR.set_xlim(-1.3, 1.3)
        axR.set_ylabel("center density / mean")
        axR.set_title("center density per rung; ticks = placed centers", fontsize=9)
        if i == 0:
            axR.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=8,
                       frameon=False)
        axR.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def mesh_examples_2d(path, budget=1024, task_ids=("2.12", "2.13", "2.16"),
                     rungs=("even", "data_p1", "surr_r1", "both_surr")):
    """Where the mesh is fine: the resolution field ``sum_k gamma_k sech^2(u_k(x))`` on a
    grid (bright = many narrow units nearby), training data overlaid, even vs adapted."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "h04run", str(Path(__file__).resolve().parent / "run.py"))
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    from h01suite.tasks import get_task

    g = np.linspace(-1, 1, 201)
    GX, GY = np.meshgrid(g, g, indexing="xy")
    G = np.stack([GX.ravel(), GY.ravel()], axis=1)
    fig, axes = plt.subplots(len(task_ids), len(rungs),
                             figsize=(4.3 * len(rungs), 4.4 * len(task_ids) + 0.6),
                             squeeze=False)
    for i, tid in enumerate(task_ids):
        task = get_task(tid)
        rows, geoms, X, y = H.run_cell(task, budget, ["even"] + [r for r in rungs if r != "even"],
                                       keep_geometry=True)
        fields = {}
        for rung in rungs:
            m = geoms[rung]
            U = m.gammas[None, :] * (G @ m.directions.T - m.centers[None, :])
            fields[rung] = ((1.0 - np.tanh(U) ** 2) * m.gammas[None, :]).sum(axis=1)
        ref = fields["even"].mean()
        for j, rung in enumerate(rungs):
            ax = axes[i][j]
            im = ax.imshow(np.log10(fields[rung] / ref).reshape(GX.shape), origin="lower",
                           extent=(-1, 1, -1, 1), cmap="magma", vmin=-1, vmax=1)
            ax.scatter(X[:1500, 0], X[:1500, 1], s=1.5, color="white", alpha=0.35)
            err = next(r for r in rows if r["rung"] == rung)["errors"]
            ax.set_title(f"{tid} {_short(task.name)}\n{LABEL[rung]}\n"
                         f"dense {err['dense_region']['rel_l2']:.1e}, "
                         f"uniform {err['uniform']['rel_l2']:.1e}", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cb.set_label("log10(resolution field / even mean)")
    fig.suptitle(f"B = {budget}: resolution field $\\sum_k \\gamma_k\\,$sech$^2(u_k(x))$ "
                 "(bright = finer mesh); white dots = training data", fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def highdim_figure(rows, path):
    """d >= 3 at one budget: bars of relative L2 per task, one bar per rung, three test
    sets side by side."""
    rows = [r for r in rows if r["d"] >= 3]
    if not rows:
        return None
    tasks = sorted({r["task"] for r in rows}, key=_key)
    rungs = [r for r in STYLE if any(x["rung"] == r for x in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    width = 0.8 / len(rungs)
    for ax, key in zip(axes, ["dense_region", "same_as_train", "uniform"]):
        for j, rung in enumerate(rungs):
            vals = []
            for tid in tasks:
                rec = next((r for r in rows if r["task"] == tid and r["rung"] == rung), None)
                vals.append(np.nan if rec is None else rec["errors"][key]["rel_l2"])
            c, _, _ = STYLE[rung]
            ax.bar(np.arange(len(tasks)) + (j - len(rungs) / 2 + 0.5) * width, vals, width,
                   color=c, label=LABEL[rung], edgecolor="k", lw=0.3,
                   alpha=0.55 if "oracle" in rung else 1.0)
        ax.set_yscale("log")
        ax.set_ylim(1e-15, 1e1)
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_xticklabels([f"{t}\n{_tiny(next(r['name'] for r in rows if r['task'] == t))}"
                            for t in tasks], fontsize=7, rotation=30, ha="right")
        ax.set_title(SET_LABEL[key], fontsize=10)
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].set_ylabel("relative $L_2$ error")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=8,
               frameon=False)
    B = sorted({r["budget"] for r in rows})
    fig.suptitle(f"d >= 3 tasks at B = {B}: even mesh vs the adapted meshes", y=0.90,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def split_figure(rows, path):
    """d = 3 even mesh: error vs centers per direction at fixed budget."""
    tasks = sorted({r["task"] for r in rows}, key=_key)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, key in zip(axes, ["dense_region", "same_as_train", "uniform"]):
        for k, tid in enumerate(tasks):
            pts = sorted([(r["n_per"], r["errors"][key]["rel_l2"]) for r in rows if r["task"] == tid])
            name = _short(next(r["name"] for r in rows if r["task"] == tid))
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], "-o", ms=4, color=cmap(k),
                      label=f"{tid} {name}")
        ax.axvline(16, color="k", ls="--", lw=0.8)
        ax.set_xlabel("centers per direction (dashed = the reference split $B^{1/3}=16$)")
        ax.set_title(SET_LABEL[key], fontsize=10)
        ax.set_ylim(1e-15, 1e1)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("relative $L_2$ error")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=8,
               frameon=False)
    B = rows[0]["budget"]
    fig.suptitle(f"d = 3, even mesh, B = {B}: directions x centers per direction = B", y=0.87,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def all_figures(results_dir: Path, fig_dir: Path):
    import glob
    rows = []
    for p in sorted(glob.glob(str(results_dir / "ladder_*.json"))):
        with open(p) as f:
            rows += json.load(f)["rows"]
    out = []
    if rows:
        for d in (1, 2):
            for key in ("dense_region", "uniform", "same_as_train"):
                out.append(ladder_figure(rows, d, key, CENTER_GROUP,
                                         fig_dir / f"ladder_centers_d{d}_{key}.png",
                                         f"Center placement rungs, d = {d}"))
                if d == 2:
                    out.append(ladder_figure(rows, d, key, DIRECTION_GROUP,
                                             fig_dir / f"ladder_directions_d{d}_{key}.png",
                                             f"Direction placement rungs, d = {d}"))
        out.append(gain_figure([r for r in rows if r["d"] <= 2], fig_dir / "gain_at_top_budget.png"))
        out.append(highdim_figure(rows, fig_dir / "highdim_bars.png"))
    sp = results_dir / "split_d3.json"
    if sp.exists():
        with open(sp) as f:
            out.append(split_figure(json.load(f)["rows"], fig_dir / "split_d3.png"))
    fp = results_dir / "floor.json"
    if fp.exists():
        with open(fp) as f:
            out.append(floor_figure(json.load(f)["rows"], fig_dir / "floor_sweep_d1.png"))
    for p in out:
        if p:
            print("saved", p)
