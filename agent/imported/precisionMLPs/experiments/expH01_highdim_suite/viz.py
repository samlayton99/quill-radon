"""Figures for expH01.

``gallery_d1.png``                 the sixteen 1-D targets with the data drawn beneath
                                   them, and the burst / step regions shaded.
``gallery_d2.png``                 the sixteen 2-D targets as heat maps with the training
                                   points scattered on top and the burst / step regions
                                   outlined.
``gallery_d3.png`` (d4, d5)        for each task, the target on the plane spanned by
                                   ``u_1`` and ``u_2``, beside a 2-D histogram of the
                                   training points projected onto ``(z_1, z_2)``.
``predicted_center_density.png``   the burst of oscillation seen three ways: on uniform
                                   data, on clustered data with the burst on the densest
                                   cluster, and on clustered data with the burst away from
                                   every cluster.
``smoke_baseline.png``             the even-geometry reference against random features.

House rules: every legend sits outside its axes, above the plot; axis ranges are fixed
and meaningful; nothing is a lone datapoint.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from h01suite.basis import u as basis_u
from h01suite.densities import make_density
from h01suite.metrics import predicted_center_density
from h01suite.tasks import get_task, tasks_for_dim

DATA_COLOR = {"even_grid": "#4c72b0", "uniform": "#55a868", "hotspots": "#c44e52",
              "stretched_hotspots": "#8172b2", "flat_sheet": "#937860",
              "flat_sheet_noisy": "#da8bc3", "curved_sheet": "#8c8c8c",
              "curved_sheet_noisy": "#ccb974"}
BURST_SHADE = "#f2c14e"
STEP_SHADE = "#e05c5c"


def _wrap(text, width=64):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


def _panel_title(t, width=64):
    return f"{t.id}  {t.name}\n{_wrap(t.what_it_tests, width)}"


# ---------------------------------------------------------------------------
# 1-D gallery
# ---------------------------------------------------------------------------

def gallery_d1(path, n_sample=2000, seed=1):
    tasks = tasks_for_dim(1)
    fig, axes = plt.subplots(4, 4, figsize=(21, 14))
    x = np.linspace(-1.0, 1.0, 4001)[:, None]
    for ax, t in zip(axes.ravel(), tasks):
        F = t.F(x)
        lo, hi = F.min(), F.max()
        for mask_fn, color, lbl in ((t.packet_mask, BURST_SHADE, "burst of oscillation"),
                                    (t.jump_mask, STEP_SHADE, "near the step / slope break")):
            m = mask_fn(x)
            if m is None:
                continue
            ax.fill_between(x[:, 0], lo, hi, where=m, color=color, alpha=0.35, lw=0,
                            zorder=0, label=lbl)
        ax.plot(x[:, 0], F, color="#222222", lw=1.1, label="target")
        Xs = t.sample(n_sample, seed=seed)[:, 0]
        ax2 = ax.twinx()
        ax2.hist(Xs, bins=80, range=(-1, 1), color=DATA_COLOR[t.density_tag],
                 alpha=0.35, label=f"training points ({t.density_tag})")
        ax2.set_ylim(0, ax2.get_ylim()[1] * 3.2)      # keep the histogram in the lower third
        ax2.set_yticks([])
        ax.set_xlim(-1, 1)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_title(_panel_title(t, 62), fontsize=8, pad=24)
        ax.grid(alpha=0.25)
        ax.set_xlabel("$z_1 = x$", fontsize=8)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.005),
                  ncol=3, fontsize=6.5, frameon=False, borderaxespad=0)
    fig.suptitle("expH01 gallery, $d=1$: the scaled target (black), the training points "
                 "beneath it, the burst of oscillation (gold) and the step or slope-break "
                 "band (red)", fontsize=13, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2-D gallery
# ---------------------------------------------------------------------------

def gallery_d2(path, n_sample=1500, seed=1, side=300):
    tasks = tasks_for_dim(2)
    g = np.linspace(-1.0, 1.0, side)
    XX, YY = np.meshgrid(g, g)
    P = np.stack([XX.ravel(), YY.ravel()], axis=1)
    fig, axes = plt.subplots(4, 4, figsize=(21, 19))
    for ax, t in zip(axes.ravel(), tasks):
        Z = t.F(P).reshape(side, side)
        lim = float(np.percentile(np.abs(Z), 99.5))
        im = ax.imshow(Z, origin="lower", extent=(-1, 1, -1, 1), cmap="RdBu_r",
                       vmin=-lim, vmax=lim, aspect="equal")
        for mask_fn, color, lbl in ((t.packet_mask, "#7a5c00", "burst of oscillation"),
                                    (t.jump_mask, "#000000", "step / slope break")):
            m = mask_fn(P)
            if m is None:
                continue
            ax.contour(g, g, m.reshape(side, side).astype(float), levels=[0.5],
                       colors=color, linewidths=1.3)
            ax.plot([], [], color=color, lw=1.3, label=lbl)
        Xs = t.sample(n_sample, seed=seed)
        ax.plot(Xs[:, 0], Xs[:, 1], ".", ms=1.6, color="#111111", alpha=0.45,
                label=f"training points ({t.density_tag})")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("$x_1$", fontsize=8)
        ax.set_ylabel("$x_2$", fontsize=8)
        ax.set_title(_panel_title(t, 62), fontsize=8, pad=24)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=6.5,
                  frameon=False, borderaxespad=0, markerscale=4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("expH01 gallery, $d=2$: the scaled target (color), the training points "
                 "(black dots), and the outlines of the burst and step regions",
                 fontsize=13, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.978])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# d = 3, 4, 5 gallery
# ---------------------------------------------------------------------------

def gallery_high(path, d, n_sample=20000, seed=1, side=220):
    """Target on the plane spanned by ``u_1`` and ``u_2``, plus the projected data.

    The slice is ``x(z_1,z_2) = z_1 ||u_1||_1 u_1 + z_2 ||u_2||_1 u_2``, drawn over the
    full nominal range ``[-1,1]^2`` so that features placed at large ``|z_k|`` are
    visible. Only the inner diamond of that square is reachable inside the cube -- a
    plane through the origin cannot reach the corner in both coordinates at once -- so
    the unreachable part is hatched and outlined; the target formula is still defined
    there.
    """
    tasks = tasks_for_dim(d)
    u1, u2 = basis_u(d, 1), basis_u(d, 2)
    s1, s2 = float(np.abs(u1).sum()), float(np.abs(u2).sum())
    a = np.linspace(-1.0, 1.0, side)
    AA, BB = np.meshgrid(a, a)
    P = (AA.ravel()[:, None] * s1) * u1[None, :] + (BB.ravel()[:, None] * s2) * u2[None, :]
    inside = (np.abs(P).max(axis=1) <= 1.0).reshape(side, side)
    fig, axes = plt.subplots(8, 4, figsize=(17, 28))
    for i, t in enumerate(tasks):
        axL = axes[i // 2][2 * (i % 2)]
        axR = axes[i // 2][2 * (i % 2) + 1]
        Z = t.F(P).reshape(side, side)
        lim = float(np.percentile(np.abs(Z), 99.5))
        im = axL.imshow(Z, origin="lower", extent=(-1, 1, -1, 1), cmap="RdBu_r",
                        vmin=-lim, vmax=lim, aspect="equal")
        axL.contour(a, a, inside.astype(float), levels=[0.5], colors="#000000", linewidths=1.1)
        axL.contourf(a, a, (~inside).astype(float), levels=[0.5, 1.5], colors="none",
                     hatches=["///"])
        fig.colorbar(im, ax=axL, fraction=0.046, pad=0.02)
        axL.set_xlabel("$z_1$", fontsize=8)
        axL.set_ylabel("$z_2$", fontsize=8)
        axL.set_title(f"{t.id} target on the $(u_1,u_2)$ plane\n{_wrap(t.what_it_tests, 58)}",
                      fontsize=7.5)

        Xs = t.sample(n_sample, seed=seed)
        z1, z2 = Xs @ u1 / s1, Xs @ u2 / s2
        axR.hist2d(z1, z2, bins=90, range=[[-1, 1], [-1, 1]], cmap="viridis")
        axR.set_aspect("equal")
        axR.set_xlabel("$z_1$", fontsize=8)
        axR.set_title(f"{t.id} training points ({t.density_tag})", fontsize=8.5, pad=8)
    fig.suptitle(f"expH01 gallery, $d={d}$: the target on the plane spanned by $u_1,u_2$ "
                 f"in normalized coordinates (left; the hatched part of the square is "
                 f"outside the cube) and the training points projected onto $(z_1,z_2)$ "
                 f"(right)", fontsize=13, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.986])
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# predicted center density
# ---------------------------------------------------------------------------

CURVE_STYLE = {"uniform reference": ("#4c72b0", "-", 2.6),
               "burst on the densest cluster": ("#c44e52", "--", 1.9),
               "burst away from the clusters": ("#dd8452", ":", 2.2)}


def predicted_center_density_figure(path, n_sample=200000, n_centers=64.0, r=1, seed=5):
    """Three ways of seeing the same burst of oscillation, in ``d=1`` and ``d=2``.

    Left column ``d=1``, right column ``d=2``. In each column: the burst-at-the-hotspot
    target evaluated on *uniform* data (the reference), the same target on clustered data
    (task 1.12 / 2.12), and the burst-away-from-the-clusters target on the same clustered
    data (task 1.13 / 2.13). Everything is measured along the direction ``u_1``.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for col, d in enumerate((1, 2)):
        aligned, away = get_task(f"{d}.12"), get_task(f"{d}.13")
        uniform_data = make_density("uniform", d)
        cases = [("uniform reference", aligned, uniform_data.sample(n_sample, seed=seed)),
                 ("burst on the densest cluster", aligned,
                  aligned.sample(n_sample, seed=seed)),
                 ("burst away from the clusters", away, away.sample(n_sample, seed=seed))]
        v = basis_u(d, 1)
        for label, task, X in cases:
            res = predicted_center_density(v, task.grad_F, X, n_centers=n_centers, r=r,
                                           differentiable=task.differentiable)
            color, ls, lw = CURVE_STYLE[label]
            kw = dict(color=color, ls=ls, lw=lw, label=label)
            axes[0][col].plot(res["t"], res["p"], **kw)
            axes[1][col].semilogy(res["t"], np.maximum(res["slope_energy"], 1e-8), **kw)
            axes[2][col].plot(res["t"], res["density"], **kw)
        for row, ylab in enumerate(["density of the projected data\n(the two clustered "
                                    "cases coincide)",
                                    "average squared slope along $u_1$",
                                    "predicted centers per unit length\n(%g centers in total)"
                                    % n_centers]):
            ax = axes[row][col]
            ax.grid(alpha=0.3)
            ax.set_ylabel(ylab, fontsize=9)
            if row == 0:
                ax.set_title(f"$d={d}$", fontsize=12, pad=10)
            if row == 2:
                ax.set_xlabel("position along the direction, $t = u_1\\cdot x$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3,
               fontsize=10, frameon=False)
    fig.suptitle("expH01: how densely the theory says centers should be placed along "
                 "$u_1$\n(data density times average squared slope, all to the power "
                 "$1/3$, scaled to the number of centers)", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# smoke baseline
# ---------------------------------------------------------------------------

MODEL_STYLE = {"even_geometry": ("#1f77b4", "o"), "random_features": ("#d62728", "s")}
SET_LABEL = {"same_as_train": "points like the training data",
             "uniform": "uniform over the cube",
             "dense_region": "densest region only"}


def smoke_baseline(path, rows, budget=None):
    """rows: the records written by ``run.py --smoke``."""
    ids = sorted({r["task"] for r in rows},
                 key=lambda s: (int(s.split(".")[0]), int(s.split(".")[1])))
    xs = np.arange(len(ids), dtype=float)
    fig = plt.figure(figsize=(17, 14.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 1.1], hspace=0.75)
    for row, key in enumerate(["same_as_train", "uniform", "dense_region"]):
        ax = fig.add_subplot(gs[row])
        for model, (color, marker) in MODEL_STYLE.items():
            ys = []
            for tid in ids:
                rec = next((r for r in rows if r["task"] == tid and r["model"] == model), None)
                ys.append(np.nan if rec is None else rec["errors"][key]["rel_l2"])
            ax.semilogy(xs, ys, marker + "-", color=color, ms=5, lw=1.2, label=model)
        ax.set_xticks(xs)
        ax.set_xticklabels(ids, rotation=45, fontsize=8)
        ax.set_ylim(1e-15, 1e3)    # fixed across the three rows so they can be compared;
                                   # the solve is unregularized, so away from the data the
                                   # error genuinely exceeds 1 on some tasks
        ax.grid(alpha=0.3, which="both")
        ax.set_ylabel(f"relative $L_2$ error\n{SET_LABEL[key]}", fontsize=9)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=9,
                  frameon=False, borderaxespad=0)
        if row == 2:
            ax.set_xlabel("task")

    ax = fig.add_subplot(gs[3])
    binned = [tid for tid in ids
              if any(r["task"] == tid and r["by_data_density"] for r in rows)]
    for tid in binned:
        rec = next(r for r in rows if r["task"] == tid and r["model"] == "even_geometry")
        vals = [b["rel_l2"] for b in rec["by_data_density"]]
        ax.semilogy(np.arange(1, len(vals) + 1), vals, "-o", ms=4, lw=1.2, label=tid)
    ax.set_xlabel("bin of the test points, ordered by how dense the data is there "
                  "(1 = sparsest, 10 = densest)")
    ax.set_ylabel("relative $L_2$ error\nin the bin", fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=8, fontsize=8,
              frameon=False, borderaxespad=0)
    title = ("expH01: the even-geometry reference against random features, on the "
             "first-pass task list")
    if budget is not None:
        title += f"   ($B={budget}$ units, $n_{{train}}=8B$)"
    title += ("\nrows: fresh points like the training data, uniform over the cube, and "
              "the densest region only")
    fig.suptitle(title, fontsize=13, y=0.985)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
