"""expF17 plots. ONE sweep, one figure per plot: all 16 tasks (8 PDE + 8 dysts)
as subplots of the same figure, 4x4.

1a/1b: rel L2 vs W, oracle / dynamic. 2: best config per method (bars + table).
3: the polynomial ablation (2-D QI arms only -- poly is not defined for dysts).
4a/4b: residual fields at best landed W -- QI-Radon / QI-tensor heatmaps for
the 2-D tasks (log10 error, train/score box, BC/IC geometry drawn in black,
outside the box masked out of the log scale) and QI(grid) per-component
|error(t)| lines for the dysts tasks (IC marked at s=-1).

House rules: legends outside the axes (single top figure legend), shared y
scale, x = W at ACTUAL column counts, geomean with GSD bars [g/s_g, g*s_g].
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import protocol as pr, store

TASK_TITLE = {
    "convection_c40": "convection c=40", "convection_c80": "convection c=80",
    "reaction": "reaction", "wave": "wave", "burgers": "Burgers",
    "poisson_man": "Poisson (perforated)", "darcy_man": "Darcy (manufactured)",
    "darcy_orig": "Darcy (FNO, ref-capped)",
}
DYSTS_ORDER = ["dysts_Lorenz", "dysts_Rossler", "dysts_Thomas",
               "dysts_Halvorsen", "dysts_Lorenz96", "dysts_InteriorSquirmer",
               "dysts_DoublePendulum", "dysts_MacArthur"]
ALL_TASKS = pr.TASK_ORDER + DYSTS_ORDER
for _k in DYSTS_ORDER:
    TASK_TITLE[_k] = _k[len("dysts_"):]

# provenance colour groups (Sam): bwler / darcy / dysts / other
FAMILY_COLOR = {"bwler": "#1f77b4", "darcy": "#d62728", "dysts": "#2ca02c",
                "other": "#7f7f7f"}
TASK_FAMILY = {"convection_c40": "bwler", "convection_c80": "bwler",
               "reaction": "bwler", "wave": "bwler", "burgers": "bwler",
               "poisson_man": "bwler", "darcy_man": "darcy",
               "darcy_orig": "darcy",
               **{k: "dysts" for k in DYSTS_ORDER}}

# published best-to-beat (dotted): BWLer Table 2; trained FNO ~1e-2 (expF08)
SOTA = {"convection_c40": 2.04e-13, "convection_c80": 1.10e-12,
        "reaction": 6.94e-11, "wave": 1.26e-11, "burgers": 4.63e-3,
        "darcy_orig": 1e-2}
# BWLer Table 2, "MLP (from literature)" column: the vanilla-PINN reference
# dots for the PINN arm (SPEC 18.8); drawn at the right edge of plot 1
PINN_LIT = {"convection_c40": 1.94e-3, "convection_c80": 6.88e-4,
            "wave": 1.27e-2, "reaction": 9.92e-3, "burgers": 1.33e-2}
METHOD_COLOR = {"qi_radon": "C3", "qi_tensor": "C0", "elm": "C2",
                "spectral": "C1", "bwler": "C8", "rbf_imq": "C4",
                "rbf_phs": "C5", "qi_grid": "C3", "pinn": "C6"}
METHOD_LABEL = {"qi_radon": "QI-Radon", "qi_tensor": "QI-tensor", "elm": "ELM",
                "spectral": "spectral (F/C)", "bwler": "BWLer (explicit)",
                "rbf_imq": "RBF-IMQ", "rbf_phs": "RBF-PHS+p1",
                "qi_grid": "QI", "pinn": "PINN (trained)"}
YLIM = (1e-16, 3e0)
EXT = 2.25  # zoomed out: the radon offset circle (~1.9) must sit fully on-canvas


def _is_1d(task):
    return task.startswith("dysts_")


def _methods_for(task):
    # the PINN arm (run_pinn.py) is 2-D only; plot 2 shows every method
    return pr.METHODS_1D if _is_1d(task) else pr.ALL_METHODS + ["pinn"]


def _agg(cells, regime, variant="base", methods=None):
    g = defaultdict(lambda: defaultdict(list))
    for rec in cells.values():
        if rec["regime"] != regime or rec["variant"] != variant:
            continue
        if methods and rec["method"] not in methods:
            continue
        g[(rec["task"], rec["method"])][rec["cols"]].append(rec["rel_l2"])
    out = {}
    for k, by_cols in g.items():
        out[k] = [(c,) + pr.geomean_gsd(v) + (len(v),)
                  for c, v in sorted(by_cols.items())]
    return out


def _grid_fig():
    fig, axes = plt.subplots(4, 4, figsize=(19, 17))
    return fig, axes.ravel()


def _task_ticks(task, data=None):
    """Discrete ticks at the protocol ladder ONLY -- the exact QI column identity
    (n_ax+1)^2 plus the declared extras. Methods whose column count is slightly
    off the identity (comparator aspect sweeps, the +3 PHS block, the +10 poly
    block) plot at their true column count with NO tick (Sam, 2026-09-02).
    `data` is accepted for call compatibility and ignored."""
    from .dicts import n_ax_for
    if _is_1d(task):
        ticks = set(pr.ladder_1d(task[len("dysts_"):])) | set(pr.EXTRA_LADDER_1D)
    else:
        Cs = list(pr.LADDER) + list(pr.EXTRA_LADDER.get(task, []))
        ticks = {(n_ax_for(C) + 1) ** 2 for C in Cs}
    return sorted(ticks)


def _style_axis(ax, task, data=None):
    import matplotlib.ticker as mtick
    ax.set_title(TASK_TITLE.get(task, task), fontsize=10,
                 color=FAMILY_COLOR.get(TASK_FAMILY.get(task, "other"), "k"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_xlim((56, 1150) if _is_1d(task) else (80, 12000))
    ticks = _task_ticks(task, data)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=55, fontsize=6.5)
    ax.xaxis.set_minor_locator(mtick.NullLocator())
    ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
    if SOTA.get(task) is not None:
        ax.axhline(SOTA[task], color="k", lw=1.2, ls=":")
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlabel("$W$")


SCALING_PLOT_METHODS = {True: ["qi_grid", "elm"],          # 1-D
                        False: ["qi_radon", "qi_tensor", "elm", "pinn"]}  # 2-D
# pinn is on the scaling axis (Sam: it is the same 1-hidden-layer tanh
# architecture, trained instead of solved); its cells land at the queue tail


def plot_scaling(cells, regime, path, title):
    """Plot 1 carries ONLY the neural architectures (single-hidden-layer tanh
    networks), where hidden-unit scaling is a shared, meaningful axis (SPEC 3,
    14.3). Spectral/BWLer/RBF are not neural networks -- they appear at best
    configuration on plot 2, the full cross-method comparison."""
    data = _agg(cells, regime)
    refit = _agg(cells, regime, "refit", methods=["pinn"])
    fig, axes = _grid_fig()
    handles = {}
    for k, task in enumerate(ALL_TASKS):
        ax = axes[k]
        _style_axis(ax, task, data)
        if k % 4 == 0:
            ax.set_ylabel("rel $L_2$")
        for method in SCALING_PLOT_METHODS[_is_1d(task)]:
            rows = data.get((task, method))
            if not rows:
                continue
            x = [r[0] for r in rows]; y = [r[1] for r in rows]
            ln = ax.plot(x, y, "o-", color=METHOD_COLOR[method], ms=4)[0]
            ax.fill_between(x, [r[1] / r[2] for r in rows],
                            [r[1] * r[2] for r in rows],
                            color=METHOD_COLOR[method], alpha=0.15)
            handles[method] = ln
        rows = refit.get((task, "pinn"))
        if rows:  # the frozen-feature certificate: same geometry, exact readout
            handles["pinn_refit"] = ax.plot(
                [r[0] for r in rows], [r[1] for r in rows], "o--",
                color=METHOD_COLOR["pinn"], ms=4, mfc="none")[0]
        if task in PINN_LIT and (task, "pinn") in data:
            handles["pinn_lit"] = ax.plot(
                [9000], [PINN_LIT[task]], "D", color=METHOD_COLOR["pinn"],
                ms=7, mfc="none")[0]
    order = [m for m in ["qi_radon", "qi_tensor", "elm", "pinn", "qi_grid"] if m in handles]
    hs = [handles[m] for m in order]
    ls = [METHOD_LABEL[m] for m in order]
    if "pinn_refit" in handles:
        hs.append(handles["pinn_refit"]); ls.append("PINN frozen-feature refit")
    if "pinn_lit" in handles:
        hs.append(handles["pinn_lit"]); ls.append("vanilla PINN (literature, BWLer Tab. 2)")
    hs.append(plt.Line2D([], [], color="k", ls=":", lw=1.2))
    ls.append("published best (BWLer/FNO)")
    fig.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 0.997),
               ncol=min(len(hs), 9), frameon=False)
    fig.suptitle(title, y=0.965, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparators(cells, path_png, path_md):
    lines = ["# expF17 Plot 2 -- best configuration per method\n",
             "geomean rel L2 over seeds at each method's best (W, config)\n"]
    fig, axes = _grid_fig()
    for k, task in enumerate(ALL_TASKS):
        ax = axes[k]
        methods = _methods_for(task)
        best = {}
        for regime in ("oracle", "dynamic"):
            data = _agg(cells, regime)
            for method in methods:
                rows = data.get((task, method))
                if rows:
                    c, gm, sg, n = min(rows, key=lambda r: r[1])
                    best[(method, regime)] = (gm, sg, c)
        if best:
            ms = [m for m in methods
                  if (m, "oracle") in best or (m, "dynamic") in best]
            xs = np.arange(len(ms))
            for off, regime, alpha in ((-0.2, "oracle", 1.0), (0.2, "dynamic", 0.55)):
                vals = [best.get((m, regime), (np.nan,) * 3) for m in ms]
                ax.bar(xs + off, [v[0] for v in vals], width=0.38,
                       color=[METHOD_COLOR[m] for m in ms], alpha=alpha, log=True)
            ax.set_xticks(xs)
            ax.set_xticklabels([METHOD_LABEL[m] for m in ms], rotation=40,
                               ha="right", fontsize=7)
        else:
            ax.set_xticks([])
            ax.text(0.5, 0.5, "blocked\n(reference pending)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_yscale("log")
        ax.set_ylim(*YLIM)
        ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
        if SOTA.get(task) is not None:
            ax.axhline(SOTA[task], color="k", lw=1.2, ls=":")
        ax.set_title(TASK_TITLE.get(task, task), fontsize=10,
                     color=FAMILY_COLOR.get(TASK_FAMILY.get(task, "other"), "k"))
        ax.grid(True, axis="y", which="both", alpha=0.2)
        lines.append(f"\n## {task}\n")
        lines.append("| method | regime | best W | geomean rel L2 | GSD |")
        lines.append("|---|---|---|---|---|")
        for (m, regime), (gm, sg, c) in sorted(best.items()):
            lines.append(f"| {METHOD_LABEL[m]} | {regime} | {c} | {gm:.2e} | {sg:.2f} |")
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="0.35", alpha=1.0, label="oracle regime (fit the known solution)"),
        Patch(facecolor="0.35", alpha=0.45, label="dynamic regime (solve the PDE / ODE)"),
        plt.Line2D([], [], color="k", ls=":", lw=1.2, label="published best (BWLer / FNO)"),
        plt.Line2D([], [], color="gray", ls=":", lw=0.8, label="fp64 floor, rel $L_2 = 10^{-13}$")],
        loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4, frameon=False,
        fontsize=10)
    fig.suptitle("Plot 2 -- best result per method: geomean rel $L_2$ over 3 seeds "
                 "at each method's best (W, tuned config); bar colour = method",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path_png, dpi=130)
    plt.close(fig)
    path_md.write_text("\n".join(lines) + "\n")


def plot_poly(cells, path):
    """Plot 3: 2-D QI arms with (dotted) / without (solid) the deg-3 block."""
    base = _agg(cells, "oracle", "base", methods=["qi_radon", "qi_tensor"])
    poly = _agg(cells, "oracle", "poly", methods=["qi_radon", "qi_tensor"])
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    axes = axes.ravel()
    for k, task in enumerate(pr.TASK_ORDER):
        ax = axes[k]
        _style_axis(ax, task, base)
        if k % 4 == 0:
            ax.set_ylabel("rel $L_2$ (oracle fit)")
        for method, col in (("qi_radon", "red"), ("qi_tensor", "blue")):
            for src, ls in ((base, "-"), (poly, ":")):
                rows = src.get((task, method))
                if rows:
                    ax.plot([r[0] for r in rows], [r[1] for r in rows], ls,
                            color=col, marker="o", ms=3)
    fig.legend(handles=[
        plt.Line2D([], [], color="red", ls="-", label="QI-Radon"),
        plt.Line2D([], [], color="red", ls=":", label="QI-Radon + deg-3 poly"),
        plt.Line2D([], [], color="blue", ls="-", label="QI-tensor"),
        plt.Line2D([], [], color="blue", ls=":", label="QI-tensor + deg-3 poly"),
        plt.Line2D([], [], color="k", ls=":", lw=1.2, label="published best")],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False)
    fig.suptitle("Plot 3 -- the polynomial block, applied to BOTH arms",
                 y=0.935, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot 4: residual fields
# ---------------------------------------------------------------------------

def _best_oracle_cell(cells, task, method):
    rows = [r for r in cells.values()
            if r["task"] == task and r["method"] == method
            and r["regime"] == "oracle" and r["variant"] == "base"]
    return min(rows, key=lambda r: r["rel_l2"]) if rows else None


def _draw_bc_geometry(ax, task):
    """Black marks where the boundary/initial conditions actually sit."""
    from .tasks import TASKS, f13
    for blk in TASKS[task]["bc_blocks"]:
        w = blk["where"]
        if w == "ic":
            ax.plot([-1, 1], [-1, -1], color="k", lw=2.5)
        elif w == "left":
            ax.plot([-1, -1], [-1, 1], color="k", lw=2.5)
        elif w == "right":
            ax.plot([1, 1], [-1, 1], color="k", lw=2.5)
        elif w == "periodic_x":
            for x in (-1, 1):
                ax.plot([x, x], [-1, 1], color="k", lw=2.0, ls="--")
        elif w == "square":
            ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec="k", lw=2.5))
        elif w == "holes":
            for cx, cy in f13.HOLE_CENTERS:
                ax.add_patch(plt.Circle((cx, cy), f13.HOLE_RADIUS, fill=False,
                                        ec="k", lw=1.8))


def _coverage_edge(ax, d):
    """Red dotted line at the dictionary's sampled-area edge, from the ACTUAL
    built geometry, so the translation is visible: radon = the centered circle
    at the outermost offset radius (the domain square must sit inside it);
    tensor = the outermost-centre square (ditto)."""
    meta = d.meta
    if meta["method"] == "qi_radon":
        r = float(np.max(np.abs(d.b) / np.hypot(d.a1, d.a2)))
        ax.add_patch(plt.Circle((0.0, 0.0), r, fill=False, ec="red", ls=":",
                                lw=1.6))
    elif meta["method"] == "qi_tensor":
        x0, x1 = d.cx.min(), d.cx.max()
        y0, y1 = d.cy.min(), d.cy.max()
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="red", ls=":", lw=1.6))


def _heat_panel(ax, cells, task, method):
    from . import dicts as fd, solve as sv
    from .tasks import TASKS
    cell = _best_oracle_cell(cells, task, method)
    if cell is None:
        ax.set_axis_off()
        return None
    t = TASKS[task]
    n_ax, cfg = cell["n_ax"], cell["config"]
    d = fd.build_dictionary(method, n_ax, cfg, fd.dict_rng(method, n_ax, 0),
                            fourier_x=t["periodic_fourier"])
    # per-panel zoom: the coverage edge must always sit on-canvas (burgers'
    # walked halo reaches r ~ 3.3; the fixed canvas hid its circle)
    if d.meta["method"] == "qi_radon":
        r_edge = float(np.max(np.abs(d.b) / np.hypot(d.a1, d.a2)))
    else:
        r_edge = float(max(np.abs(d.cx).max(), np.abs(d.cy).max()))
    ext = max(1.6, 1.12 * r_edge)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.85")
    if task == "darcy_orig":
        a, _ = sv.darcy_orig_fit(t, d, 0)
        Pe, ue = t["eval_grid"]()
        m = int(round(np.sqrt(len(Pe))))
        Eg = np.log10(np.abs(d.rows(Pe, [((0, 0), 1.0)]) @ a - ue) + 1e-18)
        pc = ax.pcolormesh(Pe[:, 0].reshape(m, m), Pe[:, 1].reshape(m, m),
                           Eg.reshape(m, m), shading="auto", cmap=cmap,
                           vmin=-17, vmax=0)
    else:
        a, _ = sv.oracle_fit(t, d, 0)
        g = np.linspace(-ext, ext, 240)
        GX, GY = np.meshgrid(g, g)
        P = np.stack([GX.ravel(), GY.ravel()], axis=1)
        inside = (np.abs(P[:, 0]) <= 1.0) & (np.abs(P[:, 1]) <= 1.0)
        if t["mask"] is not None:
            inside &= t["mask"](P)
        E = np.full(len(P), np.nan)
        Pi = P[inside]
        E[inside] = np.log10(np.abs(d.rows(Pi, [((0, 0), 1.0)]) @ a
                                    - t["exact"](Pi)) + 1e-18)
        pc = ax.pcolormesh(GX, GY, np.ma.masked_invalid(E.reshape(GX.shape)),
                           shading="auto", cmap=cmap, vmin=-17, vmax=0)
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec="k", lw=1.0))
    _coverage_edge(ax, d)
    _draw_bc_geometry(ax, task)
    ax.set_aspect("equal")
    halo = cell["config"].get("halo")
    htxt = f"  halo={halo}" if halo is not None else ""
    ax.set_title(f"{TASK_TITLE.get(task, task)}  W={cell['cols']}{htxt}  "
                 f"rel={cell['rel_l2']:.1e}", fontsize=8,
                 color=FAMILY_COLOR.get(TASK_FAMILY.get(task, "other"), "k"))
    return pc


def _dysts_panel(ax, cells, task):
    """|error(t)| per component, QI (grid), best landed W. IC (the start) is
    the thick black line at s = -1."""
    from . import dicts as fd, dicts1d as f1, solve1d as s1
    from .tasks1d import dysts_task
    cell = _best_oracle_cell(cells, task, "qi_grid")
    if cell is None:
        ax.set_axis_off()
        return
    t = dysts_task(task[len("dysts_"):])
    d = f1.build_dictionary_1d("qi_grid", cell["C"], cell["config"],
                               fd.dict_rng("qi_grid", cell["C"], 0))
    A, sigma, _ = s1.oracle_fit_1d(t, d, 0)
    srel = 2.0 * t["ts"] / t["T"] - 1.0
    E = np.abs((d.rows(srel, 0) @ A.T) * sigma[None, :] - t["Yref"]) + 1e-18
    for c in range(E.shape[1]):
        ax.semilogy(srel, E[:, c], lw=0.7)
    ax.axvline(-1.0, color="k", lw=2.5)   # IC row lives here
    ax.axvline(1.0, color="k", lw=1.0)
    ax.axvline(float(np.min(d.b / d.a)), color="red", ls=":", lw=1.6)
    ax.axvline(float(np.max(d.b / d.a)), color="red", ls=":", lw=1.6)
    ax.set_xlim(-EXT, EXT)
    ax.set_ylim(1e-17, 1e0)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_title(f"{TASK_TITLE[task]} (QI)  W={cell['cols']}  "
                 f"halo={cell['config'].get('halo')}  rel={cell['rel_l2']:.1e}",
                 fontsize=8, color=FAMILY_COLOR["dysts"])


def residual_figs(cells, tuning=None):
    from . import store as st
    for method, fname, label in [("qi_radon", "plot4a_residual_radon.png", "QI-Radon"),
                                 ("qi_tensor", "plot4b_residual_tensor.png", "QI-tensor")]:
        fig, axes = _grid_fig()
        pc = None
        for k, task in enumerate(ALL_TASKS):
            if _is_1d(task):
                _dysts_panel(axes[k], cells, task)
            else:
                pc = _heat_panel(axes[k], cells, task, method) or pc
        if pc is not None:
            fig.colorbar(pc, ax=list(axes), shrink=0.6,
                         label=r"$\log_{10}|\hat u - u^*|$ (grey = outside the box)")
        fig.legend(handles=[
            plt.Line2D([], [], color="red", ls=":", lw=1.6,
                       label="dictionary coverage edge (outermost centers/offsets)"),
            plt.Line2D([], [], color="k", lw=2.5, label="BC / IC location"),
            plt.Line2D([], [], color="k", lw=1.0, label="train + score box"),
            plt.Rectangle((0, 0), 1, 1, fc="0.85", ec="none",
                          label="outside the box (masked from the log scale)"),
            plt.Line2D([], [], color="C0", lw=0.9,
                       label="$u_1$ (state components, 1-D panels)"),
            plt.Line2D([], [], color="C1", lw=0.9, label="$u_2$"),
            plt.Line2D([], [], color="C2", lw=0.9, label="$u_3$"),
            plt.Line2D([], [], color="C3", lw=0.9, label="$u_4$"),
            plt.Line2D([], [], color="C4", lw=0.9, label=r"$u_5\,\ldots$")],
            loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5,
            frameon=False, fontsize=9)
        fig.suptitle(f"Plot 4 -- residual fields at best landed W (oracle, seed 0)"
                     f" -- 2-D panels: {label}; dysts panels: QI",
                     y=0.985, fontsize=13)
        fig.savefig(st.RESULTS_DIR / fname, dpi=130)
        plt.close(fig)


def write_status(cells):
    q = pr.build_queue()
    landed = sum(1 for e in q if pr.cell_key(e["task"], e["method"], e["C"],
                 e["seed"], e["regime"], e["variant"]) in cells)
    by = defaultdict(lambda: [0, 0])
    for e in q:
        k = (e["regime"], e["variant"])
        by[k][1] += 1
        if pr.cell_key(e["task"], e["method"], e["C"], e["seed"], e["regime"],
                       e["variant"]) in cells:
            by[k][0] += 1
    lines = [f"# expF17 status  ({time.strftime('%Y-%m-%d %H:%M:%S')})",
             f"\n{landed}/{len(q)} core cells landed\n"]
    for (regime, variant), (a, b) in sorted(by.items()):
        lines.append(f"- {regime}/{variant}: {a}/{b}")
    (store.RESULTS_DIR / "status.md").write_text("\n".join(lines) + "\n")


def replot():
    cells = store.load()
    write_status(cells)
    if not cells:
        return
    d = store.RESULTS_DIR
    plot_scaling(cells, "oracle", d / "plot1a_oracle.png",
                 "Plot 1a -- oracle regime (fit the known solution / reference)")
    plot_scaling(cells, "dynamic", d / "plot1b_dynamic.png",
                 "Plot 1b -- dynamic regime (solve the PDE / ODE)")
    plot_comparators(cells, d / "plot2_best.png", d / "plot2_table.md")
    plot_poly(cells, d / "plot3_poly.png")
