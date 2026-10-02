"""Figures for expH06. Legends sit above the axes; error axes are fixed to [1e-14, 1]."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R

ERR_LIM = (1e-14, 3.0)
ACTION_STYLE = {"open_bg": ("tab:blue", "o", "open background directions"),
                "open_atom": ("tab:red", "*", "open an atom (found direction)"),
                "refine_bg": ("tab:cyan", "s", "refine background blocks"),
                "refine_atoms": ("tab:orange", "D", "refine atom blocks"),
                "init": ("k", ".", "start")}


def _legend_above(ax, ncol=3):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol, borderaxespad=0, fontsize=8, frameon=False)


# ---------------------------------------------------------------------------
# floors
# ---------------------------------------------------------------------------

def plot_floors(path=R.FLOORS_JSON, out=R.FIG_DIR / "floors_d3.png"):
    data = json.loads(path.read_text())
    keys = data["targets"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for i, k in enumerate(keys):
        Ms = [c["M"] for c in data["e_M"]]
        es = [c["rel_l2"][k] for c in data["e_M"]]
        ax.plot(Ms, es, "-o", color=colors[i], ms=4, label=k)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(*ERR_LIM)
    ax.set_xlabel(f"directions M  (N = {data['em_N']} offsets each)"); ax.set_ylabel("rel L2 on the inner ball")
    ax.set_title("e_M(M): the directions floor", fontsize=10, pad=30); ax.grid(alpha=0.3)
    _legend_above(ax, ncol=3)
    ax = axes[1]
    for i, k in enumerate(keys):
        Ns = [c["N"] for c in data["e_N"]]
        es = [c["rel_l2"][k] for c in data["e_N"]]
        ax.plot(Ns, es, "-o", color=colors[i], ms=4, label=k)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(*ERR_LIM)
    ax.set_xlabel(f"offsets per direction N  (M = {data['en_M']} directions)")
    ax.set_title("e_N(N): the offsets floor", fontsize=10, pad=30); ax.grid(alpha=0.3)
    _legend_above(ax, ncol=3)
    ax = axes[2]
    if data["exact"]:
        for i, k in enumerate(keys):
            eM = {c["M"]: c["rel_l2"][k] for c in data["e_M"]}
            eN = {c["N"]: c["rel_l2"][k] for c in data["e_N"]}
            xs, ys = [], []
            for c in data["exact"]:
                if c["M"] in eM and c["N"] in eN:
                    xs.append(max(eM[c["M"]], eN[c["N"]])); ys.append(c["rel_l2"][k])
            ax.plot(xs, ys, "o", color=colors[i], ms=5, label=k)
        ax.plot(ERR_LIM, ERR_LIM, "k-", lw=0.8)
        ax.plot(ERR_LIM, [2 * e for e in ERR_LIM], "k:", lw=0.8)
        ax.plot(ERR_LIM, [0.5 * e for e in ERR_LIM], "k:", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*ERR_LIM); ax.set_ylim(*ERR_LIM)
    ax.set_xlabel("max(e_M(M), e_N(N))  (predicted)"); ax.set_ylabel("measured e(M, N)")
    ax.set_title("max-of-two-floors test (dotted: factor 2)", fontsize=10, pad=30); ax.grid(alpha=0.3)
    _legend_above(ax, ncol=3)
    fig.suptitle(f"d = {data['d']}, data ball r = {data['r']}, nested directions, n_train = {data['n_train']}", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# grow trajectories
# ---------------------------------------------------------------------------

def _even_best(even_rows, k):
    best = {}
    for r in even_rows:
        e = r["rel_l2"][k]
        if r["budget"] not in best or e < best[r["budget"]][0]:
            best[r["budget"]] = (e, r["N"])
    Bs = sorted(best)
    return Bs, [best[B][0] for B in Bs], [best[B][1] for B in Bs]


def plot_grow(paths=None, out=R.FIG_DIR / "grow_trajectories_d3.png"):
    if paths is None:
        paths = sorted(R.RESULTS_DIR.glob("grow_d3*.json"))
    runs, even, meta = {}, [], None
    for p in paths:
        d = json.loads(p.read_text())
        meta = d
        runs.update(d["grow"])
        if d["even"]:
            even += d["even"]
    keys = list(runs)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linestyle="none", marker=m, color=c, ms=7 if m == "*" else 5, label=lab)
               for act, (c, m, lab) in ACTION_STYLE.items() if act != "init"]
    handles += [Line2D([], [], color="k", ls="--", lw=1.2, label="even mesh, best split per budget"),
                Line2D([], [], color="0.6", ls=":", lw=0.8, label="even mesh at N = 8, 16, 32"),
                Line2D([], [], linestyle="none", marker="+", color="k", ms=12, mew=2, label="final, test set")]
    n = len(keys)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        h = runs[k]["history"]
        us = [r["units"] for r in h if r["units"] > 0]
        es = [r["val_err"] for r in h if r["units"] > 0]
        ax.plot(us, es, "-", color="0.4", lw=1, zorder=1)
        for act, (c, m, lab) in ACTION_STYLE.items():
            pts = [(r["units"], r["val_err"]) for r in h if r["action"] == act and r["units"] > 0]
            if pts:
                ax.plot(*zip(*pts), linestyle="none", marker=m, color=c, ms=7 if m == "*" else 5, label=lab, zorder=3)
        ev_rows, seen = [], set()
        for r in even:
            if k in r["rel_l2"] and (r["budget"], r["N"]) not in seen:
                seen.add((r["budget"], r["N"])); ev_rows.append(r)
        if ev_rows:
            Bs, es_b, Ns_b = _even_best(ev_rows, k)
            ax.plot(Bs, es_b, "k--", lw=1.2, label="even mesh, best split per budget")
            for N in sorted({r["N"] for r in ev_rows}):
                rows = sorted([r for r in ev_rows if r["N"] == N], key=lambda r: r["budget"])
                ax.plot([r["budget"] for r in rows], [r["rel_l2"][k] for r in rows], ":", color="0.6", lw=0.8)
        ax.plot([runs[k]["final"]["units"]], [runs[k]["final_test_rel_l2"]], "k+", ms=12, mew=2, label="final, test set")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(*ERR_LIM); ax.set_xlim(16, 8192)
        ax.grid(alpha=0.3)
        ax.set_title(f"{k}   (final: {runs[k]['final']['n_dir']} dirs, {runs[k]['final']['n_atoms']} atoms)", fontsize=10)
        ax.set_xlabel("units (total offsets)"); ax.set_ylabel("rel L2 (validation, inner ball)")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"the greedy hierarchy vs the even mesh, d = {meta['d']}, ball r = {meta['r']}, budget {meta['budget']}", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# ridge recovery
# ---------------------------------------------------------------------------

def plot_ridges(path=None, out=R.FIG_DIR / "ridge_recovery.png"):
    from modes import RIDGES_JSON
    rows = json.loads((path or RIDGES_JSON).read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    dims = sorted({r["d"] for r in rows})
    cols = {3: "tab:blue", 4: "tab:red", 5: "tab:green"}
    for d in dims:
        sub = [r for r in rows if r["d"] == d]
        rc = [r["r"] for r in sub]
        axes[0].plot(rc, [r["test_rel_l2"] for r in sub], "o", color=cols[d], ms=5, alpha=0.8, label=f"d = {d}")
        axes[1].plot(rc, [max(1e-17, max(r["direction_errors_rad"])) for r in sub], "o", color=cols[d], ms=5, alpha=0.8, label=f"d = {d}")
        axes[2].plot(rc, [r["stages"][-1]["train_rel"] for r in sub], "o", color=cols[d], ms=5, alpha=0.8, label=f"d = {d}")
    for ax, t, yl in zip(axes, ["test rel L2 after joint polish", "largest direction error (radians)",
                                 "train residual BEFORE the joint polish"],
                         ["rel L2", "radians", "rel residual"]):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("number of hidden ridges r")
        ax.set_ylabel(yl); ax.set_title(t, fontsize=10, pad=18); ax.grid(alpha=0.3); _legend_above(ax, ncol=3)
        ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    axes[0].set_ylim(*ERR_LIM); axes[1].set_ylim(1e-17, 1); axes[2].set_ylim(*ERR_LIM)
    fig.suptitle("hidden-ridge recovery: projection pursuit + Gauss-Newton polish, 3 seeds, ball r = 0.3", y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def plot_anatomy(paths=None, out=None):
    """Per target: the final mesh block by block in insertion order (offsets per block,
    atoms red / background blue), i.e. what the hierarchy actually built."""
    if paths is None:
        paths = sorted(R.RESULTS_DIR.glob("grow_d3*.json"))
    runs, meta = {}, None
    for p in paths:
        d = json.loads(p.read_text()); meta = d; runs.update(d["grow"])
    keys = list(runs); n = len(keys); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        f = runs[k]["final"]
        ns, kinds = f["n_per"], f["kinds"]
        cols = ["tab:red" if kd == "atom" else "tab:blue" for kd in kinds]
        ax.bar(np.arange(len(ns)), ns, color=cols, width=1.0, edgecolor="none")
        ax.set_yscale("log"); ax.set_ylim(4, 1024)
        ax.set_xlabel("block (insertion order)"); ax.set_ylabel("offsets in block")
        ax.set_title(f"{k}: {f['n_dir']} dirs, {f['n_atoms']} atoms, {f['units']} units, "
                     f"test {runs[k]['final_test_rel_l2']:.0e}", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"anatomy of the grown meshes (red: atoms, blue: nested background), d = {meta['d']}", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out or R.FIG_DIR / f"anatomy_d{meta['d']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def plot_push(out=R.FIG_DIR / "even_cells_d3.png"):
    """Every even 3-D cell measured (floors, exact, push): error vs units per target, marker
    color = offsets per direction N, so the reader sees which split reaches the floor."""
    cells = []
    fl = json.loads(R.FLOORS_JSON.read_text())
    for key in ("e_M", "e_N", "exact"):
        cells += fl[key]
    for p in sorted(R.RESULTS_DIR.glob("push_d3*.json")):
        cells += json.loads(p.read_text())["cells"]
    seen, uniq = set(), []
    for c in cells:
        if (c["M"], c["N"]) not in seen:
            seen.add((c["M"], c["N"])); uniq.append(c)
    keys = fl["targets"]
    Ns = sorted({c["N"] for c in uniq})
    cmap = plt.cm.viridis(np.linspace(0, 1, len(Ns)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for i, k in enumerate(keys):
        ax = axes[i // 3][i % 3]
        for j, N in enumerate(Ns):
            rows = sorted([c for c in uniq if c["N"] == N], key=lambda c: c["units"])
            ax.plot([c["units"] for c in rows], [c["rel_l2"][k] for c in rows], "-o", color=cmap[j], ms=5, lw=1, label=f"N = {N}")
        best = min(uniq, key=lambda c: c["rel_l2"][k])
        ax.annotate(f"({best['M']}, {best['N']})", (best["units"], best["rel_l2"][k]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
        ax.axhline(1e-13, color="k", lw=0.6, ls=":")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(*ERR_LIM); ax.set_xlim(128, 16384)
        ax.set_title(k, fontsize=10, pad=30); ax.grid(alpha=0.3)
        ax.set_xlabel("units = M N"); ax.set_ylabel("rel L2, inner ball")
        _legend_above(ax, ncol=5)
    fig.suptitle("every even nested-direction cell measured in d = 3 (ball r = 0.3): lines join cells with the same N;"
                 " label = (M, N) of the best cell", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def plot_alloc(out=R.FIG_DIR / "alloc_d3.png"):
    """The equal-budget allocation shootout: one panel per target, bars = the four arms at
    each budget, log error axis."""
    rows = []
    for p in sorted(R.RESULTS_DIR.glob("alloc_d3*.json")):
        rows += json.loads(p.read_text())["rows"]
    keys = list(dict.fromkeys(r["target"] for r in rows))
    arms = [("even", "0.55"), ("atoms", "tab:red"), ("spikes", "tab:orange"), ("waterfill", "tab:cyan")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.2), squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[0][i]
        sub = sorted([r for r in rows if r["target"] == k], key=lambda r: r["budget"])
        x = np.arange(len(sub))
        w = 0.2
        for a, (arm, col) in enumerate(arms):
            ax.bar(x + (a - 1.5) * w, [r[arm] for r in sub], width=w, color=col, label=arm)
        ax.set_yscale("log"); ax.set_ylim(1e-12, 1)
        ax.set_xticks(x); ax.set_xticklabels([f"B = {r['budget']}\n(even {r['even_M']}x{r['even_N']})" for r in sub], fontsize=9)
        ax.set_title(k, fontsize=10, pad=30); ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("rel L2, inner ball")
        _legend_above(ax, ncol=4)
    fig.suptitle("equal-budget allocation: even best split vs +atoms vs +spikes vs water-filled N_j (d = 3, r = 0.3, rcond 1e-14)", y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def all_figures():
    if R.FLOORS_JSON.exists():
        plot_floors()
    if (R.RESULTS_DIR / "floors_d4.json").exists():
        plot_floors(path=R.RESULTS_DIR / "floors_d4.json", out=R.FIG_DIR / "floors_d4.png")
        if list(R.RESULTS_DIR.glob("push_d3*.json")):
            plot_push()
    if list(R.RESULTS_DIR.glob("grow_d3*.json")):
        plot_grow()
        plot_anatomy()
    if list(R.RESULTS_DIR.glob("grow_d4*.json")):
        plot_grow(paths=sorted(R.RESULTS_DIR.glob("grow_d4*.json")), out=R.FIG_DIR / "grow_trajectories_d4.png")
        plot_anatomy(paths=sorted(R.RESULTS_DIR.glob("grow_d4*.json")))
    from modes import RIDGES_JSON
    if RIDGES_JSON.exists():
        plot_ridges()
    if list(R.RESULTS_DIR.glob("alloc_d3*.json")):
        plot_alloc()


if __name__ == "__main__":
    all_figures()
