"""expI02 figures, six in all, each rebuilt from the study JSON files. Legends sit above the axes, never inside.
Precision panels use a fixed log axis [1e-15, 1e1]; trajectories are drawn wherever they exist."""
import json
import math
from pathlib import Path

import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _run():
    """The experiment's run.py loaded by path (another experiment's run.py may own the name 'run')."""
    spec = importlib.util.spec_from_file_location("expi02_run", Path(__file__).resolve().parent / "run.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

YLIM = (1e-15, 1e1)
COORD_COLORS = {"raw": "#1f77b4", "val": "#2ca02c", "white": "#d62728", "val_white": "#9467bd"}
COORD_LABELS = {"raw": "raw tanh weights", "val": "function-value coords", "white": "raw + capped whitening", "val_white": "values + capped whitening"}
ARM_STYLE = {"block": ("#d62728", "o", "-"), "shallow": ("#ff7f0e", "s", "--"), "mlp_params": ("#1f77b4", "^", ":"), "mlp_flops": ("#17becf", "v", ":"), "linear": ("#7f7f7f", "x", "-.")}
ARM_LABELS = {"block": "QI block", "shallow": "shallow ridge-QI, same units", "mlp_params": "MLP, same params", "mlp_flops": "MLP, same FLOPs", "linear": "linear / ridge"}


def _load(res_dir, name):
    p = Path(res_dir) / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def _legend_above(ax, ncol, y=1.02, **kw):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, borderaxespad=0, fontsize=8, frameon=False, **kw)


def _save(fig, res_dir, name):
    out = Path(res_dir) / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return out / f"{name}.png"


# ----------------------------------------------------------------------------- fig 1: the fixed-feature ladder
def fig1(res_dir):
    A0 = _load(res_dir, "a0")
    targets = sorted({k.split("|")[0] for k in A0})
    fig, axes = plt.subplots(1, len(targets), figsize=(3.6 * len(targets), 3.6), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), targets):
        for coords, col in COORD_COLORS.items():
            for opt, ls in [("momentum", "-"), ("adam", ":")]:
                r = A0.get(f"{name}|{coords}|{opt}")
                if r:
                    ax.plot(r["step"], r["test"], ls, color=col, lw=1.4, label=f"{COORD_LABELS[coords]}" if opt == "momentum" else None)
        fl = A0.get(f"{name}|floor")
        if fl is not None:
            ax.axhline(fl, color="grey", ls="--", lw=1, label="solved head (lstsq floor)")
        ax.set_yscale("log"); ax.set_ylim(*YLIM); ax.set_title(name.replace("_", " "), fontsize=10); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("test rel L2, trained head")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=8, frameon=False)
    fig.text(0.5, 1.13, "solid: heavy-ball momentum, lr = 1/L;  dotted: Adam", ha="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, res_dir, "fig1_ladder")


# ----------------------------------------------------------------------------- fig 2: the mechanism ladder
TARGET_COLORS = {"gauss_bump": "#1f77b4", "fast_waves": "#d62728", "composition": "#2ca02c", "three_bumps": "#9467bd",
                 "product_peak": "#ff7f0e", "radial_runge": "#8c564b", "random_ridges": "#7f7f7f"}


def fig2(res_dir, ladder=None, targets=None, gap_target="gauss_bump"):
    A1 = _load(res_dir, "a1")
    if not A1:
        raise FileNotFoundError("a1.json")
    R = _run(); A1_LADDER, A1_TARGETS = R.A1_LADDER, R.A1_TARGETS
    extra = [r for r in ["mesh+lr0.02", "white+lr0.02"] if any(k.split("|")[1] == r for k in A1 if k.count("|") == 2)]
    ladder = ladder or (A1_LADDER + extra); targets = targets or A1_TARGETS
    winner = (A1.get("_winner") or {}).get("name")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    xs = list(range(len(ladder)))
    for name in targets:
        col = TARGET_COLORS.get(name, "k")
        seeds = sorted({int(k.split("|")[2]) for k in A1 if k.startswith(f"{name}|") and k.count("|") == 2})
        fin = [[A1[f"{name}|{r}|{s}"]["final"] for s in seeds if f"{name}|{r}|{s}" in A1] for r in ladder]
        gn = [[A1[f"{name}|{r}|{s}"].get("final_gn", np.nan) for s in seeds if f"{name}|{r}|{s}" in A1] for r in ladder]
        med = [np.median(v) if v else np.nan for v in fin]
        ax.plot(xs, med, "o-", color=col, lw=1.4, label=name.replace("_", " "))
        for x, v in zip(xs, fin):
            if len(v) > 1:
                ax.plot([x] * len(v), v, ".", color=col, alpha=0.5, ms=4)
        ax.plot(xs, [np.nanmedian(v) if v else np.nan for v in gn], "o", mfc="none", color=col, ms=7)
        o = A1.get(f"{name}|oracle")
        if o:
            ax.axhline(o["err"], color=col, ls=":", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([r.replace("_", "\n").replace("+", "\n+").replace("lr0.02", "lr 0.02") for r in ladder], fontsize=7)
    ax.set_yscale("log"); ax.set_ylim(*YLIM); ax.set_ylabel("test rel L2, solved head, 2000 steps"); ax.grid(alpha=0.3)
    if extra:
        ax.axvline(len(A1_LADDER) - 0.5, color="grey", lw=0.8, ls="--")
    ax.set_title("(a) from the notebook recipe to the corrected one (filled: first order; hollow: +Gauss-Newton; dotted: oracle floor)", fontsize=8, y=1.0)
    _legend_above(ax, ncol=4, y=1.07)
    # (b) the solved-head gap on the winner's trained-head variant
    if winner and f"{gap_target}|{winner}+trained|0" in A1:
        r = A1[f"{gap_target}|{winner}+trained|0"]
        bx.plot(r["step"], r["test"], "-", color="#1f77b4", label="trained head (Adam)")
        bx.plot(r["step"], r["solved"], "-", color="#d62728", label="same features, head solved")
        v = A1.get(f"{gap_target}|{winner}|0")
        if v:
            bx.plot(v["step"], v["test"], "--", color="k", lw=1, label="VarPro (head solved every step)")
        p = A1.get(f"{gap_target}|{winner}+periodic|0")
        if p:
            bx.plot(p["step"], p["solved"], ":", color="#2ca02c", lw=1.2, label="head re-solved every 250 steps")
        bx.set_title(f"(b) the solved-head gap, {gap_target.replace('_', ' ')}, recipe {winner}", fontsize=8, y=1.0)
    bx.set_yscale("log"); bx.set_ylim(*YLIM); bx.set_xlabel("step"); bx.set_ylabel("test rel L2"); bx.grid(alpha=0.3)
    _legend_above(bx, ncol=2, y=1.07)
    fig.tight_layout()
    return _save(fig, res_dir, "fig2_mechanisms")


# ----------------------------------------------------------------------------- fig 3: scaling at d = 4
KNOB_COLORS = {"N1": "#d62728", "N2": "#ff7f0e", "M": "#2ca02c", "K": "#9467bd"}


def fig3(res_dir):
    A2 = _load(res_dir, "a2")
    if not A2:
        raise FileNotFoundError("a2.json")
    R = _run(); A2_TARGETS, A2_SWEEP, A2_DEPTH, A2_MLP_W = R.A2_TARGETS, R.A2_SWEEP, R.A2_DEPTH, R.A2_MLP_W
    fig, axes = plt.subplots(1, len(A2_TARGETS), figsize=(4.2 * len(A2_TARGETS), 4.0), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), A2_TARGETS):
        for knob, col in KNOB_COLORS.items():
            pts = [(A2[k]["counts"]["params"], A2[k]["final"], v) for v in A2_SWEEP[knob] if (k := f"{name}|{knob}|{v}") in A2]
            if pts:
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, lw=1.3, label=f"sweep {knob}")
                for p in pts:
                    ax.annotate(str(p[2]), (p[0], p[1]), fontsize=6, color=col, xytext=(3, 3), textcoords="offset points")
        for res, ls, lab in [(False, "-", "depth 2,3,4"), (True, "--", "depth 3,4 + residual")]:
            pts = [(A2[k]["counts"]["params"], A2[k]["final"], dep) for dep, r in A2_DEPTH if r == res and (k := f"{name}|depth|{dep}{'r' if r else ''}") in A2]
            if not res:
                pts = [p for p in pts]
            if pts:
                pts.sort(key=lambda p: p[2])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "s" + ls, color="k", lw=1.2, mfc="none" if res else "k", label=lab)
        pts = [(A2[k]["counts"]["params"], A2[k]["final"]) for w in A2_MLP_W if (k := f"{name}|mlp|{w}") in A2]
        if pts:
            ax.plot(*zip(*pts), "^:", color="#7f7f7f", lw=1.3, label="tanh MLP, width ladder")
        o = A2.get(f"{name}|oracle")
        if o:
            ax.axhline(o["err"], color="k", ls=":", lw=1, label="oracle floor at the base config")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(*YLIM); ax.set_xlabel("parameters"); ax.grid(alpha=0.3)
        ax.set_title(f"d = 4, {name.replace('_', ' ')}", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("test rel L2, solved head, 2000 steps")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, res_dir, "fig3_scaling")


# ----------------------------------------------------------------------------- fig 4: head to head
def fig4(res_dir):
    B1, B2 = _load(res_dir, "b1"), _load(res_dir, "b2")
    if not B1 and not B2:
        raise FileNotFoundError("b1.json / b2.json")
    R = _run(); B1_TARGETS, B2_SETS = R.B1_TARGETS, R.B2_SETS
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    arms = ["block", "shallow", "mlp_params", "mlp_flops"]
    offs = np.linspace(-0.27, 0.27, len(arms))
    names = [n for n in B1_TARGETS if any(k.startswith(n + "|") for k in B1)]
    for j, arm in enumerate(arms):
        col, mk, _ = ARM_STYLE[arm]
        for i, name in enumerate(names):
            vals = [B1[k][arm]["final"] for k in B1 if k.startswith(name + "|")]
            if not vals:
                continue
            x = i + offs[j]
            ax.plot([x, x], [min(vals), max(vals)], "-", color=col, lw=1.2, alpha=0.7)
            ax.plot(x, np.median(vals), mk, color=col, ms=6, label=ARM_LABELS[arm] if i == 0 else None)
            if arm == "block":
                g = [B1[k][arm].get("final_gn") for k in B1 if k.startswith(name + "|") and "final_gn" in B1[k][arm]]
                if g:
                    ax.plot(x, min(g), mk, color=col, ms=8, mfc="none", label="QI block + Gauss-Newton" if i == 0 else None)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(*YLIM); ax.set_ylabel("test rel L2 (median of seeds, bar = min to max)"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("(a) analytic targets, d = 5, matched parameters / FLOPs, 2000 steps", fontsize=9, y=1.0)
    _legend_above(ax, ncol=3, y=1.07)
    sets = [n for n in B2_SETS if any(k.startswith(n + "|") for k in B2)]
    for j, arm in enumerate(["block", "mlp_params"]):
        col, mk, _ = ARM_STYLE[arm]
        for i, name in enumerate(sets):
            rows = [B2[k] for k in B2 if k.startswith(name + "|")]
            vals = [r[arm]["rmse"] / r["ridge"] for r in rows]
            x = i + (-0.12 if j == 0 else 0.12)
            bx.plot([x, x], [min(vals), max(vals)], "-", color=col, lw=1.2, alpha=0.7)
            bx.plot(x, np.median(vals), mk, color=col, ms=6, label=ARM_LABELS[arm] if i == 0 else None)
    bx.axhline(1.0, color="#7f7f7f", ls="-.", lw=1, label="ridge regression")
    bx.set_xticks(range(len(sets))); bx.set_xticklabels([n.replace("_", "\n") for n in sets], fontsize=8)
    bx.set_yscale("log"); bx.set_ylabel("test RMSE / ridge RMSE"); bx.grid(alpha=0.3, axis="y")
    bx.set_title("(b) structured regression, 3 splits (Lorenz map is noise free)", fontsize=9, y=1.0)
    _legend_above(bx, ncol=3, y=1.07)
    fig.tight_layout()
    return _save(fig, res_dir, "fig4_headtohead")


# ----------------------------------------------------------------------------- fig 5: Fashion-MNIST
def fig5(res_dir):
    B3 = _load(res_dir, "b3")
    if not B3:
        raise FileNotFoundError("b3.json")
    B3_SIZES = _run().B3_SIZES
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.0))
    style = {"block": ("#d62728", "-"), "mlp_tanh": ("#1f77b4", "-"), "mlp_relu": ("#17becf", "--"), "logistic": ("#7f7f7f", "-.")}
    labels = {"block": "QI block", "mlp_tanh": "tanh MLP, same params", "mlp_relu": "ReLU MLP, same params", "logistic": "logistic regression"}
    for model, (col, ls) in style.items():
        key = f"large|{model}|0" if model != "logistic" else "logistic|0"
        r = B3.get(key)
        if r:
            ep = np.arange(len(r["err"]))                                   # logged at step 0 and after every epoch
            ax.plot(ep, 100 * (1 - np.array(r["err"])), ls, color=col, marker="o", ms=3, label=labels[model])
            if model == "block":
                ax.plot(ep, 100 * (1 - np.array(r["err_solved"])), ":", color=col, label="QI block, head refit by least squares")
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy (%)"); ax.grid(alpha=0.3); ax.set_ylim(75, 92)
    ax.set_title("(a) Fashion-MNIST, large size (88k parameters)", fontsize=9, y=1.0)
    _legend_above(ax, ncol=2, y=1.08)
    for model, (col, ls) in style.items():
        if model == "logistic":
            continue
        pts = [(B3[k]["counts"]["params"], 100 * (1 - B3[k]["final_err"])) for size in B3_SIZES if (k := f"{size}|{model}|0") in B3]
        if pts:
            bx.plot(*zip(*pts), "o" + ls, color=col, label=labels[model])
    r = B3.get("logistic|0")
    if r:
        bx.axhline(100 * (1 - r["final_err"]), color="#7f7f7f", ls="-.", label="logistic regression")
    bx.set_xscale("log"); bx.set_xlabel("parameters"); bx.set_ylabel("test accuracy after 10 epochs (%)"); bx.grid(alpha=0.3)
    bx.set_title("(b) accuracy against size", fontsize=9, y=1.0)
    _legend_above(bx, ncol=2, y=1.08)
    fig.tight_layout()
    return _save(fig, res_dir, "fig5_fashion")


# ----------------------------------------------------------------------------- fig 6: PINNs
def fig6(res_dir):
    B4 = _load(res_dir, "b4")
    if not B4:
        raise FileNotFoundError("b4.json")
    B4_PDES = _run().B4_PDES
    names = [n for n in B4_PDES if f"{n}|0" in B4]
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.8), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        r = B4[f"{name}|0"]
        for arm, col, lab in [("block", "#d62728", "QI block"), ("mlp_params", "#1f77b4", "tanh MLP, same params")]:
            a = r[arm]
            ax.plot(a["step"], a["err"], "-", color=col, label=lab + " (Adam on the PINN loss)")
            if "final_solved" in a:
                ax.plot(a["step"][-1], a["final_solved"], "o", color=col, ms=8, mfc="none", label=lab + ", head solved on the operator system")
        ax.set_yscale("log"); ax.set_ylim(*YLIM); ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.set_title(name, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("rel L2 against the exact solution")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, res_dir, "fig6_pinn")


ALL = dict(fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig6=fig6)
