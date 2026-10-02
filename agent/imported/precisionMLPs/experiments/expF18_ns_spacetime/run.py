"""expF18 -- 2-D unsteady Navier-Stokes (space + time = 3 coordinates) on the expH06
space-time ridge mesh: frozen QI geometry + Gauss-Newton collocation lstsq, no training.

Modes (all resumable through cells.json; coefficient vectors cached under coef/):
    --oracle   fit the exact (u, v, p) at every (W, N) of the ladder x offsets grid: the
               representational ceiling of the geometry and the N selector per width
    --tune     w_mult sweep {0.3, 1, 3} on the dynamic solve at W_TUNE (declared effort)
    --solve    the dynamic ladder: per width, N inherited from the oracle, w_mult from
               the tune, warm-started from the previous rung (the cascade)
    --plot     scaling / split / error-vs-time / Newton / residual-heatmap figures
    --gif      the animation of the top dynamic cell
    --smoke    a tiny end-to-end pass (W = 256) of everything

Usage:
    OMP_NUM_THREADS=10 uv run --extra dev python experiments/expF18_ns_spacetime/run.py --oracle --solve --plot --gif
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import problem as pb  # noqa: E402
import ridge3d as r3  # noqa: E402

RESULTS = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF18_ns_spacetime"
FIG = RESULTS / "figures"
COEF = RESULTS / "coef"
CELLS = RESULTS / "cells.json"
LOG = RESULTS / "run.log"

LADDER = [256, 512, 1024, 2048, 4096]
N_GRID = [8, 12, 16, 24, 32, 48]
W_TUNE = 1024
W_MULTS = [0.3, 1.0, 3.0]
W_MULT_WALK = [10.0, 30.0]      # expF17 edge-walk: extend while the grid edge keeps winning
OVERSAMPLE = 3          # interior points per column (rows/cols = 3.75 with the condition rows)
SEEDS = [0]


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def load_cells():
    return json.loads(CELLS.read_text()) if CELLS.exists() else []


def save_cells(cells):
    tmp = CELLS.with_suffix(".tmp")
    tmp.write_text(json.dumps(cells, indent=1))
    tmp.replace(CELLS)


def find(cells, **key):
    for c in cells:
        if all(c.get(k) == v for k, v in key.items()):
            return c
    return None


def coef_path(regime, W, N, seed, w_mult=None):
    tag = f"{regime}_W{W}_N{N}_s{seed}" + (f"_w{w_mult:g}" if w_mult is not None else "")
    return COEF / (tag + ".npy")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def build(W, N, seed):
    M = W // N
    d = r3.SpaceTimeDict(M, N, seed=seed)
    pts = r3.point_sets(OVERSAMPLE * d.cols, seed=seed)
    return d, pts


# ---------------------------------------------------------------------------
# oracle regime
# ---------------------------------------------------------------------------

def run_oracle(ladder, seeds, cells):
    for seed in seeds:
        for W in ladder:
            for N in N_GRID:
                if W // N < 4:
                    continue
                if find(cells, regime="oracle", W=W, N=N, seed=seed):
                    continue
                d, pts = build(W, N, seed)
                t0 = time.time()
                model, info = r3.oracle_fit(d, pts)
                sc = r3.score(model)
                rec = dict(regime="oracle", W=W, N=N, M=d.M, units=d.units, seed=seed, **info,
                           **{k: v for k, v in sc.items() if k != "per_t"}, per_t=sc["per_t"],
                           wall=round(time.time() - t0, 1))
                cells.append(rec)
                save_cells(cells)
                np.save(coef_path("oracle", W, N, seed), model.a)
                log(f"oracle W={W:5d} N={N:2d} M={d.M:4d} units={d.units:5d} rank={info['rank']:5d} "
                    f"rel_v={sc['rel_l2_v']:.2e} rel_p={sc['rel_l2_p']:.2e} linf_v={sc['linf_v']:.1e} "
                    f"[{rec['wall']:.0f}s]")


def best_oracle_N(cells, W, seed=0):
    rows = [c for c in cells if c["regime"] == "oracle" and c["W"] == W and c["seed"] == seed]
    if not rows:
        raise RuntimeError(f"no oracle cells at W={W}; run --oracle first")
    return min(rows, key=lambda c: c["rel_l2_v"])["N"]


# ---------------------------------------------------------------------------
# dynamic regime
# ---------------------------------------------------------------------------

def dynamic_cell(W, N, seed, w_mult, cells, warm=None, regime="dynamic", max_iter=15):
    d, pts = build(W, N, seed)
    asm = r3.Assembly(d, pts, w_mult=w_mult)
    log(f"{regime} W={W} N={N} M={d.M} units={d.units} rows={asm.n_rows} cols={asm.n_cols} "
        f"w_mult={w_mult} warm={'yes' if warm is not None else 'zero'}")
    t0 = time.time()
    a0 = None
    if warm is not None:
        vals = warm.fields(pts["Xi"])
        a0 = r3.fit_to_function(asm, vals)
        q = r3.quick_score(r3Model(d, a0))
        log(f"    warm start fitted: rel_v {q['rel_l2_v']:.2e} rel_p {q['rel_l2_p']:.2e}")
    with open(LOG, "a") as lf:
        a, hist = r3.gauss_newton(asm, a0, max_iter=max_iter, log=lf,
                                  eval_fn=lambda a: r3.quick_score(r3.Model(d, a)))
    del asm
    model = r3.Model(d, a)
    sc = r3.score(model)
    rec = dict(regime=regime, W=W, N=N, M=d.M, units=d.units, seed=seed, w_mult=w_mult,
               iters=len(hist), rank=hist[-1]["rank"], warm=warm is not None,
               **{k: v for k, v in sc.items() if k != "per_t"}, per_t=sc["per_t"], hist=hist,
               wall=round(time.time() - t0, 1))
    cells.append(rec)
    save_cells(cells)
    np.save(coef_path(regime, W, N, seed, w_mult if regime == "tune" else None), a)
    log(f"{regime} W={W:5d} N={N:2d} done: rel_v={sc['rel_l2_v']:.2e} rel_p={sc['rel_l2_p']:.2e} "
        f"linf_v={sc['linf_v']:.1e} mom_rms={sc['mom_rms']:.1e} div_max={sc['div_max']:.1e} "
        f"iters={len(hist)} [{rec['wall']:.0f}s]")
    return model


def r3Model(d, a):
    return r3.Model(d, a)


def load_model(cells, regime, W, seed, w_mult=None):
    c = find(cells, regime=regime, W=W, seed=seed) if w_mult is None else \
        find(cells, regime=regime, W=W, seed=seed, w_mult=w_mult)
    if c is None:
        return None
    d, _ = build(W, c["N"], seed)
    return r3.Model(d, np.load(coef_path(regime, W, c["N"], seed, w_mult)))


def run_tune(cells, seed=0):
    N = best_oracle_N(cells, W_TUNE, seed)
    warm = load_model(cells, "dynamic", W_TUNE // 2, seed)
    grid = list(W_MULTS)
    for w in grid:
        if find(cells, regime="tune", W=W_TUNE, seed=seed, w_mult=w):
            continue
        dynamic_cell(W_TUNE, N, seed, w, cells, warm=warm, regime="tune")
    for w in W_MULT_WALK:                       # walk past the edge while the edge wins
        if chosen_w_mult(cells, seed) != grid[-1]:
            break
        if not find(cells, regime="tune", W=W_TUNE, seed=seed, w_mult=w):
            dynamic_cell(W_TUNE, N, seed, w, cells, warm=warm, regime="tune")
        grid.append(w)


def chosen_w_mult(cells, seed=0):
    """The w_mult tuned on seed 0 at W_TUNE is inherited by every seed and width."""
    rows = [c for c in cells if c["regime"] == "tune" and c["seed"] == 0]
    if not rows:
        return 1.0
    return min(rows, key=lambda c: c["rel_l2_v"])["w_mult"]


def run_solve(ladder, seeds, cells, w_mult=None):
    for seed in seeds:
        w = chosen_w_mult(cells, seed) if w_mult is None else w_mult
        warm = None
        for W in ladder:
            N = best_oracle_N(cells, W, seed=0)
            if find(cells, regime="dynamic", W=W, seed=seed):
                warm = load_model(cells, "dynamic", W, seed)
                continue
            if warm is None:
                prev = [c for c in cells if c["regime"] == "dynamic" and c["seed"] == seed and c["W"] < W]
                if prev:
                    pw = max(c["W"] for c in prev)
                    warm = load_model(cells, "dynamic", pw, seed)
            warm = dynamic_cell(W, N, seed, w, cells, warm=warm)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fig_legend(fig, axes, ncol, top=0.86):
    """One shared legend across the top of the figure (house rule: legends outside the axes)."""
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=ncol, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, top))


def _legend_above(ax, ncol=3):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol, borderaxespad=0,
              frameon=False, fontsize=8)


def load_split_cells():
    out = []
    for f in sorted(RESULTS.glob("cells_split*.json")):
        out += json.loads(f.read_text())
    return out


def _dyn_rows(cells, seed=None):
    rows = [c for c in cells if c["regime"] == "dynamic" and (seed is None or c["seed"] == seed)]
    return sorted(rows, key=lambda c: (c["W"], c["seed"]))


def _agg(rows, key):
    """Per width: (units, median, min, max) across seeds."""
    out = {}
    for c in rows:
        out.setdefault(c["W"], []).append(c[key])
    Ws = sorted(out)
    med = np.array([np.median(out[W]) for W in Ws])
    lo = np.array([min(out[W]) for W in Ws])
    hi = np.array([max(out[W]) for W in Ws])
    return np.array(Ws), med, lo, hi


def plot_scaling(cells, plt):
    dyn = _dyn_rows(cells)
    ora = []
    for W in sorted({c["W"] for c in cells if c["regime"] == "oracle"}):
        N = best_oracle_N(cells, W)
        ora.append(find(cells, regime="oracle", W=W, N=N, seed=0))
    # the split arm: per width, the best dynamic cell over the N candidates tried (inherited + ablation)
    split = load_split_cells()
    arm = {}
    for c in [x for x in dyn if x["seed"] == 0] + [x for x in split if x["regime"] == "split" and x["seed"] == 0]:
        if c["W"] not in arm or c["rel_l2_v"] < arm[c["W"]]["rel_l2_v"]:
            arm[c["W"]] = c
    arm_W = sorted(W for W in arm if any(x["W"] == W for x in split))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    panels = [("rel_l2_v", "velocity rel $L_2$ (whole cube)"),
              ("rel_l2_p", "pressure rel $L_2$ (whole cube)"),
              (None, "fresh NS residual")]
    for ax, (key, title) in zip(axes, panels):
        if key is not None:
            if dyn:
                Ws, med, lo, hi = _agg(dyn, key)
                ax.errorbar(Ws, med, yerr=[med - lo, hi - med], color="C2", marker="o", ms=6, lw=2,
                            capsize=3, label="dynamic solve (Gauss-Newton lstsq, no oracle)")
            if arm_W:
                ax.plot(arm_W, [arm[W][key] for W in arm_W], color="C2", ls=":", marker="D", mfc="none",
                        ms=7, lw=1.4, label="dynamic solve, split N chosen by the solve (ablation)")
            if ora:
                ax.plot([c["W"] for c in ora], [c[key] for c in ora], color="k", ls="--", marker="o",
                        mfc="none", ms=6, lw=1.4, label="oracle fit of the exact fields (ceiling)")
        else:
            if dyn:
                Ws, med, lo, hi = _agg(dyn, "mom_rms")
                ax.plot(Ws, med, color="C3", marker="s", ms=5, lw=1.8, label="momentum residual RMS")
                Ws, med, lo, hi = _agg(dyn, "div_max")
                ax.plot(Ws, med, color="C0", marker="^", ms=5, lw=1.8, label=r"max $|\nabla\cdot u|$")
            if ora:
                ax.plot([c["W"] for c in ora], [c["mom_rms"] for c in ora], color="C3", ls="--",
                        marker="s", mfc="none", ms=5, lw=1.2, label="oracle fit: momentum RMS")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(LADDER)
        ax.set_xticklabels([str(W) for W in LADDER])
        ax.set_ylim(1e-15, 1e1)
        ax.set_xlabel("hidden units per field")
        ax.set_ylabel(title, fontsize=10)
        ax.axhline(1e-13, color="gray", lw=0.6, ls=":")
        ax.grid(alpha=0.3, which="both")
        _legend_above(ax, ncol=1)
    fig.suptitle("expF18: drifting Taylor-Green NS on the space-time ridge mesh -- error vs width",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "scaling.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_split(cells, plt):
    rows = [c for c in cells if c["regime"] == "oracle" and c["seed"] == 0]
    if not rows:
        return
    dyn_all = [c for c in cells if c["regime"] == "dynamic" and c["seed"] == 0] + \
        [c for c in load_split_cells() if c["regime"] == "split" and c["seed"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    Ws = sorted({c["W"] for c in rows})
    cmap = plt.get_cmap("viridis")
    for ax, key, title in [(axes[0], "rel_l2_v", "velocity"), (axes[1], "rel_l2_p", "pressure")]:
        for i, W in enumerate(Ws):
            rr = sorted([c for c in rows if c["W"] == W], key=lambda c: c["N"])
            ax.plot([c["N"] for c in rr], [c[key] for c in rr], marker="o", ms=4,
                    color=cmap(i / max(1, len(Ws) - 1)), label=f"W = {W}")
            b = min(rr, key=lambda c: c["rel_l2_v"])
            ax.plot([b["N"]], [b[key]], marker="*", ms=13, color=cmap(i / max(1, len(Ws) - 1)))
            dd = sorted([c for c in dyn_all if c["W"] == W], key=lambda c: c["N"])
            if dd:
                ax.plot([c["N"] for c in dd], [c[key] for c in dd], ls="none", marker="D", mfc="none", ms=7,
                        color=cmap(i / max(1, len(Ws) - 1)),
                        label="dynamic solve (hollow diamonds)" if i == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_GRID)
        ax.set_xticklabels([str(n) for n in N_GRID])
        ax.set_yscale("log")
        ax.set_ylim(1e-15, 1e1)
        ax.set_xlabel("offsets per direction N  (directions M = W / N)")
        ax.set_ylabel(f"{title} rel $L_2$: oracle fit (lines, star = chosen N)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    _fig_legend(fig, axes, ncol=len(Ws) + 1, top=0.9)
    fig.savefig(FIG / "oracle_split.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_time(cells, plt):
    dyn = _dyn_rows(cells, seed=0)
    if not dyn:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    cmap = plt.get_cmap("viridis")
    keys = [("rel_l2_v", "velocity rel $L_2$ per time slice"), ("linf_v", "velocity $L_\\infty$ per time slice"),
            ("rel_l2_p", "pressure rel $L_2$ per time slice")]
    for ax, (key, title) in zip(axes, keys):
        for i, c in enumerate(dyn):
            ts = [r["t"] for r in c["per_t"]]
            ax.plot(ts, [r[key] for r in c["per_t"]], color=cmap(i / max(1, len(dyn) - 1)), lw=1.8,
                    label=f"dynamic W = {c['W']}")
        top = dyn[-1]
        o = find(cells, regime="oracle", W=top["W"], N=top["N"], seed=0)
        if o is not None:
            ax.plot([r["t"] for r in o["per_t"]], [r[key] for r in o["per_t"]], color="k", ls="--", lw=1.2,
                    label=f"oracle fit W = {o['W']}")
        ax.set_yscale("log")
        ax.set_ylim(1e-15, 1e1)
        ax.set_xlabel("t")
        ax.set_ylabel(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    _fig_legend(fig, axes, ncol=6, top=0.9)
    fig.savefig(FIG / "error_vs_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_newton(cells, plt):
    dyn = _dyn_rows(cells, seed=0)
    if not dyn:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(dyn):
        col = cmap(i / max(1, len(dyn) - 1))
        it = [h["iter"] for h in c["hist"]]
        axes[0].plot(it, [h["res_after"] for h in c["hist"]], marker="o", ms=4, color=col, label=f"W = {c['W']}")
        axes[1].plot(it, [h["rel_l2_v"] for h in c["hist"]], marker="o", ms=4, color=col, label=f"W = {c['W']}")
    axes[0].set_ylabel("stacked collocation residual after each step", fontsize=10)
    axes[1].set_ylabel("velocity rel $L_2$ (fresh points) after each step", fontsize=10)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Gauss-Newton iteration")
        ax.grid(alpha=0.3, which="both")
    axes[1].set_ylim(1e-15, 1e1)
    _fig_legend(fig, axes, ncol=len(dyn), top=0.9)
    fig.savefig(FIG / "newton.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _frame_fields(model, S, tau, m):
    X = np.concatenate([S, np.full((len(S), 1), tau)], axis=1)
    F = pb.fields(X)
    d = model.evaluate(X, (r3.VAL, r3.DX, r3.DY, r3.DT, r3.DXX, r3.DYY))
    Fh = d[r3.VAL]
    u, v = Fh[:, 0:1], Fh[:, 1:2]
    gradp = np.stack([d[r3.DX][:, 2], d[r3.DY][:, 2]], axis=1)
    mom = (pb.DT_DTAU * d[r3.DT][:, :2] + u * d[r3.DX][:, :2] + v * d[r3.DY][:, :2] + gradp
           - pb.NU * (d[r3.DXX][:, :2] + d[r3.DYY][:, :2]))
    div = d[r3.DX][:, 0] + d[r3.DY][:, 1]
    spd_e = np.hypot(F[:, 0], F[:, 1]).reshape(m, m)
    spd = np.hypot(Fh[:, 0], Fh[:, 1]).reshape(m, m)
    vort = (d[r3.DX][:, 1] - d[r3.DY][:, 0]).reshape(m, m)
    err = np.hypot(Fh[:, 0] - F[:, 0], Fh[:, 1] - F[:, 1]).reshape(m, m)
    res = np.maximum(np.abs(mom).max(axis=1), np.abs(div)).reshape(m, m)
    return dict(spd_e=spd_e, spd=spd, vort=vort, err=err, res=res, uv=Fh[:, :2].reshape(m, m, 2))


def top_dynamic(cells, seed=0):
    dyn = _dyn_rows(cells, seed=seed)
    return dyn[-1] if dyn else None


def plot_residual_heatmaps(cells, plt):
    top = top_dynamic(cells)
    if top is None:
        return
    model = load_model(cells, "dynamic", top["W"], 0)
    m = 81
    g = np.linspace(-1, 1, m)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    S = np.stack([GX.ravel(), GY.ravel()], axis=1)
    taus = [-1.0, -1 / 3, 1 / 3, 1.0]
    frames = [_frame_fields(model, S, tau, m) for tau in taus]
    emax = max(f["err"].max() for f in frames)
    rmax = max(f["res"].max() for f in frames)
    fig, axes = plt.subplots(len(taus), 4, figsize=(15, 3.4 * len(taus)))
    for i, (tau, f) in enumerate(zip(taus, frames)):
        t = (tau + 1) * pb.T_END / 2
        panels = [(f["spd_e"], f"exact speed $|u|$, t = {t:.2f}", dict(cmap="viridis", vmin=0, vmax=1.6)),
                  (np.log10(f["err"] + 1e-17), r"$\log_{10}|u - u^*|$", dict(cmap="magma", vmin=np.log10(emax) - 4, vmax=np.log10(emax))),
                  (np.log10(np.abs(f["res"]) + 1e-17), r"$\log_{10}$ NS residual (max of momentum, div)", dict(cmap="magma", vmin=np.log10(rmax) - 4, vmax=np.log10(rmax))),
                  (f["vort"], r"vorticity $\omega$ (model)", dict(cmap="RdBu_r", vmin=-3.2, vmax=3.2))]
        for ax, (data, title, kw) in zip(axes[i], panels):
            im = ax.imshow(data.T, origin="lower", extent=[-1, 1, -1, 1], **kw)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"expF18 dynamic solve, W = {top['W']} units per field (N = {top['N']}): "
                 f"error and residual maps over time", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "residual_heatmaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_gif(cells, plt, n_frames=41, m=81):
    from matplotlib import animation
    top = top_dynamic(cells)
    if top is None:
        return
    model = load_model(cells, "dynamic", top["W"], 0)
    g = np.linspace(-1, 1, m)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    S = np.stack([GX.ravel(), GY.ravel()], axis=1)
    taus = np.linspace(-1, 1, n_frames)
    frames = [_frame_fields(model, S, tau, m) for tau in taus]
    emax = max(f["err"].max() for f in frames)
    rmax = max(f["res"].max() for f in frames)
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.4))
    f0 = frames[0]
    specs = [("spd_e", "exact speed $|u^*|$", dict(cmap="viridis", vmin=0, vmax=1.6)),
             ("spd", f"model speed $|u|$ (W = {top['W']})", dict(cmap="viridis", vmin=0, vmax=1.6)),
             ("vort", r"model vorticity $\omega$", dict(cmap="RdBu_r", vmin=-3.2, vmax=3.2)),
             ("err", r"$\log_{10}|u - u^*|$", dict(cmap="magma", vmin=np.log10(emax) - 4, vmax=np.log10(emax))),
             ("res", r"$\log_{10}$ NS residual", dict(cmap="magma", vmin=np.log10(rmax) - 4, vmax=np.log10(rmax)))]
    ims = []
    for ax, (key, title, kw) in zip(axes, specs):
        data = f0[key] if key not in ("err", "res") else np.log10(f0[key] + 1e-17)
        im = ax.imshow(data.T, origin="lower", extent=[-1, 1, -1, 1], **kw)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ims.append(im)
    step = 6
    q = axes[1].quiver(GX[::step, ::step], GY[::step, ::step], f0["uv"][::step, ::step, 0],
                       f0["uv"][::step, ::step, 1], color="w", scale=25, width=0.004)
    sup = fig.suptitle("t = 0.00", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(i):
        f = frames[i]
        for im, (key, _, _) in zip(ims, specs):
            im.set_data((f[key] if key not in ("err", "res") else np.log10(f[key] + 1e-17)).T)
        q.set_UVC(f["uv"][::step, ::step, 0], f["uv"][::step, ::step, 1])
        sup.set_text(f"drifting Taylor-Green NS, dynamic solve W = {top['W']}:  t = {(taus[i] + 1) * pb.T_END / 2:.2f}")
        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(frames))
    ani.save(FIG / "ns2d.gif", writer="pillow", fps=8)
    plt.close(fig)
    log(f"gif -> {FIG / 'ns2d.gif'}")


def write_status(cells):
    lines = ["# expF18 status (auto-written)", "",
             "| regime | W | N | M | units | seed | w_mult | rel L2 v | L_inf v | rel L2 p | mom RMS | max div | iters | wall s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["regime"], c["W"], c["N"], c["seed"])):
        lines.append(f"| {c['regime']} | {c['W']} | {c['N']} | {c['M']} | {c['units']} | {c['seed']} | "
                     f"{c.get('w_mult', '')} | {c['rel_l2_v']:.2e} | {c['linf_v']:.1e} | {c['rel_l2_p']:.2e} | "
                     f"{c['mom_rms']:.1e} | {c['div_max']:.1e} | {c.get('iters', 1)} | {c['wall']:.0f} |")
    (RESULTS / "status.md").write_text("\n".join(lines) + "\n")


def run_plots(cells):
    plt = _mpl()
    plot_scaling(cells, plt)
    plot_split(cells, plt)
    plot_error_vs_time(cells, plt)
    plot_newton(cells, plt)
    plot_residual_heatmaps(cells, plt)
    write_status(cells)
    log("plots written")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--widths", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--w-mult", type=float, default=None)
    args = ap.parse_args()
    for p in (RESULTS, FIG, COEF):
        p.mkdir(parents=True, exist_ok=True)
    pb.verify()
    if args.smoke:
        global CELLS, LOG
        CELLS = RESULTS / "cells_smoke.json"
        LOG = RESULTS / "run_smoke.log"
        cells = []
        run_oracle([256], [0], cells)
        run_solve([256], [0], cells, w_mult=1.0)
        run_plots(cells)
        make_gif(cells, _mpl(), n_frames=4, m=33)
        return
    cells = load_cells()
    ladder = args.widths or LADDER
    seeds = args.seeds or SEEDS
    if args.oracle:
        run_oracle(ladder, [0], cells)
    if args.tune:
        run_tune(cells)
    if args.solve:
        run_solve(ladder, seeds, cells, w_mult=args.w_mult)
    if args.plot:
        run_plots(cells)
    if args.gif:
        make_gif(cells, _mpl())


if __name__ == "__main__":
    main()
