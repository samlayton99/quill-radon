"""expF14 -- dysts chaotic ODEs: frozen QI geometry + one collocation solve.

Applies the expF01/expF02 recipe (freeze the geometry, solve the readout; Newton
with one min-norm lstsq per step for the nonlinearity) to five chaotic systems
from the `dysts` benchmark, in the order Sam asked for:

    Lorenz -> Rossler -> Thomas -> Halvorsen -> Lorenz96.

The whole trajectory on [0, T] is solved at once -- time is a coordinate, the
initial condition is a row block, there is no time-stepping, so nothing
accumulates over t (expF01's space-time framing, one coordinate instead of two).

Sweeps
  A  width scaling at a fixed horizon lambda_max * T = 3
  B  horizon scaling at fixed width: how many Lyapunov times fit under fp64
  C  ablations: geometry, lambda basin, rcond, poly block, warm start
Reference: mpmath odefun at 30 digits (see reference.py) -- a fp64 Runge-Kutta
reference is NOT accurate enough to certify these numbers.

Usage:
    .venv/bin/python experiments/expF14_dysts_chaos/run.py [--smoke] [--plot]
                                                           [--only A,B,C]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import systems  # noqa: E402
import core  # noqa: E402
import reference as ref  # noqa: E402

RESULTS_DIR = (REPO_ROOT / "results" / "checkpoint_F_applications"
               / "expF14_dysts_chaos")
FIG_DIR = RESULTS_DIR / "figures"
DATA_PATH = RESULTS_DIR / "data.json"

SMOKE = "--smoke" in sys.argv
PLOT_ONLY = "--plot" in sys.argv
_only = [a for a in sys.argv if a.startswith("--only")]
ONLY = set((_only[0].split("=", 1)[1] if _only else "ABCD").upper())
_sys = [a for a in sys.argv if a.startswith("--systems")]
SYSTEMS = ([n.strip() for n in _sys[0].split("=", 1)[1].split(",")] if _sys
           else list(systems.SYSTEM_ORDER))

# --- protocol constants -----------------------------------------------------
LYAP_TIMES = 3.0                       # sweep A horizon: lambda_max * T
N_EVAL = 6001                          # dense evaluation / reference grid
WARM_RTOL, WARM_ATOL = 1e-8, 1e-11     # cheap fp64 RK warm start (not an oracle)
N_GRID = [48, 96, 128, 192, 256, 384]
HORIZONS = [1.0, 2.0, 3.0, 4.0, 6.0]
HORIZON_WIDTHS = [256]
ABL_N = 256
# MacArthur is d=10: the dense system is (d*n_col) x (d*(W+4)), so cost grows
# like d^3 at fixed width. Cap its grids rather than its coverage.
N_GRID_BY_SYSTEM = {"MacArthur": [48, 96, 128, 192, 256],
                    # the squirmer's drive switches over dt ~ 0.008, so it is
                    # pushed further in width to locate its resolution wall
                    "InteriorSquirmer": [96, 192, 256, 384, 512, 768]}
ABL_N_BY_SYSTEM = {"MacArthur": 128}
INIT_GRID = ["warm", "cascade", "bcfit", "cold"]
WMULT_GRID = [0.1, 1.0, 10.0]
LAM_GRID = [0.15, 0.20, 0.25, 0.30, 0.40]
RCOND_GRID = [1e-12, 1e-13, 1e-15]
WARM_TOL_GRID = [1e-6, 1e-10]
RK_TOL_GRID = [1e-6, 1e-8, 1e-10, 1e-12, 1e-13]

if SMOKE:
    N_GRID = [96, 192]
    HORIZONS = [1.0, 3.0]
    HORIZON_WIDTHS = [256]
    LAM_GRID = [0.20, 0.25, 0.30]
    INIT_GRID = ["warm", "cascade", "cold"]
    WMULT_GRID = [1.0, 10.0]
    RCOND_GRID = [1e-12, 1e-13]
    WARM_TOL_GRID = [1e-6, 1e-10]
    N_EVAL = 2001


# ---------------------------------------------------------------------------
# one measured cell
# ---------------------------------------------------------------------------

def measure(S, T, ts, Yref, N, **kw):
    t0 = time.time()
    cell = core.solve_cell(S, T, N, warm_rtol=kw.pop("warm_rtol", WARM_RTOL),
                           warm_atol=kw.pop("warm_atol", WARM_ATOL), **kw)
    secs = time.time() - t0
    Yh = core.model_trajectory(cell, ts)
    Yw = core.warm_trajectory(cell, ts)
    rel, per, linf = core.errors(Yh, Yref)
    wrel, _, wlinf = core.errors(Yw, Yref)
    return dict(N=N, W=cell["W"], p=cell["p"], n_col=cell["n_col"],
                lam=cell["lam"], rcond=cell["rcond"], geom=cell["geom"],
                use_poly=cell["use_poly"], iters=cell["iters"],
                hist=[float(h) for h in cell["hist"]], rk_nfev=cell["rk_nfev"],
                rel_l2=rel, rel_l2_percomp=per, linf=linf,
                diverged=bool(cell.get("diverged", False)),
                warm_rel_l2=wrel, warm_linf=wlinf, init=cell["init"],
                w_mult=cell["w_mult"], fresh=cell["fresh"],
                cascade_W=cell.get("cascade_W"),
                seconds=round(secs, 2)), cell


def rk_baselines(S, T, ts, Yref):
    out = []
    for tol in RK_TOL_GRID:
        t0 = time.time()
        Z, nfev = ref.rk_trajectory(S, T, ts, tol, tol * 1e-2)
        rel, per, linf = core.errors(Z, Yref)
        out.append(dict(rtol=tol, rel_l2=rel, rel_l2_percomp=per, linf=linf,
                        nfev=nfev, seconds=round(time.time() - t0, 3)))
    return out


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

def sweep_width(S):
    T = S.horizon(LYAP_TIMES)
    ts, Yref = ref.reference(S, T, N_EVAL)
    cells, traj = [], {}
    grid = N_GRID_BY_SYSTEM.get(S.name, N_GRID)
    for N in grid:
        rec, cell = measure(S, T, ts, Yref, N)
        rec["interp_floor"] = core.interpolation_floor(S, T, N, ts, Yref)[0]
        cells.append(rec)
        traj[N] = core.model_trajectory(cell, ts)
        print(f"  [A] {S.name:10s} N={N:4d} W={rec['W']:4d} it={rec['iters']:2d} "
              f"warm={rec['warm_rel_l2']:.2e} -> rel={rec['rel_l2']:.2e} "
              f"Linf={rec['linf']:.2e} floor={rec['interp_floor']:.2e} "
              f"({rec['seconds']}s)", flush=True)
    # nested-width self-consistency (deployable, needs no reference)
    for i in range(1, len(cells)):
        a, b = grid[i - 1], grid[i]
        d = traj[b] - traj[a]
        cells[i]["selfconsistency"] = float(np.linalg.norm(d) / np.linalg.norm(traj[b]))
    return dict(T=T, periods=T / S.period, lyap=S.lyapunov, d=S.d,
                cells=cells, rk=rk_baselines(S, T, ts, Yref))


def sweep_horizon(S):
    out = []
    for lt in HORIZONS:
        T = S.horizon(lt)
        ts, Yref = ref.reference(S, T, N_EVAL)
        row = dict(lyap_times=lt, T=T, periods=T / S.period, cells=[])
        for N in HORIZON_WIDTHS:
            rec, _ = measure(S, T, ts, Yref, N)
            rec["interp_floor"] = core.interpolation_floor(S, T, N, ts, Yref)[0]
            row["cells"].append(rec)
            print(f"  [B] {S.name:10s} lamT={lt:4.1f} N={N:4d} it={rec['iters']:2d} "
                  f"warm={rec['warm_rel_l2']:.2e} -> rel={rec['rel_l2']:.2e} "
                  f"floor={rec['interp_floor']:.2e} ({rec['seconds']}s)", flush=True)
        row["rk"] = rk_baselines(S, T, ts, Yref)
        out.append(row)
    return out


def sweep_ablations(S):
    T = S.horizon(LYAP_TIMES)
    ts, Yref = ref.reference(S, T, N_EVAL)
    abl = {}

    def run(tag, **kw):
        rec, _ = measure(S, T, ts, Yref, ABL_N_BY_SYSTEM.get(S.name, ABL_N), **kw)
        print(f"  [C] {S.name:10s} {tag:28s} rel={rec['rel_l2']:.2e} "
              f"it={rec['iters']:2d} ({rec['seconds']}s)", flush=True)
        return rec

    abl["geometry"] = [run(f"geom={g}", geom=g)
                       for g in ("uniform", "random", "chebyshev")]
    abl["lambda"] = [dict(**run(f"lambda={l:.2f}", lam=l)) for l in LAM_GRID]
    abl["rcond"] = [dict(**run(f"rcond={r:.0e}", rcond=r)) for r in RCOND_GRID]
    abl["poly"] = [dict(**run(f"poly={u}", use_poly=u)) for u in (True, False)]
    abl["start"] = [dict(warm=w, **run(f"warm_start={w}", warm=w))
                    for w in (True, False)]
    abl["warm_tol"] = [dict(warm_rtol=t, **run(f"warm_rtol={t:.0e}",
                                               warm_rtol=t, warm_atol=t * 1e-3))
                       for t in WARM_TOL_GRID]
    return abl


def sweep_init(S):
    """Sweep D: the Newton-init ladder and the initial-condition row weight.

    expF03 part 1 found the initialiser to be the dominant nonlinear knob and
    the condition-block weight to be two-sided -- down-weighting *hurt* their
    logistic IVP, whose single inhomogeneous row is the only data. Every
    problem here is an IVP of exactly that shape, so both are tested.
    """
    T = S.horizon(LYAP_TIMES)
    ts, Yref = ref.reference(S, T, N_EVAL)
    out = {}

    def run(tag, **kw):
        rec, _ = measure(S, T, ts, Yref, ABL_N_BY_SYSTEM.get(S.name, ABL_N), **kw)
        print(f"  [D] {S.name:10s} {tag:24s} rel={rec['rel_l2']:.2e} "
              f"it={rec['iters']:2d} fresh(pde={rec['fresh']['pde']:.1e}, "
              f"ic={rec['fresh']['ic']:.1e}) ({rec['seconds']}s)", flush=True)
        return rec

    out["init"] = [run(f"init={i}", init=i) for i in INIT_GRID]
    out["w_mult"] = [run(f"w_mult={w:g}", w_mult=w) for w in WMULT_GRID]
    return out


def run_all():
    print("verifying re-implemented RHS and Jacobians against dysts...", flush=True)
    checks = systems.verify_all()
    data = {"_meta": {"lyap_times": LYAP_TIMES, "n_eval": N_EVAL,
                      "warm_rtol": WARM_RTOL, "warm_atol": WARM_ATOL,
                      "rcond": core.RCOND, "lam": core.LAM,
                      "colloc_per_neuron": core.COLLOC_PER_NEURON,
                      "verification": {k: list(v) for k, v in checks.items()}}}
    for name in SYSTEMS:
        S = systems.System(name)
        rec = dict(d=S.d, params=S.params, ic=S.ic.tolist(), period=S.period,
                   lyapunov=S.lyapunov)
        t0 = time.time()
        if "A" in ONLY:
            rec["width"] = sweep_width(S)
            rec["ref_check"] = ref.crosscheck(S, S.horizon(LYAP_TIMES))
            rec["ref_uncertainty"] = ref.reference_uncertainty(
                S, S.horizon(LYAP_TIMES))
        if "B" in ONLY:
            rec["horizon"] = sweep_horizon(S)
        if "C" in ONLY:
            rec["ablations"] = sweep_ablations(S)
        if "D" in ONLY:
            rec["init_sweep"] = sweep_init(S)
        rec["seconds"] = round(time.time() - t0, 1)
        data[name] = rec
        _save_data(data)          # incremental: one crashed system must not
                                  # discard every system that already finished
        print(f"{name} done in {rec['seconds']}s\n", flush=True)
    return _save_data(data)


def _save_data(data):
    """Merge into the on-disk record and rewrite it. Called after every system."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    merged = json.loads(DATA_PATH.read_text()) if DATA_PATH.exists() else {}
    for k, v in data.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k].update(v)
        else:
            merged[k] = v
    DATA_PATH.write_text(json.dumps(merged, indent=1, default=_jsonable))
    return merged


def _jsonable(o):
    """Several systems carry array-valued parameters (mode amplitudes, the
    consumer matrices); json cannot encode those on its own."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _int_xticks(ax, vals):
    """Plain integer tick labels on a log x-axis (log ticks read badly here)."""
    from matplotlib.ticker import NullFormatter
    v = sorted(set(int(x) for x in vals))
    ax.set_xticks(v)
    ax.set_xticklabels([str(x) for x in v], fontsize=7.5)
    ax.xaxis.set_minor_formatter(NullFormatter())


def _panel_axes(n=None, ncol=3, figsize=None):
    n = len(systems.SYSTEM_ORDER) if n is None else n
    nrow = int(np.ceil(n / ncol))
    if figsize is None:
        figsize = (15.5, 4.3 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def fig_width(data):
    fig, axes = _panel_axes()
    for i, name in enumerate(systems.SYSTEM_ORDER):
        w = data[name]["width"]
        cells = w["cells"]
        W = [c["W"] for c in cells]
        ax = axes[i]
        ax.loglog(W, [c["rel_l2"] for c in cells], "o-", color="C0",
                  label="QI solve, rel $L_2$")
        ax.loglog(W, [c["linf"] for c in cells], "s--", color="C1",
                  label="QI solve, $L_\\infty$")
        ax.loglog(W, [c["interp_floor"] for c in cells], "^:", color="0.45",
                  label="interpolation floor")
        ax.axhline(cells[-1]["warm_rel_l2"], color="C2", ls="-.", lw=1.2,
                   label=f"RK warm start (rtol {WARM_RTOL:.0e})")
        # Suppressed where the reference IS a Runge-Kutta run: comparing DOP853
        # to itself measures nothing.
        if name not in ref.RK_REFERENCE:
            best_rk = min(w["rk"], key=lambda r: r["rel_l2"])
            ax.axhline(best_rk["rel_l2"], color="C3", ls=":", lw=1.4,
                       label=f"DOP853 rtol {best_rk['rtol']:.0e}")
        ax.set_title(f"{name}  ($d={w['d']}$, $\\lambda_{{max}}T={LYAP_TIMES:.0f}$, "
                     f"{w['periods']:.1f} periods)", fontsize=10)
        ax.set_xlabel("total width $W$ (neurons)")
        if i % 3 == 0:
            ax.set_ylabel("error vs mpmath reference")
        ax.set_ylim(1e-16, 1e2)
        _int_xticks(ax, W)
        ax.grid(True, which="both", alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5,
               frameon=False, fontsize=10)
    fig.suptitle(f"expF14 -- width scaling of the frozen-QI collocation solve on "
                 f"{len(systems.SYSTEM_ORDER)} dysts chaotic systems", y=1.02,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "error_vs_width.png")


def fig_horizon(data):
    fig, axes = _panel_axes()
    for i, name in enumerate(systems.SYSTEM_ORDER):
        rows = data[name]["horizon"]
        ax = axes[i]
        lts = [r["lyap_times"] for r in rows]
        for k, N in enumerate(HORIZON_WIDTHS):
            ys = [r["cells"][k]["rel_l2"] for r in rows]
            fs = [r["cells"][k]["interp_floor"] for r in rows]
            W = rows[0]["cells"][k]["W"]
            ax.semilogy(lts, ys, "o-", color=f"C{k}", label=f"QI solve, $W$={W}")
            ax.semilogy(lts, fs, "^:", color=f"C{k}", alpha=0.55,
                        label=f"interp. floor, $W$={W}")
        lt = np.array(lts, dtype=float)
        ax.semilogy(lt, 2.2e-16 * np.exp(lt), "k--", lw=1.2,
                    label=r"$\varepsilon_{mach}\,e^{\lambda_{max}T}$")
        if name not in ref.RK_REFERENCE:
            best = [min(r["rk"], key=lambda q: q["rel_l2"])["rel_l2"] for r in rows]
            ax.semilogy(lts, best, "v-.", color="C3", lw=1.1, label="best DOP853")
        ax.set_title(f"{name}", fontsize=11)
        ax.set_xlabel(r"horizon $\lambda_{max} T$ (Lyapunov times)")
        if i % 3 == 0:
            ax.set_ylabel("rel $L_2$ vs mpmath reference")
        ax.set_ylim(1e-16, 1e2)
        ax.grid(True, which="both", alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=6,
               frameon=False, fontsize=9)
    fig.suptitle("expF14 -- how far in Lyapunov times a fixed neuron budget "
                 "reaches at fp64", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "error_vs_horizon.png")


def fig_representations(data):
    """Per system: attractor (exact vs solved), components, error-in-time."""
    n = len(systems.SYSTEM_ORDER)
    fig = plt.figure(figsize=(15.0, 3.0 * n))
    for i, name in enumerate(systems.SYSTEM_ORDER):
        S = systems.System(name)
        w = data[name]["width"]
        best = min(w["cells"], key=lambda c: c["rel_l2"])
        T = w["T"]
        ts, Yref = ref.reference(S, T, N_EVAL)
        cell = core.solve_cell(S, T, best["N"], warm_rtol=WARM_RTOL,
                               warm_atol=WARM_ATOL)
        Yh = core.model_trajectory(cell, ts)
        Yw = core.warm_trajectory(cell, ts)

        if S.d == 3:
            ax1 = fig.add_subplot(n, 3, 3 * i + 1, projection="3d")
            ax1.plot(Yref[:, 0], Yref[:, 1], Yref[:, 2], color="k", lw=1.5,
                     label="reference (mpmath)")
            ax1.plot(Yh[:, 0], Yh[:, 1], Yh[:, 2], color="C1", lw=0.9, ls="--",
                     label="QI solve")
            ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$"); ax1.set_zlabel("$z$")
        else:
            ax1 = fig.add_subplot(n, 3, 3 * i + 1)
            for c in range(S.d):
                ax1.plot(ts, Yref[:, c], color="k", lw=1.4)
                ax1.plot(ts, Yh[:, c], color=f"C{c % 10}", lw=0.9, ls="--")
            ax1.set_xlabel("$t$"); ax1.set_ylabel("state")
        ax1.set_title(f"{name}: phase portrait", fontsize=10, pad=26)

        ax2 = fig.add_subplot(n, 3, 3 * i + 2)
        for c in range(S.d):
            ax2.plot(ts, Yref[:, c], color=f"C{c % 10}", lw=1.4,
                     label=(f"$u_{c}$ ref" if S.d <= 5 else None))
            ax2.plot(ts, Yh[:, c], color="k", lw=0.7, ls="--",
                     label="QI solve" if c == 0 else None)
        ax2.set_xlabel("$t$"); ax2.set_ylabel("state")
        ax2.set_title(f"{name}: components over $[0,T]$, $T={T:.2f}$",
                      fontsize=10, pad=26)
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=S.d + 1,
                   frameon=False, fontsize=7, borderaxespad=0)

        ax3 = fig.add_subplot(n, 3, 3 * i + 3)
        e_h = np.linalg.norm(Yh - Yref, axis=1) + 1e-18
        e_w = np.linalg.norm(Yw - Yref, axis=1) + 1e-18
        ax3.semilogy(ts, e_w, color="C2", lw=1.0, label="RK warm start")
        ax3.semilogy(ts, e_h, color="C1", lw=1.0, label="QI solve")
        guide = e_h[e_h > 0].min() * np.exp(S.lyapunov * ts)
        ax3.semilogy(ts, guide, "k--", lw=1.0,
                     label=r"$\propto e^{\lambda_{max} t}$")
        ax3.set_xlabel("$t$"); ax3.set_ylabel(r"$\|\hat u - u^*\|_2$")
        ax3.set_ylim(1e-16, max(1e-2, e_w.max() * 5))
        ax3.set_title(f"{name}: error in time ($W={best['W']}$)",
                      fontsize=10, pad=26)
        ax3.grid(True, which="both", alpha=0.25)
        for ax in (ax1, ax3):
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
                      frameon=False, fontsize=7, borderaxespad=0)
    fig.tight_layout()
    _save(fig, "representations.png")


def fig_ablations(data):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    ax = axes[0, 0]
    labels = ["uniform", "random", "chebyshev"]
    xs = np.arange(len(systems.SYSTEM_ORDER))
    for k, g in enumerate(labels):
        ys = [next(c["rel_l2"] for c in data[n]["ablations"]["geometry"]
                   if c["geom"] == g) for n in systems.SYSTEM_ORDER]
        ax.bar(xs + (k - 1) * 0.27, ys, 0.26, color=f"C{k}", label=g)
    ax.set_yscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels(systems.SYSTEM_ORDER, rotation=20, fontsize=8)
    ax.set_ylabel("rel $L_2$")
    ax.set_title("centre placement", fontsize=11, pad=30)

    for j, (key, xkey, xlabel, logx) in enumerate([
            ("lambda", "lam", r"$\lambda = \gamma h$", False),
            ("rcond", "rcond", "lstsq rcond", True),
            ("warm_tol", "warm_rtol", "warm-start RK rtol", True)]):
        a = axes[(j + 1) // 3, (j + 1) % 3]
        for k, n in enumerate(systems.SYSTEM_ORDER):
            rows = data[n]["ablations"][key]
            xv = [r[xkey] for r in rows]
            yv = [r["rel_l2"] for r in rows]
            (a.loglog if logx else a.semilogy)(xv, yv, "o-", color=f"C{k}", label=n)
        a.set_xlabel(xlabel); a.set_ylabel("rel $L_2$")
        a.set_title({"lambda": "bandwidth basin", "rcond": "rcond",
                     "warm_tol": "warm-start tolerance"}[key], fontsize=11, pad=30)
        a.grid(True, which="both", alpha=0.25)

    a = axes[1, 1]
    ys_on = [data[n]["ablations"]["poly"][0]["rel_l2"] for n in systems.SYSTEM_ORDER]
    ys_off = [data[n]["ablations"]["poly"][1]["rel_l2"] for n in systems.SYSTEM_ORDER]
    a.bar(xs - 0.16, ys_on, 0.3, color="C0", label="with cubic block")
    a.bar(xs + 0.16, ys_off, 0.3, color="C1", label="tanh only")
    a.set_yscale("log"); a.set_xticks(xs)
    a.set_xticklabels(systems.SYSTEM_ORDER, rotation=20, fontsize=8)
    a.set_ylabel("rel $L_2$")
    a.set_title("polynomial block", fontsize=11, pad=30)

    a = axes[1, 2]
    ys_w = [data[n]["ablations"]["start"][0]["rel_l2"] for n in systems.SYSTEM_ORDER]
    ys_c = [data[n]["ablations"]["start"][1]["rel_l2"] for n in systems.SYSTEM_ORDER]
    a.bar(xs - 0.16, ys_w, 0.3, color="C0", label="warm (cheap RK)")
    a.bar(xs + 0.16, ys_c, 0.3, color="C3", label="cold ($a_0 = 0$)")
    a.set_yscale("log"); a.set_xticks(xs)
    a.set_xticklabels(systems.SYSTEM_ORDER, rotation=20, fontsize=8)
    a.set_ylabel("rel $L_2$")
    a.set_title("Newton start", fontsize=11, pad=30)

    for a in axes.ravel():
        a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
                 frameon=False, fontsize=7.5, borderaxespad=0)
    fig.suptitle(f"expF14 -- ablations at $N$={ABL_N}, "
                 rf"$\lambda_{{max}}T={LYAP_TIMES:.0f}$", y=1.02,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "ablations.png")


def fig_newton(data):
    fig, axes = _panel_axes()
    for i, name in enumerate(systems.SYSTEM_ORDER):
        ax = axes[i]
        for c in data[name]["width"]["cells"]:
            ax.semilogy(range(1, len(c["hist"]) + 1), c["hist"], "o-",
                        label=f"$W$={c['W']}", lw=1.0, ms=3)
        ax.set_xlabel("Gauss-Newton step")
        if i % 3 == 0:
            ax.set_ylabel(r"$\max_j |r_j|$ (collocation residual)")
        ax.set_title(name, fontsize=11, pad=32)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
                  frameon=False, fontsize=6.5, borderaxespad=0)
    fig.suptitle("expF14 -- Gauss-Newton convergence (one lstsq per step)",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "newton_convergence.png")


def fig_selfconsistency(data):
    fig, axes = _panel_axes()
    for i, name in enumerate(systems.SYSTEM_ORDER):
        cells = data[name]["width"]["cells"]
        sc = [(c["W"], c["selfconsistency"], c["rel_l2"])
              for c in cells if "selfconsistency" in c]
        ax = axes[i]
        ax.loglog([s[0] for s in sc], [s[1] for s in sc], "o-", color="C4",
                  label=r"$\|u_{W_2}-u_{W_1}\|/\|u_{W_2}\|$ (no reference)")
        ax.loglog([s[0] for s in sc], [s[2] for s in sc], "s--", color="C0",
                  label="true rel $L_2$")
        ax.set_xlabel("total width $W$")
        if i % 3 == 0:
            ax.set_ylabel("error estimate")
        ax.set_ylim(1e-16, 1e2)
        _int_xticks(ax, [s0[0] for s0 in sc])
        ax.set_title(name, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2,
               frameon=False, fontsize=10)
    fig.suptitle("expF14 -- nested-width self-consistency tracks the true error",
                 y=1.05, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "self_consistency.png")


def _all_cells(data):
    """Every measured cell in the file, for the signal-vs-error scatter."""
    out = []
    for name in systems.SYSTEM_ORDER:
        rec = data.get(name, {})
        buckets = []
        if "width" in rec:
            buckets.append(rec["width"]["cells"])
        for row in rec.get("horizon", []):
            buckets.append(row["cells"])
        for v in rec.get("ablations", {}).values():
            buckets.append(v)
        for v in rec.get("init_sweep", {}).values():
            buckets.append(v)
        for b in buckets:
            for c in b:
                if isinstance(c, dict) and c.get("fresh"):
                    out.append((name, c))
    return out


def fig_init_and_signal(data):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))
    xs = np.arange(len(systems.SYSTEM_ORDER))

    ax = axes[0]
    inits = [c["init"] for c in data[systems.SYSTEM_ORDER[0]]["init_sweep"]["init"]]
    wbar = 0.8 / len(inits)
    for k, ini in enumerate(inits):
        ys = [next(c["rel_l2"] for c in data[n]["init_sweep"]["init"]
                   if c["init"] == ini) for n in systems.SYSTEM_ORDER]
        ax.bar(xs + (k - (len(inits) - 1) / 2) * wbar, ys, wbar * 0.92,
               color=f"C{k}", label=ini)
    ax.set_yscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels(systems.SYSTEM_ORDER, rotation=20, fontsize=8)
    ax.set_ylabel("rel $L_2$")
    ax.set_title("Newton initialisation ladder", fontsize=11, pad=30)

    ax = axes[1]
    for k, n in enumerate(systems.SYSTEM_ORDER):
        rows = data[n]["init_sweep"]["w_mult"]
        ax.loglog([r["w_mult"] for r in rows], [r["rel_l2"] for r in rows],
                  "o-", color=f"C{k}", label=n)
    ax.set_xlabel("initial-condition row weight multiplier")
    ax.set_ylabel("rel $L_2$")
    ax.set_title("condition-row weight", fontsize=11, pad=30)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[2]
    cells = _all_cells(data)
    ax.loglog([c["fresh"]["pde"] for _, c in cells],
              [c["rel_l2"] for _, c in cells], "o", ms=3.2, alpha=0.5,
              color="C3", label="PDE rows only")
    ax.loglog([c["fresh"]["stacked"] for _, c in cells],
              [c["rel_l2"] for _, c in cells], "s", ms=3.2, alpha=0.5,
              color="C0", label="stacked (PDE + IC)")
    lo = 1e-16
    ax.plot([lo, 1e4], [lo, 1e4], "k:", lw=1.0, label="$y=x$")
    ax.set_xlabel("fresh-point residual (no reference used)")
    ax.set_ylabel("true rel $L_2$")
    ax.set_title("is the residual a usable error signal?", fontsize=11, pad=30)
    ax.grid(True, which="both", alpha=0.25)

    for a in axes:
        a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
                 frameon=False, fontsize=8, borderaxespad=0)
    fig.suptitle(f"expF14 -- Newton start, condition weight, and the "
                 f"reference-free error signal ($N$={ABL_N})", y=1.06,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "init_and_signal.png")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def make_figures(data):
    if "A" in ONLY:
        fig_width(data)
        fig_newton(data)
        fig_selfconsistency(data)
        fig_representations(data)
    if "B" in ONLY:
        fig_horizon(data)
    if "C" in ONLY:
        fig_ablations(data)
    if "D" in ONLY:
        fig_init_and_signal(data)


def main():
    t0 = time.time()
    if PLOT_ONLY and DATA_PATH.exists():
        data = json.loads(DATA_PATH.read_text())
    else:
        data = run_all()
    make_figures(data)
    print(f"\nall outputs in {RESULTS_DIR}   total {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
