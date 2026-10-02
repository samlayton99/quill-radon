"""Experiment expF03 -- ablations, deployable model selection, baselines.

Parts (each a separate invocation; each writes its own JSON and figures):
  ablate     oracle ablations, one-axis-at-a-time around the default config
  rescue     targeted levers on the hard problems (built after ablate is inspected)
  select     deployable model selection (built after part 1 check-in)
  baselines  other-method comparison (built after part 2 check-in)

Usage:
    uv run --extra dev python experiments/expF03_ablation_and_baselines/run.py ablate [--smoke] [--round2]
    uv run --extra dev python experiments/expF03_ablation_and_baselines/run.py plot-ablate
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
from common import (DEFAULT_CFG, PART1_DIR, PART2_DIR, PART3_DIR, PART4_DIR,
                    SolveConfig, assemble_linear, load_problem_suite, solve_linear,
                    solve_nonlinear, stack, svd_factor, svd_solve, _model_from,
                    evaluate, fresh_residual_linear, coeff_norms)

SMOKE = "--smoke" in sys.argv

ABLATE_PATH = PART1_DIR / "ablate.json"

# ---------------------------------------------------------------------------
# part: ablate
# ---------------------------------------------------------------------------

# probe widths (round 1): 1D gets floor + drift region, 2D one representative width
PROBE_SIZES = {
    "f01": {"ode1d": [32, 1024], "pde2d_steady": [1024], "pde2d_time": [1024]},
    "f02": {"ode1d": [32], "pde2d_steady": [576], "pde2d_time": [576]},
}

AXES_LINEAR = {
    "lam": [0.10, 0.15, 0.20, 0.30, 0.35, 0.45],
    "rcond": [1e-11, 1e-12, 1e-14, 1e-15, 1e-16, 0.0],
    "w_mult": [0.03, 0.1, 0.3, 3.0, 10.0, 30.0],
    "halo": ["0", "20", "70", "0.8N"],                     # 1D only
    "collar": [1.0, 1.6, 2.0],                             # 2D only (per-category default marked)
    "oversample": [2, 8, 16],
    "poly_degree": [None, 1, 5],
    "equilibrate": [True],
}

AXES_NONLINEAR = {
    "lam": [0.15, 0.20, 0.30, 0.35],
    "rcond": [1e-12, 1e-14, 1e-15],
    "w_mult": [0.1, 0.3, 3.0, 10.0],
    "halo": ["0", "70"],
    "collar": [1.0, 2.0],
    "oversample": [2, 8],
    "poly_degree": [None, 5],
    "equilibrate": [True],
    "init": ["zero", "bcfit", "linpart", "cascade", "picard", "homotopy"],
    "lm": [True],
    "rcond_tighten": [True],
}


def _cfg_with(cfg, axis, setting, category):
    if axis == "collar":
        if category == "pde2d_steady":
            return replace(cfg, collar_steady=setting)
        return replace(cfg, collar_time=setting)
    if axis == "oversample":
        return replace(cfg, oversample_1d=setting, oversample_2d=setting)
    return replace(cfg, **{axis: setting})


def _axis_applies(axis, category):
    if axis == "halo":
        return category == "ode1d"
    if axis == "collar":
        return category != "ode1d"
    return True


def _sanitize(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _record(setup, key, axis, setting, size, seed, res):
    sig = dict(res["signals"])
    sig.pop("newton_hist", None)  # keep the JSON small; histories go in rescue
    rec = dict(setup=setup, key=key, axis=axis,
               setting=repr(setting), size=size, W=res["W"], seed=seed,
               rel_l2=res["rel_l2"], linf=res["linf"], **sig)
    rec = {k: _sanitize(v) for k, v in rec.items()}
    if rec["rel_l2"] is None:
        rec["diverged"] = True
    return rec


def _solve_linear_from_fac(prob, asm, fac, y, cfg):
    a, info = svd_solve(fac, y, cfg.rcond)
    model = _model_from(asm, a)
    rel, linf, _ = evaluate(prob, model)
    sig = dict(info, **coeff_norms(model),
               fresh_res_phys=fresh_residual_linear(prob, model, cfg))
    return dict(model=model, rel_l2=rel, linf=linf, signals=sig, W=model["W"])


def run_ablate(only_axes=None):
    """only_axes: set of axis names to run (merged into the existing ablate.json);
    None runs everything and overwrites."""
    records = []
    t_start = time.time()
    for setup in ["f01", "f02"]:
        suite = load_problem_suite(setup)
        axes = AXES_LINEAR if setup == "f01" else AXES_NONLINEAR
        if only_axes is not None:
            axes = {a: v for a, v in axes.items() if a in only_axes}
        probs = suite.PROBLEMS
        if SMOKE:
            probs = {k: probs[k] for k in ["ode1d_o2", "pde2d_steady_o2"]}
        for key, prob in probs.items():
            cat = prob["category"]
            sizes = PROBE_SIZES[setup][cat]
            if SMOKE:
                sizes = sizes[:1]
                axes = {a: v[:1] for a, v in axes.items()}
            for size in sizes:
                seed = 42
                cfg0 = replace(DEFAULT_CFG, seed=seed)
                t0 = time.time()
                base = None
                if setup == "f01":
                    if only_axes is None or {"rcond", "w_mult"} & set(axes):
                        asm = assemble_linear(prob, size, cfg0)
                        A, y, s = stack(asm, cfg0)
                        fac = svd_factor(A)
                        # default + whole rcond axis from ONE factorization
                        res = _solve_linear_from_fac(prob, asm, fac, y, cfg0)
                        records.append(_record(setup, key, "default", "default", size, seed, res))
                        base = res["rel_l2"]
                        for rc in axes.get("rcond", []):
                            res = _solve_linear_from_fac(prob, asm, fac, y, replace(cfg0, rcond=rc))
                            records.append(_record(setup, key, "rcond", rc, size, seed, res))
                        del fac
                        # weighting axis reuses the assembly (re-stack + solve)
                        for m in axes.get("w_mult", []):
                            cfg = replace(cfg0, w_mult=m)
                            res = solve_linear(prob, size, cfg, asm=asm)
                            records.append(_record(setup, key, "w_mult", m, size, seed, res))
                        del asm
                    # remaining axes need re-assembly
                    for axis in ["lam", "halo", "collar", "oversample", "poly_degree",
                                 "equilibrate"]:
                        if axis not in axes or not _axis_applies(axis, cat):
                            continue
                        for setting in axes[axis]:
                            cfg = _cfg_with(cfg0, axis, setting, cat)
                            res = solve_linear(prob, size, cfg)
                            records.append(_record(setup, key, axis, setting, size, seed, res))
                else:
                    if only_axes is None:
                        res = solve_nonlinear(prob, size, cfg0)
                        records.append(_record(setup, key, "default", "default", size, seed, res))
                        base = res["rel_l2"]
                    for axis, settings in axes.items():
                        if not _axis_applies(axis, cat):
                            continue
                        for setting in settings:
                            cfg = _cfg_with(cfg0, axis, setting, cat)
                            res = solve_nonlinear(prob, size, cfg)
                            if res is None:
                                records.append(dict(setup=setup, key=key, axis=axis,
                                                    setting=repr(setting), size=size,
                                                    W=None, seed=seed, rel_l2=None,
                                                    linf=None, not_applicable=True))
                                continue
                            records.append(_record(setup, key, axis, setting, size, seed, res))
                n_rec = sum(1 for r in records if r["key"] == key and r["size"] == size)
                msg = f"{setup} {key} size={size}: {n_rec} cells"
                if base is not None:
                    msg += f", default relL2={base:.2e}"
                print(msg + f", {time.time() - t0:.1f}s", flush=True)
    if only_axes is not None and ABLATE_PATH.exists():
        old = json.loads(ABLATE_PATH.read_text())
        kept = [r for r in old if r["axis"] not in only_axes]
        records = kept + records
    PART1_DIR.mkdir(parents=True, exist_ok=True)
    ABLATE_PATH.write_text(json.dumps(records, indent=1))
    print(f"\nablate: {len(records)} records in {time.time() - t_start:.0f}s -> {ABLATE_PATH}",
          flush=True)
    return records


# ---------------------------------------------------------------------------
# ablate figures
# ---------------------------------------------------------------------------

PROBLEM_ORDER = [f"{c}_o{o}" for c in ["ode1d", "pde2d_steady", "pde2d_time"] for o in [1, 2, 3]]
CAT_COLOR = {"ode1d": "C0", "pde2d_steady": "C1", "pde2d_time": "C2"}


def _load_ablate():
    return json.loads(ABLATE_PATH.read_text())


def _defaults_map(records):
    """(setup, key, size) -> default rel_l2."""
    return {(r["setup"], r["key"], r["size"]): r["rel_l2"]
            for r in records if r["axis"] == "default"}


def plot_ablate_summary(records):
    defaults = _defaults_map(records)
    axes_names = sorted({r["axis"] for r in records if r["axis"] != "default"})
    rows = []
    labels = []
    for setup in ["f01", "f02"]:
        for key in PROBLEM_ORDER:
            sizes = sorted({r["size"] for r in records
                            if r["setup"] == setup and r["key"] == key})
            for size in sizes:
                base = defaults.get((setup, key, size))
                if base is None:
                    continue
                row = []
                for ax in axes_names:
                    cells = [r["rel_l2"] for r in records
                             if (r["setup"], r["key"], r["size"], r["axis"]) == (setup, key, size, ax)
                             and r["rel_l2"] is not None]
                    row.append(np.log10(min(cells) / base) if cells else np.nan)
                rows.append(row)
                labels.append(f"{setup}:{key} W~{size}")
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(1.1 * len(axes_names) + 4, 0.32 * len(labels) + 2))
    vmax = max(1.0, np.nanmax(np.abs(M)))
    pc = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(axes_names)), axes_names, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(pc, ax=ax, label="$\\log_{10}$(best on axis / default)  "
                                  "(blue = knob helps; red = every setting hurts)")
    ax.set_title("Ablation sensitivity: best achievable improvement per knob (oracle)")
    fig.tight_layout()
    out = PART1_DIR / "ablate_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


# knobs with a shape worth seeing (a sweep with structure). The single-value / null
# knobs (equilibrate, lm, rcond_tighten) are captured by the summary-heatmap columns
# and add nothing as a one-point line, so they are not drawn here.
SHAPE_AXES = ["lam", "rcond", "w_mult", "poly_degree",
              "collar", "halo", "oversample", "init"]
AXIS_TITLE = {
    "lam": r"$\lambda$ bandwidth", "rcond": "rcond (SVD cutoff)",
    "w_mult": "condition-block weight", "poly_degree": "poly degree",
    "collar": "collar (2D only)", "halo": "halo (1D only)",
    "oversample": "collocation $\\times$ width", "init": "Newton init (nonlin only)",
}


def _setting_key(s):
    t = str(s).strip("'\"")
    try:
        return (0, float(t), "")
    except (ValueError, TypeError):
        if t.endswith("N"):                       # halo "0.8N": a width-scaled count,
            try:                                  # order it after the fixed counts
                return (0, 1e4 * float(t[:-1]), "")
            except ValueError:
                pass
        return (1, 0.0, t)


def plot_ablate_axes(records):
    """One condensed figure. Per knob: the typical effect (median over all
    problem/width cells) and the full range across problems (shaded band), for the
    linear and nonlinear setups, normalized to the default so every knob shares the
    '<1 = better' scale. Each panel autoscales so the shape -- basin (lam, collar),
    cliff (halo), two-sided (rcond, w_mult), flat (oversample) -- stays legible;
    per-problem magnitudes live in the summary heatmap."""
    defaults = _defaults_map(records)
    ncol, nrow = 4, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow))
    setup_style = [("f01", "linear", "C0"), ("f02", "nonlinear", "C3")]
    for ax, axis in zip(axes.ravel(), SHAPE_AXES):
        recs = [r for r in records if r["axis"] == axis and r["rel_l2"] is not None]
        settings = sorted({r["setting"] for r in recs}, key=_setting_key)
        xpos = {s: i for i, s in enumerate(settings)}
        for setup, _lab, color in setup_style:
            xs, med, lo, hi = [], [], [], []
            for s in settings:
                vals = [r["rel_l2"] / defaults[(r["setup"], r["key"], r["size"])]
                        for r in recs if r["setup"] == setup and r["setting"] == s
                        and defaults.get((r["setup"], r["key"], r["size"]))]
                if vals:
                    xs.append(xpos[s]); med.append(np.median(vals))
                    lo.append(min(vals)); hi.append(max(vals))
            if not xs:
                continue
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(xs, med, "o-", color=color, lw=1.7, ms=4.5)
        ax.axhline(1.0, color="k", lw=0.9, ls=":")
        ax.set_yscale("log")
        ax.set_xticks(range(len(settings)),
                      [s.strip("'") for s in settings], rotation=30, ha="right", fontsize=7.5)
        ax.set_title(AXIS_TITLE.get(axis, axis), fontsize=9.5)
        ax.grid(True, which="both", alpha=0.2)
    for ax in axes.ravel()[len(SHAPE_AXES):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("rel $L_2$ / default\n($<$1 = better)", fontsize=8.5)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=1.7, label=lab)
               for _s, lab, c in setup_style]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.005),
               frameon=False, fontsize=10,
               title="line = median over problems; band = full range",
               title_fontsize=9)
    fig.suptitle("Ablation knob shapes (each panel normalized to its default setting)",
                 y=1.05, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PART1_DIR / "ablate_shapes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# part: refine -- round-2 combos of the round-1 winners
# ---------------------------------------------------------------------------

REFINE_PATH = PART1_DIR / "refine.json"

# (setup, key, size, label, cfg overrides) -- built from the round-1 signal:
# 1D drift responds to small w_mult (+ rcond/oversample/halo); 2D steady to
# collar 1.6 + poly 5 (+ rcond); nonlinear to cascade/bcfit init + lm + rcond.
REFINE_SPECS = [
    # --- linear 1D drift region ---
    ("f01", "ode1d_o2", 1024, "wm0.1", dict(w_mult=0.1)),
    ("f01", "ode1d_o2", 1024, "wm0.1+rcond0", dict(w_mult=0.1, rcond=0.0)),
    ("f01", "ode1d_o2", 1024, "wm0.03+rcond0", dict(w_mult=0.03, rcond=0.0)),
    ("f01", "ode1d_o2", 1024, "wm0.1+os8", dict(w_mult=0.1, oversample_1d=8)),
    ("f01", "ode1d_o2", 1024, "wm0.1+rcond0+os8", dict(w_mult=0.1, rcond=0.0, oversample_1d=8)),
    ("f01", "ode1d_o3", 1024, "wm0.03", dict(w_mult=0.03)),
    ("f01", "ode1d_o3", 1024, "wm0.03+halo20", dict(w_mult=0.03, halo="20")),
    ("f01", "ode1d_o3", 1024, "wm0.03+rcond1e-15", dict(w_mult=0.03, rcond=1e-15)),
    ("f01", "ode1d_o3", 1024, "wm0.03+halo20+os16", dict(w_mult=0.03, halo="20", oversample_1d=16)),
    ("f01", "ode1d_o3", 1024, "wm0.03+halo20+rc1e-15+os16",
     dict(w_mult=0.03, halo="20", rcond=1e-15, oversample_1d=16)),
    # --- linear 2D steady combos ---
    ("f01", "pde2d_steady_o1", 1024, "collar1.6+rc1e-15", dict(collar_steady=1.6, rcond=1e-15)),
    ("f01", "pde2d_steady_o1", 1024, "collar1.6+rc1e-15+poly5",
     dict(collar_steady=1.6, rcond=1e-15, poly_degree=5)),
    ("f01", "pde2d_steady_o2", 1024, "collar1.6+poly5", dict(collar_steady=1.6, poly_degree=5)),
    ("f01", "pde2d_steady_o2", 1024, "collar1.6+poly5+rc1e-15",
     dict(collar_steady=1.6, poly_degree=5, rcond=1e-15)),
    ("f01", "pde2d_steady_o3", 1024, "os16+poly5", dict(oversample_2d=16, poly_degree=5)),
    ("f01", "pde2d_steady_o3", 1024, "collar1.6+os16+poly5",
     dict(collar_steady=1.6, oversample_2d=16, poly_degree=5)),
    # --- nonlinear combos at the probe width ---
    ("f02", "pde2d_steady_o1", 576, "cascade+lm", dict(init="cascade", lm=True)),
    ("f02", "pde2d_steady_o1", 576, "cascade+rc1e-15", dict(init="cascade", rcond=1e-15)),
    ("f02", "pde2d_steady_o1", 576, "cascade+lm+rc1e-15",
     dict(init="cascade", lm=True, rcond=1e-15)),
    ("f02", "pde2d_steady_o2", 576, "collar2+poly5", dict(collar_steady=2.0, poly_degree=5)),
    ("f02", "pde2d_steady_o2", 576, "collar2+poly5+rc1e-15",
     dict(collar_steady=2.0, poly_degree=5, rcond=1e-15)),
    ("f02", "pde2d_steady_o3", 576, "cascade+lm", dict(init="cascade", lm=True)),
    ("f02", "pde2d_steady_o3", 576, "cascade+rc1e-15", dict(init="cascade", rcond=1e-15)),
    ("f02", "pde2d_steady_o3", 576, "cascade+lm+rc1e-15",
     dict(init="cascade", lm=True, rcond=1e-15)),
    ("f02", "pde2d_time_o1", 576, "lm+cascade", dict(lm=True, init="cascade")),
    ("f02", "pde2d_time_o1", 576, "lm+rc1e-15", dict(lm=True, rcond=1e-15)),
    ("f02", "pde2d_time_o1", 576, "lm+max_newton60", dict(lm=True, max_newton=60)),
    ("f02", "pde2d_time_o2", 576, "bcfit+rc1e-15", dict(init="bcfit", rcond=1e-15)),
    ("f02", "pde2d_time_o2", 576, "bcfit+lm", dict(init="bcfit", lm=True)),
    ("f02", "pde2d_time_o2", 576, "bcfit+lm+rc1e-15", dict(init="bcfit", lm=True, rcond=1e-15)),
    ("f02", "pde2d_time_o2", 576, "cascade+rc1e-15", dict(init="cascade", rcond=1e-15)),
    ("f02", "pde2d_time_o3", 576, "cascade+rc1e-15", dict(init="cascade", rcond=1e-15)),
    ("f02", "pde2d_time_o3", 576, "cascade+lm", dict(init="cascade", lm=True)),
    ("f02", "pde2d_time_o3", 576, "cascade+lm+rc1e-15", dict(init="cascade", lm=True, rcond=1e-15)),
]

# winners re-run for stability: other widths + fresh collocation seeds
REFINE_CONFIRM_SEEDS = [7, 19]
REFINE_CONFIRM_SIZES = {"f02": [1024, 2304]}

# part-1 winning config per problem (the width is where its headline number lives)
WINNERS = {
    ("f01", "pde2d_steady_o1"): (1024, dict(collar_steady=1.6, rcond=1e-15, poly_degree=5)),
    ("f01", "pde2d_steady_o2"): (1024, dict(poly_degree=5)),
    ("f01", "pde2d_steady_o3"): (1024, dict(oversample_2d=16)),
    ("f02", "pde2d_steady_o1"): (1024, dict(init="cascade", rcond=1e-15)),
    ("f02", "pde2d_steady_o2"): (576, dict(collar_steady=2.0, poly_degree=5, rcond=1e-15)),
    ("f02", "pde2d_steady_o3"): (2304, dict(init="cascade", rcond=1e-15, w_mult=0.1)),
    ("f02", "pde2d_time_o1"): (1024, dict(lm=True, max_newton=60)),
    ("f02", "pde2d_time_o2"): (2304, dict(init="cascade", rcond=1e-15)),
    ("f02", "pde2d_time_o3"): (2304, dict(init="cascade", rcond=1e-15)),
}

LAM_WINNER_GRID = [0.15, 0.20, 0.30, 0.35]


def run_lam_winners():
    """Sweep lambda ON TOP of each part-1 winning config: is the flat-0.25 choice
    still near-optimal once the other knobs are right, or did the winners shift
    the lambda basin? Appends axis='lam_winner' records to refine.json."""
    rows = json.loads(REFINE_PATH.read_text()) if REFINE_PATH.exists() else []
    rows = [r for r in rows if r["axis"] != "lam_winner"]
    suites = {"f01": load_problem_suite("f01"), "f02": load_problem_suite("f02")}
    t_start = time.time()
    for (setup, key), (size, overrides) in WINNERS.items():
        prob = suites[setup].PROBLEMS[key]
        for lam in [0.25] + LAM_WINNER_GRID:
            t0 = time.time()
            cfg = replace(DEFAULT_CFG, lam=lam, **overrides)
            res = (solve_linear if setup == "f01" else solve_nonlinear)(prob, size, cfg)
            rec = _record(setup, key, "lam_winner", lam, size, 42, res)
            rows.append(rec)
            rl = rec["rel_l2"]
            print(f"  {setup} {key} W~{size} lam={lam:.2f} relL2="
                  + (f"{rl:.2e}" if rl is not None else "diverged")
                  + f"  ({time.time() - t0:.1f}s)", flush=True)
    REFINE_PATH.write_text(json.dumps(rows, indent=1))
    print(f"\nlam-winners done in {time.time() - t_start:.0f}s -> {REFINE_PATH}", flush=True)
    return rows


def run_refine():
    rows = []
    t_start = time.time()
    suites = {"f01": load_problem_suite("f01"), "f02": load_problem_suite("f02")}

    def solve_one(setup, key, size, label, overrides, seed):
        prob = suites[setup].PROBLEMS[key]
        cfg = replace(DEFAULT_CFG, seed=seed, **overrides)
        res = (solve_linear if setup == "f01" else solve_nonlinear)(prob, size, cfg)
        if res is None:
            return None
        rec = _record(setup, key, "refine", label, size, seed, res)
        rows.append(rec)
        print(f"  {setup} {key} W~{size} s{seed} {label:32s} relL2="
              f"{rec['rel_l2']:.2e}" if rec['rel_l2'] is not None else
              f"  {setup} {key} {label} diverged", flush=True)
        return rec

    best_per_target = {}
    for setup, key, size, label, overrides in REFINE_SPECS:
        rec = solve_one(setup, key, size, label, overrides, 42)
        if rec and rec["rel_l2"] is not None:
            tgt = (setup, key)
            cur = best_per_target.get(tgt)
            if cur is None or rec["rel_l2"] < cur[0]:
                best_per_target[tgt] = (rec["rel_l2"], label, overrides, size)

    print("\nconfirming winners across widths/seeds...", flush=True)
    for (setup, key), (rl, label, overrides, size) in sorted(best_per_target.items()):
        if setup == "f02" and key.startswith("pde2d"):
            for s2 in REFINE_CONFIRM_SIZES["f02"]:
                solve_one(setup, key, s2, label + "|confirm", overrides, 42)
        if key.startswith("pde2d"):
            for seed in REFINE_CONFIRM_SEEDS:
                solve_one(setup, key, size, label + "|seed", overrides, seed)

    PART1_DIR.mkdir(parents=True, exist_ok=True)
    REFINE_PATH.write_text(json.dumps(rows, indent=1))
    print(f"\nrefine: {len(rows)} records in {time.time() - t_start:.0f}s -> {REFINE_PATH}",
          flush=True)
    return rows


# ---------------------------------------------------------------------------
# part: rescue -- targeted levers on the hard problems, probe-first
# ---------------------------------------------------------------------------

RESCUE_PATH = PART1_DIR / "rescue.json"


def continuation(prob, size, cfg, ladder, term_fn):
    """Warm-started parameter continuation: solve at each eps in ladder (descending),
    passing the solution as the next Newton start. term_fn(eps) -> extra lin terms."""
    trail = []
    a = None
    res = None
    for eps in ladder:
        extra = term_fn(eps) if eps > 0 else None
        res = solve_nonlinear(prob, size, cfg, extra_lin=extra, a0_override=a)
        a = res["model"]["sol"]
        trail.append(dict(eps=eps, rel_l2=_sanitize(res["rel_l2"]),
                          newton_res=_sanitize(res["signals"]["newton_final_res"]),
                          iters=res["signals"]["newton_iters"]))
    return res, trail


def _rescue_record(label, res, trail=None):
    if res is None:
        return dict(label=label, not_applicable=True)
    rec = dict(label=label, rel_l2=_sanitize(res["rel_l2"]), linf=_sanitize(res["linf"]),
               W=res["W"], newton_hist=[_sanitize(h) for h in res["signals"].get("newton_hist", [])],
               stalled=res["signals"].get("stalled"))
    if trail is not None:
        rec["trail"] = trail
    return rec


# viscous regularization terms (space-time coords: x is column 0)
_VISC_1TERM = lambda eps: [((2, 0), -eps)]
_VISC_2TERM = lambda eps: [((2, 0), -eps), ((0, 2), -eps)]

NU_LADDER = [0.25, 0.1, 0.04, 0.015, 0.006, 0.002, 0.0]
EPS_LADDER = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0]


def run_rescue():
    """Informed by ablate + refine: eikonal, Monge-Ampere, viscous Burgers, and KdV
    are already rescued (cascade/collar+poly + rcond=1e-15 reach the floor). What
    remains: inviscid Burgers' last 1.5 orders, stress o3 at W=2304, and the
    linear 1D order-3 drift."""
    f02 = load_problem_suite("f02")
    out = {}
    t_start = time.time()

    def probe(name, prob, size, variants, include_default=True):
        rows = []
        if include_default:
            rows.append(_rescue_record("default", solve_nonlinear(prob, size, DEFAULT_CFG)))
            print(f"{name} @ {size}: default relL2={rows[0]['rel_l2']:.2e}", flush=True)
        else:
            print(f"{name} @ {size}:", flush=True)
        for label, spec in variants:
            t0 = time.time()
            if "ladder" in spec:
                res, trail = continuation(prob, size,
                                          replace(DEFAULT_CFG, **spec.get("cfg", {})),
                                          spec["ladder"], spec["term"])
                rows.append(_rescue_record(label, res, trail))
            else:
                res = solve_nonlinear(prob, size, replace(DEFAULT_CFG, **spec["cfg"]))
                rows.append(_rescue_record(label, res))
            rl = rows[-1].get("rel_l2")
            print(f"  {label:34s} relL2={rl:.2e}  ({time.time() - t0:.1f}s)"
                  if rl is not None else f"  {label:34s} n/a", flush=True)
        out[name] = dict(size=size, rows=rows)

    # 1. inviscid Burgers: lm+max_newton=60 reached 9.6e-8 @576 / 5.8e-12 @2304.
    #    Does vanishing-viscosity continuation beat it or push it to the floor?
    burgers = f02.PROBLEMS["pde2d_time_o1"]
    probe("inviscid_burgers_576", burgers, 576, [
        ("lm+mn60", dict(cfg=dict(lm=True, max_newton=60))),
        ("nu cont", dict(ladder=NU_LADDER, term=_VISC_1TERM)),
        ("nu cont + lm+mn60", dict(ladder=NU_LADDER, term=_VISC_1TERM,
                                   cfg=dict(lm=True, max_newton=60))),
        ("nu cont + rc1e-15", dict(ladder=NU_LADDER, term=_VISC_1TERM,
                                   cfg=dict(rcond=1e-15))),
        ("cascade+rc1e-15+mn60", dict(cfg=dict(init="cascade", rcond=1e-15,
                                               max_newton=60))),
    ])
    probe("inviscid_burgers_2304", burgers, 2304, [
        ("lm+mn60", dict(cfg=dict(lm=True, max_newton=60))),
        ("nu cont + lm+mn60", dict(ladder=NU_LADDER, term=_VISC_1TERM,
                                   cfg=dict(lm=True, max_newton=60))),
        ("nu cont + rc1e-15", dict(ladder=NU_LADDER, term=_VISC_1TERM,
                                   cfg=dict(rcond=1e-15))),
    ], include_default=False)

    # 2. stress o3 @2304: cascade+rc1e-15 gives 8.9e-11; residual drift remains
    probe("stress_o3_2304", f02.PROBLEMS["pde2d_steady_o3"], 2304, [
        ("cascade+rc1e-15", dict(cfg=dict(init="cascade", rcond=1e-15))),
        ("cascade+rc1e-15+equil", dict(cfg=dict(init="cascade", rcond=1e-15,
                                                equilibrate=True))),
        ("cascade+rcond0", dict(cfg=dict(init="cascade", rcond=0.0))),
        ("cascade+rc1e-15+wm0.1", dict(cfg=dict(init="cascade", rcond=1e-15,
                                                w_mult=0.1))),
    ], include_default=False)

    # 3. linear 1D order-3 drift @ N=1024 (best knob so far: w_mult=0.03 -> 4.6e-12
    #    vs 2.7e-15 at the optimal width; can any combo close the rest?)
    f01 = load_problem_suite("f01")
    lin_rows = []
    for label, cfg in [("default", DEFAULT_CFG),
                       ("wm0.03", replace(DEFAULT_CFG, w_mult=0.03)),
                       ("wm0.03+equil", replace(DEFAULT_CFG, w_mult=0.03, equilibrate=True)),
                       ("wm0.03+rcond0", replace(DEFAULT_CFG, w_mult=0.03, rcond=0.0)),
                       ("wm0.01", replace(DEFAULT_CFG, w_mult=0.01)),
                       ("wm0.01+rc1e-15", replace(DEFAULT_CFG, w_mult=0.01, rcond=1e-15))]:
        res = solve_linear(f01.PROBLEMS["ode1d_o3"], 1024, cfg)
        lin_rows.append(dict(key="ode1d_o3", size=1024, label=label,
                             rel_l2=_sanitize(res["rel_l2"]), linf=_sanitize(res["linf"])))
        print(f"f01 ode1d_o3 N=1024 {label:20s} relL2={res['rel_l2']:.2e}", flush=True)
    out["linear_drift"] = lin_rows

    PART1_DIR.mkdir(parents=True, exist_ok=True)
    RESCUE_PATH.write_text(json.dumps(out, indent=1))
    print(f"\nrescue done in {time.time() - t_start:.0f}s -> {RESCUE_PATH}", flush=True)
    return out


def plot_rescue(out):
    probs = [k for k in out if k != "linear_drift"]
    fig, axes = plt.subplots(2, len(probs), figsize=(3.6 * len(probs), 8.5))
    for j, name in enumerate(probs):
        rows = out[name]["rows"]
        ax = axes[0, j]
        labels = [r["label"] for r in rows if r.get("rel_l2") is not None]
        vals = [r["rel_l2"] for r in rows if r.get("rel_l2") is not None]
        colors = ["0.5" if lab == "default" else "C0" for lab in labels]
        best = int(np.argmin(vals))
        colors[best] = "C2"
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_xscale("log")
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(f"{name} @ W~{out[name]['size']}", fontsize=9)
        ax.grid(True, axis="x", alpha=0.25)
        if j == 0:
            ax.set_xlabel("rel $L_2$")
        ax2 = axes[1, j]
        for r in rows:
            h = [v for v in r.get("newton_hist", []) if v is not None]
            if h:
                ax2.semilogy(range(len(h)), h, "-", lw=1,
                             label=r["label"], alpha=0.85)
        ax2.set_xlabel("Newton step")
        if j == 0:
            ax2.set_ylabel("PDE residual $\\max|r|$")
        ax2.grid(True, which="both", alpha=0.25)
        ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                   fontsize=5, frameon=False)
    fig.tight_layout()
    outp = PART1_DIR / "rescue.png"
    fig.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outp}", flush=True)


# ---------------------------------------------------------------------------
# part: select -- deployable model selection (no u* anywhere in the protocol)
# ---------------------------------------------------------------------------

SELECT_PATH = PART2_DIR / "select.json"
SELECT_HELDOUT_PATH = PART2_DIR / "select_heldout.json"

# The frozen deployed recipe (per setup x category). Chosen from part 1 on the
# 18-problem development set; the held-out test is the honest evaluation.
DEPLOY = {
    ("f01", "ode1d"): {},
    ("f01", "pde2d_steady"): dict(poly_degree=5, collar_steady=1.6),
    ("f01", "pde2d_time"): {},
    ("f02", "ode1d"): {},
    # unification probe (dev set): cascade+rc15+poly5+collar1.6 dominates plain
    # cascade+rc15 on all three steady problems (8.3e-14 / 8.1e-14 / 4.1e-12 @576)
    ("f02", "pde2d_steady"): dict(init="cascade", rcond=1e-15, poly_degree=5,
                                  collar_steady=1.6),
    ("f02", "pde2d_time"): dict(init="cascade", rcond=1e-15),
}

LADDER_1D = [8, 16, 32, 64, 128, 256, 512, 1024]
LADDER_2D = [144, 256, 576, 1024, 2304]
LADDER_TIME_LIN = LADDER_2D + [4096]
SEEDS_2D = [42, 7, 19]


def ladder_for(setup, category):
    if category == "ode1d":
        return LADDER_1D
    if category == "pde2d_time" and setup == "f01":
        return LADDER_TIME_LIN
    return LADDER_2D


def deployed_solve(setup, prob, size, cfg, cache_tag):
    """One deployed solve. Nonlinear adds the stall fallback: if Newton's residual
    stalls, retry with LM + more iterations and keep the retry if its residual is
    lower -- a rule that needs no u*."""
    if setup == "f01":
        return solve_linear(prob, size, cfg, cache_tag=cache_tag), False
    res = solve_nonlinear(prob, size, cfg, cache_tag=cache_tag)
    if res["signals"]["stalled"]:
        retry = solve_nonlinear(prob, size, replace(cfg, lm=True, max_newton=60,
                                                    init="default"),
                                cache_tag=cache_tag + "_lm")
        if (retry["signals"]["newton_final_res"] or np.inf) < \
                (res["signals"]["newton_final_res"] or np.inf):
            return retry, True
    return res, False


def run_select_ladders(problems_by_setup, out_path, tag_prefix):
    """Solve the width ladder for every problem under the deployed recipe,
    caching predictions; returns the raw cell records."""
    cells = []
    t_start = time.time()
    for setup, probs in problems_by_setup.items():
        for key, prob in probs.items():
            cat = prob["category"]
            cfg_over = DEPLOY[(setup, cat)]
            seeds = [42] if cat == "ode1d" else SEEDS_2D
            t0 = time.time()
            for size in ladder_for(setup, cat):
                for seed in seeds:
                    tag = f"{tag_prefix}_{setup}_{key}_W{size}_s{seed}"
                    cfg = replace(DEFAULT_CFG, seed=seed, **cfg_over)
                    res, fb = deployed_solve(setup, prob, size, cfg, tag)
                    rec = _record(setup, key, "select", size, size, seed, res)
                    rec["tag"] = tag + ("_lm" if fb else "")
                    rec["lm_fallback"] = fb
                    cells.append(rec)
                print(f"  {setup} {key} W~{size}: " + " ".join(
                    f"{c['rel_l2']:.1e}" if c['rel_l2'] is not None else "div"
                    for c in cells if c["key"] == key and c["size"] == size
                    and c["setup"] == setup), flush=True)
            print(f"{setup} {key} ladder done ({time.time() - t0:.0f}s)", flush=True)
    out_path.write_text(json.dumps(cells, indent=1))
    print(f"ladders: {len(cells)} cells in {time.time() - t_start:.0f}s -> {out_path}",
          flush=True)
    return cells


def _rel_diff(u_a, u_b):
    return float(np.linalg.norm(u_a - u_b) / max(np.linalg.norm(u_b), 1e-300))


def select_width(cells, setup, key):
    """The frozen selector. Input: this problem's ladder cells (all seeds).
    delta_k = median over seeds of rel L2 diff between consecutive-width solutions
    (cached predictions); est(W_k) = max(delta_{k-1}, delta_k) at interior k,
    delta_0 / delta_last at the ends; W* = argmin est (ties -> smaller W).
    Returns dict with W*, est, per-width diagnostics."""
    sub = [c for c in cells if c["setup"] == setup and c["key"] == key
           and c["rel_l2"] is not None]
    sizes = sorted({c["size"] for c in sub})
    seeds = sorted({c["seed"] for c in sub})
    preds = {}
    for c in sub:
        preds[(c["size"], c["seed"])] = common.load_pred(c["tag"])
    deltas = []
    for k in range(len(sizes) - 1):
        ds = [_rel_diff(preds[(sizes[k], s)], preds[(sizes[k + 1], s)])
              for s in seeds if (sizes[k], s) in preds and (sizes[k + 1], s) in preds]
        deltas.append(float(np.median(ds)))
    est = {}
    for k, size in enumerate(sizes):
        if k == 0:
            est[size] = deltas[0]
        elif k == len(sizes) - 1:
            est[size] = deltas[-1]
        else:
            est[size] = max(deltas[k - 1], deltas[k])
    w_star = min(sizes, key=lambda s: (est[s], s))
    # negative control: pick by fresh-point physical residual (median over seeds)
    res_by_size = {s: float(np.median([c["fresh_res_phys"] for c in sub if c["size"] == s
                                       and c.get("fresh_res_phys") is not None]))
                   for s in sizes}
    w_resid = min(sizes, key=lambda s: (res_by_size.get(s, np.inf), s))
    true_by_size = {s: float(np.median([c["rel_l2"] for c in sub if c["size"] == s]))
                    for s in sizes}
    return dict(sizes=sizes, deltas=deltas, est={str(k): v for k, v in est.items()},
                w_star=w_star, est_err=est[w_star],
                true_at_star=true_by_size[w_star],
                w_oracle=min(sizes, key=lambda s: true_by_size[s]),
                true_at_oracle=min(true_by_size.values()),
                w_resid=w_resid, true_at_resid=true_by_size[w_resid],
                true_at_default=true_by_size.get(1024),
                true_by_size={str(k): v for k, v in true_by_size.items()})


def run_select():
    problems = {"f01": load_problem_suite("f01").PROBLEMS,
                "f02": load_problem_suite("f02").PROBLEMS}
    cells = run_select_ladders(problems, SELECT_PATH, "sel")
    summary = {}
    for setup in problems:
        for key in problems[setup]:
            summary[f"{setup}:{key}"] = select_width(cells, setup, key)
            s = summary[f"{setup}:{key}"]
            print(f"{setup}:{key}: W*={s['w_star']} est={s['est_err']:.1e} "
                  f"true={s['true_at_star']:.1e} | oracle W={s['w_oracle']} "
                  f"true={s['true_at_oracle']:.1e} | resid-pick W={s['w_resid']} "
                  f"true={s['true_at_resid']:.1e}", flush=True)
    (PART2_DIR / "select_summary.json").write_text(json.dumps(summary, indent=1))
    return cells, summary


def run_select_heldout():
    """The one-shot held-out evaluation. Selector and recipe are FROZEN; this runs
    after the dev-set selection is inspected, exactly once."""
    import heldout_problems as hp
    hp.verify_all(verbose=False)
    problems = {"f01": hp.HELDOUT_LINEAR, "f02": hp.HELDOUT_NONLINEAR}
    cells = run_select_ladders(problems, SELECT_HELDOUT_PATH, "hld")
    summary = {}
    for setup in problems:
        for key in problems[setup]:
            summary[f"{setup}:{key}"] = select_width(cells, setup, key)
            s = summary[f"{setup}:{key}"]
            print(f"HELD-OUT {setup}:{key}: W*={s['w_star']} est={s['est_err']:.1e} "
                  f"true={s['true_at_star']:.1e} | oracle W={s['w_oracle']} "
                  f"true={s['true_at_oracle']:.1e}", flush=True)
    (PART2_DIR / "select_heldout_summary.json").write_text(json.dumps(summary, indent=1))
    return cells, summary


# ---------------------------------------------------------------------------
# part: baselines -- other methods, incremental (fastest first), per-method JSON
# ---------------------------------------------------------------------------

def _qi_ladder_from_select():
    """Our method's ladder cells, reused verbatim from part 2 (same recipe)."""
    cells = json.loads(SELECT_PATH.read_text())
    out = []
    for c in cells:
        if c["setup"] != "f01" or c.get("rel_l2") is None:
            continue
        out.append(dict(key=c["key"], dof=c["W"], size=c["size"], seed=c["seed"],
                        variant="deployed", rel_l2=c["rel_l2"], linf=c["linf"],
                        t_assemble=c.get("t_assemble"), t_solve=c.get("t_solve")))
    return out


def _dof_ladder(key):
    """Per-problem DOF rungs = the QI ladder's actual W values (halo included)."""
    cells = json.loads(SELECT_PATH.read_text())
    sizes = sorted({c["size"] for c in cells if c["key"] == key and c["setup"] == "f01"})
    return [(s, next(c["W"] for c in cells
                     if c["key"] == key and c["setup"] == "f01" and c["size"] == s))
            for s in sizes]


ELM_R_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
RBF_C_MULT = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
BASE_SEEDS = [42, 7, 19]


def run_baseline_method(method):
    import baselines as B
    f01 = load_problem_suite("f01")
    t_start = time.time()
    rows = []
    if method == "qi":
        rows = _qi_ladder_from_select()
    elif method == "spectral":
        for key, prob in f01.PROBLEMS.items():
            ladder = _dof_ladder(key)
            if prob["category"] == "ode1d":
                # the QI ladder starts at W~149 (halo included); spectral's sweet
                # spot is much lower degree, so give it extra small rungs
                ladder = [(d, d) for d in [17, 33, 65, 97]] + ladder
            done = set()
            for size, dof in ladder:
                basis = B.spectral_basis_for(prob["category"], dof)
                if basis.dof in done:
                    continue
                done.add(basis.dof)
                r = B.solve_with_basis(prob, basis)
                rows.append(dict(key=key, size=size, seed=0, variant="cheb", **r))
                print(f"  spectral {key} dof={basis.dof}: relL2={r['rel_l2']:.1e}",
                      flush=True)
    elif method == "elm":
        for key, prob in f01.PROBLEMS.items():
            dim = 1 if prob["category"] == "ode1d" else 2
            ladder = _dof_ladder(key)
            mid_dof = ladder[len(ladder) // 2][1]
            # Dong & Yang: tune R once per problem by the collocation residual
            tune = []
            for R in ELM_R_GRID:
                r = B.solve_with_basis(prob, B.ElmBasis(mid_dof, R, dim, seed=42))
                tune.append((r["tune_signal"], R, r["rel_l2"]))
            R_tuned = min(tune)[1]
            print(f"  elm {key}: tuned R={R_tuned} (signal-picked; oracle would pick "
                  f"R={min(tune, key=lambda t: t[2])[1]})", flush=True)
            for variant, R, seeds in [("default_R1", 1.0, [42]),
                                      (f"tuned_R{R_tuned:g}", R_tuned, BASE_SEEDS)]:
                for size, dof in _dof_ladder(key):
                    if dof > 2400:
                        continue  # stochastic-baseline ladder capped at 2304 (cost)
                    for seed in seeds:
                        r = B.solve_with_basis(prob, B.ElmBasis(dof, R, dim, seed=seed))
                        rows.append(dict(key=key, size=size, seed=seed,
                                         variant=variant, **r))
            print(f"  elm {key} done", flush=True)
    elif method == "rbf":
        for key, prob in f01.PROBLEMS.items():
            dim = 1 if prob["category"] == "ode1d" else 2
            ladder = _dof_ladder(key)
            mid_dof = ladder[len(ladder) // 2][1]
            diam = 2.0 if dim == 1 or prob["category"] == "pde2d_steady" else 2.83
            # tune (kind, c) jointly by the u*-free signal, then ladder the winner
            best_by_kind = {}
            for kind in ["mq", "gauss"]:
                c_fr = B.franke_c(mid_dof, diameter=diam)
                tune = []
                for m in RBF_C_MULT:
                    r = B.solve_with_basis(prob, B.RbfBasis(mid_dof, m * c_fr, dim,
                                                            seed=42, kind=kind))
                    tune.append((r["tune_signal"], m, r["rel_l2"]))
                best_by_kind[kind] = min(tune)
            kind = min(best_by_kind, key=lambda k: best_by_kind[k][0])
            m_tuned = best_by_kind[kind][1]
            print(f"  rbf {key}: picked {kind}, c mult {m_tuned} (signal-tuned)",
                  flush=True)
            for variant, mult, seeds in [(f"{kind}_franke", 1.0, [42]),
                                         (f"{kind}_tuned", m_tuned, BASE_SEEDS)]:
                for size, dof in _dof_ladder(key):
                    if dof > 2400:
                        continue
                    c_val = mult * B.franke_c(dof, diameter=diam)
                    for seed in seeds:
                        r = B.solve_with_basis(prob, B.RbfBasis(dof, c_val, dim,
                                                                seed=seed, kind=kind))
                        rows.append(dict(key=key, size=size, seed=seed,
                                         variant=variant, **r))
            print(f"  rbf {key} done", flush=True)
    elif method == "pinn":
        from baselines_pinn import run_pinn
        rows = run_pinn()
    elif method == "fd":
        from baselines_fd import run_fd
        rows = run_fd(_dof_ladder)
    else:
        sys.exit(f"unknown baseline method {method!r}")
    rows = [{k: _sanitize(v) for k, v in r.items()} for r in rows]
    out = PART3_DIR / f"baselines_{method}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"{method}: {len(rows)} cells in {time.time() - t_start:.0f}s -> {out}", flush=True)
    return rows


METHOD_STYLE = {  # label, color
    "qi": ("QI/Radon (ours, deployed)", "C0"),
    "elm": ("random features (ELM/PIELM)", "C1"),
    "rbf": ("Kansa RBF (MQ/Gauss, best)", "C2"),
    "spectral": ("Chebyshev spectral LS", "C3"),
    "pinn": ("trained PINN", "C4"),
    "fd": ("finite differences", "C5"),
}


def _method_rows():
    out = {}
    for m in METHOD_STYLE:
        p = PART3_DIR / f"baselines_{m}.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
    return out


def _best_line(rows, key):
    """Per DOF rung: median over seeds of the best u*-free variant (tuned where it
    exists, else the only one). QI uses its deployed cells directly."""
    sub = [r for r in rows if r["key"] == key and r.get("rel_l2") is not None]
    if not sub:
        return [], []
    variants = {r["variant"] for r in sub}
    pick = [v for v in variants if "tuned" in v] or list(variants)
    pts = {}
    for r in sub:
        if r["variant"] in pick or len(variants) == 1 or r["variant"] == "deployed":
            pts.setdefault(r["dof"], []).append(r["rel_l2"])
    dofs = sorted(pts)
    return dofs, [float(np.median(pts[d])) for d in dofs]


# QI oracle-best per problem: the best true error QI achieves with oracle access to
# its OWN config space -- the symmetric analog of letting each baseline pick its best
# variant/scale. For most problems oracle width (part-2 true_at_oracle) is already the
# best; these are the hard cases where part-1 per-problem CONFIG tuning beats it (values
# from the part-1 hard-problem table, verified). Applied as min(true_at_oracle, this).
QI_PART1_BEST = {
    "f02:pde2d_steady_o1": 6.6e-15,   # eikonal: cascade + rcond 1e-15 @1024
    "f02:pde2d_steady_o3": 1.3e-14,   # stress:  cascade + rcond 1e-15 + w_mult 0.1 @2304
    "f02:pde2d_time_o1": 5.8e-12,     # inviscid Burgers: LM + max_newton 60 @2304
}


def _qi_oracle(key, summary):
    """QI's best true error with oracle access to width AND config."""
    return min(summary[key]["true_at_oracle"], QI_PART1_BEST.get(key, float("inf")))


def plot_baselines():
    methods = _method_rows()
    summary = json.loads((PART2_DIR / "select_summary.json").read_text())
    f01 = load_problem_suite("f01")
    # --- line figure: rel L2 vs DOF, one line per method ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5))
    for col, cat in enumerate(["ode1d", "pde2d_steady", "pde2d_time"]):
        for row_i, order in enumerate([1, 2, 3]):
            key = f"{cat}_o{order}"
            ax = axes[row_i, col]
            for m, rows in methods.items():
                dofs, vals = _best_line(rows, key)
                if not dofs:
                    continue
                label, color = METHOD_STYLE[m]
                ax.loglog(dofs, vals, "o-", color=color, ms=3.5, lw=1.2, label=label)
            ax.set_title(f01.PROBLEMS[key]["title"], fontsize=9)
            ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
            ax.grid(True, which="both", alpha=0.25)
            if row_i == 2:
                ax.set_xlabel("degrees of freedom")
            if col == 0:
                ax.set_ylabel(f"order {order}\nrel $L_2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=min(len(labels), 6), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(PART3_DIR / "baseline_error_vs_width.png", dpi=150)
    plt.close(fig)
    print(f"wrote {PART3_DIR / 'baseline_error_vs_width.png'}", flush=True)

    # --- bar figures: (a) ours deployed vs baselines oracle (tilted toward baselines);
    #     (b) FAIR -- ours also at oracle-best, both sides oracle ---
    def _bars(fair):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        for col, cat in enumerate(["ode1d", "pde2d_steady", "pde2d_time"]):
            for row_i, order in enumerate([1, 2, 3]):
                key = f"{cat}_o{order}"
                ax = axes[row_i, col]
                names, vals, colors = [], [], []
                fk = f"f01:{key}"
                names.append("QI (oracle)" if fair else "QI (deployed $W^*$)")
                vals.append(_qi_oracle(fk, summary) if fair else summary[fk]["true_at_star"])
                colors.append(METHOD_STYLE["qi"][1])
                for m, rows in methods.items():
                    if m == "qi":
                        continue
                    sub = [r for r in rows if r["key"] == key and r.get("rel_l2") is not None]
                    if not sub:
                        continue
                    best = min(sub, key=lambda r: r["rel_l2"])
                    names.append(f"{METHOD_STYLE[m][0].split(' (')[0]}\n(oracle, dof "
                                 f"{best['dof']})")
                    vals.append(best["rel_l2"])
                    colors.append(METHOD_STYLE[m][1])
                ax.bar(range(len(vals)), vals, color=colors)
                ax.set_yscale("log")
                ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
                ax.set_xticks(range(len(names)), names, fontsize=6, rotation=25, ha="right")
                ax.set_title(f01.PROBLEMS[key]["title"], fontsize=9)
                ax.grid(True, axis="y", which="both", alpha=0.2)
                if col == 0:
                    ax.set_ylabel(f"order {order}\nrel $L_2$")
        fig.suptitle("Fair comparison: every method at its oracle-best cell (QI oracle "
                     "over width + config)" if fair else
                     "Ours at the no-oracle deployed width vs baselines at their "
                     "oracle-best cell (tuned variants included)", y=0.995, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = PART3_DIR / ("baseline_bars_fair.png" if fair else "baseline_bars.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}", flush=True)
    _bars(False)
    _bars(True)

    # --- tradeoff figure: wall-clock vs precision ---
    fig, ax = plt.subplots(figsize=(7, 5.6))
    for m, rows in methods.items():
        label, color = METHOD_STYLE[m]
        xs, ys = [], []
        for key in PROBLEM_ORDER:
            sub = [r for r in rows if r["key"] == key and r.get("rel_l2") is not None
                   and r.get("t_solve") is not None]
            if not sub:
                continue
            best = min(sub, key=lambda r: r["rel_l2"])
            xs.append((best.get("t_assemble") or 0) + best["t_solve"])
            ys.append(best["rel_l2"])
        ax.loglog(xs, ys, "o", color=color, ms=6, label=label, alpha=0.8)
    ax.set_xlabel("wall-clock per solve (s, assembly + solve/training)")
    ax.set_ylabel("best rel $L_2$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(PART3_DIR / "baseline_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PART3_DIR / 'baseline_tradeoff.png'}", flush=True)


# ---------------------------------------------------------------------------
# select figures
# ---------------------------------------------------------------------------

def plot_select(summary, heldout_summary=None):
    # 1. does the signal track truth? est(W) vs true(W) over every ladder point
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for name, s in summary.items():
        cat = name.split(":")[1].rsplit("_", 1)[0]
        xs = [s["true_by_size"][k] for k in s["est"]]
        ys = [s["est"][k] for k in s["est"]]
        ax.loglog(xs, ys, "o", ms=4, color=CAT_COLOR[cat], alpha=0.7)
    lims = [1e-16, 1e0]
    ax.loglog(lims, lims, "k:", lw=0.8)
    ax.set_xlabel("true rel $L_2$")
    ax.set_ylabel("selector estimate (nested-width $\\delta$)")
    ax.grid(True, which="both", alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
               for l, c in CAT_COLOR.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(PART2_DIR / "selector_signal_vs_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PART2_DIR / 'selector_signal_vs_error.png'}", flush=True)

    # 2. what does deploying cost? oracle vs W* pick vs fixed default vs resid pick
    entries = list(summary.items())
    if heldout_summary:
        entries += [("HELD-OUT " + k, v) for k, v in heldout_summary.items()]
    fig, ax = plt.subplots(figsize=(1.05 * len(entries) + 2.5, 5))
    xs = np.arange(len(entries))
    wbar = 0.2
    for off, (field, label, color) in zip(
            [-1.5, -0.5, 0.5, 1.5],
            [("true_at_oracle", "oracle width", "0.35"),
             ("true_at_star", "selector $W^*$", "C2"),
             ("true_at_default", "fixed $W{=}1024$", "C1"),
             ("true_at_resid", "residual pick (control)", "C3")]):
        vals = [e[1].get(field) or np.nan for e in entries]
        ax.bar(xs + off * wbar, vals, wbar, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(xs, [e[0] for e in entries], rotation=45, ha="right", fontsize=7)
    ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("true rel $L_2$ at the picked width")
    ax.grid(True, axis="y", which="both", alpha=0.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(PART2_DIR / "selection_oracle_vs_deployed.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PART2_DIR / 'selection_oracle_vs_deployed.png'}", flush=True)

    # 3. trajectories: est + true vs width, W* marked. The shaded gap shows the
    # estimate sitting on-or-above the true error (conservative); the red line is
    # the selector's automatic pick.
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
    n = len(entries)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow), sharex=False)
    for ax, (name, s) in zip(axes.ravel(), entries):
        sizes = s["sizes"]
        true = [s["true_by_size"][str(k)] for k in sizes]
        est = [s["est"][str(k)] for k in sizes]
        ax.fill_between(sizes, true, est, color="C2", alpha=0.12, lw=0)
        ax.loglog(sizes, true, "o-", color="k", lw=1.2, ms=3, label="true error")
        ax.loglog(sizes, est, "s--", color="C2", lw=1.2, ms=3, label="estimate $\\delta$")
        ax.axvline(s["w_star"], color="C3", lw=1.4, alpha=0.85)
        ax.annotate(f"$W^*${s['w_star']}", xy=(s["w_star"], 1),
                    xycoords=("data", "axes fraction"), xytext=(2, -2),
                    textcoords="offset points", fontsize=6, color="C3", va="top")
        ax.set_title(name.replace("HELD-OUT ", "★ "), fontsize=7.5)
        ax.grid(True, which="major", alpha=0.25)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=6.5)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles + [plt.Line2D([], [], color="C3", lw=1.4)],
               labels + ["selected $W^*$"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=9)
    fig.suptitle("Model selection: nested-width estimate vs true error "
                 "(★ = held-out; band = est$-$true gap)", y=1.02, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(PART2_DIR / "selection_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PART2_DIR / 'selection_trajectories.png'}", flush=True)


# ---------------------------------------------------------------------------
# part 4: nonlinear-zoo baseline bar chart
# ---------------------------------------------------------------------------

def run_baseline_nonlinear(method):
    import baselines_nonlinear as BN
    return BN.run_method_nonlinear(method)


def _best_per_problem(records):
    """oracle-best (min finite rel_l2) per problem key."""
    best = {}
    for r in records:
        rl = r.get("rel_l2")
        if rl is None or not np.isfinite(rl):
            continue
        if r["key"] not in best or rl < best[r["key"]]:
            best[r["key"]] = rl
    return best


def plot_part4(fair=False):
    f02 = load_problem_suite("f02")
    summ = json.loads((PART2_DIR / "select_summary.json").read_text())

    def load(m):
        p = PART4_DIR / f"nl_{m}.json"
        return _best_per_problem(json.loads(p.read_text())) if p.exists() else {}

    if fair:   # QI also at oracle-best (width + config), symmetric with the baselines
        qi = {k.split(":", 1)[1]: _qi_oracle(k, summ) for k in summ if k.startswith("f02:")}
        qi_label = "QI/Radon (oracle)"
    else:
        qi = {k.split(":", 1)[1]: v["true_at_star"] for k, v in summ.items()
              if k.startswith("f02:")}
        qi_label = "QI/Radon (deployed)"
    # (label, color, per-problem dict, applies?) -- FD greyed everywhere (not run on
    # the nonlinear zoo: disk excluded, and it is the algebraic control from part 3)
    methods = [(qi_label, "C0", qi, True),
               ("random features (ELM+Newton)", "C1", load("elm"), True),
               ("Kansa RBF (+Newton)", "C2", load("rbf"), True),
               ("Chebyshev spectral (+Newton)", "C3", load("spectral"), True),
               ("trained PINN", "C4", load("pinn"), True),
               ("finite differences", "0.6", {}, False)]
    order = list(f02.PROBLEMS)
    LO, HI = 1e-16, 3e0
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, key in zip(axes.ravel(), order):
        for i, (lab, color, dct, applies) in enumerate(methods):
            if not applies:                     # greyed N/A bar at full height
                ax.bar(i, HI / LO, bottom=LO, color="0.75", hatch="//",
                       edgecolor="0.5", width=0.8)
                ax.text(i, HI * 0.3, "N/A", ha="center", va="top", fontsize=8,
                        color="0.35", rotation=90)
                continue
            v = dct.get(key)
            if v is None or not np.isfinite(v):  # ran but failed: bar off the top
                ax.bar(i, HI / LO, bottom=LO, color=color, alpha=0.35, width=0.8)
                ax.text(i, HI * 0.3, "fail", ha="center", va="top", fontsize=8,
                        color=color, rotation=90)
            else:
                ax.bar(i, max(v, LO) - LO, bottom=LO, color=color, width=0.8)
        ax.set_yscale("log")
        ax.set_ylim(LO, HI)
        ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m[0].split(" (")[0].split(" +")[0] for m in methods],
                           rotation=35, ha="right", fontsize=7)
        ax.set_title(f02.PROBLEMS[key]["title"], fontsize=8.5)
        ax.grid(True, axis="y", which="both", alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("rel $L_2$")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _l, c, _d, _a in methods]
    fig.legend(handles, [m[0] for m in methods], loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=8.5)
    fig.suptitle("Nonlinear zoo, fair comparison: every method at its oracle-best cell "
                 "(QI oracle over width + config), all via our Newton harness" if fair else
                 "Nonlinear zoo: ours (deployed, no oracle) vs baselines given our "
                 "Newton harness + cascade init (baselines at oracle-best cell)",
                 y=1.03, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PART4_DIR / ("nl_baseline_bars_fair.png" if fair else "nl_baseline_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


# ---------------------------------------------------------------------------

def main():
    part = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    print("verifying problem definitions + reproduction gate...", flush=True)
    load_problem_suite("f01").verify_all(verbose=False)
    load_problem_suite("f02").verify_all(verbose=False)
    if not SMOKE:
        common.reproduction_gate()
    print("ok\n", flush=True)
    only = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--only=")), None)
    if part == "ablate":
        records = run_ablate(only_axes=set(only.split(",")) if only else None)
        plot_ablate_summary(records)
        plot_ablate_axes(records)
    elif part == "lamwin":
        run_lam_winners()
    elif part == "select":
        cells, summary = run_select()
        plot_select(summary)
    elif part == "select-heldout":
        cells, hsummary = run_select_heldout()
        summary = json.loads((PART2_DIR / "select_summary.json").read_text())
        plot_select(summary, heldout_summary=hsummary)
    elif part == "baselines":
        import baselines as B
        B.verify_bases()
        which = only.split(",") if only else ["qi", "elm", "rbf", "spectral"]
        for m in which:
            run_baseline_method(m)
            plot_baselines()   # figures update after every method (cancellable)
    elif part == "plot-baselines":
        plot_baselines()
    elif part == "part4":
        import baselines as B
        B.verify_bases()
        which = only.split(",") if only else ["spectral", "elm", "rbf", "pinn"]
        for m in which:
            if m == "pinn":
                import baselines_pinn as PN
                PN.run_pinn_nonlinear()
            else:
                run_baseline_nonlinear(m)
            plot_part4()   # bar chart updates after each method (cancellable)
        plot_part4(fair=True)
    elif part == "plot-part4":
        plot_part4()
        plot_part4(fair=True)
    elif part == "plot-select":
        summary = json.loads((PART2_DIR / "select_summary.json").read_text())
        hp = PART2_DIR / "select_heldout_summary.json"
        plot_select(summary, heldout_summary=json.loads(hp.read_text()) if hp.exists() else None)
    elif part == "plot-ablate":
        records = _load_ablate()
        plot_ablate_summary(records)
        plot_ablate_axes(records)
    elif part == "refine":
        run_refine()
    elif part == "rescue":
        out = run_rescue()
        plot_rescue(out)
    elif part == "plot-rescue":
        out = json.loads(RESCUE_PATH.read_text())
        plot_rescue(out)
    else:
        sys.exit(f"unknown or missing part: {part!r} (ablate | plot-ablate | rescue | plot-rescue)")


if __name__ == "__main__":
    main()
