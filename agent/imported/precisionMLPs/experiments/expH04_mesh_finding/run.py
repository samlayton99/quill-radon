"""expH04 -- the mesh-finding ladder: can a monitor tell us where the centers should go?

The suite's even reference puts centers evenly along evenly spread directions and lets
one least-squares solve do the rest. This experiment keeps the solve and changes only
the mesh (where the centers, and in 2-D the directions, sit), following the theory's
prescription that the center density along a direction should follow
``(p_v * roughness)^{1/(2r+1)}``. The rungs go from the safest reading of that idea
to the most speculative; every rung is a fixed geometry plus one solve, no training.

Rungs (``mesh.py`` has the definitions):

    even          the reference, nothing adapted
    data_p13      centers follow p_v^{1/3}       (data density only, no target information)
    data_p1       centers follow p_v            (match the data outright)
    oracle_r1     centers follow (p_v R_1)^{1/3}, R_1 from the TRUE gradient (a ceiling)
    oracle_r2     centers follow (p_v R_2)^{1/5}, R_2 from the true second derivative
    surr_r1       as oracle_r1 with R_1 taken from a first even fit  (practical, 2 solves)
    surr_r2       as oracle_r2 from the first even fit
    residual      centers follow (p_v E[e^2|t])^{1/3}, e = residual of the first even fit
    surr_r1_x3    surr_r1 iterated: the monitor is re-read from each new fit, 3 solves
    freq_oracle   centers follow the local frequency omega_v(t) = sqrt(R_2/R_1) of the
                  TRUE target: the spectral rule (data- and amplitude-independent)
    freq          the same from the first even fit (practical, 2 solves)
    active_oracle directions from the gradient covariance E[grad F grad F^T] of the TRUE
                  target (the active subspace): the budget is split as an m-dimensional
                  problem inside the leading-m eigenvector subspace, m chosen by the
                  eigenvalue spectrum; centers then follow the true slope monitor
    active        the same with the gradient covariance and slope monitor read from the
                  first even fit (2 solves)
    active_x3     iterated: the covariance and monitor are re-read from the active fit and
                  the mesh rebuilt, twice (3 solves). A fixed-point iteration on the
                  subspace: a better fit gives a better subspace gives a better fit.
    -- d = 2 only --
    dir_oracle    directions follow A(theta)^{1/3}, A = mean |dF/dv|^2, true gradient;
                  centers even
    dir_surr      the same with A from the first even fit
    both_surr     directions and centers from the first even fit
    joint_surr    both_surr, plus per-direction center counts proportional to a floor
                  plus that direction's monitor mass

All rungs use the floor ``s = 2/3`` (one third of the centers always spread evenly),
grading ``|dh/dt| <= 0.15`` and monitor smoothing at 5.8 even spacings (derived from the mesh-map resolution limit, see mesh.py). ``--floor``
sweeps ``s`` for one rung.

Usage:
    uv run --extra dev python experiments/expH04_mesh_finding/run.py --dims 1
    uv run --extra dev python experiments/expH04_mesh_finding/run.py --dims 2 --tasks 2.11,2.12
    uv run --extra dev python experiments/expH04_mesh_finding/run.py --floor
    uv run --extra dev python experiments/expH04_mesh_finding/run.py --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "experiments" / "expH01_highdim_suite"))
sys.path.append(str(Path(__file__).resolve().parent))

from h01suite.baseline import EvenGeometry                      # noqa: E402
from h01suite.metrics import error_metrics                       # noqa: E402
from h01suite.tasks import get_task                              # noqa: E402
from mesh import (AdaptiveGeometry, Monitors, oracle_derivatives,  # noqa: E402
                  surrogate_derivatives, FLOOR_S, gradient_covariance,
                  active_dimension, active_subspace_geometry)

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH04_mesh_finding"
FIG_DIR = RESULTS_DIR / "figures"

TASKS = {1: ["1.1", "1.7", "1.8", "1.11", "1.12", "1.13", "1.14", "1.15", "1.16"],
         2: ["2.1", "2.3", "2.7", "2.8", "2.11", "2.12", "2.13", "2.14", "2.15", "2.16"]}
BUDGETS = {1: [32, 64, 128, 256, 512, 1024], 2: [256, 512, 1024, 2048, 4096]}
RATIO = 8.0
TEST_SETS = ("same_as_train", "uniform", "dense_region")
TRAIN_SEED, TEST_SEED = 0, 10_000
DIR_SUBSAMPLE = 4096          # training points used for the direction monitor

CENTER_RUNGS = ["even", "data_p13", "data_p1", "oracle_r1", "oracle_r2", "surr_r1",
                "surr_r2", "residual", "surr_r1_x3", "freq_oracle", "freq"]
DIRECTION_RUNGS = ["dir_oracle", "dir_surr", "both_surr", "joint_surr"]
FLOOR_LEVELS = [0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0]
FLOOR_TASKS = ["1.7", "1.13", "1.14", "1.16"]


# ---------------------------------------------------------------------------
# the rungs
# ---------------------------------------------------------------------------

def _dir_monitor(kind, task, X, model=None):
    Xs = X[:DIR_SUBSAMPLE]

    def A(Vg):
        if kind == "oracle":
            D = (task.grad_F(Xs) @ Vg.T).T
        else:
            D = surrogate_derivatives(model, Xs, Vg, 1)
        return (D * D).mean(axis=1)
    return A


def _monitor_mass(geo: AdaptiveGeometry, X, monitors: Monitors) -> np.ndarray:
    """Integral of each direction's center monitor (used by the joint rung)."""
    from mesh import N_GRID
    Z = []
    for i, v in enumerate(geo.unique_directions):
        T = geo.margin * float(np.abs(v).sum())
        grid = np.linspace(-T, T, N_GRID)
        h0 = 2.0 * T / geo.n_per_direction
        m = monitors.center_monitor(i, X @ v, grid, geo.bw_mult * h0)
        Z.append(np.trapezoid(m, dx=grid[1] - grid[0]))
    return np.asarray(Z)


def _active_part(model):
    """The units of an active-subspace mesh that live in the subspace, as a model-like
    object: re-estimating the subspace from these alone removes the bias that the
    full-space (background) units would put into the gradient covariance."""
    from types import SimpleNamespace
    info = getattr(model, "mesh_info", {}).get("active")
    if not info:
        return model
    n_units = int(np.sum(model.per_direction[:info["n_active_dirs"]]))
    return SimpleNamespace(directions=model.directions[:n_units], centers=model.centers[:n_units],
                           gammas=model.gammas[:n_units], weights=model.weights[:n_units])


def build_rung(rung: str, task, X, y, budget: int, s: float, even_model=None):
    """Return ``(model, n_solves)`` or ``(None, 0)`` when the rung does not apply."""
    d = task.d
    if rung == "even":
        return EvenGeometry(d=d, budget=budget).fit(X, y), 1
    geo = AdaptiveGeometry(d=d, budget=budget, s=s, name=rung)
    V = geo.unique_directions
    solves = 0

    def need_even():
        nonlocal solves
        if even_model is None:
            raise RuntimeError("even fit required")
        solves += 1
        return even_model

    if rung == "data_p13":
        return geo.build(X, Monitors("data", beta=1.0 / 3.0)).fit(X, y), 1
    if rung == "data_p1":
        return geo.build(X, Monitors("data", beta=1.0)).fit(X, y), 1
    if rung in ("oracle_r1", "oracle_r2"):
        if not task.differentiable:
            return None, 0
        r = int(rung[-1])
        D = oracle_derivatives(task, X, V, r)
        return geo.build(X, Monitors("roughness", r=r, deriv=D)).fit(X, y), 1
    if rung in ("surr_r1", "surr_r2"):
        r = int(rung[-1])
        D = surrogate_derivatives(need_even(), X, V, r)
        return geo.build(X, Monitors("roughness", r=r, deriv=D)).fit(X, y), solves + 1
    if rung == "residual":
        e = need_even().predict(X) - y
        return geo.build(X, Monitors("residual", resid=e)).fit(X, y), solves + 1
    if rung == "freq_oracle":
        if not task.differentiable:
            return None, 0
        D1 = oracle_derivatives(task, X, V, 1)
        D2 = oracle_derivatives(task, X, V, 2)
        return geo.build(X, Monitors("frequency", deriv=D1, deriv2=D2)).fit(X, y), 1
    if rung == "freq":
        ev = need_even()
        D1 = surrogate_derivatives(ev, X, V, 1)
        D2 = surrogate_derivatives(ev, X, V, 2)
        return geo.build(X, Monitors("frequency", deriv=D1, deriv2=D2)).fit(X, y), solves + 1
    if rung in ("active_oracle", "active", "active_x3"):
        if rung == "active_oracle":
            if not task.differentiable:
                return None, 0
            G = task.grad_F(X)
            evals, W = gradient_covariance(G)
            m = active_dimension(evals)
            geo = active_subspace_geometry(d, budget, W, m, s=s, name=rung)
            geo.mesh_info["eigenvalues"] = (evals / evals[0]).tolist()
            D = oracle_derivatives(task, X, geo.unique_directions, 1)
            return geo.build(X, Monitors("roughness", r=1, deriv=D)).fit(X, y), 1
        model = need_even()
        n_iter = 3 if rung == "active_x3" else 1
        for _ in range(n_iter):
            G = surrogate_derivatives(_active_part(model), X, np.eye(d), 1).T
            evals, W = gradient_covariance(G)
            m = active_dimension(evals)
            geo = active_subspace_geometry(d, budget, W, m, s=s, name=rung)
            geo.mesh_info["eigenvalues"] = (evals / evals[0]).tolist()
            D = surrogate_derivatives(model, X, geo.unique_directions, 1)
            model = geo.build(X, Monitors("roughness", r=1, deriv=D)).fit(X, y)
            solves += 1
        return model, solves
    if rung == "surr_r1_x3":
        model = need_even()
        for _ in range(2):
            D = surrogate_derivatives(model, X, V, 1)
            model = AdaptiveGeometry(d=d, budget=budget, s=s, name=rung).build(
                X, Monitors("roughness", r=1, deriv=D)).fit(X, y)
            solves += 1
        return model, solves
    if d != 2:
        return None, 0
    if rung == "dir_oracle":
        if not task.differentiable:
            return None, 0
        geo.set_directions(X, _dir_monitor("oracle", task, X))
        return geo.build(X, Monitors("even")).fit(X, y), 1
    if rung == "dir_surr":
        geo.set_directions(X, _dir_monitor("surr", task, X, need_even()))
        return geo.build(X, Monitors("even")).fit(X, y), solves + 1
    if rung in ("both_surr", "joint_surr"):
        ev = need_even()
        geo.set_directions(X, _dir_monitor("surr", task, X, ev))
        D = surrogate_derivatives(ev, X, geo.unique_directions, 1)
        mon = Monitors("roughness", r=1, deriv=D)
        if rung == "joint_surr":
            geo.set_counts(_monitor_mass(geo, X, mon))
        return geo.build(X, mon).fit(X, y), solves + 1
    raise KeyError(rung)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run_cell(task, budget, rungs, s=FLOOR_S, keep_geometry=False):
    n_train = int(round(RATIO * budget))
    X, y = task.train_set(n_train, seed=TRAIN_SEED)
    sets = task.test_sets(seed=TEST_SEED)
    y_true = {k: task.F(sets[k]) for k in TEST_SETS}
    rows, geoms = [], {}
    even = None
    for rung in rungs:
        t0 = time.time()
        model, solves = build_rung(rung, task, X, y, budget, s, even_model=even)
        if model is None:
            continue
        if rung == "even":
            even = model
        errors = {k: error_metrics(model.predict(sets[k]), y_true[k]) for k in TEST_SETS}
        mesh_info = getattr(model, "mesh_info", {})
        rec = {"task": task.id, "name": task.name, "d": task.d, "data": task.density_tag,
               "rung": rung, "budget": budget, "n_train": n_train, "s": s,
               "solves": solves, "rank": model.info.get("rank"),
               "max_neighbor_ratio": mesh_info.get("max_neighbor_ratio", 1.0),
               "min_spacing_over_even": mesh_info.get("min_spacing_over_even", 1.0),
               "max_spacing_over_even": mesh_info.get("max_spacing_over_even", 1.0),
               "active": mesh_info.get("active"), "eigenvalues": mesh_info.get("eigenvalues"),
               "seconds": time.time() - t0, "errors": errors}
        rows.append(rec)
        if keep_geometry:
            geoms[rung] = model
        print(f"  {task.id:5s} B={budget:5d} {rung:11s} "
              f"same={errors['same_as_train']['rel_l2']:.1e} "
              f"unif={errors['uniform']['rel_l2']:.1e} "
              f"dense={errors['dense_region']['rel_l2']:.1e} "
              f"ratio={rec['max_neighbor_ratio']:.2f} {rec['seconds']:.0f}s", flush=True)
    return (rows, geoms, X, y) if keep_geometry else rows


def run_ladder(task_ids, budgets, out_path, rungs=None):
    rows = []
    t0 = time.time()
    for tid in task_ids:
        task = get_task(tid)
        if rungs is None:
            rungs = CENTER_RUNGS + (DIRECTION_RUNGS if task.d == 2 else [])
        elif "even" not in rungs:
            rungs = ["even"] + list(rungs)
        for B in budgets or BUDGETS[task.d]:
            rows += run_cell(task, B, rungs)
        with open(out_path, "w") as f:
            json.dump(_jsonable({"rows": rows}), f)
    print(f"saved {out_path} ({time.time() - t0:.0f}s)")
    return rows


SPLIT_TASKS = ["3.3", "3.7", "3.11", "3.12", "3.13", "3.16"]
SPLIT_NPER = [8, 12, 16, 24, 32, 48, 64]


def run_split(out_path, budget=4096, tasks=None):
    """d = 3: even mesh, but vary how the budget is split between directions and
    centers per direction (the reference uses B^(1/3) = 16 per direction)."""
    rows = []
    for tid in tasks or SPLIT_TASKS:
        task = get_task(tid)
        n_train = int(round(RATIO * budget))
        X, y = task.train_set(n_train, seed=TRAIN_SEED)
        sets = task.test_sets(seed=TEST_SEED)
        y_true = {k: task.F(sets[k]) for k in TEST_SETS}
        for n_per in SPLIT_NPER:
            t0 = time.time()
            geo = AdaptiveGeometry(d=task.d, budget=budget, s=0.0, name="split",
                                   n_per_override=n_per)
            model = geo.build(X, Monitors("even")).fit(X, y)
            errors = {k: error_metrics(model.predict(sets[k]), y_true[k]) for k in TEST_SETS}
            rows.append({"task": tid, "name": task.name, "d": task.d, "budget": budget,
                         "n_per": n_per, "n_dir": geo.n_directions, "rank": model.info["rank"],
                         "seconds": time.time() - t0, "errors": errors})
            print(f"  {tid:5s} B={budget} n_per={n_per:3d} n_dir={geo.n_directions:4d} "
                  f"same={errors['same_as_train']['rel_l2']:.1e} "
                  f"unif={errors['uniform']['rel_l2']:.1e} "
                  f"dense={errors['dense_region']['rel_l2']:.1e} {time.time() - t0:.0f}s",
                  flush=True)
            with open(out_path, "w") as f:
                json.dump(_jsonable({"rows": rows}), f)
    return rows


def run_floor(out_path):
    rows = []
    for tid in FLOOR_TASKS:
        task = get_task(tid)
        for B in BUDGETS[1]:
            for s in FLOOR_LEVELS:
                rows += run_cell(task, B, ["even", "surr_r1", "data_p1"], s=s)
    with open(out_path, "w") as f:
        json.dump(_jsonable({"rows": rows}), f)
    return rows


def load_rows(pattern="ladder_*.json"):
    rows = []
    for p in sorted(glob.glob(str(RESULTS_DIR / pattern))):
        with open(p) as f:
            rows += json.load(f)["rows"]
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=str, default=None, help="comma-separated: 1,2")
    ap.add_argument("--tasks", type=str, default=None, help="comma-separated task ids")
    ap.add_argument("--budgets", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None, help="output file tag")
    ap.add_argument("--rungs", type=str, default=None,
                    help="comma-separated rungs to run (even is always added first)")
    ap.add_argument("--floor", action="store_true", help="sweep the floor s in 1-D")
    ap.add_argument("--split", action="store_true",
                    help="d = 3: sweep the directions/centers split of the even mesh")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else None
    if args.tasks or args.dims:
        ids = args.tasks.split(",") if args.tasks else sum(
            (TASKS[int(d)] for d in args.dims.split(",")), [])
        tag = args.tag or ("d" + (args.dims or "x").replace(",", ""))
        rungs = args.rungs.split(",") if args.rungs else None
        run_ladder(ids, budgets, RESULTS_DIR / f"ladder_{tag}.json", rungs)
    if args.floor:
        run_floor(RESULTS_DIR / "floor.json")
    if args.split:
        run_split(RESULTS_DIR / "split_d3.json", budget=int(args.budgets or 4096),
                  tasks=args.tasks.split(",") if args.tasks else None)
    if args.plot or args.tasks or args.dims or args.floor or args.split:
        import viz
        viz.all_figures(RESULTS_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
