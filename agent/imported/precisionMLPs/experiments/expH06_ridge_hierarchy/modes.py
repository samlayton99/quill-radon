"""The --ridges and --grow modes of expH06 (kept out of run.py for size)."""

from __future__ import annotations

import json
import time

import numpy as np

from h06.core import (Geometry, make_block, nested_directions, fibonacci_directions,
                      fit_geometry, rel_l2, max_abs, ball, origin, RCOND as RCOND_DEFAULT)
from h06.targets import get_target
from h06.atoms import projection_pursuit, varpro_polish
from h06.grow import Grower
import run as R


# ---------------------------------------------------------------------------
# --ridges: hidden-ridge recovery, d = 3 and 4
# ---------------------------------------------------------------------------

RIDGE_DIMS = [3, 4]
RIDGE_COUNTS = [1, 2, 4, 8]
RIDGE_R = 0.3
RIDGE_N_ATOM = 48
RIDGE_N_TRAIN = 12000
RIDGE_SEEDS = [0, 1, 2]
RIDGES_JSON = R.RESULTS_DIR / "ridges.json"


def recover_ridges(d, r_count, seed, bump=False, n_train=RIDGE_N_TRAIN, n_atom=RIDGE_N_ATOM, verbose=True):
    """Forward stagewise projection pursuit (one atom at a time on the residual) followed by
    a joint Gauss-Newton polish; returns errors and the direction errors in radians."""
    tgt = get_target(f"ridge{r_count}" + ("_bump" if bump else ""), d)
    x0 = origin(d)
    Ztr = ball(n_train, d, RIDGE_R, np.random.default_rng(seed))
    Zte = ball(R.N_TEST, d, R.TEST_SHRINK * RIDGE_R, np.random.default_rng(seed + 100))
    ytr, yte = tgt(x0 + Ztr), tgt(x0 + Zte)
    t0 = time.time()
    found, res, stage = [], ytr.copy(), []
    idx = np.random.default_rng(seed).choice(n_train, size=min(6000, n_train), replace=False)
    for i in range(r_count):
        v, sc, sp = projection_pursuit(Ztr[idx], res[idx], n_off=32, seed=seed)
        found.append(v)
        g = Geometry([make_block(u, Ztr, n_atom, "atom") for u in found])
        fit = fit_geometry(g, Ztr, ytr)
        res = ytr - g.augmented(Ztr) @ fit.coef
        stage.append({"atom": i, "pp_coarse": sc, "pp_polished": sp,
                      "train_rel": float(np.linalg.norm(res) / np.linalg.norm(ytr))})
    g, hist = varpro_polish(g, Ztr, ytr, which=list(range(r_count)), iters=20)
    # if the joint polish did not reach the floor, keep pulling atoms from the residual
    # (a stuck direction is left in place; an extra atom is harmless) up to 2r atoms
    extra = 0
    while hist[-1]["rel_residual"] > 1e-10 and len(found) < 2 * r_count:
        fit = fit_geometry(g, Ztr, ytr)
        res = ytr - g.augmented(Ztr) @ fit.coef
        v, sc, sp = projection_pursuit(Ztr[idx], res[idx], n_off=32, seed=seed + 7 * len(found))
        found.append(v)
        g.blocks.append(make_block(v, Ztr, n_atom, "atom"))
        g, hist = varpro_polish(g, Ztr, ytr, which=list(range(len(found))), iters=20)
        extra += 1
    fit = fit_geometry(g, Ztr, ytr)
    pred = fit.predict(g, Zte)
    dir_err = []
    for b in g.blocks:
        dir_err.append(float(min(np.arccos(min(1.0, abs(float(b.v @ u)))) for u in tgt.U)))
    rec = {"d": d, "r": r_count, "bump": bump, "seed": seed, "n_atom": n_atom, "units": g.units,
           "test_rel_l2": rel_l2(pred, yte), "test_max_abs": max_abs(pred, yte),
           "train_rel_after_polish": hist[-1]["rel_residual"], "polish_iters": len(hist) - 1, "extra_atoms": extra,
           "direction_errors_rad": dir_err, "stages": stage, "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"  d={d} r={r_count} bump={bump} seed={seed}: test={rec['test_rel_l2']:.2e} "
              f"max_dir_err={max(dir_err):.1e} rad extra_atoms={extra} [{rec['seconds']:.0f}s]", flush=True)
    return rec


def ridges(args):
    out = []
    for d in RIDGE_DIMS:
        for rc in RIDGE_COUNTS:
            for seed in RIDGE_SEEDS:
                out.append(recover_ridges(d, rc, seed))
                RIDGES_JSON.write_text(json.dumps(out, indent=1))
    print("saved", RIDGES_JSON)


# ---------------------------------------------------------------------------
# --grow: the hierarchy vs the even reference
# ---------------------------------------------------------------------------

GROW_D = 3
GROW_R = 0.3
GROW_BUDGET = 4096
GROW_N_TRAIN = 24576
GROW_TARGETS = ["ridge2", "ridge4", "ridge4_bump", "product_sines", "gauss_bump",
                "composition", "radial_runge", "fast_waves", "spatial_packet"]
EVEN_BUDGETS = [128, 256, 512, 1024, 2048, 4096]
EVEN_NPER = [8, 16, 32]
GROW_JSON = R.RESULTS_DIR / "grow_d3.json"


def even_reference(d, Ztr, Ytr, Zte, Yte, keys, budgets=EVEN_BUDGETS, npers=EVEN_NPER, verbose=True):
    """Nested-direction even mesh at every budget and several splits (multi-RHS)."""
    Vseq = nested_directions(d, max(budgets) // min(npers) + 1)
    rows = []
    for B in budgets:
        for N in npers:
            M = B // N
            if M < 2:
                continue
            g = Geometry([make_block(v, Ztr, N) for v in Vseq[:M]])
            t0 = time.time()
            fit = fit_geometry(g, Ztr, Ytr)
            pred = fit.predict(g, Zte)
            rec = {"budget": B, "M": M, "N": N, "units": g.units, "rank": fit.rank,
                   "rel_l2": {k: rel_l2(pred[:, i], Yte[:, i]) for i, k in enumerate(keys)},
                   "seconds": round(time.time() - t0, 1)}
            rows.append(rec)
            if verbose:
                print(f"  even B={B:5d} M={M:4d} N={N:3d} [{rec['seconds']:5.1f}s] "
                      + " ".join(f"{k[:6]}={rec['rel_l2'][k]:.1e}" for k in keys), flush=True)
    return rows


def grow(args):
    d, r = (args.dim or GROW_D), GROW_R
    keys = args.targets.split(",") if args.targets else GROW_TARGETS
    json_path = R.RESULTS_DIR / (f"grow_d{d}_{args.tag}.json" if args.tag else f"grow_d{d}.json")
    x0 = origin(d)
    Zall = ball(GROW_N_TRAIN, d, r, np.random.default_rng(R.SEED_TRAIN))
    Zte = ball(R.N_TEST, d, R.TEST_SHRINK * r, np.random.default_rng(R.SEED_TEST))
    fns = {k: get_target(k, d) for k in keys}
    Yall = np.stack([fns[k](x0 + Zall) for k in keys], axis=1)
    Yte = np.stack([fns[k](x0 + Zte) for k in keys], axis=1)
    n_val = GROW_N_TRAIN // 8
    Zval, Yval, Ztr, Ytr = Zall[:n_val], Yall[:n_val], Zall[n_val:], Yall[n_val:]
    out = {"d": d, "r": r, "budget": (args.budget or GROW_BUDGET), "targets": keys, "n_train": len(Ztr), "n_val": n_val,
           "even": None, "grow": {}}
    print("even reference")
    out["even"] = even_reference(d, Ztr, Ytr, Zte, Yte, keys)
    json_path.write_text(json.dumps(out, indent=1))
    for i, k in enumerate(keys):
        print(f"grow: {k}")
        gr = Grower(d, Ztr, Ytr[:, i], Zval, Yval[:, i], budget=(args.budget or GROW_BUDGET))
        geom, fit, hist = gr.run()
        # test error along the trajectory is recorded at the end only for the final geometry;
        # re-evaluate the test error for every recorded round from the saved geometries is
        # expensive, so store the final one and the validation trajectory.
        pred = fit.predict(geom, Zte)
        out["grow"][k] = {"history": hist, "final_test_rel_l2": rel_l2(pred, Yte[:, i]),
                          "final_test_max_abs": max_abs(pred, Yte[:, i]), "seconds": round(gr.total_seconds, 1),
                          "final": geom.describe(),
                          "atom_directions": [b.v.tolist() for b in geom.blocks if b.kind == "atom"]}
        print(f"  final: units={geom.units} dirs={geom.n_dir} atoms={geom.describe()['n_atoms']} "
              f"test={out['grow'][k]['final_test_rel_l2']:.2e} [{gr.total_seconds:.0f}s]", flush=True)
        json_path.write_text(json.dumps(out, indent=1))
    print("saved", json_path)


# ---------------------------------------------------------------------------
# --push: 3-D to the floor along the law-chosen cells (directions bind)
# ---------------------------------------------------------------------------

PUSH_CELLS = [(256, 16), (320, 24), (384, 20), (512, 16), (256, 32)]
PUSH_JSON = R.RESULTS_DIR / "push_d3.json"


def push(args):
    d, r, keys = R.FLOORS_D, R.FLOORS_R, R.FLOORS_TARGETS
    n_train = args.rows or R.FLOORS_N_TRAIN
    Ztr, Ytr, Zte, Yte = R.data_sets(d, r, n_train, keys)
    cells = [tuple(int(x) for x in c.split("x")) for c in args.cells.split(",")] if args.cells else PUSH_CELLS
    json_path = R.RESULTS_DIR / (f"push_d3_{args.tag}.json" if args.tag else "push_d3.json")
    Vseq = nested_directions(d, max(M for M, _ in cells))
    out = {"d": d, "r": r, "targets": keys, "n_train": n_train, "cells": []}
    for M, N in cells:
        out["cells"].append(R._cell(Vseq[:M], N, Ztr, Ytr, Zte, Yte, keys, "push"))
        json_path.write_text(json.dumps(out, indent=1))
    print("saved", json_path)


# ---------------------------------------------------------------------------
# --polish: learn ALL directions of a law-chosen even mesh by Gauss-Newton
# ---------------------------------------------------------------------------

POLISH_D = 3
POLISH_R = 0.3
POLISH_TARGETS = ["radial_runge", "fast_waves", "spatial_packet", "composition", "gauss_bump", "product_sines"]
POLISH_CELLS = [(256, 16), (128, 32)]
POLISH_N_TRAIN = 20480
POLISH_ITERS = 12
POLISH_JSON = R.RESULTS_DIR / "polish_d3.json"


def polish(args):
    """Start from an even nested-direction mesh at ``(M, N)``; run the variable-projection
    Gauss-Newton polish on every direction (``M (d-1)`` parameters); record the test error
    before and after, per target. Nothing else changes: same blocks, same readout solve."""
    d, r = (args.dim or POLISH_D), POLISH_R
    keys = args.targets.split(",") if args.targets else POLISH_TARGETS
    cells = [tuple(int(x) for x in c.split("x")) for c in args.cells.split(",")] if args.cells else POLISH_CELLS
    json_path = R.RESULTS_DIR / (f"polish_d{d}_{args.tag}.json" if args.tag else f"polish_d{d}.json")
    x0 = origin(d)
    rc = args.rcond or RCOND_DEFAULT
    iters = args.iters or POLISH_ITERS
    Ztr = ball(args.rows or POLISH_N_TRAIN, d, r, np.random.default_rng(R.SEED_TRAIN))
    Zte = ball(R.N_TEST, d, R.TEST_SHRINK * r, np.random.default_rng(R.SEED_TEST))
    out = {"d": d, "r": r, "n_train": len(Ztr), "iters": iters, "rcond": rc, "rows": []}
    for M, N in cells:
        Vseq = nested_directions(d, M)
        for k in keys:
            f = get_target(k, d)
            ytr, yte = f(x0 + Ztr), f(x0 + Zte)
            g0 = Geometry([make_block(v, Ztr, N) for v in Vseq])
            fit0 = fit_geometry(g0, Ztr, ytr, rcond=rc)
            e0 = rel_l2(fit0.predict(g0, Zte), yte)
            t0 = time.time()
            g1, hist = varpro_polish(g0, Ztr, ytr, which=list(range(M)), iters=iters, tol=1e-3, verbose=True, rcond=rc)
            fit1 = fit_geometry(g1, Ztr, ytr, rcond=rc)
            e1 = rel_l2(fit1.predict(g1, Zte), yte)
            moved = [float(np.degrees(np.arccos(min(1.0, abs(float(a.v @ b.v)))))) for a, b in zip(g0.blocks, g1.blocks)]
            row = {"M": M, "N": N, "units": g0.units, "target": k, "rcond": rc, "test_before": e0, "test_after": e1,
                   "train_rel_hist": [h["rel_residual"] for h in hist], "max_move_deg": max(moved),
                   "median_move_deg": float(np.median(moved)), "seconds": round(time.time() - t0, 1)}
            out["rows"].append(row)
            json_path.write_text(json.dumps(out, indent=1))
            print(f"  polish M={M} N={N} {k:14s} before={e0:.2e} after={e1:.2e} "
                  f"train {hist[0]['rel_residual']:.1e}->{hist[-1]['rel_residual']:.1e} "
                  f"moved max {max(moved):.2f} deg [{row['seconds']:.0f}s]", flush=True)
    print("saved", json_path)


# ---------------------------------------------------------------------------
# --polish-grown: learn ALL directions of the grown meshes
# ---------------------------------------------------------------------------

POLISH_GROWN_ITERS = 10
POLISH_GROWN_SKIP_BELOW = 3e-13


def rebuild_grown(d, rec, Ztr, seed=0):
    """The final geometry of a grow run from its record: blocks in insertion order, background
    directions from the nested sequence in order, atoms from the saved directions."""
    Vseq = nested_directions(d, 2048 if d <= 3 else 4096, seed=seed)
    atoms = iter(rec["atom_directions"])
    blocks, i_bg = [], 0
    for n, kind in zip(rec["final"]["n_per"], rec["final"]["kinds"]):
        if kind == "bg":
            v = Vseq[i_bg]; i_bg += 1
        else:
            v = np.asarray(next(atoms))
        blocks.append(make_block(v, Ztr, n, kind=kind))
    return Geometry(blocks)


def polish_grown(args):
    d = args.dim or GROW_D
    r = GROW_R
    paths = sorted(R.RESULTS_DIR.glob(f"grow_d{d}_*.json"))
    runs = {}
    for p in paths:
        runs.update(json.load(open(p))["grow"])
    keys = args.targets.split(",") if args.targets else list(runs)
    json_path = R.RESULTS_DIR / (f"polish_grown_d{d}_{args.tag}.json" if args.tag else f"polish_grown_d{d}.json")
    x0 = origin(d)
    Zall = ball(GROW_N_TRAIN, d, r, np.random.default_rng(R.SEED_TRAIN))
    Zte = ball(R.N_TEST, d, R.TEST_SHRINK * r, np.random.default_rng(R.SEED_TEST))
    n_val = GROW_N_TRAIN // 8
    Ztr = Zall[n_val:]
    out = {"d": d, "r": r, "iters": POLISH_GROWN_ITERS, "rows": []}
    for k in keys:
        rec = runs[k]
        if rec["final_test_rel_l2"] < POLISH_GROWN_SKIP_BELOW:
            print(f"  {k}: already at {rec['final_test_rel_l2']:.1e}, skipped", flush=True)
            continue
        f = get_target(k, d)
        ytr, yte = f(x0 + Ztr), f(x0 + Zte)
        g0 = rebuild_grown(d, rec, Ztr)
        fit0 = fit_geometry(g0, Ztr, ytr)
        e0 = rel_l2(fit0.predict(g0, Zte), yte)
        t0 = time.time()
        g1, hist = varpro_polish(g0, Ztr, ytr, which=list(range(g0.n_dir)), iters=POLISH_GROWN_ITERS, tol=1e-3)
        fit1 = fit_geometry(g1, Ztr, ytr)
        e1 = rel_l2(fit1.predict(g1, Zte), yte)
        moved = [float(np.degrees(np.arccos(min(1.0, abs(float(a.v @ b.v)))))) for a, b in zip(g0.blocks, g1.blocks)]
        row = {"target": k, "units": g0.units, "n_dir": g0.n_dir, "n_atoms": g0.describe()["n_atoms"],
               "recorded_final": rec["final_test_rel_l2"], "test_before": e0, "test_after": e1,
               "train_rel_hist": [h["rel_residual"] for h in hist], "max_move_deg": max(moved),
               "median_move_deg": float(np.median(moved)), "seconds": round(time.time() - t0, 1),
               "directions_after": [b.v.tolist() for b in g1.blocks]}
        out["rows"].append(row)
        json_path.write_text(json.dumps(out, indent=1))
        print(f"  polish-grown {k:14s} units={g0.units} dirs={g0.n_dir} before={e0:.2e} (recorded {rec['final_test_rel_l2']:.1e}) "
              f"after={e1:.2e} train {hist[0]['rel_residual']:.1e}->{hist[-1]['rel_residual']:.1e} "
              f"moved max {max(moved):.2f} med {np.median(moved):.2f} deg [{row['seconds']:.0f}s]", flush=True)
    print("saved", json_path)


# ---------------------------------------------------------------------------
# --rcond-scan: the truncation threshold at large widths (one factorization, many rcond)
# ---------------------------------------------------------------------------

RCONDS = [1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
RCOND_CELL_JSONS = {}
RCOND_JSON = R.RESULTS_DIR / "rcond_scan_d3.json"


def rcond_scan(args):
    import scipy.linalg as sla
    d, r, keys = R.FLOORS_D, R.FLOORS_R, R.FLOORS_TARGETS
    n_train = args.rows or R.FLOORS_N_TRAIN
    Ztr, Ytr, Zte, Yte = R.data_sets(d, r, n_train, keys)
    cells = [tuple(int(x) for x in c.split("x")) for c in args.cells.split(",")]
    out = {"d": d, "r": r, "targets": keys, "n_train": n_train, "rconds": RCONDS, "rows": []}
    for M, N in cells:
        t0 = time.time()
        g = R.even_geometry(nested_directions(d, M), Ztr, N)
        A = g.augmented(Ztr)
        Q, Rm = sla.qr(A, mode="economic", overwrite_a=True, check_finite=False)
        del A
        Ur, s, Vt = np.linalg.svd(Rm, full_matrices=False)
        del Rm
        QtY = Ur.T @ (Q.T @ Ytr)
        del Q
        Ate = g.augmented(Zte)
        for rc in RCONDS:
            keep = s > rc * s[0]
            s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
            coef = Vt.T @ (s_inv[:, None] * QtY)
            pred = Ate @ coef
            row = {"M": M, "N": N, "units": g.units, "rcond": rc, "rank": int(keep.sum()),
                   "s_max": float(s[0]), "s_min": float(s[-1]),
                   "rel_l2": {k: rel_l2(pred[:, i], Yte[:, i]) for i, k in enumerate(keys)},
                   "weight_norm": {k: float(np.linalg.norm(coef[:-1, i])) for i, k in enumerate(keys)}}
            out["rows"].append(row)
            print(f"  M={M} N={N} rcond={rc:.0e} rank={row['rank']:5d} s_max={s[0]:.1e} "
                  + " ".join(f"{k[:5]}={row['rel_l2'][k]:.1e}" for k in keys), flush=True)
        print(f"  [{time.time() - t0:.0f}s]", flush=True)
        RCOND_JSON.write_text(json.dumps(out, indent=1))
    print("saved", RCOND_JSON)


# ---------------------------------------------------------------------------
# --alloc: equal-budget allocation shootout (even / +atoms / +spikes / waterfill)
# ---------------------------------------------------------------------------

ALLOC_D = 3
ALLOC_R = 0.3
ALLOC_TARGETS = ["spatial_packet", "fast_waves", "off_packet"]
ALLOC_BUDGETS = [2048, 4096]
ALLOC_NPERS = [8, 16, 24]
ALLOC_RCOND = 1e-14
ALLOC_N_TRAIN = 24576
ALLOC_BLOCK_N = 32          # offsets in each added atom or spike
ALLOC_BASE_FRAC = 0.75      # the even base for the +atoms / +spikes arms
ALLOC_JSON = R.RESULTS_DIR / "alloc_d3.json"


def _even_best(d, Ztr, ytr, Zte, yte, B, rcond):
    """Best balanced even mesh at budget B over the ALLOC_NPERS splits."""
    best = None
    for N in ALLOC_NPERS:
        M = B // N
        g = Geometry([make_block(v, Ztr, N) for v in nested_directions(d, M)])
        fit = fit_geometry(g, Ztr, ytr, rcond=rcond)
        e = rel_l2(fit.predict(g, Zte), yte)
        if best is None or e < best[0]:
            best = (e, M, N, g, fit)
    return best


def _add_blocks(g0, kind, d, Ztr, ytr, Zte, yte, budget, rcond, seed=0, polish_last=12):
    """Fill the remaining budget one block at a time -- each aimed by a fresh scan of the
    updated residual (the sequential adaptivity is what earns the gain; a batched scan of
    one residual was measured to earn nothing). Atoms get the GN direction polish (capped
    to the last ``polish_last`` atoms); spikes are graded by the residual profile."""
    from h06.spikes import best_spike
    g = g0.copy()
    fit = fit_geometry(g, Ztr, ytr, rcond=rcond)
    rng = np.random.default_rng(seed)
    while g.units + ALLOC_BLOCK_N <= budget:
        resid = ytr - g.augmented(Ztr) @ fit.coef
        if kind == "spike":
            blk, sc = best_spike(Ztr, resid, ALLOC_BLOCK_N, rcond=rcond, seed=seed)
            g.blocks.append(blk)
            fit = fit_geometry(g, Ztr, ytr, rcond=rcond)
        else:
            idx = rng.choice(len(Ztr), size=min(6000, len(Ztr)), replace=False)
            v, _, _ = projection_pursuit(Ztr[idx], resid[idx], n_off=ALLOC_BLOCK_N, rcond=rcond, seed=seed)
            g.blocks.append(make_block(v, Ztr, ALLOC_BLOCK_N, kind="atom"))
            which = [i for i, b in enumerate(g.blocks) if b.kind == "atom"][-polish_last:]
            sub = rng.choice(len(Ztr), size=min(max(6000, 4 * g.units), len(Ztr)), replace=False)
            g, _ = varpro_polish(g, Ztr[sub], ytr[sub], which=which, iters=3, rcond=rcond)
            fit = fit_geometry(g, Ztr, ytr, rcond=rcond)
    e = rel_l2(fit.predict(g, Zte), yte)
    return e, g


def _waterfill(d, Ztr, ytr, Zte, yte, B, N0, M, rcond):
    """One reallocation round: per-block demand = roughness of the fitted block profile
    (norm of second differences of its readout coefficients); N_j proportional to demand,
    floored at 6, at the same total."""
    g = Geometry([make_block(v, Ztr, N0) for v in nested_directions(d, M)])
    fit = fit_geometry(g, Ztr, ytr, rcond=rcond)
    sl = g.block_slices()
    demand = np.array([float(np.linalg.norm(np.diff(fit.coef[s], n=2))) + 1e-30 for s in sl])
    total = N0 * M
    Nj = np.maximum(6, np.round(demand / demand.sum() * total).astype(int))
    while Nj.sum() > total:
        Nj[np.argmax(Nj)] -= 1
    while Nj.sum() < total:
        Nj[np.argmin(Nj)] += 1
    g2 = Geometry([make_block(b.v, Ztr, int(n)) for b, n in zip(g.blocks, Nj)])
    fit2 = fit_geometry(g2, Ztr, ytr, rcond=rcond)
    e2 = rel_l2(fit2.predict(g2, Zte), yte)
    return e2, {"N_min": int(Nj.min()), "N_max": int(Nj.max()), "N_med": float(np.median(Nj))}


def alloc(args):
    d, r = ALLOC_D, ALLOC_R
    keys = args.targets.split(",") if args.targets else ALLOC_TARGETS
    rcond = args.rcond or ALLOC_RCOND
    x0 = origin(d)
    Ztr = ball(args.rows or ALLOC_N_TRAIN, d, r, np.random.default_rng(R.SEED_TRAIN))
    Zte = ball(R.N_TEST, d, R.TEST_SHRINK * r, np.random.default_rng(R.SEED_TEST))
    out = {"d": d, "r": r, "rcond": rcond, "n_train": len(Ztr), "rows": []}
    json_path = R.RESULTS_DIR / (f"alloc_d3_{args.tag}.json" if args.tag else "alloc_d3.json")
    for k in keys:
        f = get_target(k, d)
        ytr, yte = f(x0 + Ztr), f(x0 + Zte)
        for B in ALLOC_BUDGETS:
            t0 = time.time()
            e_even, M, N, g_even, _ = _even_best(d, Ztr, ytr, Zte, yte, B, rcond)
            base_B = int(ALLOC_BASE_FRAC * B)
            _, Mb, Nb, g_base, _ = _even_best(d, Ztr, ytr, Zte, yte, base_B, rcond)
            e_atoms, g_a = _add_blocks(g_base, "atom", d, Ztr, ytr, Zte, yte, B, rcond)
            e_spikes, g_s = _add_blocks(g_base, "spike", d, Ztr, ytr, Zte, yte, B, rcond)
            e_wf, wf_info = _waterfill(d, Ztr, ytr, Zte, yte, B, N, M, rcond)
            row = {"target": k, "budget": B, "even": e_even, "even_M": M, "even_N": N,
                   "base": f"{Mb}x{Nb}", "atoms": e_atoms, "n_atoms": g_a.describe()["n_atoms"],
                   "spikes": e_spikes, "n_spikes": sum(b.kind == "spike" for b in g_s.blocks),
                   "waterfill": e_wf, "wf": wf_info, "seconds": round(time.time() - t0, 1)}
            out["rows"].append(row)
            json_path.write_text(json.dumps(out, indent=1))
            print(f"  {k:14s} B={B:5d} even={e_even:.1e}({M}x{N}) atoms={e_atoms:.1e} "
                  f"spikes={e_spikes:.1e} waterfill={e_wf:.1e} (N {wf_info['N_min']}..{wf_info['N_max']}) "
                  f"[{row['seconds']:.0f}s]", flush=True)
    print("saved", json_path)
