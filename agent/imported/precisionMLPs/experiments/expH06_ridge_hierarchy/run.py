"""expH06: the hierarchical ridge mesh -- nested background directions, atoms found from
the residual, and refine-vs-open arbitration by the two-floor law; pushed to d = 3, 4.

Modes:
    --floors   d = 3 on a data ball: the two floor curves e_M(M) (offsets held generous) and
               e_N(N) (directions held generous), the max-of-two-floors test on exact
               cells, and the predicted split/budget for the floor.
    --ridges   hidden-ridge recovery: sum of r = 1, 2, 4, 8 ridges in d = 3, 4; projection
               pursuit + Gauss-Newton polish of the directions; error vs r.
    --grow     the greedy hierarchy on a set of targets vs the even reference at equal
               budgets; the trajectory error-vs-units with the actions annotated.
    --plot     redraw figures from the JSONs.

Usage:
    OMP_NUM_THREADS=8 uv run --extra dev python experiments/expH06_ridge_hierarchy/run.py --floors
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h06.core import (Geometry, make_block, nested_directions, fibonacci_directions,   # noqa: E402
                      fit_geometry, solve_augmented, rel_l2, max_abs, ball, origin, RCOND)
from h06.targets import get_target                                                   # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH06_ridge_hierarchy"
FIG_DIR = RESULTS_DIR / "figures"

N_TEST = 20000
TEST_SHRINK = 0.9
SEED_TRAIN, SEED_TEST = 0, 1

# --- floors (d = 3) --------------------------------------------------------
FLOORS_D = 3
FLOORS_R = 0.3
FLOORS_TARGETS = ["fast_waves", "radial_runge", "composition", "gauss_bump", "product_sines", "spatial_packet"]
FLOORS_EM_N = 48                                   # offsets held fixed while M moves
FLOORS_EM_M = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160]
FLOORS_EN_M = 128                                  # directions held fixed while N moves
FLOORS_EN_N = [4, 6, 8, 12, 16, 24, 32, 48, 64]
FLOORS_EXACT = [(32, 16), (64, 16), (32, 32), (64, 32), (16, 64), (96, 24), (24, 48), (48, 8)]
FLOORS_N_TRAIN = 32768
FLOORS_JSON = RESULTS_DIR / "floors_d3.json"


def data_sets(d, r, n_train, targets, seed_train=SEED_TRAIN, seed_test=SEED_TEST):
    """Centered training/test points on the ball and the target values (absolute coords
    are ``x0 + Z``)."""
    x0 = origin(d)
    Ztr = ball(n_train, d, r, np.random.default_rng(seed_train))
    Zte = ball(N_TEST, d, TEST_SHRINK * r, np.random.default_rng(seed_test))
    fns = [get_target(k, d) for k in targets]
    Ytr = np.stack([f(x0[None, :] + Ztr) for f in fns], axis=1)
    Yte = np.stack([f(x0[None, :] + Zte) for f in fns], axis=1)
    return Ztr, Ytr, Zte, Yte


def even_geometry(V, Z, n_per):
    return Geometry([make_block(v, Z, n_per) for v in V])


def _cell(V, n_per, Ztr, Ytr, Zte, Yte, keys, tag="", rcond=None):
    from h06.core import RCOND
    t0 = time.time()
    geom = even_geometry(V, Ztr, n_per)
    fit = fit_geometry(geom, Ztr, Ytr, rcond=(rcond or RCOND))
    pred = fit.predict(geom, Zte)
    rec = {"M": len(V), "N": int(n_per), "units": geom.units, "rank": fit.rank,
           "n_cols": fit.n_cols, "rel_l2": {}, "max_abs": {}, "weight_norm": {}}
    for k, name in enumerate(keys):
        rec["rel_l2"][name] = rel_l2(pred[:, k], Yte[:, k])
        rec["max_abs"][name] = max_abs(pred[:, k], Yte[:, k])
        rec["weight_norm"][name] = float(np.linalg.norm(fit.coef[:-1, k]))
    rec["seconds"] = round(time.time() - t0, 1)
    print(f"  {tag:8s} M={rec['M']:4d} N={rec['N']:3d} units={rec['units']:6d} rank={fit.rank:5d}/{fit.n_cols:5d}"
          f" [{rec['seconds']:6.1f}s] " + " ".join(f"{n[:5]}={rec['rel_l2'][n]:.1e}" for n in keys), flush=True)
    return rec


FLOORS_D4_EM_N = 24
FLOORS_D4_EM_M = [16, 32, 64, 96, 128, 192, 256, 384, 512]
FLOORS_D4_EN_M = 384
FLOORS_D4_EN_N = [6, 8, 12, 16, 24, 32]
FLOORS_D4_EXACT = [(64, 12), (128, 8), (128, 16), (256, 12), (256, 16), (192, 24)]


def floors(args=None):
    d = (getattr(args, "dim", 0) or FLOORS_D) if args else FLOORS_D
    rcond = getattr(args, "rcond", 0.0) or None if args else None
    n_train = (getattr(args, "rows", 0) or FLOORS_N_TRAIN) if args else FLOORS_N_TRAIN
    r, keys = FLOORS_R, FLOORS_TARGETS
    if d == 4:
        em_N, em_M, en_M, en_N, exact = FLOORS_D4_EM_N, FLOORS_D4_EM_M, FLOORS_D4_EN_M, FLOORS_D4_EN_N, FLOORS_D4_EXACT
    else:
        em_N, em_M, en_M, en_N, exact = FLOORS_EM_N, FLOORS_EM_M, FLOORS_EN_M, FLOORS_EN_N, FLOORS_EXACT
    json_path = RESULTS_DIR / (f"floors_d{d}.json" if d != 3 else "floors_d3.json")
    Ztr, Ytr, Zte, Yte = data_sets(d, r, n_train, keys)
    Vseq = nested_directions(d, max(em_M + [en_M]))
    out = {"d": d, "r": r, "targets": keys, "n_train": n_train, "em_N": em_N,
           "en_M": en_M, "rcond": rcond, "e_M": [], "e_N": [], "exact": [], "fibonacci": []}

    def dump():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(out, indent=1))

    print(f"e_M(M) at N={em_N}")
    for M in em_M:
        out["e_M"].append(_cell(Vseq[:M], em_N, Ztr, Ytr, Zte, Yte, keys, "e_M", rcond=rcond))
        dump()
    print(f"e_N(N) at M={en_M}")
    for N in en_N:
        if N == em_N:
            out["e_N"].append(next(c for c in out["e_M"] if c["M"] == en_M))
            continue
        out["e_N"].append(_cell(Vseq[:en_M], N, Ztr, Ytr, Zte, Yte, keys, "e_N", rcond=rcond))
        dump()
    print("exact cells for the max-of-two-floors test")
    for M, N in exact:
        out["exact"].append(_cell(Vseq[:M], N, Ztr, Ytr, Zte, Yte, keys, "exact", rcond=rcond))
        dump()
    if d == 3:
        print("Fibonacci (expH01) directions at the same cells, for reference")
        for M, N in [(64, 32), (128, 32)]:
            out["fibonacci"].append(_cell(fibonacci_directions(d, M), N, Ztr, Ytr, Zte, Yte, keys, "fib", rcond=rcond))
            dump()
    print("saved", json_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", action="store_true")
    ap.add_argument("--ridges", action="store_true")
    ap.add_argument("--grow", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--targets", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--cells", default="")
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--rcond", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--polish-grown", action="store_true")
    ap.add_argument("--rcond-scan", action="store_true")
    ap.add_argument("--alloc", action="store_true")
    args = ap.parse_args()
    if args.floors:
        floors(args)
    if args.ridges:
        from modes import ridges
        ridges(args)
    if args.grow:
        from modes import grow
        grow(args)
    if args.push:
        from modes import push
        push(args)
    if args.polish:
        from modes import polish
        polish(args)
    if args.polish_grown:
        from modes import polish_grown
        polish_grown(args)
    if args.rcond_scan:
        from modes import rcond_scan
        rcond_scan(args)
    if args.alloc:
        from modes import alloc
        alloc(args)
    if args.plot:
        import viz
        viz.all_figures()


if __name__ == "__main__":
    main()
