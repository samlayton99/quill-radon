"""expF13 width extension: convection c=40/c=80 and wave at W in {6400, 9216}.

These three are linear and were still descending ~2 orders per width step at
W=4096, so this is the direct test of whether the method reaches BWLer's
numbers with one more width octave. Linear problems need exactly ONE lstsq
(the GN loop in run.py wastes 3-4 redundant solves on them), so this is cheap.

Memory discipline (16GB box): collocation oversampling 4x at 6400, 3x at 9216;
PDE rows built directly (no D dict), J assembled once.

Usage: uv run --extra dev python experiments/expF13_bwler_suite/extend_width.py
Appends results to results/.../expF13_bwler_suite/extend_width.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import problems as pb
import run as R

OUT = (REPO_ROOT / "results" / "checkpoint_F_applications" / "expF13_bwler_suite"
       / "extend_width.json")
RCOND = 1e-15
SEED = 42


def linear_solve(prob, W, lam, n_mult):
    rng = np.random.default_rng(SEED)
    dirs, offs, gamma = R.radon_geometry(W, lam)
    P = rng.uniform(-1, 1, (n_mult * W, 2))
    lin_rows = R.rows_2d(P, dirs, offs, gamma, prob["lin_terms"])
    s = np.abs(lin_rows).max()
    f = prob["forcing"]
    fv = f(P) if callable(f) else np.full(len(P), float(f))
    bcs = R.build_bcs(prob, dirs, offs, gamma, len(P))
    J = np.vstack([lin_rows / s] + [w * B for (B, g, w) in bcs])
    del lin_rows
    y = np.concatenate([fv / s] + [w * g for (B, g, w) in bcs])
    a = np.linalg.lstsq(J, y, rcond=RCOND)[0]
    del J
    return dict(dirs=dirs, offs=offs, gamma=gamma, sol=a, W=len(offs), iters=1)


CASES = [
    ("convection_c40", 6400, [0.20, 0.25], 4),
    ("convection_c40", 9216, None, 3),          # None: reuse the 6400 winner
    ("convection_c80", 6400, [0.16, 0.20, 0.25], 4),
    ("convection_c80", 9216, None, 3),
    ("wave", 6400, [0.20, 0.25], 4),
    ("wave", 9216, None, 3),
]


def main():
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    winners = {}
    for key, W, lams, n_mult in CASES:
        prob = pb.PROBLEMS[key]
        P_ev, u_true = R.eval_set(prob)
        lams = lams if lams is not None else [winners[key]]
        best = None
        for lam in lams:
            t0 = time.time()
            model = linear_solve(prob, W, lam, n_mult)
            rel, linf = R.metrics(R.eval_model(model, P_ev), u_true)
            dt = time.time() - t0
            print(f"  {key} W={W} lam={lam:.2f} n={n_mult}W  relL2={rel:.2e} "
                  f"Linf={linf:.2e}  ({dt:.0f}s)", flush=True)
            if best is None or rel < best["rel_l2"]:
                best = dict(W=W, lam=lam, rel_l2=rel, linf=linf, n_mult=n_mult,
                            seconds=round(dt, 1))
        winners[key] = best["lam"]
        results.setdefault(key, []).append(best)
        OUT.write_text(json.dumps(results, indent=1))
        print(f"{key} W={W} best lam={best['lam']:.2f} relL2={best['rel_l2']:.2e}",
              flush=True)


if __name__ == "__main__":
    main()
