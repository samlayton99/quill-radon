"""expF09 Stage-A driver: manufactured Stokes, W/lambda sweep.

Usage (from repo root):
    uv run --extra dev python experiments/expF09_navier_stokes/run_stokes.py [--smoke] [--plot]

Writes results incrementally to
results/checkpoint_F_applications/expF09_navier_stokes/data.json.
"""
from __future__ import annotations

import argparse
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
sys.path.append(str(HERE))

import core_system as cs
import stokes as st

RESULTS_DIR = (REPO_ROOT / "results" / "checkpoint_F_applications"
               / "expF09_navier_stokes")

N_INTERIOR = 8000
SEED = 0
W_GRID = [576, 1024, 1600, 2304]
LAM_GRID = [0.20, 0.25, 0.30]


def evaluate_cell(W, lam, n_interior=N_INTERIOR, seed=SEED):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    Pi = rng.uniform(-1.0, 1.0, (n_interior, 2))
    Pb = cs.boundary_points_square(480)
    model = cs.solve_system(["u", "v", "p"], st.stokes_equations(Pi, Pb),
                            W, lam, seed=seed)
    g = np.linspace(-1.0, 1.0, 151)
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    uh, vh, ph = (cs.eval_field(model, f, P) for f in ("u", "v", "p"))
    us, vs, ps = st.u_star(P), st.v_star(P), st.p_star(P)
    vel = float(np.linalg.norm(np.concatenate([uh - us, vh - vs]))
                / np.linalg.norm(np.concatenate([us, vs])))
    pr = float(np.linalg.norm(ph - ps) / np.linalg.norm(ps))
    Pr = rng.uniform(-1.0, 1.0, (5000, 2))
    div = float(np.max(np.abs(cs.eval_field(model, "u", Pr, [((1, 0), 1.0)])
                              + cs.eval_field(model, "v", Pr, [((0, 1), 1.0)]))))
    return dict(W=W, lam=lam, vel_rel_l2=vel, p_rel_l2=pr, max_div=div,
                t_solve=time.time() - t0)


def load_records():
    p = RESULTS_DIR / "data.json"
    return json.loads(p.read_text()) if p.exists() else []


def save_records(recs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data.json").write_text(json.dumps(recs, indent=1))


def run_sweep(smoke=False):
    Ws = [576, 1024] if smoke else W_GRID
    lams = [0.25] if smoke else LAM_GRID
    recs = load_records()
    done = {(r["W"], r["lam"]) for r in recs}
    for W in Ws:
        for lam in lams:
            if (W, lam) in done:
                continue
            rec = evaluate_cell(W, lam)
            recs.append(rec)
            save_records(recs)
            print(f"W={W} lam={lam}: vel={rec['vel_rel_l2']:.2e} "
                  f"p={rec['p_rel_l2']:.2e} div={rec['max_div']:.2e} "
                  f"({rec['t_solve']:.1f}s)", flush=True)
    return recs


def plot(recs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    Ws = sorted({r["W"] for r in recs})
    for key, lab in [("vel_rel_l2", "velocity rel L2"),
                     ("p_rel_l2", "pressure rel L2"), ("max_div", "max |div u|")]:
        ax.loglog(Ws, [min(r[key] for r in recs if r["W"] == W) for W in Ws],
                  "o-", label=lab)
    ax.set_xlabel("width W")
    ax.set_ylabel("error")
    ax.set_title("expF09 Stage A: Stokes convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stokes_convergence.png", dpi=150)
    print(f"wrote {RESULTS_DIR / 'stokes_convergence.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    recs = load_records() if args.plot else run_sweep(smoke=args.smoke)
    plot(recs)


if __name__ == "__main__":
    main()
