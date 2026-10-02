"""expF08 Darcy sweep driver: smooth control + real darcy_421 instances.

The expF01 tanh collocation core (vendored here as core.py) applied to the FNO
darcy_421 benchmark. Two problems:
  - a smooth manufactured control that verifies the machine-precision claim on
    the Darcy operator (reaches ~3e-14 rel L2 vs the exact solution);
  - 16 rough benchmark instances swept over Gaussian coefficient pre-smoothing
    sigma in {0,1,2,4}px, width W, and lambda, scored in rel L2 vs the dataset
    reference and in raw PDE residual.

Usage (from repo root):
    python experiments/expF08_darcy_sweep/run_darcy.py [--smoke] [--plot] [--darcy-path PATH]

--smoke: 2 instances, W in {256, 576}, lambda 0.25 only, sigma in {0, 2}.
--plot:  skip solving, regenerate figures from the results json.

Writes results incrementally to
results/checkpoint_F_applications/expF08_darcy_sweep/darcy_sweep.json
(safe to re-run: finished cells are skipped).
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import core
import darcy_problems as dp

RESULTS_DIR = (REPO_ROOT / "results" / "checkpoint_F_applications"
               / "expF08_darcy_sweep")

SEED = 42
N_RESIDUAL_PTS = 20000
FNO_BASELINE = 1e-2


def eval_grid_points(n, cell_centered=False):
    g = dp.grid_1d(n, cell_centered)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def zero_dirichlet_bc():
    Pb = core.boundary_points_square(480)
    return [dict(points=Pb, terms=[((0, 0), 1.0)], values=np.zeros(len(Pb)))]


def pde_residual(model, terms, forcing, rng):
    """Median/max |L u_hat - f| at fresh random interior points (scalar forcing)."""
    P = rng.uniform(-1.0, 1.0, (N_RESIDUAL_PTS, 2))
    r = np.abs(core.eval_model(model, P, terms=terms) - forcing)
    return float(np.median(r)), float(np.max(r))


def solve_control_cell(W, lam):
    t0 = time.time()
    model = core.solve_square(dp.control_terms(), dp.control_forcing,
                              zero_dirichlet_bc(), W, lam, seed=SEED)
    P = eval_grid_points(241)
    u_hat, u_true = core.eval_model(model, P), dp.control_u(P)
    # control forcing is a callable, so evaluate the residual directly here
    # (pde_residual assumes scalar forcing and serves the instance path)
    rng = np.random.default_rng(SEED + 1)
    Pr = rng.uniform(-1.0, 1.0, (N_RESIDUAL_PTS, 2))
    r = np.abs(core.eval_model(model, Pr, terms=dp.control_terms())
               - dp.control_forcing(Pr))
    return dict(kind="control", W=W, lam=lam,
                rel_l2=core.rel_l2(u_hat, u_true), linf=core.linf(u_hat, u_true),
                res_med=float(np.median(r)), res_max=float(np.max(r)),
                t_solve=time.time() - t0)


def solve_instance_cell(a_grid, u_ref, inst, sigma, W, lam):
    t0 = time.time()
    coeff = dp.DarcyCoefficient(a_grid, sigma_px=sigma, cell_centered=True)
    model = core.solve_square(coeff.terms(), dp.DARCY_FORCING,
                              zero_dirichlet_bc(), W, lam, seed=SEED)
    n = u_ref.shape[0]
    P = eval_grid_points(n, cell_centered=True)
    u_hat = core.eval_model(model, P).reshape(n, n)
    rng = np.random.default_rng(SEED + 1)
    res_med, res_max = pde_residual(model, coeff.terms(), dp.DARCY_FORCING, rng)
    return dict(kind="instance", instance=inst, sigma=sigma, W=W, lam=lam,
                rel_l2=core.rel_l2(u_hat.ravel(), u_ref.ravel()),
                linf=core.linf(u_hat.ravel(), u_ref.ravel()),
                res_med=res_med, res_max=res_max, t_solve=time.time() - t0)


def cell_key(rec):
    if rec["kind"] == "control":
        return ("control", rec["W"], rec["lam"])
    return ("instance", rec["instance"], rec["sigma"], rec["W"], rec["lam"])


def load_results():
    path = RESULTS_DIR / "darcy_sweep.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_results(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "darcy_sweep.json").write_text(json.dumps(records, indent=1))


def run_sweep(cfg):
    dp.verify_control()
    records = load_results()
    done = {cell_key(r) for r in records}

    for W in cfg["W_grid"]:
        for lam in cfg["lam_grid"]:
            if ("control", W, lam) in done:
                continue
            rec = solve_control_cell(W, lam)
            records.append(rec)
            save_results(records)
            print(f"control W={W} lam={lam}: rel_l2={rec['rel_l2']:.2e} "
                  f"({rec['t_solve']:.1f}s)", flush=True)

    a_all, u_all = dp.load_darcy_test(cfg["darcy_path"], cfg["n_instances"])
    bnd = np.concatenate([u_all[:, 0, :].ravel(), u_all[:, -1, :].ravel(),
                          u_all[:, :, 0].ravel(), u_all[:, :, -1].ravel()])
    print(f"dataset: {a_all.shape}, max|u_ref| on boundary = {np.max(np.abs(bnd)):.2e}",
          flush=True)

    for inst in range(len(a_all)):
        for sigma in cfg["sigmas"]:
            for W in cfg["W_grid"]:
                for lam in cfg["lam_grid"]:
                    if ("instance", inst, sigma, W, lam) in done:
                        continue
                    rec = solve_instance_cell(a_all[inst], u_all[inst],
                                              inst, sigma, W, lam)
                    records.append(rec)
                    save_results(records)
                    print(f"inst={inst} sig={sigma} W={W} lam={lam}: "
                          f"rel_l2={rec['rel_l2']:.2e} res_med={rec['res_med']:.2e} "
                          f"({rec['t_solve']:.1f}s)", flush=True)
    return records


def best_by(records, criterion):
    """Per (sigma, W): pick lam minimizing `criterion` per instance, then the
    median rel_l2 across instances of those picks."""
    out = {}
    inst_recs = [r for r in records if r["kind"] == "instance"]
    sigmas = sorted({r["sigma"] for r in inst_recs})
    Ws = sorted({r["W"] for r in inst_recs})
    for sigma in sigmas:
        for W in Ws:
            picks = []
            insts = sorted({r["instance"] for r in inst_recs})
            for inst in insts:
                cands = [r for r in inst_recs
                         if r["sigma"] == sigma and r["W"] == W and r["instance"] == inst]
                if cands:
                    picks.append(min(cands, key=lambda r: r[criterion]))
            if picks:
                out[(sigma, W)] = float(np.median([p["rel_l2"] for p in picks]))
    return out


def plot_results(records, darcy_path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ctrl = [r for r in records if r["kind"] == "control"]
    if ctrl:
        Ws = sorted({r["W"] for r in ctrl})
        ax.loglog(Ws, [min(r["rel_l2"] for r in ctrl if r["W"] == W) for W in Ws],
                  "k^-", label="smooth control (vs exact)")
    agg = best_by(records, "res_med")
    sigmas = sorted({s for s, _ in agg})
    for sigma in sigmas:
        Ws_s = sorted({W for s, W in agg if s == sigma})
        ax.loglog(Ws_s, [agg[(sigma, W)] for W in Ws_s], "o-",
                  label=f"real a, sigma={sigma:g}px (vs reference)")
    ax.axhline(FNO_BASELINE, color="gray", ls=":", label="trained FNO ~1e-2")
    ax.set_xlabel("width W")
    ax.set_ylabel("rel $L_2$")
    ax.set_title("Darcy: collocation lstsq, median over instances")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "error_vs_width.png", dpi=150)
    print(f"wrote {RESULTS_DIR / 'error_vs_width.png'}")

    # heatmaps: instance 0, best cell per sigma
    inst_recs = [r for r in records if r["kind"] == "instance" and r["instance"] == 0]
    if inst_recs:
        a_all, u_all = dp.load_darcy_test(darcy_path, 1)
        n = u_all.shape[-1]
        P = eval_grid_points(n, cell_centered=True)
        sigmas0 = sorted({r["sigma"] for r in inst_recs})
        fig, axes = plt.subplots(1, len(sigmas0) + 1,
                                 figsize=(4 * (len(sigmas0) + 1), 3.6))
        im = axes[0].imshow(a_all[0], origin="lower", extent=[-1, 1, -1, 1])
        axes[0].set_title("a(x, y) raw")
        fig.colorbar(im, ax=axes[0])
        for k, sigma in enumerate(sigmas0):
            best = min([r for r in inst_recs if r["sigma"] == sigma],
                       key=lambda r: r["res_med"])
            coeff = dp.DarcyCoefficient(a_all[0], sigma_px=sigma, cell_centered=True)
            model = core.solve_square(coeff.terms(), dp.DARCY_FORCING,
                                      zero_dirichlet_bc(), best["W"], best["lam"],
                                      seed=SEED)
            u_hat = core.eval_model(model, P).reshape(n, n)
            err = np.log10(np.abs(u_hat - u_all[0]) + 1e-18)
            im = axes[k + 1].imshow(err, origin="lower", extent=[-1, 1, -1, 1])
            axes[k + 1].set_title(f"log10|err| sig={sigma:g} W={best['W']}")
            fig.colorbar(im, ax=axes[k + 1])
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "error_heatmaps.png", dpi=150)
        print(f"wrote {RESULTS_DIR / 'error_heatmaps.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--darcy-path", default=dp.DEFAULT_NPZ)
    args = ap.parse_args()

    smoke = args.smoke
    cfg = dict(
        W_grid=[256, 576] if smoke else [256, 576, 1024, 1600, 2304],
        lam_grid=[0.25] if smoke else [0.20, 0.25, 0.30],
        sigmas=[0.0, 2.0] if smoke else [0.0, 1.0, 2.0, 4.0],
        n_instances=2 if smoke else 16,
        darcy_path=args.darcy_path,
    )
    records = load_results() if args.plot else run_sweep(cfg)
    plot_results(records, args.darcy_path)


if __name__ == "__main__":
    main()
