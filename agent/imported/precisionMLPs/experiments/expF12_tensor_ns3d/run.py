"""expF12 -- 3D unsteady Navier-Stokes via a tensor-product sech^2 basis PINN.

Stages (run.py --stage <s>, default all):
  ceiling   Supervised expressivity ceiling: Kronecker-lstsq fit of the readout
            A to the exact Beltrami fields over (N, lambda, rcond) -- how
            precise can this basis be AT ALL. No PDE involved.
  pinn      The moonshot: PINN training (Adam -> LBFGS) of A on the NS
            residual + IC + BC + gauge, geometry frozen, warm-started from a
            Kronecker fit to the t-constant extension of the IC.
  newton    THE HEADLINE: training-free solve -- frozen QI geometry +
            Gauss-Newton collocation lstsq for the readout in the reduced
            product-SVD basis (first step = Stokes). See newton_solve.py.
  baseline  Standard tanh-MLP PINN (autograd), same losses/eval, matched
            wall-clock -- the "other method" for the benchmark plot.
  plots     Benchmark figure, error-vs-time figure, ceiling figure, slice
            animation GIFs.

Results land in results/checkpoint_F_applications/expF12_tensor_ns3d/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.append(str(HERE))

import beltrami as bel
import pinn
import tensor_basis as tb

RESULTS = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF12_tensor_ns3d"

torch.set_default_dtype(torch.float64)


# -- stage: ceiling ------------------------------------------------------------

def stage_ceiling(args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    P = pinn.sample_interior(rng, 20000)
    exact = bel.fields(P.numpy())
    rows = []
    Ns = [8, 12, 16, 20, 24, 32] if not args.smoke else [8, 12]
    lams = [0.1, 0.125, 0.15, 0.2, 0.25] if not args.smoke else [0.15]
    rconds = [1e-13, 1e-15] if not args.smoke else [1e-15]
    for n in Ns:
        for lam in lams:
            for rc in rconds:
                t0 = time.time()
                net = tb.TensorBasisNet(n_centers=n, lam=lam,
                                        domains=bel.DOMAINS)
                grids = [np.linspace(lo, hi, 2 * n) for lo, hi in bel.DOMAINS]
                _, info = tb.kron_fit(net, grids, bel.grid_targets(grids),
                                      rcond=rc)
                t_solve = time.time() - t0
                pred = net.evaluate_chunked(P, [(0, 0, 0, 0)])[(0, 0, 0, 0)].numpy()
                rel_v = float(np.linalg.norm(pred[:, :3] - exact[:, :3])
                              / np.linalg.norm(exact[:, :3]))
                rel_p = float(np.linalg.norm(pred[:, 3] - exact[:, 3])
                              / np.linalg.norm(exact[:, 3]))
                mom, cont = tb.ns_residuals(net, P[:5000], bel.NU)
                rows.append(dict(N=n, lam=lam, rcond=rc, rel_l2_v=rel_v,
                                 rel_l2_p=rel_p,
                                 mom_rms=float((mom**2).mean().sqrt()),
                                 div_max=float(cont.abs().max()),
                                 max_cond_1d=max(info["conds"]),
                                 kept=info["n_kept"] / info["n_total"],
                                 t_solve=t_solve))
                print(rows[-1], flush=True)
    (RESULTS / "ceiling.json").write_text(json.dumps(rows, indent=1))


# -- stage: pinn ---------------------------------------------------------------

def stage_pinn(args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = RESULTS / "pinn_history.jsonl"
    log.unlink(missing_ok=True)
    kw = dict(n_centers=args.N, lam=args.lam, seed=args.seed,
              adam_steps=args.adam_steps, lbfgs_iters=args.lbfgs_iters,
              log_path=str(log))
    if args.smoke:
        kw.update(adam_steps=10, lbfgs_iters=20, adam_batch=256,
                  lbfgs_batch=512, n_ic=128, n_bc=128, eval_every=5)
    net, hist = pinn.train_tensor_pinn(**kw)
    torch.save(net.state_dict(), RESULTS / "pinn_state.pt")
    (RESULTS / "pinn_config.json").write_text(json.dumps(
        dict(N=args.N, lam=args.lam, seed=args.seed)))
    _final_metrics(net, hist, "pinn")


def _final_metrics(net, hist, tag):
    rng = np.random.default_rng(7)
    P = pinn.sample_interior(rng, 10000)
    if isinstance(net, tb.TensorBasisNet):
        mom, cont = tb.ns_residuals(net, P, bel.NU)
        mom, cont = mom.detach(), cont.detach()
    else:
        mom, cont = pinn.mlp_residuals(net, P, bel.NU)
        mom, cont = mom.detach(), cont.detach()
    out = dict(tag=tag, rel_l2_v=hist[-1]["rel_l2_v"],
               rel_l2_p=hist[-1]["rel_l2_p"],
               mom_rms=float((mom**2).mean().sqrt()),
               div_max=float(cont.abs().max()),
               wall=hist[-1]["wall"])
    print("FINAL", out, flush=True)
    path = RESULTS / f"{tag}_final.json"
    path.write_text(json.dumps(out, indent=1))


def stage_newton(args):
    import newton_solve as ns
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = RESULTS / "newton_history.jsonl"
    log.unlink(missing_ok=True)
    kw = dict(n_centers=args.N_newton, lam=args.lam_newton, K_vel=args.K_vel,
              K_p=args.K_p, newton_iters=args.newton_iters, seed=args.seed,
              log_path=str(log))
    if args.smoke:
        kw.update(n_centers=8, K_vel=300, K_p=150, n_int=1200, n_ic=400,
                  n_bc=400, newton_iters=2)
    basis_v, basis_p, theta, hist = ns.solve_navier_stokes(**kw)
    np.save(RESULTS / "newton_theta.npy", theta)
    (RESULTS / "newton_config.json").write_text(json.dumps(
        dict(N=kw["n_centers"], lam=kw["lam"], K_vel=kw["K_vel"],
             K_p=kw["K_p"])))
    out = dict(tag="newton", rel_l2_v=hist[-1]["rel_l2_v"],
               rel_l2_p=hist[-1]["rel_l2_p"], mom_rms=hist[-1]["mom_rms"],
               div_max=hist[-1]["div_max"], wall=hist[-1]["wall"])
    print("FINAL", out, flush=True)
    (RESULTS / "newton_final.json").write_text(json.dumps(out, indent=1))


def stage_baseline(args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = RESULTS / "baseline_history.jsonl"
    log.unlink(missing_ok=True)
    kw = dict(seed=args.seed, adam_steps=args.mlp_adam_steps,
              lbfgs_iters=args.mlp_lbfgs_iters, log_path=str(log))
    if args.smoke:
        kw.update(adam_steps=10, lbfgs_iters=20, adam_batch=128,
                  lbfgs_batch=256, n_ic=64, n_bc=64, eval_every=5)
    net, hist = pinn.train_mlp_pinn(**kw)
    torch.save(net.state_dict(), RESULTS / "baseline_state.pt")
    _final_metrics(net, hist, "baseline")


# -- stage: plots --------------------------------------------------------------

def _load_hist(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def _load_pinn_net():
    cfg = json.loads((RESULTS / "pinn_config.json").read_text())
    net = tb.TensorBasisNet(n_centers=cfg["N"], lam=cfg["lam"],
                            domains=bel.DOMAINS)
    net.load_state_dict(torch.load(RESULTS / "pinn_state.pt",
                                   weights_only=True))
    return net, cfg


def _legend_above(ax, ncol=3):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol,
              borderaxespad=0, frameon=False)


def stage_plots(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ceiling = json.loads((RESULTS / "ceiling.json").read_text())
    ph = _load_hist(RESULTS / "pinn_history.jsonl")
    bh = _load_hist(RESULTS / "baseline_history.jsonl")
    net, cfg = _load_pinn_net()

    nh = None
    if (RESULTS / "newton_history.jsonl").exists():
        nh = _load_hist(RESULTS / "newton_history.jsonl")

    # -- fig 1: benchmark -- error vs wall-clock -------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if nh:
        ax.plot([r["wall"] for r in nh], [r["rel_l2_v"] for r in nh],
                color="C2", marker="s", ms=5, lw=2,
                label="QI geometry + Gauss-Newton lstsq (no training)")
    for hist, label, color in [(ph, "tensor sech$^2$ PINN, Adam$\\to$LBFGS", "C0"),
                               (bh, "tanh-MLP PINN baseline", "C1")]:
        wall = [r["wall"] for r in hist if r["step"] >= 0]
        err = [r["rel_l2_v"] for r in hist if r["step"] >= 0]
        ax.plot(wall, err, color=color, label=label)
        sw = [r["wall"] for r in hist if r.get("phase") == "lbfgs"]
        if sw:
            ax.axvline(sw[0], color=color, ls=":", lw=0.8, alpha=0.5)
    same_n = [r for r in ceiling if r["N"] == cfg["N"] and r["rcond"] == 1e-15]
    if same_n:
        c = min(r["rel_l2_v"] for r in same_n)
        ax.axhline(c, color="C0", ls="--", lw=1.2,
                   label=f"supervised ceiling, same width (N={cfg['N']})")
    best = min(r["rel_l2_v"] for r in ceiling)
    ax.axhline(best, color="k", ls="--", lw=1.2,
               label="supervised ceiling, best (fp64)")
    ax.axhline(1e-3, color="gray", ls="-.", lw=1.0,
               label="typical literature PINN (~1e-3)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock [s]")
    ax.set_ylabel(r"velocity rel $L_2$ vs exact Beltrami")
    _legend_above(ax, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_error_vs_wallclock.png", dpi=160)
    plt.close(fig)

    # -- fig 2: ceiling convergence -------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    lams = sorted({r["lam"] for r in ceiling})
    for lam in lams:
        for rc, ls in [(1e-13, ":"), (1e-15, "-")]:
            rows = sorted([r for r in ceiling
                           if r["lam"] == lam and r["rcond"] == rc],
                          key=lambda r: r["N"])
            if rows:
                ax.plot([r["N"] for r in rows], [r["rel_l2_v"] for r in rows],
                        ls=ls, marker="o", ms=3,
                        label=f"$\\lambda$={lam}, rcond={rc:g}")
    ax.set_yscale("log")
    ax.set_xlabel("per-axis width N")
    ax.set_ylabel(r"velocity rel $L_2$ (supervised fit)")
    _legend_above(ax, ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS / "ceiling_convergence.png", dpi=160)
    plt.close(fig)

    # -- newton predictor (if run) --------------------------------------------
    newton_predict = None
    if nh and (RESULTS / "newton_theta.npy").exists():
        import newton_solve as nsv
        ncfg = json.loads((RESULTS / "newton_config.json").read_text())
        theta = np.load(RESULTS / "newton_theta.npy")
        bv = nsv.ReducedTensorBasis(ncfg["N"], ncfg["lam"], K=ncfg["K_vel"])
        bp = nsv.ReducedTensorBasis(ncfg["N"], ncfg["lam"], K=ncfg["K_p"])

        def newton_predict(P):
            Cv = bv.op_columns(P, [((0, 0, 0, 0), 1.0)])
            Cp = bp.op_columns(P, [((0, 0, 0, 0), 1.0)])
            Kv = bv.K
            cols = [Cv @ theta[i * Kv:(i + 1) * Kv] for i in range(3)]
            cols.append(Cp @ theta[3 * Kv:])
            return np.stack(cols, axis=1)

    # -- fig 3: error vs time --------------------------------------------------
    g = np.linspace(-1, 1, 25)
    ts = np.linspace(0, 1, 21)
    S = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    mnet = pinn.MLP()
    mnet.load_state_dict(torch.load(RESULTS / "baseline_state.pt",
                                    weights_only=True))
    curves = {"tensor sech$^2$ PINN": [], "tanh-MLP PINN": []}
    if newton_predict:
        curves["QI + Gauss-Newton"] = []
    for t in ts:
        P = np.concatenate([S, np.full((len(S), 1), t)], axis=1)
        exact = bel.fields(P)
        X = torch.tensor(P)
        pred_t = net.evaluate_chunked(X, [(0, 0, 0, 0)])[(0, 0, 0, 0)].numpy()
        with torch.no_grad():
            pred_m = mnet(X).numpy()
        pairs = [("tensor sech$^2$ PINN", pred_t), ("tanh-MLP PINN", pred_m)]
        if newton_predict:
            pairs.append(("QI + Gauss-Newton", newton_predict(P)))
        for k, pred in pairs:
            curves[k].append(np.linalg.norm(pred[:, :3] - exact[:, :3])
                             / np.linalg.norm(exact[:, :3]))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for k, v in curves.items():
        ax.plot(ts, v, marker="o", ms=3, label=k)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"velocity rel $L_2(t)$")
    _legend_above(ax, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "error_vs_time.png", dpi=160)
    plt.close(fig)

    # -- animation: z=0 slice --------------------------------------------------
    m = 81
    gs = np.linspace(-1, 1, m)
    XX, YY = np.meshgrid(gs, gs, indexing="ij")
    S2 = np.stack([XX.ravel(), YY.ravel(), np.zeros(m * m)], axis=1)
    frames = np.linspace(0, 1, 41)
    combos = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)]

    # the animation shows the headline model: Newton solve if run, else the PINN
    def slice_fields(t):
        P = np.concatenate([S2, np.full((len(S2), 1), t)], axis=1)
        if newton_predict:
            Kv = bv.K
            val = newton_predict(P)
            gx = bv.op_columns(P, [((1, 0, 0, 0), 1.0)])
            gy = bv.op_columns(P, [((0, 1, 0, 0), 1.0)])
            wz = (gx @ theta[Kv:2 * Kv]
                  - gy @ theta[:Kv]).reshape(m, m)
        else:
            X = torch.tensor(P)
            d = net.evaluate_chunked(X, combos)
            val = d[(0, 0, 0, 0)].numpy()
            wz = (d[(1, 0, 0, 0)][:, 1]
                  - d[(0, 1, 0, 0)][:, 0]).numpy().reshape(m, m)
        spd = np.linalg.norm(val[:, :3], axis=1).reshape(m, m)
        ex = bel.fields(P)
        spd_e = np.linalg.norm(ex[:, :3], axis=1).reshape(m, m)
        err = np.abs(np.linalg.norm(val[:, :3] - ex[:, :3], axis=1)).reshape(m, m)
        return spd_e, spd, wz, err

    s0 = slice_fields(0.0)
    vmax = s0[0].max()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    who = "Newton solve" if newton_predict else "PINN"
    titles = [r"exact $|\mathbf{u}|$", f"{who} " + r"$|\mathbf{u}|$",
              f"{who} " + r"$\omega_z$",
              r"$\log_{10}|\mathbf{u}-\mathbf{u}^*|$"]
    ims = []
    for ax_, title, data, kwargs in zip(
            axes, titles,
            [s0[0], s0[1], s0[2], np.log10(s0[3] + 1e-16)],
            [dict(vmin=0, vmax=vmax, cmap="viridis"),
             dict(vmin=0, vmax=vmax, cmap="viridis"),
             dict(cmap="RdBu_r"),
             dict(vmin=-9, vmax=-1, cmap="magma")]):
        im = ax_.imshow(data.T, origin="lower", extent=[-1, 1, -1, 1], **kwargs)
        ax_.set_title(title)
        fig.colorbar(im, ax=ax_, fraction=0.046)
        ims.append(im)
    sup = fig.suptitle("t = 0.00  (z = 0 slice)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(t):
        spd_e, spd, wz, err = slice_fields(float(t))
        for im, data in zip(ims, [spd_e, spd, wz, np.log10(err + 1e-16)]):
            im.set_data(data.T)
        sup.set_text(f"t = {float(t):.2f}  (z = 0 slice)")
        return ims

    ani = animation.FuncAnimation(fig, update, frames=frames)
    ani.save(RESULTS / "ns3d_slice.gif", writer="pillow", fps=8)
    plt.close(fig)
    print("plots done ->", RESULTS, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["ceiling", "newton", "pinn", "baseline", "plots",
                             "all"])
    ap.add_argument("--N-newton", type=int, default=12)
    ap.add_argument("--lam-newton", type=float, default=0.1)
    ap.add_argument("--K-vel", type=int, default=2000)
    ap.add_argument("--K-p", type=int, default=1000)
    ap.add_argument("--newton-iters", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam-steps", type=int, default=1500)
    ap.add_argument("--lbfgs-iters", type=int, default=300)
    ap.add_argument("--mlp-adam-steps", type=int, default=5000)
    ap.add_argument("--mlp-lbfgs-iters", type=int, default=300)
    args = ap.parse_args()
    stages = ([args.stage] if args.stage != "all"
              else ["ceiling", "newton", "pinn", "baseline", "plots"])
    for s in stages:
        print(f"=== stage: {s} ===", flush=True)
        dict(ceiling=stage_ceiling, newton=stage_newton, pinn=stage_pinn,
             baseline=stage_baseline, plots=stage_plots)[s](args)


if __name__ == "__main__":
    main()
