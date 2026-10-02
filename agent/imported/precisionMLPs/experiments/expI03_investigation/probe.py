"""Controlled checkpoint-I probes. Existing I01/I02 implementations remain untouched.

Run with the repository environment; all data are synthetic and seeded. Comparisons
share samples and use final, independently evaluated errors, never best-test selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precision-mpl-cache")
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/expI02_block_prior"))
import qi2
import tasks

OUT = ROOT / "results/checkpoint_I_depth_theory/investigation_20260906"


def cheb(x, degree):
    terms = [torch.ones_like(x), x]
    for k in range(2, degree + 1):
        terms.append(2 * x * terms[-1] - terms[-2])
    return torch.stack(terms[:degree + 1], -1)


class SmoothMixer(qi2.Mixer):
    """Restrict each inner profile to degree-D functions represented in the QI bank.

    The constant mode is absorbed by the channel bias. The basis is target independent;
    coefficients and directions are learned from end-to-end labels only.
    """
    def __init__(self, old, degree=3):
        super().__init__(old.M, old.N, old.K)
        x, bank = qi2._ref_bank(old.N)
        mask = x.abs() <= 0.9
        # Each basis column has unit RMS on the occupied interval. No target labels.
        y = cheb(x[mask] / 0.9, degree)[:, 1:]
        y = y / y.square().mean(0).sqrt()
        B, _ = qi2.tsvd_solve(bank[mask], y, rcond=1e-12)
        self.P = B
        self.theta = nn.Parameter(torch.zeros(old.M, degree, old.K))
        self.degree = degree

    def counts(self):
        return dict(params=self.theta.numel() + self.K, macs=self.M * self.N * self.K)


@torch.no_grad()
def configure(net, X, kind, seed):
    gen = torch.Generator().manual_seed(seed + 9000)
    if kind.startswith("poly") or kind.startswith("quadratic"):
        degree = (int(kind[4:]) if len(kind) > 4 else 3) if kind.startswith("poly") else 2
        for layer in net.layers[:-1]:
            layer.mixer = SmoothMixer(layer.mixer, degree)
            layer.mixer.theta.copy_(torch.randn(layer.mixer.theta.shape, generator=gen)
                                    / torch.arange(1, degree + 1).view(1, -1, 1)
                                    / math.sqrt(layer.mixer.M))
            if kind.startswith("quadratic"):
                # Label-free collection of positive quadratic summaries with small
                # random shifts. Direction rows form repeated orthogonal frames.
                if layer.proj:
                    frames = []
                    for _ in range(math.ceil(layer.M / layer.d_in)):
                        Q, _ = torch.linalg.qr(torch.randn(layer.d_in, layer.d_in, generator=gen))
                        frames.append(Q)
                    layer.V.copy_(torch.cat(frames, 0)[:layer.M])
                layer.mixer.theta[:, 0].mul_(.1)
                layer.mixer.theta[:, 1].fill_(1 / math.sqrt(layer.M))
                layer.mixer.theta[:, 1].add_(.1 * torch.randn(layer.M, layer.d_out, generator=gen))
    elif kind == "derivative":
        # Quadrature law v_j = h f'(c_j)/2; initializes smooth functions with small
        # coefficients, avoiding an ill-conditioned polynomial interpolation solve.
        for layer in net.layers[:-1]:
            c = layer.bank.c
            a = torch.randn(layer.M, 3, layer.d_out, generator=gen)
            deriv = a[:, 0:1] + 2 * a[:, 1:2] * c[None, :, None] + 3 * a[:, 2:3] * c[None, :, None] ** 2
            layer.mixer.theta.copy_(0.5 * layer.bank.h * deriv / math.sqrt(layer.M))
    qi2.calibrate(net, X)
    if kind == "quadratic_free":
        # Same initial function as the quadratic subspace, then release its
        # coefficient restriction. This isolates initialization from function class.
        for layer in net.layers[:-1]:
            old = layer.mixer
            raw = qi2.Mixer(old.M, old.N, old.K)
            raw.set_raw(old.coeffs())
            raw.bias.copy_(old.bias)
            layer.mixer = raw
    return net


@torch.no_grad()
def solve_head(net, X, Y, rcond=1e-14, ridge=0.):
    if ridge <= 0:
        return net.solve_readout(X, Y, rcond)
    ridge_from_features(net, net.feats(X), Y, ridge)


@torch.no_grad()
def ridge_from_features(net, F, Y, ridge):
    """SVD solve for mean squared error + ridge * ||head weights||^2; bias unpenalized."""
    Y = Y.reshape(len(F), -1)
    fm, ym = F.mean(0), Y.mean(0)
    U, s, Vh = torch.linalg.svd(F - fm, full_matrices=False)
    W = (Vh.T * (s / (s.square() + len(F) * ridge))) @ (U.T @ (Y - ym))
    if isinstance(net, qi2.MLP):
        net.W.copy_(W)
        net.b.copy_(ym - fm @ W)
    else:
        mix = net.layers[-1].mixer
        mix.set_raw(W.reshape(mix.M, mix.N, mix.K))
        mix.bias.copy_(ym - fm @ W)


def ridge_forward(net, X, Y, ridge):
    if isinstance(net, qi2.MLP):
        F = net.feats(X)
        ridge_from_features(net, F.detach(), Y, ridge)
        return F @ net.W + net.b, []
    pres = []
    for i, layer in enumerate(net.layers):
        p = layer.pre(X)
        pres.append(p)
        H = layer.bank(p)
        if i == len(net.layers) - 1:
            ridge_from_features(net, H.detach().reshape(len(X), -1), Y, ridge)
        out = layer.mixer(H)
        X = X + out if layer.residual else out
    return X, pres


def metrics(net, D, rcond=1e-14, ridge=0.):
    with torch.no_grad():
        solve_head(net, D["Xtr"], D["Ytr"], rcond, ridge)
        pred = net(D["Xte"]).ravel()
        train = net(D["Xtr"]).ravel()
        # A second test distribution covers the full training ball, including the shell.
        Xfull = tasks.ball(len(D["Xte"]), D["d"], D["r"], 45678) - D["x0"]
        Yfull = tasks.target(D["name"], Xfull + D["x0"])
        return dict(train=qi2.rel_l2(train, D["Ytr"]), test=qi2.rel_l2(pred, D["Yte"]),
                    full_test=qi2.rel_l2(net(Xfull).ravel(), Yfull),
                    linf=float((pred - D["Yte"]).abs().max()), band=qi2.band_stats(net.pres(D["Xtr"])),
                    coeff_norm=[float(l.mixer.coeffs().norm()) for l in net.layers] if hasattr(net, "layers") else [],
                    actual_params=sum(p.numel() for p in net.parameters()))


def train(net, D, steps, lr, mixer_scale=1.0, log_every=100, beta=0.0, train_rcond=1e-14, ridge=0.):
    nl = net.nonlinear_params()
    head_ids = {id(p) for p in net.readout()}
    for p in net.readout():
        p.requires_grad_(False)
    mixer_ids = {id(p) for l in net.layers[:-1] for p in l.mixer.parameters()} if hasattr(net, "layers") else set()
    groups = [dict(params=[p for p in nl if id(p) not in mixer_ids], lr=lr),
              dict(params=[p for p in nl if id(p) in mixer_ids], lr=lr * mixer_scale)]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.Adam(groups)
    rates = [g["lr"] for g in groups]
    curve = [dict(step=0, **metrics(net, D, train_rcond, ridge))]
    y = D["Ytr"].view(-1, 1)
    t0 = time.time()
    for step in range(steps):
        factor = min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g, rate in zip(opt.param_groups, rates):
            g["lr"] = rate * factor
        opt.zero_grad(set_to_none=True)
        out, pres = ridge_forward(net, D["Xtr"], y, ridge) if ridge > 0 else net.forward_pres(D["Xtr"], solve=(y, train_rcond))
        loss = (out - y).square().mean() + beta * qi2.band_penalty(pres)
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            curve.append(dict(step=step + 1, **metrics(net, D, train_rcond, ridge)))
    precision_head = metrics(net, D)
    solve_head(net, D["Xtr"], D["Ytr"], train_rcond, ridge)
    return dict(curve=curve, time=time.time()-t0, final=curve[-1], precision_head=precision_head, counts=net.counts())


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2))
    tmp.replace(path)


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.loads(path.read_text())
    names = list(dict.fromkeys(r["target"] for r in rows.values()))
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2), squeeze=False)
    colors = dict(zip(sorted({r["arm"] for r in rows.values()}), plt.get_cmap("tab10").colors))
    for ax, name in zip(axes[0], names):
        for arm, color in colors.items():
            runs = [r for r in rows.values() if r["target"] == name and r["arm"] == arm]
            if not runs:
                continue
            ts = [s["step"] for s in runs[0]["curve"]]
            e = np.array([[s["test"] for s in r["curve"]] for r in runs])
            ax.semilogy(ts, np.median(e, 0), color=color, label=arm)
            ax.fill_between(ts, e.min(0), e.max(0), color=color, alpha=.12)
        ax.set(title=name, xlabel="Adam steps", ylabel="Independent test relative L2")
        ax.set_ylim(1e-7, 2)
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .82))
    fig.savefig(path.with_suffix(".png"), dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="screen")
    ap.add_argument("--targets", default="fast_waves,composition,random_ridges")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="raw,raw_scaled,derivative,poly3")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--ntrain", type=int, default=1024)
    ap.add_argument("--ntest", type=int, default=2048)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--N1", type=int, default=32)
    ap.add_argument("--N2", type=int, default=64)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--lr", type=float, default=.02)
    ap.add_argument("--beta", type=float, default=0.)
    ap.add_argument("--train-rcond", type=float, default=1e-14)
    ap.add_argument("--ridge", type=float, default=0.)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    path = OUT / f"{args.name}.json"
    if args.plot:
        plot(path)
        return
    rows = json.loads(path.read_text()) if path.exists() else {}
    for seed in map(int, args.seeds.split(",")):
        for name in args.targets.split(","):
            D = tasks.analytic(name, args.d, args.ntrain, args.ntest, seed=seed + 1200)
            for arm in args.arms.split(","):
                key = f"{name}|{seed}|{arm}"
                if key in rows:
                    continue
                net = qi2.make_block(args.d, args.M, args.N1, args.K, args.N2, seed=seed)
                if arm == "mlp":
                    width = qi2.mlp_width_for(net.counts()["params"], args.d, "params")
                    net = qi2.MLP(args.d, [width, width], seed=seed)
                else:
                    configure(net, D["Xtr"], "raw" if arm == "raw_scaled" else arm, seed)
                result = train(net, D, args.steps, args.lr, mixer_scale=1 / args.N1 if arm == "raw_scaled" else 1,
                               beta=args.beta, train_rcond=args.train_rcond, ridge=args.ridge)
                result.update(target=name, seed=seed, arm=arm, config=vars(args))
                rows[key] = result
                save(path, rows)
                if args.save_models:
                    torch.save(dict(state=net.state_dict(), config=vars(args), arm=arm, target=name, seed=seed),
                               OUT / f"{args.name}_{name}_{seed}_{arm}.pt")
                print(key, f"test={result['final']['test']:.4g} full={result['final']['full_test']:.4g}",
                      f"seconds={result['time']:.1f}", flush=True)
    plot(path)


if __name__ == "__main__":
    main()
