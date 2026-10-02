"""Read-only audit of checkpoint I records plus small correctness probes.

Run from the repository root with .venv/bin/python <this file>.
Writes only beside this script; no training beyond one optimization step.
"""
import json
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
sys.path.insert(0, str(REPO / "experiments/expI02_block_prior"))
import qi2
import tasks

RES = REPO / "results/checkpoint_I_depth_theory/expI02_block_prior"
A2 = json.loads((RES / "a2.json").read_text())


def numerical_probes():
    out = {}
    # A free shallow projection bias is differentiable, but absent from the optimizer.
    D = tasks.analytic("fast_waves", 3, 512, n_test=512)
    net = qi2.make_block(3, 6, 16, 1, 1, depth=1)
    qi2.calibrate(net, D["Xtr"])
    net.solve_readout(D["Xtr"], D["Ytr"])
    loss = ((net(D["Xtr"])[:, 0] - D["Ytr"]) ** 2).mean()
    grad = torch.autograd.grad(loss, net.layers[0].bV)[0]
    before = net.layers[0].bV.detach().clone()
    qi2.fit(net, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], steps=1,
            beta=0, calibrate_first=False)
    out["shallow_bias"] = dict(gradient_norm=float(grad.norm()),
        optimizer_contains_bias=any(p is net.layers[0].bV for p in net.nonlinear_params()),
        change_after_one_step=float((net.layers[0].bV - before).norm()))
    assert float(grad.norm()) > 1e-7
    assert out["shallow_bias"]["change_after_one_step"] == 0

    # Rank-deficient head: reduced QR supplies arbitrary extra complement columns.
    x = torch.linspace(-1, 1, 32)
    A = torch.stack([x, x, torch.ones_like(x)], dim=1)
    Q, _ = torch.linalg.qr(A)
    U, s, _ = torch.linalg.svd(A, full_matrices=False)
    rank = int((s > s[0] * 1e-14).sum())
    leftover = Q - U[:, :rank] @ (U[:, :rank].T @ Q)
    J = leftover[:, leftover.norm(dim=0).argmax()].reshape(-1, 1)
    J = J / J.norm()
    bad = J - Q @ (Q.T @ J)
    good = J - U[:, :rank] @ (U[:, :rank].T @ J)
    out["rank_projection"] = dict(columns=A.shape[1], rank=rank,
        current_projected_norm=float(bad.norm()), retained_rank_projected_norm=float(good.norm()))
    assert float(bad.norm()) < 1e-12 and float(good.norm()) > 0.9

    # Vector output flattening is sample-major; the current implementation slices output-major.
    gen = torch.Generator().manual_seed(3)
    Q, _ = torch.linalg.qr(torch.randn(11, 3, generator=gen))
    J0 = torch.randn(11 * 2, 5, generator=gen)
    bad = J0 - torch.cat([Q @ (Q.T @ J0[i * 11:(i + 1) * 11]) for i in range(2)], 0)
    J3 = J0.reshape(11, 2, 5)
    good = (J3 - torch.einsum("nr,rs,sqp->nqp", Q, Q.T, J3)).reshape(22, 5)
    orthog_bad = torch.einsum("rn,nqp->rqp", Q.T, bad.reshape(11, 2, 5)).norm()
    orthog_good = torch.einsum("rn,nqp->rqp", Q.T, good.reshape(11, 2, 5)).norm()
    out["multioutput_projection"] = dict(relative_difference=float((bad-good).norm()/good.norm()),
        bad_readout_span_component=float(orthog_bad), correct_readout_span_component=float(orthog_good))
    assert float(orthog_bad) > 1 and float(orthog_good) < 1e-12
    return out


def figures():
    targets = ["fast_waves", "composition", "product_peak"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, targets):
        for knob, color in [("N1", "#2166ac"), ("M", "#1b7837"), ("K", "#762a83")]:
            rows = [(int(k.split("|")[-1]), v) for k, v in A2.items() if k.startswith(f"{name}|{knob}|")]
            rows.sort()
            ax.plot([v["counts"]["params"] for _, v in rows], [v["final"] for _, v in rows],
                    "o-", color=color, label=f"QI: vary {knob}")
        rows = sorted([(int(k.split("|")[-1]), v) for k, v in A2.items() if k.startswith(f"{name}|mlp|")])
        ax.plot([v["counts"]["params"] for _, v in rows], [v["final"] for _, v in rows], "s--", color="0.25", label="tanh MLP")
        ax.set(title=name.replace("_", " "), xlabel="Parameters", xscale="log", yscale="log", ylim=(1e-5, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Saved test relative L2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.01))
    fig.suptitle("I02 A2 contains useful small-block results (d=4, seed 0, 2000 steps)", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / "saved_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    depth_labels = [("2", "#2166ac"), ("3", "#b2182b"), ("4", "#f4a582"), ("3r", "#1b7837")]
    for col, name in enumerate(targets):
        for depth, color in depth_labels:
            v = A2[f"{name}|depth|{depth}"]
            axes[0, col].semilogy(v["step"], v["solved"], color=color, label=f"depth {depth}")
            # Each saved layer statistic is [mean |mu|, mean sd, fraction outside |p|>.9].
            max_mu = [max(layer[0] for layer in row) for row in v["band"]]
            max_exc = [max(layer[2] for layer in row) for row in v["band"]]
            axes[1, col].plot(v["step"], max_mu, color=color)
            axes[2, col].plot(v["step"], max_exc, color=color)
        axes[0, col].set(title=name.replace("_", " "), ylim=(1e-4, 2))
        axes[1, col].set(yscale="symlog", ylim=(0, 100), yticks=[0, 1, 10, 100])
        axes[2, col].set(ylim=(0, 1.02), xlabel="Adam step")
        for row in range(3):
            axes[row, col].grid(alpha=.2)
    axes[0, 0].set_ylabel("Saved test relative L2")
    axes[1, 0].set_ylabel("Largest layer mean |channel mean|")
    axes[2, 0].set_ylabel("Largest fraction outside band")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.01))
    fig.suptitle("Failed deep runs leave the mesh; they do not test stable deep composition", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / "saved_depth_escape.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    evidence = numerical_probes()
    (OUT / "correctness_probes.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    figures()
