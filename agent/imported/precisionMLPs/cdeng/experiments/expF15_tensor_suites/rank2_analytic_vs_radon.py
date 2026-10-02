"""Constructive tensor Toeplitz-QI versus Radon on sin(x+y)/sqrt(1+x^2).

The target has exact separation rank two.  Four analytic 1-D factors are built
by the original derivative-convolution/cardinal Toeplitz QI (no fit).  The Radon
baseline gets the same number of trainable readout coefficients and is oracle-
tuned over the expF13 lambda grid using value least squares.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cdeng/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.construction.qi_mpmath import construct_qi, evaluate_qi  # noqa: E402

OUT = ROOT / "cdeng" / "results" / "checkpoint_F_applications" / "expF15_tensor_suites"
N = 47
HALO = 120
KC = 160
LAMS = [0.12, 0.16, 0.20, 0.25, 0.30]
RCOND = 1e-15


def target(P):
    return np.sin(P[:, 0] + P[:, 1]) / np.sqrt(1.0 + P[:, 0] ** 2)


def factors():
    def root(x):
        return np.sqrt(1.0 + x * x)

    return [
        (lambda x: np.sin(x) / root(x),
         lambda x: np.cos(x) / root(x) - x * np.sin(x) / root(x) ** 3),
        (lambda x: np.cos(x) / root(x),
         lambda x: -np.sin(x) / root(x) - x * np.cos(x) / root(x) ** 3),
        (lambda y: np.cos(y), lambda y: -np.sin(y)),
        (lambda y: np.sin(y), lambda y: np.cos(y)),
    ]


def tensor_qi(P):
    qis = [construct_qi(f, fp, N=N, precision="mpmath", lambda_star=0.25,
                        Kc=KC, halo=HALO, mp_dps=100)
           for f, fp in factors()]
    x, y = P[:, 0], P[:, 1]
    vals = [evaluate_qi(q, x if i < 2 else y, kahan=True)
            for i, q in enumerate(qis)]
    pred = vals[0] * vals[2] + vals[1] * vals[3]
    width = len(qis[0].centers)
    # Four factor readouts, each with one bias. Product/mixing weights are fixed
    # by the analytic rank-2 identity and add no fitted scalar.
    return pred, 4 * (width + 1), width


def radon_geometry(max_readout):
    width = max_readout - 1  # one scalar output bias
    J = int(round(np.sqrt(width)))
    M = int(np.ceil(width / J))
    theta = np.pi * (np.arange(J) + 0.5) / J
    offsets = np.linspace(-1.6, 1.6, M)
    dirs = np.repeat(np.column_stack([np.cos(theta), np.sin(theta)]), M, axis=0)
    offs = np.tile(offsets, J)
    return dirs[:width], offs[:width]


def radon_fit(Peval, budget):
    dirs, offs = radon_geometry(budget)
    width = len(offs)
    rng = np.random.default_rng(42)
    Ptr = rng.uniform(-1.0, 1.0, (max(5 * width, 3000), 2))
    ytr, yev = target(Ptr), target(Peval)
    projection_train = Ptr @ dirs.T
    h_ref = 2.8 / np.sqrt(width)
    best = None
    for lam in LAMS:
        gamma = lam / h_ref
        Dtr = np.tanh(gamma * (projection_train - offs[None, :]))
        Dtr = np.hstack([Dtr, np.ones((len(Ptr), 1))])
        sol = np.linalg.lstsq(Dtr, ytr, rcond=RCOND)[0]
        err2, max_abs = 0.0, 0.0
        for start in range(0, len(Peval), 4096):
            sl = slice(start, min(start + 4096, len(Peval)))
            Dev = np.tanh(gamma * (Peval[sl] @ dirs.T - offs[None, :]))
            pred = Dev @ sol[:-1] + sol[-1]
            residual = pred - yev[sl]
            err2 += float(residual @ residual)
            max_abs = max(max_abs, float(np.max(np.abs(residual))))
        err = float(np.sqrt(err2) / np.linalg.norm(yev))
        rec = {"relative_l2": err, "max_abs": max_abs,
               "lambda": lam, "ridge_width": width,
               "readout_parameters": width + 1}
        if best is None or err < best["relative_l2"]:
            best = rec
    return best


def main():
    grid = np.linspace(-0.999, 0.999, 257)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel()])
    truth = target(P)
    pred, budget, shared_width = tensor_qi(P)
    tensor = {"relative_l2": float(np.linalg.norm(pred - truth) / np.linalg.norm(truth)),
              "max_abs": float(np.max(np.abs(pred - truth))),
              "factor_readout_parameters": budget,
              "shared_per_axis_tanh_width": shared_width,
              "rank": 2, "core_intervals": N}
    radon = radon_fit(P, budget)
    data = {"target": "sin(x+y)/sqrt(1+x^2)",
            "scope": "constructive tensor Toeplitz-QI versus oracle-tuned value-lstsq Radon",
            "tensor_qi": tensor, "radon": radon}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rank2_analytic_vs_radon.json").write_text(json.dumps(data, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, Z, title in zip(axes, [truth, pred, np.abs(pred - truth)],
                            ["truth", "tensor Toeplitz-QI", "absolute QI error"]):
        im = ax.pcolormesh(X, Y, Z.reshape(X.shape), shading="auto")
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"rank-2 analytic target: tensor {tensor['relative_l2']:.2e} vs "
                 f"Radon {radon['relative_l2']:.2e} at <= {budget} coefficients")
    fig.tight_layout()
    fig.savefig(OUT / "rank2_analytic_vs_radon.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
