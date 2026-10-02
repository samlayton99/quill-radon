"""Matched-coefficient tensor-QI versus 1-D Radon ridges on three dysts flows.

For t -> R^d, Radon directions collapse to +/-1.  The baseline is therefore W
frozen tanh ridges with uniform offsets on a collar, plus degree-3 polynomials,
and a dense d x (W+4) least-squares readout.  At each tensor rank r we choose the
largest W satisfying d*(W+4) <= r*(p+d).  Lambda, collar, and rcond are oracle-
swept, making the result a representation ceiling rather than an ODE solve.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cdeng/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "cdeng" / "results" / "checkpoint_F_applications" / "expF15_tensor_suites"

spec = importlib.util.spec_from_file_location("extra", HERE / "run_extra_dysts.py")
extra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extra)

LAMBDAS = [0.12, 0.16, 0.20, 0.25, 0.30]
COLLARS = [1.0, 1.3, 1.6]
RCONDS = [1e-13, 1e-15]


def radon_rows(s, width, lam, collar):
    offsets = np.linspace(-collar, collar, width)
    h_ref = 2.8 / width
    gamma = lam / h_ref
    ridges = np.tanh(gamma * (s[:, None] - offsets[None, :]))
    polys = np.column_stack([np.ones_like(s), s, s**2, s**3])
    return np.hstack([ridges, polys])


def best_radon(s, Y, width):
    # Uniform training subset with at least four rows per feature; evaluate on
    # the full 6001-point trajectory.  Endpoints are always retained.
    n_train = min(len(s), max(2001, 4 * (width + 4)))
    idx = np.unique(np.linspace(0, len(s) - 1, n_train).round().astype(int))
    best = None
    for collar in COLLARS:
        for lam in LAMBDAS:
            D = radon_rows(s, width, lam, collar)
            Dt = D[idx]
            for rcond in RCONDS:
                A = np.linalg.lstsq(Dt, Y[idx], rcond=rcond)[0]
                err = extra.base._rel(D @ A, Y)
                rec = {"relative_l2": err, "width": width,
                       "features_with_poly": width + 4,
                       "parameters": Y.shape[1] * (width + 4),
                       "lambda": lam, "collar": collar, "rcond": rcond,
                       "n_train": int(len(idx))}
                if best is None or err < best["relative_l2"]:
                    best = rec
    return best


def run():
    import dysts.flows as flows

    tensor_data = json.loads((OUT / "extra_dysts_data.json").read_text())
    systems = {}
    for name, trec in tensor_data["systems"].items():
        model = getattr(flows, name)()
        T = 3.0 / float(model.maximum_lyapunov_estimated)
        ts = np.linspace(0.0, T, extra.N_EVAL)
        Y, nfev = extra.reference(model, T, ts)
        s = 2.0 * ts / T - 1.0
        cells = []
        for tensor in trec["cells"]:
            # Full-rank tensor factorization is larger than the dense readout and
            # is not a compression candidate.
            if tensor["parameters"] >= trec["full_parameters"]:
                continue
            width = max(4, tensor["parameters"] // trec["d"] - 4)
            radon = best_radon(s, Y, width)
            cells.append({"tensor_rank": tensor["rank"],
                          "budget": tensor["parameters"],
                          "tensor_relative_l2": tensor["relative_l2"],
                          "radon": radon})
            ratio = tensor["relative_l2"] / radon["relative_l2"]
            winner = "tensor" if ratio < 1 else "Radon"
            print(f"{name:18s} P<={tensor['parameters']:4d} rank={tensor['rank']:2d} "
                  f"tensor={tensor['relative_l2']:.3e} Radon={radon['relative_l2']:.3e} "
                  f"({winner}, ratio={ratio:.2e})", flush=True)
        systems[name] = {"d": trec["d"], "reference_nfev": nfev, "cells": cells}
    return {
        "scope": "Oracle representation ceilings at matched trainable coefficients; not ODE solves.",
        "radon": {"geometry": "1-D uniform-offset tanh ridges plus degree-3 polynomial",
                  "lambda_grid": LAMBDAS, "collar_grid": COLLARS,
                  "rcond_grid": RCONDS},
        "systems": systems,
    }


def plot(data):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, (name, rec) in zip(axes, data["systems"].items()):
        p = [c["budget"] for c in rec["cells"]]
        tensor = [c["tensor_relative_l2"] for c in rec["cells"]]
        radon = [c["radon"]["relative_l2"] for c in rec["cells"]]
        ax.semilogy(p, tensor, "o-", label="rank-factorized QI")
        ax.semilogy(p, radon, "s--", label="dense Radon")
        ax.set_title(name)
        ax.set_xlabel("trainable coefficients")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("relative L2")
    axes[0].legend(fontsize=8)
    fig.suptitle("Tensor-QI vs 1-D Radon at matched coefficient budgets")
    fig.tight_layout()
    fig.savefig(OUT / "extra_dysts_tensor_vs_radon.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def main():
    data = run()
    (OUT / "extra_dysts_tensor_vs_radon.json").write_text(json.dumps(data, indent=2))
    plot(data)
    print(f"wrote {OUT / 'extra_dysts_tensor_vs_radon.json'}", flush=True)


if __name__ == "__main__":
    main()
