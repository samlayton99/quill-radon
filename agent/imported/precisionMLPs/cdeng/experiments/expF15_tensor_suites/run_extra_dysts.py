"""Shared-QI representation and output-rank audit for three larger dysts systems.

Unlike the five expF14 systems, these do not all have a vectorised analytic
Jacobian in this repository, and MacArthur's d=10 dense Gauss-Newton matrix is
too large for the current solver.  This script therefore measures the oracle
interpolation floor of the same N=384 shared time dictionary, then optimally
SVD-compresses the represented trajectory in function space.  Its temporal
factors remain in the shared-QI span.  It answers whether a
rank-constrained solve is worth building; it does not claim to be that solve.
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
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_PATH = HERE / "run.py"
SPEC = importlib.util.spec_from_file_location("expf15_base", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

OUT = ROOT / "cdeng" / "results" / "checkpoint_F_applications" / "expF15_tensor_suites"
N = 384
N_EVAL = 6001
SYSTEMS = ["InteriorSquirmer", "DoublePendulum", "MacArthur"]


def reference(model, T, ts):
    ic = np.atleast_1d(np.asarray(model.ic, dtype=np.float64))

    def rhs(t, y):
        return np.asarray(model.rhs(y, t), dtype=np.float64)

    sol = solve_ivp(rhs, (0.0, T), ic, method="DOP853", rtol=1e-13,
                    atol=1e-14, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.sol(ts).T, int(sol.nfev)


def fit_and_compress(name):
    import dysts.flows as flows

    model = getattr(flows, name)()
    d = int(np.atleast_1d(model.ic).size)
    lyap = float(model.maximum_lyapunov_estimated)
    T = 3.0 / lyap
    ts = np.linspace(0.0, T, N_EVAL)
    Y, nfev = reference(model, T, ts)

    centers, gamma = base.dysts_core.geometry(N)
    p = base.dysts_core.n_params(centers)
    sgrid = 2.0 * ts / T - 1.0
    D = base.dysts_core.dict_rows(sgrid, centers, gamma, 0)
    sigma = np.maximum(np.sqrt(np.mean(Y ** 2, axis=0)), 1e-12)
    A = np.linalg.lstsq(D, Y / sigma[None, :], rcond=base.dysts_core.RCOND)[0]
    C = A * sigma[None, :]
    full = D @ C
    full_error = base._rel(full, Y)

    # Function-space SVD is the optimal rank-r compression.  Raw coefficient
    # SVD is invalid here because the tanh dictionary is non-orthogonal.  Since
    # all columns of `full` lie in span(D), the retained temporal singular
    # vectors do too and can be stored as r QI readouts.
    U, singular, Vt = np.linalg.svd(full, full_matrices=False)
    Uy, sy, Vty = np.linalg.svd(Y, full_matrices=False)
    cells = []
    for rank in range(1, d + 1):
        pred = (U[:, :rank] * singular[:rank]) @ Vt[:rank]
        intrinsic = (Uy[:, :rank] * sy[:rank]) @ Vty[:rank]
        cells.append({
            "rank": rank,
            "parameters": rank * (p + d),
            "relative_l2": base._rel(pred, Y),
            "intrinsic_trajectory_relative_l2": base._rel(intrinsic, Y),
            "coefficient_energy_fraction": float(
                np.sum(singular[:rank] ** 2) / np.sum(singular ** 2)
            ),
        })
    rec = {
        "d": d, "N": N, "W": int(len(centers)), "p": int(p),
        "T": T, "lyapunov": lyap, "period": float(model.period),
        "full_parameters": d * p, "full_interpolation_relative_l2": full_error,
        "reference": "DOP853 rtol=1e-13, atol=1e-14",
        "reference_nfev": nfev, "cells": cells,
    }
    compressed = [c for c in cells if c["parameters"] < rec["full_parameters"]]
    preserving = [c for c in compressed if c["relative_l2"] <= 1.1 * full_error]
    if preserving:
        best = preserving[0]
        suffix = (f"floor preserved at rank-{best['rank']}: "
                  f"{best['relative_l2']:.3e}, P={best['parameters']}")
    else:
        best = compressed[-1]
        suffix = (f"best compressed rank-{best['rank']}: "
                  f"{best['relative_l2']:.3e}, P={best['parameters']}")
    print(f"{name:18s} d={d:2d} floor={full_error:.3e} P={d*p} | {suffix}",
          flush=True)
    return rec


def make_plot(data):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for name, rec in data["systems"].items():
        ax.semilogy([c["parameters"] for c in rec["cells"]],
                    [c["relative_l2"] for c in rec["cells"]], "o-", label=name)
        ax.scatter([rec["full_parameters"]], [rec["full_interpolation_relative_l2"]],
                   marker="x", s=55)
    ax.set_xlabel("stored coefficients (x = dense shared-QI readout)")
    ax.set_ylabel("relative L2")
    ax.set_title("Additional dysts systems: output-rank compression ceiling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "extra_dysts_tensor_compression.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "scope": (
            "Oracle shared-QI interpolation plus optimal function-space rank truncation; "
            "not a collocation ODE solve and not a rank-constrained solve."
        ),
        "systems": {name: fit_and_compress(name) for name in SYSTEMS},
    }
    (OUT / "extra_dysts_data.json").write_text(json.dumps(data, indent=2))
    make_plot(data)
    print(f"wrote {OUT / 'extra_dysts_data.json'}", flush=True)


if __name__ == "__main__":
    main()
