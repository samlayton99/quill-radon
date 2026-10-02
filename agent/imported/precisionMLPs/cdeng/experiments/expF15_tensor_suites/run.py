"""Matched-parameter low-rank tensor audit on BWLer and dysts.

This experiment asks a deliberately narrow question: do the reference solutions
have enough separation structure for a tensor representation to explain the
large parameter-efficiency gain seen on wave and convection?

BWLer: fit an oracle tensor-Chebyshev representation

    u(x, t) = sum_{j=1}^r a_j(x) b_j(t),

where every factor is a degree-m Chebyshev expansion.  The stored coefficient
count is exactly 2*r*(m+1).  Rectangle problems use a Chebyshev-grid transform;
the perforated Poisson problems use alternating least squares on scattered
in-domain points.  These are representation ceilings, NOT PDE solves.

dysts: solve the published shared-QI model at N=384, then optimally truncate the
represented trajectory in function space.  The temporal factors remain in the
shared-QI span.  Rank r stores r*(p+d) coefficients versus d*p for the
uncompressed readout.  This is post-solve compression, NOT a rank-constrained
ODE solve.  A tight DOP853 trajectory is sufficient here because rejected ranks
have errors many orders above its fp64 floor; the uncompressed errors reported
by expF14 remain the certified mpmath numbers.

Run from the repository root:
    .venv/bin/python cdeng/experiments/expF15_tensor_suites/run.py
    .venv/bin/python cdeng/experiments/expF15_tensor_suites/run.py --smoke
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
from numpy.polynomial.chebyshev import chebvander

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BWLER = ROOT / "experiments" / "expF13_bwler_suite"
DYSTS = ROOT / "experiments" / "expF14_dysts_chaos"
sys.path.insert(0, str(BWLER))
import problems  # noqa: E402
sys.path.pop(0)
sys.path.insert(0, str(DYSTS))
import core as dysts_core  # noqa: E402
import reference as dysts_ref  # noqa: E402
import systems as dysts_systems  # noqa: E402

OUT = ROOT / "cdeng" / "results" / "checkpoint_F_applications" / "expF15_tensor_suites"
DEGREES = [16, 24, 32, 48, 64, 96, 128]
RANKS = [1, 2, 4, 8, 16, 32]
BWLER_BASELINES = {
    "convection_c40": {"error": 6.2e-12, "parameters": 9216},
    "convection_c80": {"error": 1.0e-9, "parameters": 9216},
    "reaction": {"error": 4.6e-7, "parameters": 6400},
    "wave": {"error": 1.6e-13, "parameters": 9216},
    "burgers": {"error": 2.9e-1, "parameters": 4096},
    "poisson_cg": {"error": 9.4e-5, "parameters": 4096},
    "poisson_man": {"error": 1.1e-6, "parameters": 4096},
}
DYSTS_CERTIFIED = {
    "Lorenz": 1.1e-13, "Rossler": 4.3e-13, "Thomas": 1.4e-12,
    "Halvorsen": 4.9e-13, "Lorenz96": 1.3e-13,
}


def _rel(pred, truth):
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def _lobatto(degree):
    return np.cos(np.pi * np.arange(degree + 1) / degree)


def _rectangle_target(name, x, y):
    """Matrix with rows x and columns y in scaled BWLer coordinates."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel()])
    if name == "burgers":
        from scipy.interpolate import RegularGridInterpolator

        u, t, xr = problems.load_burgers_reference()
        interp = RegularGridInterpolator((xr, 2.0 * t - 1.0), u.T,
                                         bounds_error=False, fill_value=None)
        return interp(P).reshape(X.shape)
    return problems.PROBLEMS[name]["exact"](P).reshape(X.shape)


def _rectangle_cells(name, degrees, ranks, rng):
    cells = []
    # Independent random evaluation avoids flattering Lobatto interpolation nodes.
    pe = rng.uniform(-1.0, 1.0, (30000, 2))
    if name == "burgers":
        from scipy.interpolate import RegularGridInterpolator

        u, t, x = problems.load_burgers_reference()
        ref = RegularGridInterpolator((x, 2.0 * t - 1.0), u.T,
                                      bounds_error=False, fill_value=None)
        truth = ref(pe)
    else:
        truth = problems.PROBLEMS[name]["exact"](pe)
    for degree in degrees:
        z = _lobatto(degree)
        V = chebvander(z, degree)
        F = _rectangle_target(name, z, z)
        # F = V C V^T.  V is a well-conditioned DCT-like Lobatto transform.
        C = np.linalg.solve(V, F)
        C = np.linalg.solve(V, C.T).T
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        Ve0 = chebvander(pe[:, 0], degree)
        Ve1 = chebvander(pe[:, 1], degree)
        for rank in ranks:
            rr = min(rank, len(s))
            left = Ve0 @ (U[:, :rr] * np.sqrt(s[:rr]))
            right = Ve1 @ (Vt[:rr].T * np.sqrt(s[:rr]))
            pred = np.sum(left * right, axis=1)
            cells.append(dict(degree=degree, rank=rank,
                              parameters=2 * rank * (degree + 1),
                              relative_l2=_rel(pred, truth)))
    return cells


def _poisson_samples(name, n_train, n_eval, rng):
    if name == "poisson_cg":
        P, v = problems.load_poisson_reference()
        order = rng.permutation(len(P))
        split = int(0.75 * len(P))
        return P[order[:split]], v[order[:split]], P[order[split:]], v[order[split:]]

    def draw(n):
        out = []
        while sum(len(a) for a in out) < n:
            q = rng.uniform(-1.0, 1.0, (2 * n, 2))
            out.append(q[problems.in_poisson_domain(q)])
        return np.vstack(out)[:n]

    ptr, pev = draw(n_train), draw(n_eval)
    exact = problems.PROBLEMS[name]["exact"]
    return ptr, exact(ptr), pev, exact(pev)


def _als_tensor(P, values, degree, rank, rng, sweeps=12, ridge=1e-12):
    """Fit sum_j (Tx a_j)(Ty b_j) by alternating linear least squares."""
    Tx, Ty = chebvander(P[:, 0], degree), chebvander(P[:, 1], degree)
    q = degree + 1
    B = rng.standard_normal((q, rank)) / np.sqrt(q)
    A = np.zeros_like(B)
    eye = np.eye(q * rank)
    for _ in range(sweeps):
        Gy = Ty @ B
        DA = np.einsum("nk,nr->nkr", Tx, Gy).reshape(len(P), q * rank)
        A = np.linalg.solve(DA.T @ DA + ridge * eye, DA.T @ values).reshape(q, rank)
        Gx = Tx @ A
        DB = np.einsum("nk,nr->nkr", Ty, Gx).reshape(len(P), q * rank)
        B = np.linalg.solve(DB.T @ DB + ridge * eye, DB.T @ values).reshape(q, rank)
    return A, B


def _poisson_cells(name, degrees, ranks, rng):
    Ptr, ytr, Pev, yev = _poisson_samples(name, 6000, 3000, rng)
    cells = []
    for degree in degrees:
        for rank in ranks:
            # Two deterministic starts; retain by training error, report heldout.
            best = None
            for start in range(2):
                local = np.random.default_rng(1000 * degree + 10 * rank + start)
                A, B = _als_tensor(Ptr, ytr, degree, rank, local, sweeps=8)
                tr = np.sum((chebvander(Ptr[:, 0], degree) @ A) *
                            (chebvander(Ptr[:, 1], degree) @ B), axis=1)
                score = _rel(tr, ytr)
                if best is None or score < best[0]:
                    best = (score, A, B)
            _, A, B = best
            pred = np.sum((chebvander(Pev[:, 0], degree) @ A) *
                          (chebvander(Pev[:, 1], degree) @ B), axis=1)
            cells.append(dict(degree=degree, rank=rank,
                              parameters=2 * rank * (degree + 1),
                              relative_l2=_rel(pred, yev)))
    return cells


def run_bwler(smoke=False):
    degrees = [24, 48] if smoke else DEGREES
    ranks = [1, 2, 4] if smoke else RANKS
    rng = np.random.default_rng(20260826)
    out = {}
    rectangle = ["convection_c40", "convection_c80", "reaction", "wave", "burgers"]
    for name in rectangle:
        cells = _rectangle_cells(name, degrees, ranks, rng)
        out[name] = dict(cells=cells, bwler=BWLER_BASELINES[name], geometry="rectangle")
        best = min(cells, key=lambda c: c["relative_l2"])
        print(f"BWLer {name:16s} best={best['relative_l2']:.3e} "
              f"P={best['parameters']} (r={best['rank']}, m={best['degree']})", flush=True)
    for name in ("poisson_cg", "poisson_man"):
        # Dense normal equations scale with [rank*(degree+1)]^2.  The perforated
        # cases have a float32/reference or geometry ceiling long before the
        # largest rectangle grid, so keep their audit in the useful range.
        poisson_degrees = degrees if smoke else [16, 24, 32, 48, 64]
        poisson_ranks = ranks if smoke else [1, 2, 4, 8]
        cells = _poisson_cells(name, poisson_degrees, poisson_ranks, rng)
        out[name] = dict(cells=cells, bwler=BWLER_BASELINES[name],
                         geometry="perforated_scattered_ALS")
        best = min(cells, key=lambda c: c["relative_l2"])
        print(f"BWLer {name:16s} best={best['relative_l2']:.3e} "
              f"P={best['parameters']} (r={best['rank']}, m={best['degree']})", flush=True)
    return out


def run_dysts(smoke=False):
    N = 128 if smoke else 384
    n_eval = 2001 if smoke else 6001
    out = {}
    for name in dysts_systems.SYSTEM_ORDER:
        S = dysts_systems.System(name)
        T = S.horizon(3.0)
        ts = np.linspace(0.0, T, n_eval)
        Yref, nfev = dysts_ref.rk_trajectory(S, T, ts, 1e-13, 1e-14)
        cell = dysts_core.solve_cell(S, T, N, warm_rtol=1e-8, warm_atol=1e-11)
        scaled_t = 2.0 * ts / T - 1.0
        D = dysts_core.dict_rows(scaled_t, cell["centers"], cell["gamma"], 0)[:, :cell["p"]]
        full_pred = dysts_core.model_trajectory(cell, ts)
        # Compress in the represented-function norm, not the raw coefficient
        # norm: the tanh dictionary is strongly non-orthogonal.  Every temporal
        # left factor below remains in span(D), so this is still a rank-r QI
        # coefficient factorization with r*(p+d) stored numbers.
        U, s, Vt = np.linalg.svd(full_pred, full_matrices=False)
        cells = []
        for rank in range(1, S.d + 1):
            pred = (U[:, :rank] * s[:rank]) @ Vt[:rank]
            cells.append(dict(rank=rank, parameters=rank * (cell["p"] + S.d),
                              relative_l2=_rel(pred, Yref)))
        out[name] = dict(N=N, W=cell["W"], p=cell["p"], d=S.d,
                         full_parameters=S.d * cell["p"],
                         full_vs_dop853_relative_l2=_rel(full_pred, Yref),
                         certified_full_relative_l2=DYSTS_CERTIFIED[name] if not smoke else None,
                         reference="DOP853 rtol=1e-13, atol=1e-14",
                         reference_nfev=nfev, cells=cells)
        best_compressed = cells[-2] if S.d > 1 else cells[-1]
        print(f"dysts  {name:16s} full={out[name]['full_vs_dop853_relative_l2']:.3e} "
              f"P={out[name]['full_parameters']} | rank-{S.d-1}="
              f"{best_compressed['relative_l2']:.3e} P={best_compressed['parameters']}",
              flush=True)
    return out


def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    for name, rec in data["bwler"].items():
        # Plot the attainable Pareto envelope, not every degree/rank crossing.
        ordered = sorted(rec["cells"], key=lambda c: c["parameters"])
        ps, errs, running = [], [], np.inf
        for c in ordered:
            if c["relative_l2"] < running:
                running = c["relative_l2"]
                ps.append(c["parameters"])
                errs.append(running)
        ax.loglog(ps, errs, "o-", ms=3, label=name)
        ax.scatter([rec["bwler"]["parameters"]], [rec["bwler"]["error"]],
                   marker="x", s=40)
    ax.set_xlabel("stored tensor coefficients (x = published Radon point)")
    ax.set_ylabel("relative L2")
    ax.set_title("BWLer suite: oracle tensor representation ceiling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for name, rec in data["dysts"].items():
        ax.semilogy([c["parameters"] for c in rec["cells"]],
                    [c["relative_l2"] for c in rec["cells"]], "o-", label=name)
        if rec["certified_full_relative_l2"] is not None:
            ax.scatter([rec["full_parameters"]], [rec["certified_full_relative_l2"]],
                       marker="x", s=40)
    ax.set_xlabel("stored readout/factor coefficients (x = certified full QI)")
    ax.set_ylabel("relative L2")
    ax.set_title("dysts: optimal output-rank compression of shared-QI trajectories")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "tensor_suite.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    smoke = "--smoke" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    if "--dysts-only" in sys.argv:
        path = OUT / "data.json"
        data = json.loads(path.read_text())
        data["dysts"] = run_dysts(smoke=False)
        path.write_text(json.dumps(data, indent=2))
        plot(data)
        print(f"updated {path} and {OUT / 'tensor_suite.png'}", flush=True)
        return
    data = {
        "scope": {
            "bwler": "oracle tensor-Chebyshev representation ceiling; not a PDE solve",
            "dysts": "optimal function-space rank compression of solved QI trajectory; not rank-constrained solve",
        },
        "bwler": run_bwler(smoke),
        "dysts": run_dysts(smoke),
    }
    path = OUT / ("smoke.json" if smoke else "data.json")
    path.write_text(json.dumps(data, indent=2))
    plot(data)
    print(f"wrote {path} and {OUT / 'tensor_suite.png'}", flush=True)


if __name__ == "__main__":
    main()
