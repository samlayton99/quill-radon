"""Full matched-coefficient interpolation audit: BWLer, Radon, tensor-QI.

This is an oracle *representation* comparison, not a PDE/ODE solve.  Every
method sees values of the reference solution on training points and is scored
on common held-out points.  The default budget is 1156 stored scalar
coefficients.  Fixed nodes, ridge directions, offsets, and QI centers are not
counted; every fitted nodal value/readout coefficient is counted.

Run from the precision-mlps root with the jaxpi environment (which has dysts):
  /scr/cdeng/miniconda3/envs/jaxpi/bin/python \
      cdeng/experiments/expF16_full_matched_suite/run.py
"""

from __future__ import annotations

import argparse
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
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "cdeng" / "results" / "checkpoint_F_applications" / "expF16_full_matched_suite"
sys.path.insert(0, str(ROOT / "experiments" / "expF13_bwler_suite"))
import problems  # noqa: E402
sys.path.pop(0)
sys.path.insert(0, str(ROOT / "experiments" / "expF14_dysts_chaos"))
import reference as original_reference  # noqa: E402
import systems as original_systems  # noqa: E402

PUBLISHED_BWLER = {
    "convection_c40": 2.04e-13, "convection_c80": 1.10e-12,
    "reaction": 6.94e-11, "wave": 1.26e-11, "burgers": 4.63e-3,
    "poisson_cg": 1.08e-2, "poisson_man": None,
}
PDE_NAMES = list(PUBLISHED_BWLER)
EXTRA_DYSTS = ["InteriorSquirmer", "DoublePendulum", "MacArthur"]
LAMS_2D = [0.12, 0.16, 0.25]
LAMS_1D = [0.12, 0.16, 0.25, 0.30]
COLLARS = [1.0, 1.6]
RCOND = 1e-13


def rel(pred, truth):
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def cgl(n):
    return np.cos(np.pi * np.arange(n) / (n - 1))


def cheb_cardinal(x, nodes):
    """Barycentric cardinal matrix at x for CGL nodes."""
    n = len(nodes)
    w = (-1.0) ** np.arange(n)
    w[[0, -1]] *= 0.5
    delta = x[:, None] - nodes[None, :]
    hit = np.abs(delta) < 5e-15
    safe = np.where(hit, 1.0, delta)
    a = w[None, :] / safe
    B = a / np.sum(a, axis=1, keepdims=True)
    rows = np.where(np.any(hit, axis=1))[0]
    for i in rows:
        B[i] = hit[i].astype(float)
    return B


def fourier_cardinal(x, nodes):
    """Real Fourier cardinal matrix for an even uniform periodic grid."""
    n = len(nodes)
    # Direct DFT evaluation.  The Nyquist coefficient is represented by cos.
    k = np.fft.fftfreq(n) * n
    phase_eval = np.exp(1j * np.pi * (x[:, None] + 1.0) * k[None, :])
    phase_nodes = np.exp(-1j * np.pi * (nodes[:, None] + 1.0) * k[None, :])
    return np.real(phase_eval @ phase_nodes.T / n)


def balanced_shape(budget, periodic_first=False):
    best = None
    for nx in range(6, int(np.sqrt(budget)) * 3 + 2):
        if periodic_first and nx % 2:
            continue
        nt = budget // nx
        if nt < 6:
            continue
        score = (nx * nt, -abs(nx - nt))
        if best is None or score > best[0]:
            best = (score, nx, nt)
    return best[1], best[2]


def target_pde(name, P):
    if name == "burgers":
        u, t, x = problems.load_burgers_reference()
        f = RegularGridInterpolator((x, 2.0 * t - 1.0), u.T,
                                    bounds_error=False, fill_value=None)
        return f(P)
    if name == "poisson_cg":
        raise ValueError("poisson_cg only has a scattered reference")
    return problems.PROBLEMS[name]["exact"](P)


def draw_domain(name, n, rng):
    out = []
    while sum(map(len, out)) < n:
        q = rng.uniform(-1.0, 1.0, (2 * n, 2))
        if name.startswith("poisson"):
            q = q[problems.in_poisson_domain(q)]
        out.append(q)
    return np.vstack(out)[:n]


def pde_data(name, rng, n_train=4200, n_eval=10000):
    if name == "poisson_cg":
        P, y = problems.load_poisson_reference()
        order = rng.permutation(len(P))
        split = min(int(0.72 * len(P)), n_train)
        return P[order[:split]], y[order[:split]], P[order[split:]], y[order[split:]]
    Ptr = draw_domain(name, n_train, rng)
    Pev = draw_domain(name, n_eval, rng)
    return Ptr, target_pde(name, Ptr), Pev, target_pde(name, Pev)


def bwler_pde(name, Ptr, ytr, Pev, yev, budget):
    periodic = name.startswith("convection")
    if name in ("poisson_cg", "poisson_man"):
        # A maximum-degree continuation is not automatically the best one on a
        # perforated domain.  Sweep nested BWLer spaces under the same cap.
        best_poisson = None
        max_n = int(np.sqrt(budget))
        sizes = sorted(set([9, 13, 17, 21, 25, 29, max_n]))
        for n in sizes:
            Tx, Ty = chebvander(Ptr[:, 0], n - 1), chebvander(Ptr[:, 1], n - 1)
            D = np.einsum("ni,nj->nij", Tx, Ty).reshape(len(Ptr), n * n)
            U, sv, Vh = np.linalg.svd(D, full_matrices=False)
            uy = U.T @ ytr
            Ex, Ey = chebvander(Pev[:, 0], n - 1), chebvander(Pev[:, 1], n - 1)
            for rcond in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-13):
                keep = sv > rcond * sv[0]
                flat = Vh[keep].T @ (uy[keep] / sv[keep])
                pred = np.einsum("ni,ij,nj->n", Ex, flat.reshape(n, n), Ey)
                rec = {"relative_l2": rel(pred, yev), "parameters": n * n,
                       "shape": [n, n], "bases": ["chebyshev", "chebyshev"],
                       "rcond": rcond, "effective_rank": int(np.sum(keep)),
                       "fit": "scattered value truncated-SVD in equivalent Chebyshev basis"}
                if best_poisson is None or rec["relative_l2"] < best_poisson["relative_l2"]:
                    best_poisson = rec
        return best_poisson
    nx, nt = balanced_shape(budget, periodic)
    xn = np.linspace(-1.0, 1.0, nx, endpoint=False) if periodic else cgl(nx)
    tn = cgl(nt)
    X, T = np.meshgrid(xn, tn, indexing="ij")
    vals = target_pde(name, np.column_stack([X.ravel(), T.ravel()])).reshape(nx, nt)
    Bx = fourier_cardinal(Pev[:, 0], xn) if periodic else cheb_cardinal(Pev[:, 0], xn)
    Bt = cheb_cardinal(Pev[:, 1], tn)
    pred = np.einsum("ni,ij,nj->n", Bx, vals, Bt)
    return {"relative_l2": rel(pred, yev), "parameters": nx * nt,
            "shape": [nx, nt], "bases": ["fourier" if periodic else "chebyshev", "chebyshev"]}


def radon_2d_rows(P, width, lam, collar=1.6):
    j = max(4, int(round(np.sqrt(width))))
    m = int(np.ceil(width / j))
    theta = np.pi * (np.arange(j) + 0.5) / j
    dirs = np.repeat(np.column_stack([np.cos(theta), np.sin(theta)]), m, axis=0)[:width]
    offs = np.tile(np.linspace(-collar, collar, m), j)[:width]
    gamma = lam / (2.8 / np.sqrt(width))
    return np.tanh(gamma * (P @ dirs.T - offs[None, :]))


def radon_pde(Ptr, ytr, Pev, yev, budget):
    width = budget - 1
    best = None
    for collar in COLLARS:
        for lam in LAMS_2D:
            D = np.column_stack([radon_2d_rows(Ptr, width, lam, collar), np.ones(len(Ptr))])
            a = np.linalg.lstsq(D, ytr, rcond=RCOND)[0]
            pred = np.empty(len(Pev))
            for lo in range(0, len(Pev), 2048):
                sl = slice(lo, min(lo + 2048, len(Pev)))
                pred[sl] = radon_2d_rows(Pev[sl], width, lam, collar) @ a[:-1] + a[-1]
            rec = {"relative_l2": rel(pred, yev), "parameters": budget,
                   "width": width, "lambda": lam, "collar": collar}
            if best is None or rec["relative_l2"] < best["relative_l2"]:
                best = rec
    return best


def qi_rows(x, p, lam, collar):
    nr = max(1, p - 4)
    centers = np.linspace(-collar, collar, nr)
    gamma = lam / (2.8 / nr)
    return np.column_stack([np.tanh(gamma * (x[:, None] - centers)),
                            np.ones(len(x)), x, x*x, x*x*x])


def tensor_qi_rectangle(name, budget, Pev, yev):
    # SVD supplies the best separated target factors; each factor is then fit
    # in the original fixed-center 1-D QI dictionary by value least squares.
    n = 257
    z = np.linspace(-1.0, 1.0, n)
    X, T = np.meshgrid(z, z, indexing="ij")
    F = target_pde(name, np.column_stack([X.ravel(), T.ravel()])).reshape(n, n)
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    best = None
    for rank in (1, 2, 4, 8, 16):
        p = min(n, budget // (2 * rank))
        if p < 8:
            continue
        for lam in LAMS_1D:
            for collar in COLLARS:
                D = qi_rows(z, p, lam, collar)
                Ax = np.linalg.lstsq(D, U[:, :rank] * np.sqrt(s[:rank]), rcond=RCOND)[0]
                At = np.linalg.lstsq(D, Vt[:rank].T * np.sqrt(s[:rank]), rcond=RCOND)[0]
                Px = qi_rows(Pev[:, 0], p, lam, collar) @ Ax
                Pt = qi_rows(Pev[:, 1], p, lam, collar) @ At
                pred = np.sum(Px * Pt, axis=1)
                rec = {"relative_l2": rel(pred, yev), "parameters": 2 * rank * p,
                       "rank": rank, "features_per_factor": p,
                       "lambda": lam, "collar": collar, "fit": "SVD factors + value LS"}
                if best is None or rec["relative_l2"] < best["relative_l2"]:
                    best = rec
    return best


def als_qi(Ptr, ytr, Pev, yev, budget):
    best = None
    for rank in (2, 4, 8, 16):
        p = min(128, budget // (2 * rank))
        if p < 8:
            continue
        for lam in (0.16, 0.25):
            Dx = qi_rows(Ptr[:, 0], p, lam, 1.6)
            Dy = qi_rows(Ptr[:, 1], p, lam, 1.6)
            Ex = qi_rows(Pev[:, 0], p, lam, 1.6)
            Ey = qi_rows(Pev[:, 1], p, lam, 1.6)
            for start in range(2):
                rng = np.random.default_rng(10000 + rank * 10 + start)
                B = rng.standard_normal((p, rank)) / np.sqrt(p)
                A = np.zeros_like(B)
                ridge = 1e-10
                eye = np.eye(p * rank)
                for _ in range(7):
                    G = Dy @ B
                    DA = np.einsum("np,nr->npr", Dx, G).reshape(len(Ptr), p * rank)
                    A = np.linalg.solve(DA.T @ DA + ridge * eye, DA.T @ ytr).reshape(p, rank)
                    G = Dx @ A
                    DB = np.einsum("np,nr->npr", Dy, G).reshape(len(Ptr), p * rank)
                    B = np.linalg.solve(DB.T @ DB + ridge * eye, DB.T @ ytr).reshape(p, rank)
                pred = np.sum((Ex @ A) * (Ey @ B), axis=1)
                rec = {"relative_l2": rel(pred, yev), "parameters": 2 * rank * p,
                       "rank": rank, "features_per_factor": p, "lambda": lam,
                       "collar": 1.6, "fit": "scattered value ALS"}
                if best is None or rec["relative_l2"] < best["relative_l2"]:
                    best = rec
    return best


def original_dysts(name, ts=None):
    S = original_systems.System(name)
    T = S.horizon(3.0)
    ts = np.linspace(0.0, T, 6001) if ts is None else np.asarray(ts)
    Y, nfev = original_reference.rk_trajectory(S, T, ts, 1e-13, 1e-14)
    return ts, Y, T, nfev


def extra_dysts(name, ts=None):
    import dysts.flows as flows
    from scipy.integrate import solve_ivp

    model = getattr(flows, name)()
    T = 3.0 / float(model.maximum_lyapunov_estimated)
    ts = np.linspace(0.0, T, 6001) if ts is None else np.asarray(ts)
    y0 = np.asarray(model.ic, dtype=float)
    sol = solve_ivp(lambda t, y: np.asarray(model.rhs(y, t), dtype=float),
                    (0.0, T), y0, t_eval=ts, method="DOP853", rtol=1e-13, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    return ts, sol.y.T, T, sol.nfev


def bwler_1d(s, Y, budget, node_values):
    d = Y.shape[1]
    n = budget // d
    nodes = cgl(n)
    pred = cheb_cardinal(s, nodes) @ node_values
    return {"relative_l2": rel(pred, Y), "parameters": n * d,
            "nodes_per_output": n, "basis": "chebyshev"}


def radon_1d(s, Y, budget):
    d = Y.shape[1]
    p = budget // d
    best = None
    for lam in LAMS_1D:
        for collar in COLLARS:
            D = qi_rows(s, p, lam, collar)
            A = np.linalg.lstsq(D, Y, rcond=RCOND)[0]
            rec = {"relative_l2": rel(D @ A, Y), "parameters": p * d,
                   "features_per_output": p, "lambda": lam, "collar": collar}
            if best is None or rec["relative_l2"] < best["relative_l2"]:
                best = rec
    return best


def tensor_qi_1d(s, Y, budget):
    d = Y.shape[1]
    U, sv, Vt = np.linalg.svd(Y, full_matrices=False)
    best = None
    for rank in range(1, d + 1):
        p = min(len(s), budget // rank - d)
        if p < 8:
            continue
        for lam in LAMS_1D:
            for collar in COLLARS:
                D = qi_rows(s, p, lam, collar)
                A = np.linalg.lstsq(D, U[:, :rank] * sv[:rank], rcond=RCOND)[0]
                pred = (D @ A) @ Vt[:rank]
                rec = {"relative_l2": rel(pred, Y), "parameters": rank * (p + d),
                       "rank": rank, "temporal_features": p,
                       "lambda": lam, "collar": collar,
                       "fit": "trajectory SVD + value LS"}
                if best is None or rec["relative_l2"] < best["relative_l2"]:
                    best = rec
    return best


def run(budget):
    rng = np.random.default_rng(20260826)
    data = {"protocol": {"kind": "oracle representation/interpolation ceiling",
                         "budget": budget, "metric": "relative L2",
                         "counting": "all fitted scalar nodal values/readout coefficients",
                         "not_counted": "fixed nodes, centers, offsets, ridge directions",
                         "selection": "best validation/evaluation configuration (oracle)"},
            "pdes": {}, "dysts": {}}
    for name in PDE_NAMES:
        Ptr, ytr, Pev, yev = pde_data(name, rng)
        b = bwler_pde(name, Ptr, ytr, Pev, yev, budget)
        r = radon_pde(Ptr, ytr, Pev, yev, budget)
        q = (als_qi(Ptr, ytr, Pev, yev, budget) if name.startswith("poisson")
             else tensor_qi_rectangle(name, budget, Pev, yev))
        data["pdes"][name] = {"bwler": b, "radon": r, "tensor_qi": q,
                              "published_bwler_pde_solve_relative_l2": PUBLISHED_BWLER[name],
                              "n_train": len(Ptr), "n_eval": len(Pev)}
        print(f"PDE   {name:18s} B={b['relative_l2']:.3e} R={r['relative_l2']:.3e} "
              f"TQI={q['relative_l2']:.3e}", flush=True)
    for name in original_systems.SYSTEM_ORDER + EXTRA_DYSTS:
        ts, Y, T, nfev = original_dysts(name) if name in original_systems.SYSTEM_ORDER else extra_dysts(name)
        s = 2.0 * ts / T - 1.0
        n_bwler = budget // Y.shape[1]
        bw_nodes = cgl(n_bwler)
        node_ts_ascending = T * (bw_nodes[::-1] + 1.0) / 2.0
        node_run = (original_dysts(name, node_ts_ascending)
                    if name in original_systems.SYSTEM_ORDER else extra_dysts(name, node_ts_ascending))
        node_values = node_run[1][::-1]
        b = bwler_1d(s, Y, budget, node_values)
        r, q = radon_1d(s, Y, budget), tensor_qi_1d(s, Y, budget)
        data["dysts"][name] = {"dimension": Y.shape[1], "horizon": T,
                               "reference_nfev": nfev, "bwler": b,
                               "radon": r, "tensor_qi": q}
        print(f"DYSTS {name:18s} B={b['relative_l2']:.3e} R={r['relative_l2']:.3e} "
              f"TQI={q['relative_l2']:.3e}", flush=True)
    return data


def plot(data):
    names = list(data["pdes"]) + list(data["dysts"])
    rows = [data["pdes"].get(n, data["dysts"].get(n)) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(16, 6.2))
    for method, marker in (("bwler", "o"), ("radon", "s"), ("tensor_qi", "^")):
        ax.semilogy(x, [r[method]["relative_l2"] for r in rows], marker + "-", label=method)
    ax.axvline(len(data["pdes"]) - 0.5, color="0.5", lw=1)
    ax.set_xticks(x, names, rotation=55, ha="right")
    ax.set_ylabel("held-out relative L2")
    ax.set_title(f"Full matched-coefficient oracle benchmark (P <= {data['protocol']['budget']})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "full_matched_suite.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def markdown(data):
    lines = ["# Full matched-parameter interpolation benchmark", "",
             f"Budget: **P <= {data['protocol']['budget']} scalar fitted coefficients**. ",
             "These are oracle representation errors, not PDE/ODE solve errors.", "",
             "| suite | problem | BWLer rel L2 | Radon rel L2 | tensor-QI rel L2 | winner |",
             "|---|---|---:|---:|---:|---|"]
    for suite in ("pdes", "dysts"):
        for name, rec in data[suite].items():
            vals = {m: rec[m]["relative_l2"] for m in ("bwler", "radon", "tensor_qi")}
            winner = min(vals, key=vals.get)
            lines.append(f"| {suite} | {name} | {vals['bwler']:.3e} | {vals['radon']:.3e} | "
                         f"{vals['tensor_qi']:.3e} | {winner} |")
    lines += ["", "BWLer's published PDE-solve errors are retained in `data.json` as a separate field; "
              "they are not directly comparable to the oracle interpolation columns above.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1156)
    ap.add_argument("--repair-bwler-poisson", action="store_true",
                    help="recompute only the two conditioned BWLer Poisson cells in existing data")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.repair_bwler_poisson:
        data = json.loads((OUT / "data.json").read_text())
        rng = np.random.default_rng(20260826)
        # Advance through the identical per-problem draws to reproduce the
        # original train/eval partitions exactly.
        for name in PDE_NAMES:
            Ptr, ytr, Pev, yev = pde_data(name, rng)
            if name.startswith("poisson"):
                b = bwler_pde(name, Ptr, ytr, Pev, yev, args.budget)
                data["pdes"][name]["bwler"] = b
                print(f"repaired {name}: {b['relative_l2']:.3e}", flush=True)
        (OUT / "data.json").write_text(json.dumps(data, indent=2))
        (OUT / "README.md").write_text(markdown(data))
        plot(data)
        return
    data = run(args.budget)
    (OUT / "data.json").write_text(json.dumps(data, indent=2))
    (OUT / "README.md").write_text(markdown(data))
    plot(data)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
