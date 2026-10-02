"""expF03 part 3: finite-difference baseline (queued last, cancellable).

Scope per plan: 1D ODEs and the space-time square only (irregular-boundary FD on
the disk is its own project). Stencils via Fornberg's algorithm (Fornberg,
Math. Comp. 1988) -- the standard generator of FD weights of any order and
accuracy on arbitrary nodes, which also yields the correct one-sided stencils
at boundaries. Variants: nominal 2nd- and 4th-order accuracy.

1D: differentiation matrices D^(o) on a uniform grid; stack [PDE rows at all
nodes; condition rows] and solve by lstsq (same harness conventions).
Space-time: global 2D FD on a tensor grid (time treated as a coordinate, the
same way our solver treats it): kron(D_x, I) etc., IC/BC rows on the edges.

Verification: differentiation matrices are checked against exact derivatives
of a smooth function with the expected order of accuracy before any run.
"""

from __future__ import annotations

import time

import numpy as np

from common import eval_grid, load_problem_suite, metrics


def fornberg_weights(z, x, m):
    """FD weights at nodes x for derivatives 0..m evaluated at z (Fornberg 1988).
    Returns [len(x), m+1]."""
    n = len(x)
    c = np.zeros((n, m + 1))
    c1, c4 = 1.0, x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2, c5 = 1.0, c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


def diff_matrix(x, order, acc):
    """Dense differentiation matrix of given derivative order and nominal accuracy
    on (uniform) nodes x, with one-sided Fornberg stencils near the ends."""
    n = len(x)
    width = order + acc          # nodes per stencil (order+acc gives >= acc accuracy)
    D = np.zeros((n, n))
    for i in range(n):
        lo = min(max(0, i - width // 2), n - width)
        idx = np.arange(lo, lo + width)
        D[i, idx] = fornberg_weights(x[i], x[idx], order)[:, order]
    return D


def verify_fd(verbose=True):
    f = np.sin
    exact = {1: np.cos, 2: lambda x: -np.sin(x), 3: lambda x: -np.cos(x)}
    for acc in [2, 4]:
        for order in [1, 2, 3]:
            errs = []
            for n in [80, 160]:
                x = np.linspace(-1, 1, n)
                D = diff_matrix(x, order, acc)
                errs.append(np.max(np.abs(D @ f(3 * x) * 3.0 ** (-order)
                                          - exact[order](3 * x))))
            rate = np.log2(errs[0] / errs[1])
            assert rate > acc - 0.7, f"FD order {order} acc {acc}: rate {rate:.2f}"
    if verbose:
        print("FD differentiation matrices verified (convergence rates ok)")


def solve_fd_1d(prob, n, acc):
    t0 = time.time()
    x = np.linspace(-1, 1, n)
    Ds = {0: np.eye(n)}
    for o, _ in prob["terms"]:
        if o not in Ds:
            Ds[o] = diff_matrix(x, o, acc)
    A_pde = np.zeros((n, n))
    for o, coeff in prob["terms"]:
        c = coeff(x)[:, None] if callable(coeff) else float(coeff)
        A_pde += c * Ds[o]
    f = prob["forcing"]
    y_pde = f(x) if callable(f) else np.full(n, float(f))
    s = np.abs(A_pde).max()
    rows, vals = [A_pde / s], [y_pde / s]
    w = np.sqrt(n / max(len(prob["conditions"]), 1))
    for (x0, o) in prob["conditions"]:
        i = int(np.argmin(np.abs(x - x0)))
        if o == 0:
            B = np.zeros((1, n))
            B[0, i] = 1.0
        else:
            if o not in Ds:
                Ds[o] = diff_matrix(x, o, acc)
            B = Ds[o][i:i + 1]
        val = float(np.asarray(prob["exact_derivs"][o](np.array([x0]))).ravel()[0])
        rows.append(w * B)
        vals.append(w * np.array([val]))
    A = np.vstack(rows)
    y = np.concatenate(vals)
    t_asm = time.time() - t0
    t0 = time.time()
    u = np.linalg.lstsq(A, y, rcond=1e-13)[0]
    t_solve = time.time() - t0
    # evaluate on the standard grid by interpolation (cubic via np.interp is linear;
    # use barycentric-free approach: dense grid comparison at the FD nodes is the
    # honest FD metric, but for comparability interpolate with a local quintic fit)
    pts, _ = eval_grid("ode1d")
    u_hat = _poly_interp(x, u, pts, k=min(8, n))
    rel, linf = metrics(u_hat, prob["exact"](pts))
    return dict(rel_l2=rel, linf=linf, dof=n, t_assemble=t_asm, t_solve=t_solve)


def _poly_interp(x, u, pts, k=8):
    """Local Lagrange interpolation of degree k-1 (evaluation off the FD grid)."""
    out = np.empty(len(pts))
    n = len(x)
    for j, p in enumerate(pts):
        i = int(np.searchsorted(x, p))
        lo = min(max(0, i - k // 2), n - k)
        w = fornberg_weights(p, x[lo:lo + k], 0)[:, 0]
        out[j] = w @ u[lo:lo + k]
    return out


def solve_fd_time(prob, nx, nt, acc):
    """Global space-time FD on the (x, tau) tensor grid; same treatment of time
    as the main solver (a coordinate, IC as rows on the tau = -1 edge)."""
    t0 = time.time()
    x = np.linspace(-1, 1, nx)
    tau = np.linspace(-1, 1, nt)
    XX, TT = np.meshgrid(x, tau, indexing="ij")
    P = np.stack([XX.ravel(), TT.ravel()], axis=1)
    N = nx * nt
    Ix, It = np.eye(nx), np.eye(nt)
    Dcache = {}

    def D2d(ax, ay):
        if (ax, ay) not in Dcache:
            Dx = diff_matrix(x, ax, acc) if ax else Ix
            Dt = diff_matrix(tau, ay, acc) if ay else It
            Dcache[(ax, ay)] = np.kron(Dx, Dt)
        return Dcache[(ax, ay)]

    A_pde = np.zeros((N, N))
    for (ax, ay), coeff in prob["terms"]:
        c = coeff(P)[:, None] if callable(coeff) else float(coeff)
        A_pde += c * D2d(ax, ay)
    f = prob["forcing"]
    y_pde = f(P) if callable(f) else np.full(N, float(f))
    s = np.abs(A_pde).max()
    rows, vals = [A_pde / s], [y_pde / s]
    idx = np.arange(N).reshape(nx, nt)
    edges = {"ic": idx[:, 0], "left": idx[0, :], "right": idx[-1, :]}
    for blk in prob["bc_blocks"]:
        sel = edges[blk["where"]]
        B = np.zeros((len(sel), N))
        for (ax, ay), coeff in blk["terms"]:
            c = coeff(P[sel])[:, None] if callable(coeff) else float(coeff)
            B += c * D2d(ax, ay)[sel]
        w = np.sqrt(N / len(sel))
        rows.append(w * B)
        vals.append(w * blk["value"](P[sel]))
    A = np.vstack(rows)
    y = np.concatenate(vals)
    t_asm = time.time() - t0
    t0 = time.time()
    u = np.linalg.lstsq(A, y, rcond=1e-13)[0]
    t_solve = time.time() - t0
    U = u.reshape(nx, nt)
    pts, _ = eval_grid("pde2d_time")
    u_hat = _interp2(x, tau, U, pts)
    rel, linf = metrics(u_hat, prob["exact"](pts))
    return dict(rel_l2=rel, linf=linf, dof=N, t_assemble=t_asm, t_solve=t_solve)


def _interp2(x, tau, U, pts, k=6):
    ux = np.empty((len(pts), len(tau)))
    nx = len(x)
    for j in range(len(tau)):
        ux[:, j] = _poly_interp(x, U[:, j], pts[:, 0], k=min(k, nx))
    out = np.empty(len(pts))
    nt = len(tau)
    for i, p in enumerate(pts):
        jj = int(np.searchsorted(tau, p[1]))
        lo = min(max(0, jj - k // 2), nt - k)
        w = fornberg_weights(p[1], tau[lo:lo + k], 0)[:, 0]
        out[i] = w @ ux[i, lo:lo + k]
    return out


def run_fd(dof_ladder_fn):
    verify_fd()
    f01 = load_problem_suite("f01")
    rows = []
    t_all = time.time()
    for key, prob in f01.PROBLEMS.items():
        if prob["category"] == "pde2d_steady":
            continue  # disk skipped (documented)
        for acc in [2, 4]:
            for size, dof in dof_ladder_fn(key):
                t0 = time.time()
                if prob["category"] == "ode1d":
                    if dof > 4000:
                        continue
                    r = solve_fd_1d(prob, dof, acc)
                else:
                    n = int(round(np.sqrt(dof)))
                    if n * n > 4300:
                        continue
                    r = solve_fd_time(prob, n, n, acc)
                rows.append(dict(key=key, size=size, seed=0, variant=f"fd{acc}", **r))
                print(f"  fd{acc} {key} dof={r['dof']}: relL2={r['rel_l2']:.1e} "
                      f"({time.time() - t0:.1f}s)", flush=True)
    print(f"fd done in {(time.time() - t_all) / 60:.1f} min", flush=True)
    return rows
