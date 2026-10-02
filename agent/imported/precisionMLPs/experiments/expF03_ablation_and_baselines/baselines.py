"""expF03 part 3: baseline dictionaries in the identical collocation-lstsq harness.

Implemented per literature practice (researched, not from memory -- sources in the
expF03 writeup):
  ELM/PIELM   tanh random features, weights/biases ~ U[-R,R] (Dong & Li 2021);
              R tuned by minimizing the stacked collocation residual (Dong & Yang
              2022 tune R the same way, via differential evolution; we use a grid).
  Kansa RBF   multiquadric sqrt(r^2+c^2) (Kansa's standard) and Gaussian;
              c defaults: Hardy c = 0.815 d_mean-NN, Franke c = 1.25 D/sqrt(N);
              tuned c by the same stacked-residual rule (LOOCV/Rippa is the
              interpolation-setting standard; residual minimization is its PDE
              analogue, cf. Fasshauer & Zhang 2007).
  Spectral    1D: Chebyshev basis, least-squares collocation at CGL nodes
              (Trefethen conventions); square: tensor Chebyshev on a CGL grid;
              disk: total-degree Chebyshev-in-x,y least squares on scattered
              points (Boyd & Yu 2011 name Zernike / Chebyshev-Fourier as the
              disk standards; the total-degree LS variant is the matched-harness
              stand-in and its conditioning is reported honestly -- Vandermonde
              ill-conditioning per Brubeck-Nakatsukasa-Trefethen 2021).

All PDE/condition rows are FD-verified (verify_bases()) before any run.
Every solve records wall-clock assembly/solve, DOF, and matrix shape.

The trained-PINN and finite-difference baselines live in baselines_pinn.py /
baselines_fd.py (built and run separately, fastest-first).
"""

from __future__ import annotations

import time

import numpy as np

from common import (DEFAULT_CFG, bc_points_2d, interior_points_2d, psi,
                    load_problem_suite, eval_grid, metrics, svd_factor, svd_solve)

RCOND = 1e-13


# ---------------------------------------------------------------------------
# generic stacked solve for a "basis" object
# A basis provides rows(points, terms) -> [n, dof] and works for one problem
# category. Terms use the problem's format (1D: (order, coeff); 2D: ((ax,ay),c)).
# ---------------------------------------------------------------------------

def solve_with_basis(prob, basis, cfg=DEFAULT_CFG, n_col_override=None, rng=None):
    """Stack [PDE rows; weighted condition rows], min-norm lstsq. Same scaling and
    weights as the QI harness. Returns dict(rel_l2, linf, signals...)."""
    rng = rng or np.random.default_rng(cfg.seed)
    t0 = time.time()
    if prob["category"] == "ode1d":
        n_col = n_col_override or 4 * basis.dof
        # a basis may demand its own collocation nodes (spectral needs CGL --
        # a high-degree polynomial basis at uniform nodes is Runge-territory)
        x = basis.colloc_1d(n_col) if hasattr(basis, "colloc_1d") else \
            np.linspace(-1, 1, n_col)
        A_pde = basis.rows(x, prob["terms"])
        f = prob["forcing"]
        y_pde = f(x) if callable(f) else np.full(n_col, float(f))
        s = np.abs(A_pde).max()
        rows, vals = [A_pde / s], [y_pde / s]
        w = np.sqrt(n_col / max(len(prob["conditions"]), 1))
        cond_rows = []
        for (x0, o) in prob["conditions"]:
            B = basis.rows(np.array([x0]), [(o, 1.0)])
            val = float(np.asarray(prob["exact_derivs"][o](np.array([x0]))).ravel()[0])
            rows.append(w * B)
            vals.append(w * np.array([val]))
            cond_rows.append((B, np.array([val])))
    else:
        if hasattr(basis, "colloc_2d"):
            P = basis.colloc_2d(max(5 * basis.dof, 2000))
        else:
            P = interior_points_2d(prob["category"], basis.dof, cfg, rng)
        A_pde = basis.rows(P, prob["terms"])
        f = prob["forcing"]
        y_pde = f(P) if callable(f) else np.full(len(P), float(f))
        s = np.abs(A_pde).max()
        rows, vals = [A_pde / s], [y_pde / s]
        cond_rows = []
        from common import _bc_points_for
        suite_f01 = load_problem_suite("f01")
        for blk in prob["bc_blocks"]:
            Pb = _bc_points_for(blk, prob, suite_f01)
            B = basis.rows(Pb, blk["terms"])
            w = np.sqrt(len(P) / len(Pb))
            rows.append(w * B)
            vals.append(w * blk["value"](Pb))
            cond_rows.append((B, blk["value"](Pb)))
    if getattr(basis, "row_normalize", False):
        # spectral derivative rows vary by k^{2r} across CGL points (T_k'' ~ k^4 at
        # the endpoints), so max-entry block scaling drowns the interior rows;
        # least-squares spectral collocation standardly row-equilibrates instead
        for i in range(len(rows)):
            rn = np.abs(rows[i]).max(axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            rows[i] = rows[i] / rn
            vals[i] = vals[i] / rn.ravel()
    A = np.vstack(rows)
    y = np.concatenate(vals)
    t_asm = time.time() - t0
    t0 = time.time()
    fac = svd_factor(A)
    a, info = svd_solve(fac, y, RCOND)
    t_solve = time.time() - t0
    # metrics on the standard fixed grid
    pts, _ = eval_grid(prob["category"])
    u_hat = basis.rows(pts, [((0, 0), 1.0)] if prob["category"] != "ode1d"
                       else [(0, 1.0)]) @ a
    rel, linf = metrics(u_hat, prob["exact"](pts))
    # u*-free tuning signals: fresh stacked residual (PDE + conditions)
    fresh = _fresh_points_for(prob, cfg)
    A_f = basis.rows(fresh, prob["terms"])
    fv = prob["forcing"](fresh) if callable(prob["forcing"]) else \
        np.full(len(np.atleast_1d(fresh[..., 0] if fresh.ndim > 1 else fresh)),
                float(prob["forcing"]))
    res_pde = float(np.max(np.abs(A_f @ a - fv)))
    res_bc = max((float(np.max(np.abs(B @ a - g))) for B, g in cond_rows), default=0.0)
    return dict(rel_l2=rel, linf=linf, dof=basis.dof,
                fresh_res=res_pde, fresh_bc_err=res_bc,
                tune_signal=max(res_pde / max(s, 1e-300), res_bc),
                rank=info["rank"], sigma_ratio=info["sigma_min_kept"] / info["sigma_max"],
                t_assemble=t_asm, t_solve=t_solve,
                n_rows=A.shape[0], n_cols=A.shape[1])


def _fresh_points_for(prob, cfg):
    rng = np.random.default_rng(cfg.seed + 77)
    if prob["category"] == "ode1d":
        return rng.uniform(-1, 1, 2000)
    if prob["category"] == "pde2d_steady":
        rr = np.sqrt(rng.random(2000))
        th = 2 * np.pi * rng.random(2000)
        return np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
    return rng.uniform(-1, 1, (2000, 2))


# ---------------------------------------------------------------------------
# ELM / PIELM random tanh features (Dong & Li: w, b ~ U[-R, R])
# ---------------------------------------------------------------------------

class ElmBasis:
    def __init__(self, dof, R, dim, seed):
        rng = np.random.default_rng(seed)
        self.dof = dof
        self.dim = dim
        self.Wm = rng.uniform(-R, R, (dof, dim))
        self.b = rng.uniform(-R, R, dof)

    def rows(self, pts, terms):
        if self.dim == 1:
            z = pts[:, None] * self.Wm[:, 0][None, :] + self.b[None, :]
            t = np.tanh(z)
            A = np.zeros_like(t)
            for o, coeff in terms:
                cc = coeff(pts).reshape(-1, 1) if callable(coeff) else float(coeff)
                A += cc * (self.Wm[:, 0][None, :] ** o) * psi(o, t)
            return A
        z = pts @ self.Wm.T + self.b[None, :]
        t = np.tanh(z)
        A = np.zeros_like(t)
        for (ax, ay), coeff in terms:
            o = ax + ay
            cc = coeff(pts).reshape(-1, 1) if callable(coeff) else float(coeff)
            fac = (self.Wm[:, 0] ** ax * self.Wm[:, 1] ** ay)[None, :]
            A += cc * fac * psi(o, t)
        return A


# ---------------------------------------------------------------------------
# Kansa RBF: multiquadric and Gaussian, pure-coordinate derivatives to order 3
# ---------------------------------------------------------------------------

def _mq_derivs(s, phi, order):
    """d^n/ds^n of sqrt(s^2 + c^2) given phi = sqrt(s^2+c^2); s = coordinate offset."""
    if order == 0:
        return phi
    if order == 1:
        return s / phi
    if order == 2:
        return 1.0 / phi - s**2 / phi**3
    if order == 3:
        return -3.0 * s / phi**3 + 3.0 * s**3 / phi**5
    raise ValueError(order)


def _gauss_derivs(s, e2, g, order):
    """d^n/ds^n of exp(-e2*r^2) with respect to one coordinate offset s (g = the
    exponential including the OTHER coordinates, i.e. full phi)."""
    if order == 0:
        return g
    if order == 1:
        return -2 * e2 * s * g
    if order == 2:
        return (4 * e2**2 * s**2 - 2 * e2) * g
    if order == 3:
        return (12 * e2**2 * s - 8 * e2**3 * s**3) * g
    raise ValueError(order)


class RbfBasis:
    def __init__(self, dof, c, dim, seed, kind="mq"):
        rng = np.random.default_rng(seed)
        self.dof, self.dim, self.c, self.kind = dof, dim, c, kind
        if dim == 1:
            self.centers = np.sort(rng.uniform(-1, 1, (dof, 1)))
        else:
            n_bnd = max(8, dof // 10)   # denser boundary centers (Wendland/Fedoseyev)
            rr = np.sqrt(rng.random(dof - n_bnd))
            th = 2 * np.pi * rng.random(dof - n_bnd)
            interior = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
            ang = np.linspace(0, 2 * np.pi, n_bnd, endpoint=False)
            bnd = np.stack([np.cos(ang), np.sin(ang)], axis=1)
            self.centers = np.vstack([interior, bnd])

    def rows(self, pts, terms):
        P = pts.reshape(-1, 1) if self.dim == 1 else pts
        diff = P[:, None, :] - self.centers[None, :, :]   # [n, dof, dim]
        r2 = (diff ** 2).sum(axis=2)
        if self.kind == "mq":
            phi = np.sqrt(r2 + self.c ** 2)
        else:
            e2 = 1.0 / (self.c ** 2)
            phi = np.exp(-e2 * r2)
        A = np.zeros((len(P), self.dof))
        for term in terms:
            if self.dim == 1:
                o, coeff = term
                ax, ay = o, 0
            else:
                (ax, ay), coeff = term
            cc = coeff(pts).reshape(-1, 1) if callable(coeff) else float(coeff)
            if ax > 0 and ay > 0:
                # mixed d^2/dxdy (needed for the nonlinear Monge-Ampere u_xy term)
                if not (ax == 1 and ay == 1):
                    raise NotImplementedError("only the (1,1) mixed derivative is supported")
                sx, sy = diff[:, :, 0], diff[:, :, 1]
                if self.kind == "mq":       # d2/dxdy sqrt(sx^2+sy^2+c^2) = -sx sy / phi^3
                    d = -sx * sy / phi ** 3
                else:                        # d2/dxdy exp(-e2 r^2) = 4 e2^2 sx sy g
                    d = 4 * (1.0 / self.c ** 2) ** 2 * sx * sy * phi
                A += cc * d
                continue
            # derivative in ONE coordinate of sqrt(s^2 + (other^2 + c^2)); the other
            # coordinate is absorbed into phi, so the 1D formulas apply directly
            s = diff[:, :, 0] if ax > 0 or ay == 0 else diff[:, :, 1]
            o = ax + ay
            if self.kind == "mq":
                d = _mq_derivs(s, phi, o)
            else:
                d = _gauss_derivs(s, 1.0 / (self.c ** 2), phi, o)
            A += cc * d
        return A


def hardy_c(centers):
    d = centers if centers.ndim == 2 else centers.reshape(-1, 1)
    dists = np.sqrt(((d[:, None, :] - d[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    return 0.815 * float(dists.min(axis=1).mean())


def franke_c(n, diameter=2.0):
    return 1.25 * diameter / np.sqrt(n)


# ---------------------------------------------------------------------------
# spectral: Chebyshev values/derivatives at arbitrary points by recurrence
# ---------------------------------------------------------------------------

def cheb_all(x, deg, max_der):
    """T_k(x) and derivatives, k=0..deg, orders 0..max_der. Returns [orders+1, n, deg+1].
    Recurrences: T_{k+1} = 2xT_k - T_{k-1}, differentiated max_der times."""
    n = len(x)
    out = np.zeros((max_der + 1, n, deg + 1))
    out[0, :, 0] = 1.0
    if deg >= 1:
        out[0, :, 1] = x
        if max_der >= 1:
            out[1, :, 1] = 1.0
    for k in range(1, deg):
        for d in range(max_der + 1):
            val = 2 * x * out[d, :, k] - out[d, :, k - 1]
            if d >= 1:
                val += 2 * d * out[d - 1, :, k]
            out[d, :, k + 1] = val
    return out


def cgl_points(n):
    return np.cos(np.pi * np.arange(n) / (n - 1))[::-1]


class Cheb1dBasis:
    row_normalize = True

    def __init__(self, dof):
        self.dof = dof
        self.deg = dof - 1

    def colloc_1d(self, n):
        return cgl_points(n)

    def rows(self, pts, terms):
        max_o = max(o for o, _ in terms)
        T = cheb_all(pts, self.deg, max_o)
        A = np.zeros((len(pts), self.dof))
        for o, coeff in terms:
            cc = coeff(pts).reshape(-1, 1) if callable(coeff) else float(coeff)
            A += cc * T[o]
        return A


class ChebTensorBasis:
    """Tensor Chebyshev T_i(x) T_j(y) on the square, DOF = (dx+1)(dy+1)."""

    row_normalize = True

    def __init__(self, dx, dy):
        self.dx, self.dy = dx, dy
        self.dof = (dx + 1) * (dy + 1)

    def colloc_2d(self, n):
        m = max(self.dx + 2, int(np.ceil(np.sqrt(n))))
        g = cgl_points(m)
        GX, GY = np.meshgrid(g, g)
        return np.stack([GX.ravel(), GY.ravel()], axis=1)

    def rows(self, P, terms):
        max_ax = max(t[0][0] for t in terms)
        max_ay = max(t[0][1] for t in terms)
        Tx = cheb_all(P[:, 0], self.dx, max_ax)
        Ty = cheb_all(P[:, 1], self.dy, max_ay)
        A = np.zeros((len(P), self.dof))
        for (ax, ay), coeff in terms:
            cc = coeff(P) if callable(coeff) else None
            block = (Tx[ax][:, :, None] * Ty[ay][:, None, :]).reshape(len(P), -1)
            A += (cc.reshape(-1, 1) * block) if cc is not None else float(coeff) * block
        return A


class ChebTotalDegreeBasis:
    """Total-degree <= d Chebyshev-in-x,y (disk). DOF = (d+1)(d+2)/2.
    Conditioning caveat: not orthogonal on the disk; rank via truncated SVD is
    reported. Zernike / Vandermonde-with-Arnoldi are the fully standard routes."""

    row_normalize = True

    def __init__(self, d):
        self.d = d
        self.pairs = [(i, j) for tot in range(d + 1) for i in range(tot + 1)
                      for j in [tot - i]]
        self.dof = len(self.pairs)

    def rows(self, P, terms):
        max_ax = max(t[0][0] for t in terms)
        max_ay = max(t[0][1] for t in terms)
        Tx = cheb_all(P[:, 0], self.d, max_ax)
        Ty = cheb_all(P[:, 1], self.d, max_ay)
        A = np.zeros((len(P), self.dof))
        for (ax, ay), coeff in terms:
            cc = coeff(P).reshape(-1, 1) if callable(coeff) else float(coeff)
            block = np.stack([Tx[ax][:, i] * Ty[ay][:, j] for i, j in self.pairs], axis=1)
            A += cc * block
        return A


def spectral_basis_for(category, target_dof):
    if category == "ode1d":
        return Cheb1dBasis(min(target_dof, 260))          # floor hit long before this
    if category == "pde2d_time":
        d = max(4, int(round(np.sqrt(target_dof))) - 1)
        return ChebTensorBasis(d, d)
    d = max(4, int(round((np.sqrt(8 * target_dof + 1) - 3) / 2)))
    return ChebTotalDegreeBasis(min(d, 60))               # conditioning cap, documented


# ---------------------------------------------------------------------------
# FD verification of every basis's derivative rows
# ---------------------------------------------------------------------------

def _verify_basis(basis, dim, tol=2e-4):
    rng = np.random.default_rng(3)
    a = rng.standard_normal(basis.dof)
    if dim == 1:
        x = rng.uniform(-0.8, 0.8, 25)
        u = lambda xs: basis.rows(xs, [(0, 1.0)]) @ a
        for o in [1, 2, 3]:
            h = {1: 1e-5, 2: 1e-4, 3: 1e-3}[o]
            if o == 1:
                fd = (u(x + h) - u(x - h)) / (2 * h)
            elif o == 2:
                fd = (u(x + h) - 2 * u(x) + u(x - h)) / h**2
            else:
                fd = (u(x + 2 * h) - 2 * u(x + h) + 2 * u(x - h) - u(x - 2 * h)) / (2 * h**3)
            got = basis.rows(x, [(o, 1.0)]) @ a
            scale = max(1.0, np.abs(fd).max())
            assert np.abs(got - fd).max() / scale < tol, \
                f"{type(basis).__name__} order {o}: {np.abs(got - fd).max() / scale:.1e}"
    else:
        P = rng.uniform(-0.6, 0.6, (25, 2))
        u = lambda Q: basis.rows(Q, [((0, 0), 1.0)]) @ a
        for (ax, ay) in [(1, 0), (0, 1), (2, 0), (0, 2), (3, 0), (0, 3), (1, 1)]:
            o = ax + ay
            h = {1: 1e-5, 2: 1e-4, 3: 1e-3}[o]
            if ax > 0 and ay > 0:              # mixed d^2/dxdy central stencil
                def shift2(dx, dy):
                    Q = P.copy(); Q[:, 0] += dx; Q[:, 1] += dy
                    return u(Q)
                fd = (shift2(h, h) - shift2(h, -h) - shift2(-h, h) + shift2(-h, -h)) / (4 * h * h)
                got = basis.rows(P, [((ax, ay), 1.0)]) @ a
                scale = max(1.0, np.abs(fd).max())
                assert np.abs(got - fd).max() / scale < tol, \
                    f"{type(basis).__name__} (1,1): {np.abs(got - fd).max() / scale:.1e}"
                continue
            col = 0 if ax else 1
            def shift(dv):
                Q = P.copy()
                Q[:, col] += dv
                return u(Q)
            if o == 1:
                fd = (shift(h) - shift(-h)) / (2 * h)
            elif o == 2:
                fd = (shift(h) - 2 * shift(0) + shift(-h)) / h**2
            else:
                fd = (shift(2 * h) - 2 * shift(h) + 2 * shift(-h) - shift(-2 * h)) / (2 * h**3)
            got = basis.rows(P, [((ax, ay), 1.0)]) @ a
            scale = max(1.0, np.abs(fd).max())
            assert np.abs(got - fd).max() / scale < tol, \
                f"{type(basis).__name__} ({ax},{ay}): {np.abs(got - fd).max() / scale:.1e}"


def verify_bases(verbose=True):
    _verify_basis(ElmBasis(40, 3.0, 1, seed=1), 1)
    _verify_basis(ElmBasis(40, 3.0, 2, seed=1), 2)
    _verify_basis(RbfBasis(40, 0.5, 1, seed=1, kind="mq"), 1)
    _verify_basis(RbfBasis(40, 0.5, 2, seed=1, kind="mq"), 2)
    _verify_basis(RbfBasis(40, 0.5, 1, seed=1, kind="gauss"), 1)
    _verify_basis(RbfBasis(40, 0.5, 2, seed=1, kind="gauss"), 2)
    _verify_basis(Cheb1dBasis(18), 1)
    _verify_basis(ChebTensorBasis(6, 6), 2)
    _verify_basis(ChebTotalDegreeBasis(8), 2)
    if verbose:
        print("all baseline bases verified against finite differences")
