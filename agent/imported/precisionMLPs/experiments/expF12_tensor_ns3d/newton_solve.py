"""Training-free 3D Navier-Stokes solve in the tensor-product sech^2 basis:
frozen QI geometry + Gauss-Newton collocation lstsq for the readout (expF12).

This is the checkpoint-F recipe (expF02/expF06 Newton-lstsq, expF09 multi-field
blocks) carried into the d=4 tensor-product architecture. The geometry is the
QI one -- per-axis uniform centers with halo, gamma = lambda / h, frozen -- and
the readout is SOLVED, never trained.

The obstacle: the Newton-linearized NS operator has variable coefficients
(u0 . grad), which breaks Kronecker separability, so the readout solve cannot
be mode-wise, and a dense lstsq over all 4 Nt^4 coefficients is impossible
(~10^5 unknowns). The fix comes from the ceiling study: fp64 can only use the
product-SVD directions whose PRODUCT singular value survives truncation anyway
(~5-15%). So the Gauss-Newton steps are solved in the REDUCED product-SVD
basis: per axis Phi_a = U_a S_a V_a^T on a reference grid, transformed features
psi_a = feats_a V_a diag(1/S_a) (orthonormal columns on the grid), and only the
top-K tuples by product singular value s_i s_j s_k s_l are kept per field.
Each Newton step is then one dense min-norm lstsq with ~4K unknowns.

Newton linearization of the quadratic advection term B(u,u) = (u.grad)u at u0:
    B(u0, u) + B(u, u0) - B(u0, u0)
so each step solves the linear system (x-momentum shown; y,z cyclic):
    u_t + (u0.grad) u + u dx(u0) + v dy(u0) + w dz(u0) + p_x - nu lap u
        = (u0.grad) u0
    div u = 0;  u = g on faces;  u = u0_ic at t=0;  pressure tap rows.
theta = 0 start makes the first step exactly the Stokes solve. Damped step if
the interior NS residual increases. All numpy float64; block rows scaled to
O(1) max entry before stacking (core_system convention); scipy gelsd lstsq.
"""
from __future__ import annotations

import json
import time

import numpy as np
import scipy.linalg
import torch

import beltrami as bel
import pinn
import tensor_basis as tb

RCOND = 1e-15                    # expF03 part 1: 1e-15 >> 1e-13 on nonlinear


class ReducedTensorBasis:
    """Per-axis transformed sech^2 features + top-K product-direction tuples."""

    def __init__(self, n_centers=12, lam=0.15, domains=bel.DOMAINS,
                 K=2000, m_svd=None):
        self.net = tb.TensorBasisNet(n_centers=n_centers, lam=lam,
                                     domains=domains)
        m = m_svd or 4 * n_centers
        self.T = []                       # per-axis transform [Nt, R]
        Ss = []
        with torch.no_grad():
            for a, (lo, hi) in enumerate(domains):
                g = torch.tensor(np.linspace(lo, hi, m))
                Phi = self.net.axis_features(g, a, [0])[0].numpy()
                _, S, Vt = np.linalg.svd(Phi, full_matrices=False)
                self.T.append(Vt.T / S[None, :])
                Ss.append(S)
        prod = np.einsum("i,j,k,l->ijkl", *Ss)
        flat = prod.ravel()
        order = np.argsort(flat)[::-1][:K]
        self.idx = np.stack(np.unravel_index(order, prod.shape), axis=0)  # [4,K]
        self.K = len(order)
        self.prod_sv = flat[order]

    def axis_feats(self, X, orders_per_axis):
        """X [B,4] -> per-axis dict {order: [B, R]} of transformed features."""
        out = []
        with torch.no_grad():
            for a in range(4):
                fa = self.net.axis_features(torch.tensor(X[:, a]), a,
                                            sorted(orders_per_axis[a]))
                out.append({o: f.numpy() @ self.T[a] for o, f in fa.items()})
        return out

    def op_columns(self, X, terms, feats=None):
        """[B, K] columns of sum_terms coeff * d^orders applied to the K kept
        basis functions. terms: list of (orders 4-tuple, coeff scalar or [B])."""
        need = [set() for _ in range(4)]
        for orders, _ in terms:
            for a in range(4):
                need[a].add(orders[a])
        F = feats or self.axis_feats(X, need)
        out = np.zeros((len(X), self.K))
        for orders, coeff in terms:
            col = (F[0][orders[0]][:, self.idx[0]]
                   * F[1][orders[1]][:, self.idx[1]]
                   * F[2][orders[2]][:, self.idx[2]]
                   * F[3][orders[3]][:, self.idx[3]])
            out += (np.asarray(coeff).reshape(-1, 1) * col
                    if np.ndim(coeff) else float(coeff) * col)
        return out


VAL = (0, 0, 0, 0)
DT = (0, 0, 0, 1)
DX, DY, DZ = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)
DXX, DYY, DZZ = (2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0)
GRADS = (DX, DY, DZ)


def _field_ops(basis, X, theta, ops):
    """Evaluate operators `ops` of the 3 velocity fields (+ pressure last) at X.
    theta: [3K + Kp] stacked (u, v, w, p). Returns {op: [B, 3]} velocities only
    (pressure handled separately where needed)."""
    K = basis.K
    out = {}
    for op in ops:
        C = basis.op_columns(X, [(op, 1.0)])
        out[op] = np.stack([C @ theta[i * K:(i + 1) * K] for i in range(3)],
                           axis=1)
    return out


def ns_residual_interior(basis_v, basis_p, X, theta, nu):
    """Full nonlinear NS residual (RMS over momentum comps + max |div|)."""
    Kv, Kp = basis_v.K, basis_p.K
    d = _field_ops(basis_v, X, theta, [VAL, DT, DX, DY, DZ, DXX, DYY, DZZ])
    u = d[VAL]
    adv = (u[:, 0:1] * d[DX] + u[:, 1:2] * d[DY] + u[:, 2:3] * d[DZ])
    lap = d[DXX] + d[DYY] + d[DZZ]
    gp = np.stack([basis_p.op_columns(X, [(g, 1.0)]) @ theta[3 * Kv:]
                   for g in GRADS], axis=1)
    mom = d[DT] + adv + gp - nu * lap
    div = d[DX][:, 0] + d[DY][:, 1] + d[DZ][:, 2]
    return float(np.sqrt((mom ** 2).mean())), float(np.abs(div).max())


def solve_navier_stokes(n_centers=12, lam=0.15, K_vel=2000, K_p=1000,
                        n_int=5000, n_ic=1500, n_bc=2000, newton_iters=6,
                        seed=0, rcond=RCOND, log_path=None, verbose=True,
                        theta0=None, init="cascade", w_mult=1.0):
    """Gauss-Newton NS solve. Returns (basis_v, basis_p, theta, history).

    expF03 techniques carried over:
    - init="cascade" (expF03 part 1's dominant nonlinear knob): solve at K/4
      first, then polish at full K. In this basis the cascade fit is exact
      embedding -- the top-K/4 product-SVD tuples are a prefix of the top-K
      list, so the coarse solution zero-pads into the full basis.
    - rcond: default 1e-15 via RCOND (expF03: gains 4-5 orders on nonlinear
      problems over 1e-13).
    - Condition blocks (IC/BC/gauge) weighted w_mult * sqrt(n_pde / n_blk)
      after O(1) row scaling (expF03 convention) -- without it a 33-row gauge
      block is drowned by ~24k PDE rows.
    - The no-oracle tuning signal (stacked FRESH residual incl condition rows)
      is logged every iteration as `fresh_sig`.
    theta0: explicit warm start (overrides init; zeros -> first step = Stokes).
    """
    rng = np.random.default_rng(seed)
    basis_v = ReducedTensorBasis(n_centers, lam, K=K_vel)
    basis_p = ReducedTensorBasis(n_centers, lam, K=K_p)
    if theta0 is None and init == "cascade" and K_vel >= 800:
        sub_v, sub_p, th_sub, _ = solve_navier_stokes(
            n_centers, lam, K_vel // 4, K_p // 4, n_int, n_ic, n_bc,
            newton_iters=max(newton_iters, 3), seed=seed, rcond=rcond,
            verbose=verbose, init="zero", w_mult=w_mult)
        theta0 = np.zeros(3 * basis_v.K + basis_p.K)
        for i in range(3):
            theta0[i * basis_v.K:i * basis_v.K + sub_v.K] = \
                th_sub[i * sub_v.K:(i + 1) * sub_v.K]
        assert np.array_equal(sub_v.idx, basis_v.idx[:, :sub_v.K]), \
            "cascade embedding requires prefix-nested tuple selection"
        theta0[3 * basis_v.K:3 * basis_v.K + sub_p.K] = th_sub[3 * sub_v.K:]
        if verbose:
            print(f"[cascade] embedded K={sub_v.K}/{sub_p.K} solution into "
                  f"K={basis_v.K}/{basis_p.K}", flush=True)
    Kv, Kp = basis_v.K, basis_p.K
    ncols = 3 * Kv + Kp

    Xi = pinn.sample_interior(rng, n_int).numpy()
    Xic = pinn.sample_ic(rng, n_ic).numpy()
    Xbc = pinn.sample_bc(rng, n_bc).numpy()
    Xg = pinn.gauge_points().numpy()
    Fic, Fbc, Fg = bel.fields(Xic), bel.fields(Xbc), bel.fields(Xg)
    Xe, Fe = pinn.make_eval_set()
    nu = bel.NU

    # constant (theta-independent) row blocks, pre-scaled to O(1) max entry.
    # Kept as (sparse-ish) pieces and copied into a preallocated stacked matrix
    # each iteration -- vstack of a list would double the peak memory, which
    # matters at the largest K budgets.
    blocks_const = []

    n_pde = 4 * n_int                       # momentum (3) + continuity rows

    def add_block(A, y, condition=True):
        s = np.abs(A).max()
        s = s if s > 0 else 1.0
        wt = w_mult * np.sqrt(n_pde / len(y)) if condition else 1.0
        blocks_const.append((wt * A / s, wt * np.asarray(y) / s))

    # continuity (part of the PDE block: weight 1)
    zeros_p = np.zeros((n_int, Kp))
    A_cont = np.hstack([basis_v.op_columns(Xi, [(DX, 1.0)]),
                        basis_v.op_columns(Xi, [(DY, 1.0)]),
                        basis_v.op_columns(Xi, [(DZ, 1.0)]), zeros_p])
    add_block(A_cont, np.zeros(n_int), condition=False)
    # IC + BC velocity value rows (condition blocks: expF03 sqrt weighting)
    for X, F, n in [(Xic, Fic, n_ic), (Xbc, Fbc, n_bc)]:
        Cv = basis_v.op_columns(X, [(VAL, 1.0)])
        Z = np.zeros((n, Kv))
        Zp = np.zeros((n, Kp))
        for i in range(3):
            row = [Z, Z, Z, Zp]
            row[i] = Cv
            add_block(np.hstack(row), F[:, i])
    # pressure gauge (tap line)
    Cg = basis_p.op_columns(Xg, [(VAL, 1.0)])
    add_block(np.hstack([np.zeros((len(Xg), 3 * Kv)), Cg]), Fg[:, 3])

    # precompute interior features (all orders) once -- reused every iteration
    need = [{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1}]
    Fi_v = basis_v.axis_feats(Xi, need)
    Fi_p = basis_p.axis_feats(Xi, [{0, 1}] * 4)
    visc = [(DT, 1.0), (DXX, -nu), (DYY, -nu), (DZZ, -nu)]
    grad_p_cols = [basis_p.op_columns(Xi, [(g, 1.0)], feats=Fi_p)
                   for g in GRADS]

    # fresh points for the no-oracle tuning signal (never used in the solve)
    rng_f = np.random.default_rng(seed + 7777)
    Xf = pinn.sample_interior(rng_f, 2000).numpy()
    Xf_ic = pinn.sample_ic(rng_f, 500).numpy()
    Xf_bc = pinn.sample_bc(rng_f, 500).numpy()
    Ff_ic, Ff_bc = bel.fields(Xf_ic), bel.fields(Xf_bc)

    def fresh_signal(th):
        mom_rms, div_max = ns_residual_interior(basis_v, basis_p, Xf, th, nu)
        Kv_ = basis_v.K
        errs = []
        for X, F in [(Xf_ic, Ff_ic), (Xf_bc, Ff_bc)]:
            C = basis_v.op_columns(X, [(VAL, 1.0)])
            pred = np.stack([C @ th[i * Kv_:(i + 1) * Kv_] for i in range(3)], 1)
            errs.append(np.abs(pred - F[:, :3]).max())
        return max(mom_rms, div_max, *errs)

    n_const = sum(len(y) for _, y in blocks_const)
    n_rows = 3 * n_int + n_const
    theta = np.zeros(ncols)
    if theta0 is not None:
        theta = np.asarray(theta0, dtype=np.float64).copy()
    history = []
    t0 = time.time()

    def record(it, extra):
        mom_rms, div_max = ns_residual_interior(basis_v, basis_p, Xi[:3000],
                                                theta, nu)
        pred_v = np.stack(
            [basis_v.op_columns(Xe.numpy(), [(VAL, 1.0)])
             @ theta[i * Kv:(i + 1) * Kv] for i in range(3)], axis=1)
        pred_p = basis_p.op_columns(Xe.numpy(), [(VAL, 1.0)]) @ theta[3 * Kv:]
        E = Fe.numpy()
        rel_v = float(np.linalg.norm(pred_v - E[:, :3])
                      / np.linalg.norm(E[:, :3]))
        dp = pred_p - E[:, 3]
        dp -= dp.mean()
        rel_p = float(np.linalg.norm(dp)
                      / np.linalg.norm(E[:, 3] - E[:, 3].mean()))
        rec = dict(iter=it, wall=time.time() - t0, rel_l2_v=rel_v,
                   rel_l2_p=rel_p, mom_rms=mom_rms, div_max=div_max,
                   fresh_sig=fresh_signal(theta), **extra)
        history.append(rec)
        if verbose:
            print(rec, flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    record(0, dict(phase="init"))
    for it in range(1, newton_iters + 1):
        d0 = _field_ops(basis_v, Xi, theta, [VAL, DX, DY, DZ])
        u0 = d0[VAL]
        adv0 = (u0[:, 0:1] * d0[DX] + u0[:, 1:2] * d0[DY]
                + u0[:, 2:3] * d0[DZ])                       # (u0.grad)u0 [B,3]
        Amat = np.empty((n_rows, ncols))
        yvec = np.empty(n_rows)
        Cval = basis_v.op_columns(Xi, [(VAL, 1.0)], feats=Fi_v)
        for i in range(3):                                   # momentum-i
            terms_i = visc + [(DX, u0[:, 0]), (DY, u0[:, 1]), (DZ, u0[:, 2])]
            Ai = basis_v.op_columns(Xi, terms_i, feats=Fi_v)
            r0, r1 = i * n_int, (i + 1) * n_int
            for j in range(3):
                dst = Amat[r0:r1, j * Kv:(j + 1) * Kv]
                # + u_j * d_j(u0_i) coupling
                np.multiply(d0[GRADS[j]][:, i:i + 1], Cval, out=dst)
                if j == i:
                    dst += Ai
            Amat[r0:r1, 3 * Kv:] = grad_p_cols[i]
            s = np.abs(Amat[r0:r1]).max()
            Amat[r0:r1] /= s
            yvec[r0:r1] = adv0[:, i] / s
        r = 3 * n_int
        for A, y in blocks_const:
            Amat[r:r + len(y)] = A
            yvec[r:r + len(y)] = y
            r += len(y)
        t_s = time.time()
        theta_new = scipy.linalg.lstsq(Amat, yvec, cond=rcond,
                                       lapack_driver="gelsy",
                                       overwrite_a=True)[0]
        t_solve = time.time() - t_s
        del Amat
        # damping on the nonlinear interior residual
        base, _ = ns_residual_interior(basis_v, basis_p, Xi[:3000], theta, nu)
        alpha = 1.0
        for _ in range(4):
            cand = theta + alpha * (theta_new - theta)
            r, _ = ns_residual_interior(basis_v, basis_p, Xi[:3000], cand, nu)
            if r < base or it == 1:
                break
            alpha *= 0.5
        theta = theta + alpha * (theta_new - theta)
        record(it, dict(phase="newton", alpha=alpha, t_solve=t_solve))
    return basis_v, basis_p, theta, history
