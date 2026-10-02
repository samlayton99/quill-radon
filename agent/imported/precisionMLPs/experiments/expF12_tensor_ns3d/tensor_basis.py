"""Tensor-product sech^2 basis network for d=4 PINNs (expF12).

Architecture (spec):
  Layer 1 -- per-axis basis. Each input coordinate i in (x, y, z, t) gets its own
      dedicated group of N hidden units with no cross-talk:
          phi^(i)_k(s) = sech^2(w_{i,k} (s - c_{i,k})),  k = 1..N,
      each unit with its own scalar weight w and center c, plus ONE constant unit
      appended per group (so pure lower-order tensor products -- functions of a
      subset of the coordinates, including constants -- are exactly in the span).
      Group size below is Nt = N + 1.
  Layer 2 -- outer product across groups: the full order-4 tensor
          v[i,j,k,l] = phi^x_i phi^y_j phi^z_k phi^t_l   (Nt^4 features,
      no parameters; a change of basis).
  Layer 3 -- linear readout A in R^{O x Nt^4} to O outputs.

Evaluation contracts, never materializes v: pair (x,y) -> q12 in R^{Nt^2} and
(z,t) -> q34 in R^{Nt^2}, reindex A as (O, Nt^2, Nt^2) by
(i,j,k,l) -> (i*Nt + j, k*Nt + l), then
    f_o(p) = q12 . A_o . q34
-- one broadcast GEMM (q12 @ A) plus a weighted sum against q34. Same function,
same parameters, same gradients (associativity of the product); the per-sample
autograd intermediate drops from B*Nt^4 to B*Nt^2.

Derivatives are ANALYTIC per axis (with t = tanh(z), phi = sech^2(z) = 1 - t^2):
    phi'  = -2 w t phi
    phi'' = w^2 (6 t^2 - 2) phi
so any mixed partial of f is the same contraction with the corresponding
derivative row substituted in the relevant axis group. All float64.
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

AXES = ("x", "y", "z", "t")

# Derivative combos (ox, oy, oz, ot) needed for incompressible Navier-Stokes.
COMBOS_NS = [
    (0, 0, 0, 0),  # value
    (0, 0, 0, 1),  # d/dt
    (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),   # gradient
    (2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0),   # for the Laplacian
]


def uniform_axis_geometry(lo, hi, n_centers, lam, n_halo=2):
    """QI-style uniform 1-D geometry: n_centers centers with spacing h chosen so
    n_halo of them sit outside each end of [lo, hi]; gamma = lam / h."""
    if n_centers < 2 * n_halo + 2:
        raise ValueError("n_centers too small for the halo")
    h = (hi - lo) / (n_centers - 1 - 2 * n_halo)
    c = lo - n_halo * h + h * np.arange(n_centers)
    return c, lam / h


class TensorBasisNet(torch.nn.Module):
    """O outputs on R^4. Trainable: readout A (always); per-axis (w, c) only if
    train_geometry=True (default frozen, per repo convention)."""

    def __init__(self, n_centers, lam, domains, n_out=4, n_halo=2,
                 train_geometry=False):
        super().__init__()
        cs, ws = [], []
        for (lo, hi) in domains:
            c, gamma = uniform_axis_geometry(lo, hi, n_centers, lam, n_halo)
            cs.append(c)
            ws.append(np.full(n_centers, gamma))
        cs = torch.tensor(np.stack(cs))
        ws = torch.tensor(np.stack(ws))
        if train_geometry:
            self.c = torch.nn.Parameter(cs)
            self.w = torch.nn.Parameter(ws)
        else:
            self.register_buffer("c", cs)
            self.register_buffer("w", ws)
        self.n_centers = n_centers
        self.Nt = n_centers + 1                       # + constant unit
        self.n_out = n_out
        self.A = torch.nn.Parameter(torch.zeros(n_out, self.Nt**2, self.Nt**2))

    # -- per-axis features ----------------------------------------------------
    def axis_features(self, s, axis, orders):
        """s [B] -> {order: [B, Nt]} rows of the axis group (constant unit last:
        1 for order 0, 0 for derivatives)."""
        w, c = self.w[axis][None, :], self.c[axis][None, :]
        z = w * (s[:, None] - c)
        t = torch.tanh(z)
        phi = 1.0 - t * t
        B = s.shape[0]
        one = torch.ones(B, 1, dtype=s.dtype, device=s.device)
        zero = torch.zeros(B, 1, dtype=s.dtype, device=s.device)
        out = {}
        for o in orders:
            if o == 0:
                out[o] = torch.cat([phi, one], dim=1)
            elif o == 1:
                out[o] = torch.cat([-2.0 * w * t * phi, zero], dim=1)
            elif o == 2:
                out[o] = torch.cat([w * w * (6.0 * t * t - 2.0) * phi, zero], dim=1)
            else:
                raise ValueError(o)
        return out

    @staticmethod
    def _pair(f1, f2):
        """[B,N1],[B,N2] -> [B, N1*N2], index m = i*N2 + j."""
        return (f1[:, :, None] * f2[:, None, :]).reshape(f1.shape[0], -1)

    # -- contraction ----------------------------------------------------------
    def evaluate(self, X, combos):
        """X [B,4] -> {combo: [B, n_out]} for each derivative combo
        (ox, oy, oz, ot). The (q12 @ A) intermediate is cached per (ox, oy)."""
        feats = [self.axis_features(X[:, a], a, sorted({cb[a] for cb in combos}))
                 for a in range(4)]
        q12c, q34c, rc, out = {}, {}, {}, {}
        for cb in combos:
            k12, k34 = cb[:2], cb[2:]
            if k12 not in q12c:
                q12c[k12] = self._pair(feats[0][k12[0]], feats[1][k12[1]])
            if k34 not in q34c:
                q34c[k34] = self._pair(feats[2][k34[0]], feats[3][k34[1]])
            if k12 not in rc:
                rc[k12] = torch.matmul(q12c[k12], self.A)      # [O, B, Nt^2]
            out[cb] = (rc[k12] * q34c[k34][None]).sum(-1).transpose(0, 1)
        return out

    def evaluate_chunked(self, X, combos, chunk=8192):
        """No-grad evaluation of large point sets."""
        outs = {cb: torch.empty(X.shape[0], self.n_out) for cb in combos}
        with torch.no_grad():
            for i in range(0, X.shape[0], chunk):
                d = self.evaluate(X[i:i + chunk], combos)
                for cb in combos:
                    outs[cb][i:i + chunk] = d[cb]
        return outs

    # -- naive reference (tests only) -----------------------------------------
    def evaluate_naive(self, X, combos):
        """Materialize the full Nt^4 feature tensor. O(B * Nt^4) memory --
        for correctness tests at tiny Nt only."""
        feats = [self.axis_features(X[:, a], a, sorted({cb[a] for cb in combos}))
                 for a in range(4)]
        Aflat = self.A.reshape(self.n_out, -1)                 # (i,j,k,l) flat
        out = {}
        for cb in combos:
            v = torch.einsum("bi,bj,bk,bl->bijkl",
                             feats[0][cb[0]], feats[1][cb[1]],
                             feats[2][cb[2]], feats[3][cb[3]])
            out[cb] = v.reshape(X.shape[0], -1) @ Aflat.T
        return out


# -- Navier-Stokes residuals --------------------------------------------------

def ns_residuals(net, X, nu):
    """Unforced incompressible NS residuals at X [B,4] (fields u,v,w,p):
        mom_i = u_i,t + (u . grad) u_i + p_,i - nu lap u_i,   cont = div u.
    Returns (mom [B,3], cont [B])."""
    d = net.evaluate(X, COMBOS_NS)
    val = d[(0, 0, 0, 0)]
    ut = d[(0, 0, 0, 1)][:, :3]
    gx, gy, gz = d[(1, 0, 0, 0)], d[(0, 1, 0, 0)], d[(0, 0, 1, 0)]
    lap = (d[(2, 0, 0, 0)] + d[(0, 2, 0, 0)] + d[(0, 0, 2, 0)])[:, :3]
    u, v, w = val[:, 0:1], val[:, 1:2], val[:, 2:3]
    adv = u * gx[:, :3] + v * gy[:, :3] + w * gz[:, :3]
    gradp = torch.stack([gx[:, 3], gy[:, 3], gz[:, 3]], dim=1)
    mom = ut + adv + gradp - nu * lap
    cont = gx[:, 0] + gy[:, 1] + gz[:, 2]
    return mom, cont


# -- Kronecker least-squares fit ----------------------------------------------

def kron_fit(net, grids, Y, rcond=1e-13):
    """Least-squares fit of the readout A to targets on a tensor grid.

    grids: four 1-D numpy arrays (x, y, z, t axis points), lengths M_a.
    Y:     targets [Mx, My, Mz, Mt, n_out].

    The design matrix is Phi_x (x) Phi_y (x) Phi_z (x) Phi_t with Phi_a [M_a, Nt]
    the order-0 axis features. The SVD of a Kronecker product is the Kronecker
    product of the factor SVDs, so the GLOBAL min-norm truncated-SVD solution
    (truncation on the PRODUCT singular values s_i s_j s_k s_l, exactly the
    repo's dense rcond convention) is computed mode-wise:
        1. apply U_a^T along each mode,
        2. divide by the product-singular-value tensor, zeroing entries below
           rcond * max product (this is the step per-factor pinv gets wrong:
           it keeps directions whose per-factor singulars pass rcond but whose
           product is ~sigma_min^4, amplifying roundoff),
        3. apply V_a along each mode.
    The dense M^4 x Nt^4 matrix is never formed. Writes the result into net.A
    in place; returns (A, info) with per-axis conds and truncation count."""
    T = np.asarray(Y, dtype=np.float64)
    Us, Ss, Vts = [], [], []
    with torch.no_grad():
        for a in range(4):
            s = torch.tensor(np.asarray(grids[a], dtype=np.float64))
            Phi = net.axis_features(s, a, [0])[0].numpy()      # [M_a, Nt]
            U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
            Us.append(U)
            Ss.append(S)
            Vts.append(Vt)
        for a in range(4):                                     # U_a^T along modes
            T = np.moveaxis(np.tensordot(Us[a].T, T, axes=(1, 0)), 0, 3)
        prod = np.einsum("i,j,k,l->ijkl", *Ss)
        mask = prod >= rcond * prod.max()
        D = np.where(mask, prod, 1.0)
        T = np.where(mask[..., None], T / D[..., None], 0.0)
        for a in range(4):                                     # V_a along modes
            T = np.moveaxis(np.tensordot(Vts[a].T, T, axes=(1, 0)), 0, 3)
        A = np.transpose(T, (4, 0, 1, 2, 3)).reshape(
            net.n_out, net.Nt**2, net.Nt**2)
        net.A.copy_(torch.tensor(A))
    info = dict(conds=[float(S[0] / S[-1]) for S in Ss],
                n_kept=int(mask.sum()), n_total=int(mask.size))
    return net.A, info
