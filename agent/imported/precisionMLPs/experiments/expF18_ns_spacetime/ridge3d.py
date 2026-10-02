"""Space-time ridge dictionary + multi-field Gauss-Newton collocation solve (expF18).

Geometry: the expH06 recipe in d = 3 on the scaled cube X = (x, y, tau). A nested even
direction set (projective farthest-point sequence on the half sphere) and, along each
direction v_j, one 1-D QI block of N evenly spaced offsets on the band
[-T_j, T_j], T_j = 1.25 max_cube |v_j . X| (the support function of the cube: the 25%
collar is the halo), spacing h_j = 2 T_j / N and width gamma_j = 0.25 / h_j.  The three
fields (u, v, p) share the geometry; each has its own readout (+ bias).

Derivative rows: for phi(X) = tanh(g (v . X - t)),
    d^alpha phi = g^|alpha| v^alpha psi_{|alpha|}(tanh(.)),  psi_0 = t, psi_1 = 1 - t^2, ...
so every operator row is a polynomial in the stored tanh values.

The solve: Newton-linearize the advection at the current iterate (u0, v0), stack the
momentum, continuity, wall, initial and gauge rows (each block scaled to O(1) by its max
entry, condition blocks weighted w_mult sqrt(n_pde / n_blk), the expF03/expF17 recipe),
solve J delta = -r by truncated-SVD min-norm least squares (rcond 1e-15; QR-first above
6000 columns, expH06), backtrack on the stacked residual norm.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expH06_ridge_hierarchy"))

from h06.core import (Geometry, make_block, nested_directions, solve_augmented,  # noqa: E402
                      LAMBDA)

import problem as pb  # noqa: E402

RCOND = 1e-15
CUBE_CORNERS = np.array([[sx, sy, st] for sx in (-1, 1) for sy in (-1, 1) for st in (-1, 1)],
                        dtype=np.float64)
FIELDS = ("u", "v", "p")
VAL, DX, DY, DT, DXX, DYY = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0)


def psi(order, t):
    if order == 0:
        return t
    if order == 1:
        return 1.0 - t * t
    if order == 2:
        return -2.0 * t * (1.0 - t * t)
    raise ValueError(order)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def random_rotation(seed):
    """Haar-random rotation in SO(3) (seed 0 = identity): the seed perturbs the
    dictionary, never the problem (expF17 13.9)."""
    if seed == 0:
        return np.eye(3)
    rng = np.random.default_rng(1000 + seed)
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(R))[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


class SpaceTimeDict:
    """M nested directions x N offsets on the cube; `cols` = units + 1 (bias last)."""

    def __init__(self, M, N, seed=0):
        V = nested_directions(3, M) @ random_rotation(seed).T
        self.geom = Geometry([make_block(v, CUBE_CORNERS, N) for v in V])
        D, t, g = self.geom.flat()
        self.D, self.t, self.g = D, t, g          # (W, 3), (W,), (W,)
        self.M, self.N, self.seed = M, N, seed
        self.units = len(t)
        self.cols = self.units + 1
        self.meta = dict(M=M, N=N, units=self.units, seed=seed, lam=LAMBDA,
                         gamma_min=float(g.min()), gamma_max=float(g.max()))

    def tanh_arg(self, X):
        return np.tanh(self.g[None, :] * (X @ self.D.T) - self.g[None, :] * self.t[None, :])

    def rows(self, X, terms, T=None):
        """[len(X), cols] rows of sum_j coeff_j d^alpha_j phi_k(X); terms = [(alpha, coeff)],
        coeff scalar or (n,) array. Pass T = tanh values to reuse them."""
        X = np.asarray(X, dtype=np.float64)
        if T is None:
            T = self.tanh_arg(X)
        A = np.zeros((len(X), self.cols))
        for alpha, coeff in terms:
            o = sum(alpha)
            fac = (self.g ** o) * (self.D[:, 0] ** alpha[0]) * (self.D[:, 1] ** alpha[1]) \
                * (self.D[:, 2] ** alpha[2])
            c = np.asarray(coeff, dtype=np.float64)
            block = fac[None, :] * psi(o, T)
            if c.ndim:
                A[:, :-1] += c[:, None] * block
                if o == 0:
                    A[:, -1] += c
            else:
                A[:, :-1] += float(c) * block
                if o == 0:
                    A[:, -1] += float(c)
        return A

    def op_matrices(self, X, alphas):
        """{alpha: [n, cols]} for several single-derivative operators, one tanh pass."""
        T = self.tanh_arg(X)
        return {a: self.rows(X, [(a, 1.0)], T=T) for a in alphas}


# ---------------------------------------------------------------------------
# point sets on the scaled cube
# ---------------------------------------------------------------------------

def interior_points(n, rng):
    return rng.uniform(-1.0, 1.0, (n, 3))


def face_points(axis, value, n, rng):
    """n random points on the cube face {X[axis] = value}."""
    P = rng.uniform(-1.0, 1.0, (n, 3))
    P[:, axis] = value
    return P


def wall_points(n_per_wall, rng):
    return np.concatenate([face_points(0, -1.0, n_per_wall, rng), face_points(0, 1.0, n_per_wall, rng),
                           face_points(1, -1.0, n_per_wall, rng), face_points(1, 1.0, n_per_wall, rng)])


def ic_points(n, rng):
    return face_points(2, -1.0, n, rng)


def gauge_points(n):
    tau = np.linspace(-1.0, 1.0, n)
    return np.stack([np.full(n, pb.GAUGE_POINT[0]), np.full(n, pb.GAUGE_POINT[1]), tau], axis=1)


def point_sets(n_int, seed, n_gauge=128):
    """Collocation sets for one cell. Counts: walls 4 x n_int/16, IC n_int/8, gauge line
    (rows per column held at 3.75 with n_int = 3 cols)."""
    rng = np.random.default_rng(90_000 + seed)
    Xi = interior_points(n_int, rng)
    Xw = wall_points(max(64, n_int // 16), rng)
    Xic = ic_points(max(64, n_int // 8), rng)
    Xg = gauge_points(n_gauge)
    return dict(Xi=Xi, Xw=Xw, Xic=Xic, Xg=Xg)


# ---------------------------------------------------------------------------
# the model: three readouts on one dictionary
# ---------------------------------------------------------------------------

class Model:
    def __init__(self, d: SpaceTimeDict, a: np.ndarray):
        self.d = d
        self.a = np.asarray(a, dtype=np.float64)      # (3 cols,)
        self.C = d.cols

    def coef(self, field):
        k = FIELDS.index(field)
        return self.a[k * self.C:(k + 1) * self.C]

    def evaluate(self, X, alphas=(VAL,), chunk=4096):
        """{alpha: [n, 3]} derivatives of (u, v, p) at X (chunked)."""
        X = np.asarray(X, dtype=np.float64)
        out = {a: np.empty((len(X), 3)) for a in alphas}
        A = self.a.reshape(3, self.C).T                                    # (C, 3)
        for i in range(0, len(X), chunk):
            sl = slice(i, min(i + chunk, len(X)))
            T = self.d.tanh_arg(X[sl])
            for a in alphas:
                out[a][sl] = self.d.rows(X[sl], [(a, 1.0)], T=T) @ A
        return out

    def fields(self, X, chunk=4096):
        return self.evaluate(X, (VAL,), chunk)[VAL]

    def ns_residual(self, X, chunk=4096):
        """Nonlinear NS residual at X: returns (mom [n, 2], div [n]) in scaled coords."""
        d = self.evaluate(X, (VAL, DX, DY, DT, DXX, DYY), chunk)
        u, v = d[VAL][:, 0:1], d[VAL][:, 1:2]
        gradp = np.stack([d[DX][:, 2], d[DY][:, 2]], axis=1)
        mom = (pb.DT_DTAU * d[DT][:, :2] + u * d[DX][:, :2] + v * d[DY][:, :2] + gradp
               - pb.NU * (d[DXX][:, :2] + d[DYY][:, :2]))
        div = d[DX][:, 0] + d[DY][:, 1]
        return mom, div

    def vorticity(self, X, chunk=4096):
        d = self.evaluate(X, (DX, DY), chunk)
        return d[DX][:, 1] - d[DY][:, 0]


# ---------------------------------------------------------------------------
# oracle regime: fit the exact fields (the representational ceiling)
# ---------------------------------------------------------------------------

def oracle_fit(d: SpaceTimeDict, pts, rcond=RCOND):
    """Truncated lstsq of (u, v, p) on interior points + the closed cube's faces
    (expF17 18.9: the perimeter rows pin the halo columns)."""
    rng = np.random.default_rng(777)
    X = np.vstack([pts["Xi"], pts["Xw"], pts["Xic"], face_points(2, 1.0, len(pts["Xic"]), rng)])
    Y = pb.fields(X)
    t0 = time.time()
    A = d.rows(X, [(VAL, 1.0)])
    fit = solve_augmented(A, Y, rcond=rcond, overwrite_a=True)
    a = fit.coef.T.reshape(-1)                                              # field-major
    return Model(d, a), dict(n_data=len(X), rank=fit.rank, t_solve=round(time.time() - t0, 2))


# ---------------------------------------------------------------------------
# dynamic regime: Gauss-Newton collocation solve
# ---------------------------------------------------------------------------

class Assembly:
    """Everything that does not change across Newton iterations."""

    def __init__(self, d: SpaceTimeDict, pts, w_mult=1.0):
        self.d, self.pts, self.w_mult = d, pts, w_mult
        C = d.cols
        Xi = pts["Xi"]
        self.n_int = len(Xi)
        ops = d.op_matrices(Xi, (VAL, DX, DY, DT, DXX, DYY))
        self.D0, self.Dx, self.Dy = ops[VAL], ops[DX], ops[DY]
        self.Lvisc = pb.DT_DTAU * ops[DT] - pb.NU * (ops[DXX] + ops[DYY])   # time + viscous
        del ops
        n_pde = 3 * self.n_int
        # condition blocks, stored compactly as (field k, B [n, C], y [n]) with the O(1) row
        # scaling and the weight w_mult sqrt(n_pde / n_blk) already applied to B and y
        self.cond = []

        def add(k, B, y):
            s = max(np.abs(B).max(), 1e-300)
            w = w_mult * np.sqrt(n_pde / len(y))
            self.cond.append((k, w * B / s, w * np.asarray(y) / s))

        for X in (pts["Xw"], pts["Xic"]):
            F = pb.fields(X)
            B = d.rows(X, [(VAL, 1.0)])
            add(0, B, F[:, 0])
            add(1, B, F[:, 1])
        Xg = pts["Xg"]
        add(2, d.rows(Xg, [(VAL, 1.0)]), pb.fields(Xg)[:, 2])
        self.n_cond = sum(len(y) for _, _, y in self.cond)
        self.n_rows = n_pde + self.n_cond
        self.n_cols = 3 * C

    def state(self, a):
        C = self.d.cols
        au, av, ap = a[:C], a[C:2 * C], a[2 * C:]
        return dict(u=self.D0 @ au, v=self.D0 @ av, ux=self.Dx @ au, uy=self.Dy @ au,
                    vx=self.Dx @ av, vy=self.Dy @ av, Lu=self.Lvisc @ au, Lv=self.Lvisc @ av,
                    px=self.Dx @ ap, py=self.Dy @ ap)

    def pde_residual(self, a, st=None):
        """Unscaled (momentum-x, momentum-y, continuity) at the interior points."""
        st = st or self.state(a)
        rx = st["Lu"] + st["u"] * st["ux"] + st["v"] * st["uy"] + st["px"]
        ry = st["Lv"] + st["u"] * st["vx"] + st["v"] * st["vy"] + st["py"]
        rc = st["ux"] + st["vy"]
        return rx, ry, rc

    def stacked_residual(self, a, scales):
        rx, ry, rc = self.pde_residual(a)
        parts = [rx / scales[0], ry / scales[1], rc / scales[2]]
        C = self.d.cols
        parts += [B @ a[k * C:(k + 1) * C] - y for (k, B, y) in self.cond]
        r = np.concatenate(parts)
        return r, float(np.linalg.norm(r)), float(max(np.abs(rx).max(), np.abs(ry).max(), np.abs(rc).max()))

    def jacobian(self, a, J=None):
        """Linearized system at a, written into the preallocated J [n_rows, 3C]; returns
        (J, scales, state)."""
        C, n = self.d.cols, self.n_int
        st = self.state(a)
        if J is None:
            J = np.empty((self.n_rows, self.n_cols + 1), order="F")   # last column: the rhs
        # momentum-x: [Lvisc + u0 Dx + v0 Dy + u0_x D0 | u0_y D0 | Dx]
        J[:n, :C] = self.Lvisc
        J[:n, :C] += st["u"][:, None] * self.Dx
        J[:n, :C] += st["v"][:, None] * self.Dy
        J[:n, :C] += st["ux"][:, None] * self.D0
        np.multiply(st["uy"][:, None], self.D0, out=J[:n, C:2 * C])
        J[:n, 2 * C:3 * C] = self.Dx
        # momentum-y: [v0_x D0 | Lvisc + u0 Dx + v0 Dy + v0_y D0 | Dy]
        np.multiply(st["vx"][:, None], self.D0, out=J[n:2 * n, :C])
        J[n:2 * n, C:2 * C] = self.Lvisc
        J[n:2 * n, C:2 * C] += st["u"][:, None] * self.Dx
        J[n:2 * n, C:2 * C] += st["v"][:, None] * self.Dy
        J[n:2 * n, C:2 * C] += st["vy"][:, None] * self.D0
        J[n:2 * n, 2 * C:3 * C] = self.Dy
        # continuity: [Dx | Dy | 0]
        J[2 * n:3 * n, :C] = self.Dx
        J[2 * n:3 * n, C:2 * C] = self.Dy
        J[2 * n:3 * n, 2 * C:3 * C] = 0.0
        scales = []
        for k in range(3):
            blk = J[k * n:(k + 1) * n, :3 * C]
            s = max(np.abs(blk).max(), 1e-300)
            blk /= s
            scales.append(s)
        r0 = 3 * n
        for k, B, y in self.cond:
            J[r0:r0 + len(y), :3 * C] = 0.0
            J[r0:r0 + len(y), k * C:(k + 1) * C] = B
            r0 += len(y)
        return J, scales, st


def solve_stacked(buf, rcond=RCOND):
    """Min-norm truncated-SVD least squares for the F-ordered buffer ``buf = [[A | y]]``
    (a one-element list; the array is consumed and released before the SVD).

    QR of the augmented matrix [A | y] = Q [R  q; 0 rho] gives, with no Q ever formed,
    R = the triangular factor of A and q = Q^T y in the last column (the augmented-QR
    identity).  Then A = Q U_r s V^T from the SVD of R and x = V s^+ U_r^T q, truncated at
    rcond s_max exactly as expH06's ``solve_augmented``.  Peak memory is the buffer during
    the in-place Householder QR, then only the (m+1)^2 factor and its SVD."""
    import scipy.linalg as sla
    A = buf.pop()
    m = A.shape[1] - 1
    Rf = sla.qr(A, mode="r", overwrite_a=True, check_finite=False)[0]
    del A
    R, q = np.ascontiguousarray(Rf[:m, :m]), Rf[:m, m].copy()
    del Rf
    U, sv, Vt = sla.svd(R, full_matrices=False, overwrite_a=True, check_finite=False,
                        lapack_driver="gesdd")
    del R
    keep = sv > rcond * sv[0]
    s_inv = np.where(keep, 1.0 / np.where(keep, sv, 1.0), 0.0)
    x = Vt.T @ (s_inv * (U.T @ q))
    return x, int(keep.sum())


def fit_to_function(asm: Assembly, values):
    """Warm start: lstsq fit of the dictionary to given interior (u, v, p) values plus the
    condition rows (expF17's _fit_to_function, three fields at once)."""
    C, n = asm.d.cols, asm.n_int
    A = np.empty((asm.n_rows, asm.n_cols + 1), order="F")
    for k in range(3):
        A[k * n:(k + 1) * n, :3 * C] = 0.0
        A[k * n:(k + 1) * n, k * C:(k + 1) * C] = asm.D0
        A[k * n:(k + 1) * n, 3 * C] = values[:, k]
    r0 = 3 * n
    for k, B, y in asm.cond:
        A[r0:r0 + len(y), :3 * C] = 0.0
        A[r0:r0 + len(y), k * C:(k + 1) * C] = B
        A[r0:r0 + len(y), 3 * C] = y
        r0 += len(y)
    x, _ = solve_stacked([A], rcond=RCOND)
    return x


def gauss_newton(asm: Assembly, a0=None, max_iter=15, rcond=RCOND, verbose=True, log=None,
                 eval_fn=None):
    """Damped Gauss-Newton with backtracking on the stacked residual norm (expF02/expF17
    semantics). a0 = None starts from zero (step 1 = the Stokes solve). Returns (a, hist)."""
    a = np.zeros(asm.n_cols) if a0 is None else np.asarray(a0, dtype=np.float64).copy()
    hist = []
    t0 = time.time()
    for it in range(1, max_iter + 1):
        J, scales, st = asm.jacobian(a, None)      # F-ordered [J | .], consumed by the solve
        r, rnorm, rmax = asm.stacked_residual(a, scales)
        J[:, asm.n_cols] = -r
        ts = time.time()
        step, rank = solve_stacked([J], rcond=rcond)
        del J
        t_solve = time.time() - ts
        alpha = 1.0
        new_norm = np.inf
        for _ in range(8):
            _, new_norm, new_max = asm.stacked_residual(a + alpha * step, scales)
            if np.isfinite(new_norm) and (new_norm <= rnorm * (1.0 - 1e-4 * alpha) or new_norm < 1e-14):
                break
            alpha *= 0.5
        a = a + alpha * step
        rec = dict(iter=it, res_before=rnorm, res_after=float(new_norm), pde_max_after=float(new_max),
                   alpha=alpha, step=float(alpha * np.linalg.norm(step)), rank=rank,
                   t_solve=round(t_solve, 1), wall=round(time.time() - t0, 1))
        if eval_fn is not None:
            rec.update(eval_fn(a))
        hist.append(rec)
        if verbose:
            msg = (f"    GN {it:2d} stacked {rnorm:.2e} -> {new_norm:.2e}  pde_max {new_max:.2e}  "
                   f"alpha {alpha:.3g}  rank {rank}  solve {t_solve:.0f}s")
            if eval_fn is not None:
                msg += "  " + "  ".join(f"{k} {rec[k]:.2e}" for k in ("rel_l2_v", "rel_l2_p") if k in rec)
            print(msg, flush=True)
            if log is not None:
                log.write(msg + "\n"); log.flush()
        if rec["step"] < 1e-13 * max(1.0, np.linalg.norm(a)):
            break
        if new_norm > 0 and abs(rnorm / new_norm - 1.0) < 1e-3:
            break
    return a, hist


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def eval_grid(nx=65, nt=41):
    g = np.linspace(-1.0, 1.0, nx)
    taus = np.linspace(-1.0, 1.0, nt)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    S = np.stack([GX.ravel(), GY.ravel()], axis=1)
    return g, taus, S


def score(model: Model, nx=65, nt=41):
    """Whole-cube velocity/pressure rel L2 + L_inf, per-slice curves, fresh NS residual."""
    g, taus, S = eval_grid(nx, nt)
    num_v = den_v = num_p = den_p = 0.0
    linf_v = linf_p = 0.0
    per_t = []
    for tau in taus:
        X = np.concatenate([S, np.full((len(S), 1), tau)], axis=1)
        F = pb.fields(X)
        Fh = model.fields(X)
        ev = Fh[:, :2] - F[:, :2]
        ep = Fh[:, 2] - F[:, 2]
        nv, dv = float(np.sum(ev ** 2)), float(np.sum(F[:, :2] ** 2))
        npp, dp = float(np.sum(ep ** 2)), float(np.sum(F[:, 2] ** 2))
        num_v += nv; den_v += dv; num_p += npp; den_p += dp
        lv, lp = float(np.abs(ev).max()), float(np.abs(ep).max())
        linf_v, linf_p = max(linf_v, lv), max(linf_p, lp)
        per_t.append(dict(tau=float(tau), t=float((tau + 1) * pb.T_END / 2), rel_l2_v=np.sqrt(nv / dv),
                          linf_v=lv, rel_l2_p=np.sqrt(npp / dp), linf_p=lp))
    rng = np.random.default_rng(4242)
    Xf = interior_points(4000, rng)
    mom, div = model.ns_residual(Xf)
    return dict(rel_l2_v=float(np.sqrt(num_v / den_v)), linf_v=linf_v,
                rel_l2_p=float(np.sqrt(num_p / den_p)), linf_p=linf_p,
                mom_rms=float(np.sqrt((mom ** 2).mean())), mom_max=float(np.abs(mom).max()),
                div_max=float(np.abs(div).max()), per_t=per_t)


def quick_score(model: Model, n=6000):
    rng = np.random.default_rng(99)
    X = interior_points(n, rng)
    F, Fh = pb.fields(X), model.fields(X)
    return dict(rel_l2_v=float(np.linalg.norm(Fh[:, :2] - F[:, :2]) / np.linalg.norm(F[:, :2])),
                rel_l2_p=float(np.linalg.norm(Fh[:, 2] - F[:, 2]) / np.linalg.norm(F[:, 2])))
