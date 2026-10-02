"""expF17 dictionary families. One interface, seven families, poly-free baseline.

Every family exposes:
  cols          actual column count (reported on every plot axis -- never nominal)
  rows(P, terms)  -> [len(P), cols] rows of sum_j coeff_j * d^(ax_j,ay_j) phi_k(P),
                  terms = [((ax, ay), coeff)], coeff scalar or callable(P);
                  single-axis derivatives up to order 2 (all tasks need no mixed
                  partials and no order-3 terms)
  meta          dict describing the built configuration (auditable per cell)

Budget frame (SPEC 13.5, 16.2): everything is sized from n_ax so that the exact
column identity holds:
  radon    J = n_ax spokes x M = n_ax + 2 offsets + 1 bias      = (n_ax+1)^2
  tensor   [Phi_x | 1] (x) [Phi_y | 1], n_ax centres per axis   = (n_ax+1)^2
  elm      (n_ax+1)^2 - 1 random tanh units + 1 bias            = (n_ax+1)^2
  spectral (n_ax+1) x (n_ax+1) tensor modes (balanced shape)    = (n_ax+1)^2
  bwler    (n_ax+1) x (n_ax+1) CGL nodal values (value space)   = (n_ax+1)^2
  rbf_imq  (n_ax+1) x (n_ax+1) centre grid                      = (n_ax+1)^2
  rbf_phs  same grid + [1, x, y] tail (PHS is CPD order 1)      = (n_ax+1)^2 + 3

Halo is PAID (inside the budget): radon n_int = M - 2*halo, tensor
n_int = n_ax - 2*halo; interior always spans the square, halo centres sit outside.
Radon is per-spoke (16.2): extent +-T(theta_j) with T = |cos|+|sin| (the square's
support function), EQUAL offset counts per spoke, per-spoke h_j = 2T_j/(n_int-1)
and gamma_j = lam/h_j.

Seeds (13.9) perturb the DICTIONARY, never the problem: radon direction phase in
[0, pi/J) + offset phase in [-h/2, h/2); tensor/rbf grid-origin shift per axis;
elm a fresh W,b draw; spectral/bwler deterministic (collocation RNG only).
"""
from __future__ import annotations

import numpy as np

LAM_TANH = 0.25  # baked (SPEC 17.3): expC03/expC02/expC07/expF03


def psi(order, t):
    """d^order/dz^order tanh(z), written in t = tanh(z)."""
    if order == 0:
        return t
    if order == 1:
        return 1.0 - t * t
    if order == 2:
        return -2.0 * t * (1.0 - t * t)
    raise ValueError(order)


def _coeff_col(coeff, P):
    if callable(coeff):
        return np.asarray(coeff(P), dtype=np.float64).reshape(-1, 1)
    return float(coeff)


class RidgeDict:
    """tanh(a1 x + a2 y - b) columns + one bias column (radon and elm)."""

    def __init__(self, a1, a2, b, meta):
        self.a1, self.a2, self.b = a1, a2, b
        self.cols = len(a1) + 1
        self.meta = meta

    def rows(self, P, terms):
        t = np.tanh(P[:, 0:1] * self.a1[None, :] + P[:, 1:2] * self.a2[None, :]
                    - self.b[None, :])
        A = np.zeros_like(t)
        bias = np.zeros((len(P), 1))
        for (ax, ay), coeff in terms:
            cc = _coeff_col(coeff, P)
            o = ax + ay
            fac = (self.a1 ** ax) * (self.a2 ** ay)
            A += cc * fac[None, :] * psi(o, t)
            if o == 0:
                bias += cc if np.ndim(cc) else np.full((len(P), 1), cc)
        return np.hstack([A, bias])


def radon_dict(n_ax, halo, seed_phase=(0.0, 0.0), lam=LAM_TANH):
    """seed_phase = (direction phase in [0,1), offset phase in [-.5,.5)) fractions."""
    J, M = n_ax, n_ax + 2
    n_int = M - 2 * halo
    if n_int < 3:
        raise ValueError(f"radon: halo {halo} leaves n_int {n_int} < 3")
    dph, oph = seed_phase
    a1, a2, b = [], [], []
    for j in range(J):
        th = np.pi * (j + dph) / J
        c, s = np.cos(th), np.sin(th)
        T = abs(c) + abs(s)
        h = 2.0 * T / (n_int - 1)
        t = -T + (np.arange(M) - halo + oph) * h
        g = lam / h
        a1.append(np.full(M, g * c)); a2.append(np.full(M, g * s)); b.append(g * t)
    return RidgeDict(np.concatenate(a1), np.concatenate(a2), np.concatenate(b),
                     dict(method="qi_radon", n_ax=n_ax, halo=halo, lam=lam,
                          J=J, M=M, n_int=n_int, seed_phase=list(seed_phase)))


def elm_dict(n_ax, R, rng):
    n_units = (n_ax + 1) ** 2 - 1
    a1 = rng.uniform(-R, R, n_units)
    a2 = rng.uniform(-R, R, n_units)
    b = rng.uniform(-R, R, n_units)
    return RidgeDict(a1, a2, b, dict(method="elm", n_ax=n_ax, R=R, n_units=n_units))


class TensorDict:
    """[Phi_x | 1] (x) [Phi_y | 1] Khatri-Rao product, per-axis tanh factors."""

    def __init__(self, n_ax, halo, shift=(0.0, 0.0), lam=LAM_TANH):
        n_int = n_ax - 2 * halo
        if n_int < 3:
            raise ValueError(f"tensor: halo {halo} leaves n_int {n_int} < 3")
        h = 2.0 / (n_int - 1)
        self.cx = -1.0 + (np.arange(n_ax) - halo + shift[0]) * h
        self.cy = -1.0 + (np.arange(n_ax) - halo + shift[1]) * h
        self.g = lam / h
        self.cols = (n_ax + 1) ** 2
        self.meta = dict(method="qi_tensor", n_ax=n_ax, halo=halo, lam=lam,
                         n_int=n_int, shift=list(shift))

    def _factor(self, x, c, order):
        t = np.tanh(self.g * (x[:, None] - c[None, :]))
        F = (self.g ** order) * psi(order, t)
        ones = np.ones((len(x), 1)) if order == 0 else np.zeros((len(x), 1))
        return np.hstack([F, ones])

    def rows(self, P, terms):
        out = None
        for (ax, ay), coeff in terms:
            cc = _coeff_col(coeff, P)
            Fx = self._factor(P[:, 0], self.cx, ax)
            Fy = self._factor(P[:, 1], self.cy, ay)
            block = np.einsum("ip,iq->ipq", Fx, Fy).reshape(len(P), -1)
            block = block * cc if np.ndim(cc) else block * cc
            out = block if out is None else out + block
        return out


def _shape_for(cols_target, aspect):
    """(nx, ny) with nx*ny ~ cols_target and ny/nx ~ aspect. Exact at aspect 1;
    otherwise the actual product is what gets reported (never the nominal)."""
    nx = max(2, int(round(np.sqrt(cols_target / aspect))))
    ny = max(2, int(round(cols_target / nx)))
    return nx, ny


class RBFDict:
    """IMQ (positive definite, no tail) or PHS r^3 + degree-1 tail, grid centres.
    aspect = ny/nx of the centre grid at fixed budget (SPEC 18.8)."""

    def __init__(self, n_ax, kind, epsh=1.0, shift=(0.0, 0.0), aspect=1.0):
        nx, ny = _shape_for((n_ax + 1) ** 2, aspect)
        hx, hy = 2.0 / (nx - 1), 2.0 / (ny - 1)
        gx = np.linspace(-1.0, 1.0, nx) + shift[0] * hx
        gy = np.linspace(-1.0, 1.0, ny) + shift[1] * hy
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        self.C = np.stack([GX.ravel(), GY.ravel()], axis=1)
        self.kind = kind
        self.eps = epsh / np.sqrt(hx * hy) if kind == "imq" else None
        self.n_poly = 3 if kind == "phs" else 0
        self.cols = len(self.C) + self.n_poly
        self.meta = dict(method=f"rbf_{kind}", n_ax=n_ax, epsh=(epsh if kind == "imq" else None),
                         n_centres=len(self.C), shift=list(shift),
                         aspect=aspect, shape=[nx, ny])

    def _derivs(self, P, ax, ay):
        dx = P[:, 0:1] - self.C[None, :, 0]
        dy = P[:, 1:2] - self.C[None, :, 1]
        r2 = dx * dx + dy * dy
        if self.kind == "imq":
            e2 = self.eps ** 2
            s = 1.0 + e2 * r2
            if (ax, ay) == (0, 0):
                return s ** -0.5
            if (ax, ay) == (1, 0):
                return -e2 * dx * s ** -1.5
            if (ax, ay) == (0, 1):
                return -e2 * dy * s ** -1.5
            if (ax, ay) == (2, 0):
                return -e2 * s ** -1.5 + 3.0 * e2 * e2 * dx * dx * s ** -2.5
            if (ax, ay) == (0, 2):
                return -e2 * s ** -1.5 + 3.0 * e2 * e2 * dy * dy * s ** -2.5
        else:  # phs r^3
            r = np.sqrt(r2)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_r = np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), 0.0)
            if (ax, ay) == (0, 0):
                return r * r2
            if (ax, ay) == (1, 0):
                return 3.0 * r * dx
            if (ax, ay) == (0, 1):
                return 3.0 * r * dy
            if (ax, ay) == (2, 0):
                return 3.0 * r + 3.0 * dx * dx * inv_r
            if (ax, ay) == (0, 2):
                return 3.0 * r + 3.0 * dy * dy * inv_r
        raise ValueError((ax, ay))

    def _poly(self, P, ax, ay):
        n = len(P)
        one = np.ones(n) if (ax, ay) == (0, 0) else np.zeros(n)
        px = P[:, 0] if (ax, ay) == (0, 0) else (np.ones(n) if (ax, ay) == (1, 0) else np.zeros(n))
        py = P[:, 1] if (ax, ay) == (0, 0) else (np.ones(n) if (ax, ay) == (0, 1) else np.zeros(n))
        return np.stack([one, px, py], axis=1)

    def rows(self, P, terms):
        out = np.zeros((len(P), self.cols))
        for (ax, ay), coeff in terms:
            cc = _coeff_col(coeff, P)
            blk = self._derivs(P, ax, ay)
            if self.n_poly:
                blk = np.hstack([blk, self._poly(P, ax, ay)])
            out += cc * blk if np.ndim(cc) else cc * blk
        return out


def _cheb_basis(x, n, order):
    """[len(x), n] matrix of d^order T_k(x), k = 0..n-1, by derivative recurrences."""
    x = np.asarray(x, dtype=np.float64)
    T = np.empty((len(x), n)); dT = np.zeros((len(x), n)); d2T = np.zeros((len(x), n))
    T[:, 0] = 1.0
    if n > 1:
        T[:, 1] = x; dT[:, 1] = 1.0
    for k in range(1, n - 1):
        T[:, k + 1] = 2.0 * x * T[:, k] - T[:, k - 1]
        dT[:, k + 1] = 2.0 * T[:, k] + 2.0 * x * dT[:, k] - dT[:, k - 1]
        d2T[:, k + 1] = 4.0 * dT[:, k] + 2.0 * x * d2T[:, k] - d2T[:, k - 1]
    return (T, dT, d2T)[order]


def _fourier_basis(x, n, order):
    """[len(x), n] of d^order f_k, f = [1, cos(pi x), sin(pi x), cos(2 pi x), ...]."""
    x = np.asarray(x, dtype=np.float64)
    B = np.empty((len(x), n))
    for k in range(n):
        if k == 0:
            B[:, k] = 1.0 if order == 0 else 0.0
            continue
        m = (k + 1) // 2
        w = m * np.pi
        is_cos = (k % 2 == 1)
        ph = x * w
        if is_cos:
            f = [np.cos(ph), -w * np.sin(ph), -w * w * np.cos(ph)][order]
        else:
            f = [np.sin(ph), w * np.cos(ph), -w * w * np.sin(ph)][order]
        B[:, k] = f
    return B


class SpectralDict:
    """Tensor-product spectral: Fourier on xi where the task is analytically
    periodic, Chebyshev otherwise; Chebyshev on eta. Balanced (n x n) shape
    (declared benchmark rule, SPEC 16.6). Zero hyperparameters."""

    def __init__(self, n_ax, fourier_x=False, aspect=1.0):
        self.nx, self.ny = _shape_for((n_ax + 1) ** 2, aspect)
        self.fx = fourier_x
        self.cols = self.nx * self.ny
        self.meta = dict(method="spectral", n_ax=n_ax,
                         basis_x=("fourier" if fourier_x else "chebyshev"),
                         basis_y="chebyshev", aspect=aspect,
                         shape=[self.nx, self.ny])

    def _bx(self, x, order):
        return (_fourier_basis if self.fx else _cheb_basis)(x, self.nx, order)

    def rows(self, P, terms):
        out = None
        for (ax, ay), coeff in terms:
            cc = _coeff_col(coeff, P)
            Bx = self._bx(P[:, 0], ax)
            By = _cheb_basis(P[:, 1], self.ny, ay)
            blk = np.einsum("ip,iq->ipq", Bx, By).reshape(len(P), -1)
            out = (blk * cc if np.ndim(cc) else blk * cc) if out is None \
                else out + (blk * cc if np.ndim(cc) else blk * cc)
        return out


class BWLerDict:
    """Explicit BWLer, value-space: unknowns are nodal values on the CGL grid,
    rows are (Chebyshev rows) @ V^{-1} per axis, where V_{jk} = T_k(cgl_j).
    Same span as SpectralDict (chebyshev x chebyshev), different parameterization
    (SPEC 15.4: keep both, labelled honestly). Spectral derivatives (fd_k=None):
    under a direct lstsq solve the FD variants have no mechanism to win (the
    optimization term in BWLer Thm 5.1 is zero and FD misspecification is pure
    loss), so the fd_k check is closed by argument, not by sweep."""

    def __init__(self, n_ax, aspect=1.0):
        self.nx, self.ny = _shape_for((n_ax + 1) ** 2, aspect)
        self.Vinv = {}
        for n in {self.nx, self.ny}:
            nodes = np.cos(np.arange(n) * np.pi / (n - 1))  # CGL, endpoints in
            self.Vinv[n] = np.linalg.solve(_cheb_basis(nodes, n, 0), np.eye(n))
        self.cols = self.nx * self.ny
        self.meta = dict(method="bwler", n_ax=n_ax, fd_k="spectral",
                         nodes="cgl", aspect=aspect, shape=[self.nx, self.ny])

    def rows(self, P, terms):
        out = None
        for (ax, ay), coeff in terms:
            cc = _coeff_col(coeff, P)
            Lx = _cheb_basis(P[:, 0], self.nx, ax) @ self.Vinv[self.nx]
            Ly = _cheb_basis(P[:, 1], self.ny, ay) @ self.Vinv[self.ny]
            blk = np.einsum("ip,iq->ipq", Lx, Ly).reshape(len(P), -1)
            out = (blk * cc if np.ndim(cc) else blk * cc) if out is None \
                else out + (blk * cc if np.ndim(cc) else blk * cc)
        return out


MONO_2D = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
           (3, 0), (2, 1), (1, 2), (0, 3)]


def _ffact(d, o):
    out = 1
    for k in range(o):
        out *= (d - k)
    return out


class PolyAugmented:
    """base dictionary + the checkpoint-F degree-3 monomial block (Plot 3 only,
    applied to radon AND tensor identically -- never to one side)."""

    def __init__(self, base):
        self.base = base
        self.cols = base.cols + len(MONO_2D)
        self.meta = dict(base.meta, poly="deg3")

    def rows(self, P, terms):
        A = self.base.rows(P, terms)
        polys = np.zeros((len(P), len(MONO_2D)))
        x, y = P[:, 0], P[:, 1]
        for (ax, ay), coeff in terms:
            cc = _coeff_col(coeff, P)
            ccr = cc.ravel() if np.ndim(cc) else cc
            for k, (px, py) in enumerate(MONO_2D):
                if ax <= px and ay <= py:
                    polys[:, k] += ccr * (_ffact(px, ax) * _ffact(py, ay)
                                          * x ** (px - ax) * y ** (py - ay))
        return np.hstack([A, polys])


def dict_rng(method, n_ax, seed):
    """Geometry-jitter RNG, a pure function of (method, n_ax, seed) so the oracle
    and dynamic regimes of one cell see the IDENTICAL jittered dictionary."""
    tag = sum(ord(c) * (37 ** i) for i, c in enumerate(method)) % 100_000
    return np.random.default_rng(1_000_003 * seed + 101 * n_ax + tag)


def n_ax_for(C_nominal):
    """nominal column budget -> n_ax (actual columns = (n_ax+1)^2, reported)."""
    return int(round(np.sqrt(C_nominal))) - 1


def build_dictionary(method, n_ax, config, seed_rng, fourier_x=False, poly=False):
    """One entry point for every family. config = the tuned-knob dict.
    seed_rng drives ONLY the dictionary jitter (13.9)."""
    if method == "qi_radon":
        d = radon_dict(n_ax, config["halo"],
                       seed_phase=(seed_rng.uniform(0, 1), seed_rng.uniform(-0.5, 0.5)))
    elif method == "qi_tensor":
        d = TensorDict(n_ax, config["halo"],
                       shift=tuple(seed_rng.uniform(-0.5, 0.5, 2)))
    elif method == "elm":
        d = elm_dict(n_ax, config["R"], seed_rng)
    elif method == "rbf_imq":
        d = RBFDict(n_ax, "imq", epsh=config["epsh"],
                    shift=tuple(seed_rng.uniform(-0.5, 0.5, 2)),
                    aspect=config.get("aspect", 1.0))
    elif method == "rbf_phs":
        d = RBFDict(n_ax, "phs", shift=tuple(seed_rng.uniform(-0.5, 0.5, 2)),
                    aspect=config.get("aspect", 1.0))
    elif method == "spectral":
        d = SpectralDict(n_ax, fourier_x=fourier_x,
                         aspect=config.get("aspect", 1.0))
    elif method == "bwler":
        d = BWLerDict(n_ax, aspect=config.get("aspect", 1.0))
    else:
        raise ValueError(method)
    return PolyAugmented(d) if poly else d
