"""Mesh finding: choosing where the centers go along each direction, and (in 2-D)
where the directions go, from a "monitor" that says where resolution is needed.

Every rung of the ladder shares one pipeline per direction ``v`` (``T = 1.25 ||v||_1``,
the same collar the even reference uses):

    monitor  m(t) >= 0 on a fine grid over [-T, T]
    density  rho(t) = (1 - s)/(2T) + s * m(t) / integral(m)          (s = floor knob:
             with s = 2/3 a third of the centers stay even, so no gap is much wider
             than 3 even spacings)
    spacing  h(t) = 1 / (n_per * rho(t)), then graded so |dh/dt| <= g (mesh grading:
             neighboring spacings never differ by more than a factor of about 1 + g,
             the failure mode expH02 found)
    centers  c_j = C^{-1}((j + 1/2)/n_per), C the cumulative of 1/h
    widths   h_j = (c_{j+1} - c_{j-1})/2 (one-sided at the ends), gamma_j = lambda/h_j

With ``s = 0`` this is exactly the even reference. The monitors:

    even        m = 1
    data        m = p_v(t)^beta                        projected training density only
    roughness   m = (p_v(t) * R_r(t))^{1/(2r+1)}       R_r(t) = E[ |d^r F/dv^r|^2 | v.x = t ]
    residual    m = (p_v(t) * E[e^2 | v.x = t])^{1/3}  e = residual of a previous fit
    frequency   m = omega_v(t) = sqrt( E[|d^2F/dv^2|^2 | t] / E[|dF/dv|^2 | t] )
                the local frequency of the ridge profile along v. This is the spectral
                rule: a construction that is spectrally accurate needs h(t) * omega(t)
                below a constant everywhere, so the center density should follow the
                local frequency, not the amplitude, and not the data density.

``R_r`` comes either from the true target ("oracle", the ceiling on what the monitor can
give) or from a model already fitted on the same data ("surrogate": analytic derivatives
of the tanh network), which is the practical version. The residual monitor is the
classic adaptive-refinement rule and needs a previous fit too.

Directions (2-D only). The even reference spreads angles evenly on [0, pi). The
direction monitor is ``A(theta) = mean over the training points of |dF/dv_theta|^2``
(oracle or surrogate), the angle density is ``(1 - s)/pi + s * A^alpha / integral``,
graded and placed by inverse CDF exactly like the centers. Per-direction center counts
can additionally be made proportional to a floor plus the integral of that direction's
center monitor ("joint").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from h01suite.baseline import (EDGE_MARGIN, LAMBDA, RCOND, EvenGeometry,
                               even_directions, _solve_svd)
from h01suite.metrics import _gaussian_smooth

__all__ = ["grade_spacing", "place_by_density", "AdaptiveGeometry", "Monitors",
           "surrogate_derivatives", "oracle_derivatives", "gradient_covariance",
           "active_dimension", "active_subspace_geometry"]

N_GRID = 2001
GRADE = 0.15          # |dh/dt| <= GRADE: neighboring spacings within a factor ~1.15
FLOOR_S = 2.0 / 3.0   # rho = (1-s)*uniform + s*monitor; expH02 converged for s <= 2/3
# Monitor smoothing, in even spacings. The mesh is a coordinate map c_j = Phi(j h0) and
# the network is the uniform construction applied to f o Phi, so Phi must carry no content
# below the construction's own resolution limit: mesh_map_scale.py measures that limit
# at L* ~ 12 gaps (a perturbation of the map at wavelength >= 12 gaps is harmless at any
# amplitude up to 0.2 gap; shorter wavelengths cost up to 1e-9). A Gaussian of width sigma
# suppresses wavelength L by exp(-2 pi^2 sigma^2 / L^2); 1e-2 suppression at L* needs
# sigma >= 5.8 gaps. (The first ladder used 1.5 and paid 10-100x at the floor: floor_price.py.)
MESH_MAP_WAVELENGTH = 12.0
BW_MULT = float(MESH_MAP_WAVELENGTH * np.sqrt(np.log(100.0) / (2 * np.pi ** 2)))   # = 5.8
FREQ_WINDOW = 0.05    # the local-frequency ratio is averaged over at least this fraction of
                      # the range: narrower windows see the zeros of F' at extrema and
                      # report an infinite frequency there


# ---------------------------------------------------------------------------
# one-dimensional placement
# ---------------------------------------------------------------------------

def grade_spacing(h: np.ndarray, dt: float, g: float, n_total: float,
                  max_rounds: int = 50) -> np.ndarray:
    """Limit ``|dh/dt|`` to ``g`` while keeping ``integral(1/h) = n_total``.

    A forward and a backward sweep enforce ``h[i] <= h[i +- 1] + g dt`` (this only ever
    lowers ``h``); rescaling to the required count then loosens the bound by the scale
    factor, so the two steps alternate until the scale factor is 1.
    """
    h = np.array(h, dtype=np.float64)
    for _ in range(max_rounds):
        for i in range(1, len(h)):
            if h[i] > h[i - 1] + g * dt:
                h[i] = h[i - 1] + g * dt
        for i in range(len(h) - 2, -1, -1):
            if h[i] > h[i + 1] + g * dt:
                h[i] = h[i + 1] + g * dt
        scale = np.trapezoid(1.0 / h, dx=dt) / n_total
        h *= scale
        if abs(scale - 1.0) < 1e-10:
            break
    return h


def place_by_density(grid: np.ndarray, monitor: np.ndarray, n: int, s: float,
                     g: float = GRADE, margin_mask: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    """``n`` centers on ``grid``'s range from a monitor, with a uniform floor and grading.

    ``margin_mask`` marks grid positions (the collar beyond the data range) whose
    density is pinned at the even value; the monitor then only redistributes the
    centers that the even mesh would have put inside the data range.

    Returns ``(centers, local_spacings, info)``.
    """
    grid = np.asarray(grid, dtype=np.float64)
    dt = float(grid[1] - grid[0])
    L = float(grid[-1] - grid[0])
    m = np.maximum(np.asarray(monitor, dtype=np.float64), 0.0)
    Z = np.trapezoid(m, dx=dt)
    if not np.isfinite(Z) or Z <= 0.0 or s <= 0.0:
        rho = np.full_like(grid, 1.0 / L)
    else:
        rho = (1.0 - s) / L + s * m / Z
    if margin_mask is not None and margin_mask.any() and not margin_mask.all():
        rho_even = 1.0 / L
        inside = ~margin_mask
        n_inside = n * np.trapezoid(np.where(inside, rho_even, 0.0), dx=dt)   # even count inside
        rho_in = np.where(inside, rho, 0.0)
        rho_in *= n_inside / (n * np.trapezoid(rho_in, dx=dt))
        rho = np.where(inside, rho_in, rho_even)
    h = 1.0 / (n * rho)
    h = grade_spacing(h, dt, g, n)
    rho = 1.0 / h
    C = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * dt)])
    C *= n / C[-1]
    centers = np.interp((np.arange(n) + 0.5), C, grid)
    hj = np.empty(n)
    if n >= 3:
        hj[1:-1] = 0.5 * (centers[2:] - centers[:-2])
        hj[0] = centers[1] - centers[0]
        hj[-1] = centers[-1] - centers[-2]
    else:
        hj[:] = L / n
    ratio = float(np.max(hj[1:] / hj[:-1])) if n > 1 else 1.0
    ratio = max(ratio, float(np.max(hj[:-1] / hj[1:])) if n > 1 else 1.0)
    info = {"max_neighbor_ratio": ratio, "min_spacing": float(hj.min()),
            "max_spacing": float(hj.max()), "even_spacing": L / n,
            "density": rho, "grid": grid}
    return centers, hj, info


# ---------------------------------------------------------------------------
# the quantities the monitors are built from
# ---------------------------------------------------------------------------

def _bin_mean(t: np.ndarray, w: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Smoothed conditional mean of ``w`` given position ``t`` on the grid (numerator and
    denominator smoothed separately, as in ``h01suite.metrics``)."""
    dt = float(grid[1] - grid[0])
    edges = np.concatenate([grid - 0.5 * dt, [grid[-1] + 0.5 * dt]])
    total, _ = np.histogram(t, bins=edges, weights=w)
    count, _ = np.histogram(t, bins=edges)
    num = _gaussian_smooth(total, dt, bw)
    den = _gaussian_smooth(count.astype(np.float64), dt, bw)
    floor = 1e-8 * float(den.max()) if den.max() > 0 else 1.0
    return np.where(den > floor, num / np.maximum(den, floor), 0.0)


def _bin_density(t: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    dt = float(grid[1] - grid[0])
    edges = np.concatenate([grid - 0.5 * dt, [grid[-1] + 0.5 * dt]])
    count, _ = np.histogram(t, bins=edges)
    p = count.astype(np.float64) / (len(t) * dt)
    return _gaussian_smooth(p, dt, bw)


def oracle_derivatives(task, X: np.ndarray, V: np.ndarray, r: int,
                       eps: float = 1e-4) -> np.ndarray:
    """``d^r F / dv^r`` at the points ``X`` for every row ``v`` of ``V``: ``[n_dir, n]``.

    ``r = 1`` is the analytic gradient of the centered-and-scaled target; ``r = 2`` is a
    central difference of that gradient along ``v``.
    """
    X = np.asarray(X, dtype=np.float64)
    if r == 1:
        return (task.grad_F(X) @ V.T).T
    out = np.empty((len(V), len(X)))
    for i, v in enumerate(V):
        vn = v / np.linalg.norm(v)
        gp = task.grad_F(X + eps * vn) @ v
        gm = task.grad_F(X - eps * vn) @ v
        out[i] = (gp - gm) / (2.0 * eps) * np.linalg.norm(v)
    return out


def surrogate_derivatives(model, X: np.ndarray, V: np.ndarray, r: int) -> np.ndarray:
    """``d^r Fhat / dv^r`` of a fitted tanh network, analytically: ``[n_dir, n]``.

    ``Fhat = sum_k w_k tanh(gamma_k (v_k.x - c_k)) + b``, so along ``v``
    ``dFhat/dv = sum_k w_k gamma_k (v_k.v) sech^2(u_k)`` and
    ``d^2Fhat/dv^2 = sum_k w_k gamma_k^2 (v_k.v)^2 (-2 tanh(u_k) sech^2(u_k))``.
    """
    X = np.asarray(X, dtype=np.float64)
    U = model.gammas[None, :] * (X @ model.directions.T - model.centers[None, :])
    th = np.tanh(U)
    sech2 = 1.0 - th * th
    dots = V @ model.directions.T                                   # [n_dir, B]
    if r == 1:
        coef = dots * (model.weights * model.gammas)[None, :]
        return coef @ sech2.T
    coef = dots * dots * (model.weights * model.gammas ** 2)[None, :]
    return coef @ (-2.0 * th * sech2).T


# ---------------------------------------------------------------------------
# monitors
# ---------------------------------------------------------------------------

@dataclass
class Monitors:
    """Everything needed to build the center (and direction) monitors for one fit.

    ``deriv``: ``[n_dir, n]`` values of ``d^r F/dv^r`` at the training points, or None.
    ``resid``: residual of a previous fit at the training points, or None.
    """
    kind: str                     # even | data | roughness | residual | frequency
    r: int = 1
    beta: float = 1.0 / 3.0       # exponent on p_v for the data monitor
    deriv: np.ndarray | None = None
    resid: np.ndarray | None = None
    deriv2: np.ndarray | None = None      # second derivatives, for the frequency monitor

    def center_monitor(self, i_dir: int, t: np.ndarray, grid: np.ndarray, bw: float):
        p = _bin_density(t, grid, bw)
        if self.kind == "even":
            return np.ones_like(grid)
        if self.kind == "data":
            return np.power(p, self.beta)
        if self.kind == "roughness":
            R = _bin_mean(t, self.deriv[i_dir] ** 2, grid, bw)
            return np.power(p * R, 1.0 / (2 * self.r + 1))
        if self.kind == "residual":
            E = _bin_mean(t, self.resid ** 2, grid, bw)
            return np.power(p * E, 1.0 / 3.0)
        if self.kind == "frequency":
            bw_f = max(bw, FREQ_WINDOW * float(grid[-1] - grid[0]))
            E1 = _bin_mean(t, self.deriv[i_dir] ** 2, grid, bw_f)
            E2 = _bin_mean(t, self.deriv2[i_dir] ** 2, grid, bw_f)
            # regularize where the profile is flat (both energies ~ 0): the ratio is
            # then meaningless and the floor takes over
            d1 = E1 + 1e-6 * E1.max() if E1.max() > 0 else E1 + 1.0
            return np.sqrt(E2 / d1)
        raise KeyError(self.kind)


# ---------------------------------------------------------------------------
# the adaptive geometry
# ---------------------------------------------------------------------------

def _angles_from_density(n_dir: int, A: np.ndarray, th_grid: np.ndarray, s: float,
                         alpha: float) -> np.ndarray:
    """Angles on [0, pi) from a direction monitor, graded and placed like centers."""
    m = np.power(np.maximum(A, 0.0), alpha)
    th, _, _ = place_by_density(th_grid, m, n_dir, s)
    return th


@dataclass
class AdaptiveGeometry:
    """Ridge geometry whose centers (and optionally directions) follow a monitor."""

    d: int
    budget: int
    s: float = FLOOR_S
    grade: float = GRADE
    bw_mult: float = BW_MULT
    lam: float = LAMBDA
    margin: float = EDGE_MARGIN
    rcond: float = RCOND
    name: str = "adaptive"
    n_per_override: int | None = None     # force the centers-per-direction split
    keep_margin: bool = False             # hold the collar (|t| > ||v||_1) at even density
    directions: np.ndarray = field(init=False)
    centers: np.ndarray = field(init=False)
    gammas: np.ndarray = field(init=False)
    unique_directions: np.ndarray = field(init=False)
    per_direction: np.ndarray = field(init=False)
    weights: np.ndarray | None = field(init=False, default=None)
    bias: float = field(init=False, default=0.0)
    info: dict = field(init=False, default_factory=dict)
    mesh_info: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        ref = EvenGeometry(d=self.d, budget=self.budget, lam=self.lam, margin=self.margin)
        self.n_per_direction, self.n_directions = ref.n_per_direction, ref.n_directions
        self.unique_directions = ref.geometry()["unique_directions"]
        if self.n_per_override is not None:
            self.n_per_direction = int(self.n_per_override)
            self.n_directions = max(1, int(round(self.budget / self.n_per_direction)))
            self.unique_directions = even_directions(self.d, self.n_directions)
        self.per_direction = np.full(self.n_directions, self.n_per_direction)

    # -- building ---------------------------------------------------------
    def set_directions(self, X: np.ndarray, dir_monitor, s: float | None = None,
                       alpha: float = 1.0 / 3.0, n_theta: int = 1441):
        """2-D only: place the angles from ``dir_monitor(V) -> A per direction``.

        ``dir_monitor`` is called on a fine set of angles and must return the mean
        squared directional derivative at the training points for each of them.
        """
        if self.d != 2:
            raise ValueError("direction allocation is implemented for d = 2 only")
        s = self.s if s is None else s
        th_grid = np.linspace(0.0, np.pi, n_theta)
        Vg = np.stack([np.cos(th_grid), np.sin(th_grid)], axis=1)
        A = np.asarray(dir_monitor(Vg), dtype=np.float64)
        dth = th_grid[1] - th_grid[0]
        A = _gaussian_smooth(A, dth, self.bw_mult * np.pi / self.n_directions)
        th = _angles_from_density(self.n_directions, A, th_grid, s, alpha)
        V = np.stack([np.cos(th), np.sin(th)], axis=1)
        self.unique_directions = V
        self.mesh_info["direction_monitor"] = {"theta": th_grid, "A": A, "angles": th}
        return self

    def set_counts(self, weights: np.ndarray, s: float | None = None):
        """Per-direction center counts ``n_i ∝ (1-s)/n_dir + s * w_i/sum(w)``, total kept."""
        s = self.s if s is None else s
        w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
        if w.sum() <= 0:
            return self
        frac = (1.0 - s) / self.n_directions + s * w / w.sum()
        total = self.n_directions * self.n_per_direction
        counts = np.maximum(3, np.floor(frac * total)).astype(int)
        # hand out the remainder to the largest fractional parts
        rem = total - counts.sum()
        if rem > 0:
            order = np.argsort(-(frac * total - counts))
            counts[order[:rem]] += 1
        self.per_direction = counts
        return self

    def build(self, X: np.ndarray, monitors: Monitors):
        """Place the centers along every direction from the monitors."""
        X = np.asarray(X, dtype=np.float64)
        V = self.unique_directions
        dirs, cens, gams, per = [], [], [], []
        ratios, minh, maxh = [], [], []
        for i, v in enumerate(V):
            n = int(self.per_direction[i])
            T = self.margin * float(np.abs(v).sum())
            grid = np.linspace(-T, T, N_GRID)
            h0 = 2.0 * T / n
            t = X @ v
            m = monitors.center_monitor(i, t, grid, self.bw_mult * h0)
            margin_mask = np.abs(grid) > float(np.abs(v).sum()) if self.keep_margin else None
            c, hj, info = place_by_density(grid, m, n, self.s, self.grade, margin_mask)
            dirs.append(np.repeat(v[None, :], n, axis=0))
            cens.append(c)
            gams.append(self.lam / hj)
            per.append(info)
            ratios.append(info["max_neighbor_ratio"])
            minh.append(info["min_spacing"] / h0)
            maxh.append(info["max_spacing"] / h0)
        self.directions = np.vstack(dirs)
        self.centers = np.concatenate(cens)
        self.gammas = np.concatenate(gams)
        self.mesh_info.update({
            "per_direction": per,
            "max_neighbor_ratio": float(np.max(ratios)),
            "min_spacing_over_even": float(np.min(minh)),
            "max_spacing_over_even": float(np.max(maxh)),
            "monitor": monitors.kind, "r": monitors.r, "s": self.s})
        return self

    # -- interface --------------------------------------------------------
    def features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.tanh(self.gammas[None, :] * (X @ self.directions.T - self.centers[None, :]))

    def fit(self, X, y):
        self.weights, self.bias, self.info = _solve_svd(self.features(X), y, self.rcond)
        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("call fit() first")
        return self.features(X) @ self.weights + self.bias

    def geometry(self) -> dict:
        return {"directions": self.directions, "centers": self.centers,
                "gammas": self.gammas, "n_directions": self.n_directions,
                "n_per_direction": self.n_per_direction,
                "unique_directions": self.unique_directions,
                "per_direction": self.per_direction,
                "lambda": self.lam, "margin": self.margin}


# ---------------------------------------------------------------------------
# active subspace: directions from the gradient covariance
# ---------------------------------------------------------------------------

def gradient_covariance(grads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors (columns) of ``E[grad F grad F^T]``."""
    G = np.asarray(grads, dtype=np.float64)
    C = G.T @ G / len(G)
    w, W = np.linalg.eigh(C)
    order = np.argsort(-w)
    return w[order], W[:, order]


def active_dimension(evals: np.ndarray, energy: float = 0.999) -> int:
    """Smallest ``m`` whose leading eigenvalues carry ``energy`` of the trace."""
    c = np.cumsum(evals) / evals.sum()
    return int(np.searchsorted(c, energy) + 1)


def active_subspace_geometry(d: int, budget: int, W: np.ndarray, m: int,
                             frac: float = 0.8, s: float = FLOOR_S,
                             name: str = "active") -> AdaptiveGeometry:
    """A mesh that treats the problem as ``m``-dimensional inside the subspace spanned by
    the first ``m`` columns of ``W``: a fraction ``frac`` of the budget is split as an
    ``m``-dimensional even mesh would split it (``B_a^(1/m)`` centers per direction,
    directions spread evenly *inside the subspace*), the rest stays a ``d``-dimensional
    even mesh so nothing outside the subspace is starved.

    ``m = 1`` puts the single subspace direction in once and gives it the whole active
    share of the budget.
    """
    geo = AdaptiveGeometry(d=d, budget=budget, s=s, name=name)
    if m >= d:
        return geo
    B_a = int(round(frac * budget))
    B_r = budget - B_a
    Wm = np.asarray(W, dtype=np.float64)[:, :m]
    if m == 1:
        V_a = Wm.T                                    # one direction
        n_a = np.array([B_a])
    else:
        n_per = max(3, int(round(B_a ** (1.0 / m))))
        n_dir = max(1, int(round(B_a / n_per)))
        V_a = even_directions(m, n_dir) @ Wm.T        # embedded, unit length
        n_a = np.full(n_dir, n_per)
    n_per_r = max(3, int(round(B_r ** (1.0 / d))))
    n_dir_r = max(1, int(round(B_r / n_per_r)))
    V_r = even_directions(d, n_dir_r)
    geo.unique_directions = np.vstack([V_a, V_r])
    geo.per_direction = np.concatenate([n_a, np.full(n_dir_r, n_per_r)])
    geo.n_directions = len(geo.unique_directions)
    geo.n_per_direction = int(np.median(geo.per_direction))
    geo.mesh_info["active"] = {"m": m, "frac": frac, "n_active_dirs": len(V_a),
                               "centers_per_active_dir": n_a.tolist()[:3],
                               "n_rest_dirs": n_dir_r, "centers_per_rest_dir": n_per_r}
    return geo
