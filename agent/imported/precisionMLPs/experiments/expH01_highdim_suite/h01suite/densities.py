"""Where the training points come from, and the three test sets.

Eight data geometries, all living inside ``[-1,1]^d``:

``even_grid``            deterministic even coverage: equispaced midpoints in 1-D,
                         Halton points with bases ``2,3,5,7,11`` for ``d > 1``.
``uniform``              independent uniform draws on the cube.
``hotspots``             most of the data in three tight clusters, the rest spread thinly:
                         ``.20 uniform + .40 N_T(mu_+, .22^2 I) + .25 N_T(mu_-, .28^2 I)
                         + .15 N_T(mu_perp, .25^2 I)``, with ``mu_+ = .45*(1..1)``,
                         ``mu_- = -.45*(1..1)`` and ``mu_perp = .35 u_2/||u_2||_inf``
                         (in ``d = 1``: means ``.45, -.45, 0`` with the same widths).
``stretched_hotspots``   two clusters stretched 3:1, so movement along ``u_2`` is barely
                         visible in the data: ``.20 uniform + .40 N_T(.35*1, Sigma)
                         + .40 N_T(-.35*1, Sigma)``, ``Sigma = Q diag(.25^2, .083^2,
                         .15^2, ...) Q^T``.
``flat_sheet``           the data lies exactly on a flat sheet through the origin:
                         ``y = (t, 0)`` in ``d = 2``, ``(s, t, 0, ..., 0)`` for ``d >= 3``,
                         parameters uniform on ``[-.75, .75]``, ``x = Q y``.
``flat_sheet_noisy``     the same sheet, thickened by ``N(0, .015^2)`` in the directions
                         perpendicular to it.
``curved_sheet``         a sheet that bends: ``y = (.75t, .30 sin pi t)`` in ``d = 2`` and
                         ``y = (.65s, .65t, .25 sin pi s, .20 sin pi t, .15 sin pi(s+t))_{1:d}``
                         for ``d >= 3``, with ``s, t`` uniform on ``[-1, 1]``.
``curved_sheet_noisy``   the same curved sheet thickened by ``N(0, .015^2)`` perpendicular
                         to it, using the analytic tangent Jacobian so the noise really
                         is perpendicular.

Truncated normals are drawn with **per-cluster rejection**, so the realized mixture
fractions equal the nominal weights exactly. Rejecting globally would silently reweight
the clusters by how much of each one falls outside the cube.

Three test sets, built by ``test_sets``:

``same_as_train``   fresh points drawn the same way as the training data.
``uniform``         uniform over the whole cube: what did adaptation give up elsewhere?
``dense_region``    points from the densest part of the data only, kept away from its
                    edge by a margin of training data on every side. This is the set on
                    which machine-precision is a fair question: it contains no outliers,
                    no boundary points, and no points the data does not surround.
                    Per family: ``even_grid``/``uniform`` -> uniform on the shrunken cube
                    ``[-0.8, 0.8]^d``, the outer band of width ``0.2`` being the margin;
                    the hotspot families -> the single densest cluster, restricted to one
                    standard deviation, with that cluster's own 1-3 sd shell plus the other
                    clusters and the uniform background as the margin; the sheets -> points
                    exactly on the sheet whose parameters lie in the inner 80% of their
                    range (for the noisy variants the sheet is the densest part of the slab,
                    and the parameter margin keeps the test away from the sheet's rim).

``logpdf`` returns the *unnormalized* log density where one exists (``uniform`` and the
two hotspot families) and ``None`` otherwise. The truncation constants are dropped
because the only use is ranking points from sparsest to densest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from h01suite.basis import dct_basis, u

__all__ = ["Density", "EvenGrid", "Uniform", "Hotspots", "StretchedHotspots",
           "FlatSheet", "CurvedSheet", "make_density", "SHEET_DISTANCES",
           "test_set_size", "DENSITY_TAGS"]

SHEET_DISTANCES = (0.02, 0.05, 0.10, 0.20)
DENSE_CUBE = 0.8       # even_grid/uniform dense region: |x_i| <= 0.8 (margin band 0.2)
DENSE_SD = 1.0         # hotspot dense region: within this many sd of the densest cluster
DENSE_PARAM = 0.8      # sheet dense region: inner 80% of each parameter range
HALTON_BASES = (2, 3, 5, 7, 11)
SHEET_NOISE_SD = 0.015
FLAT_HALF_RANGE = 0.75   # flat sheets: parameters uniform on [-0.75, 0.75]

DENSITY_TAGS = ("even_grid", "uniform", "hotspots", "stretched_hotspots",
                "flat_sheet", "flat_sheet_noisy", "curved_sheet", "curved_sheet_noisy")


def test_set_size(d: int) -> int:
    """Test-set size: 20000 points for ``d <= 2``, 40000 for ``d >= 3``."""
    return 20000 if d <= 2 else 40000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def halton(n: int, base: int, skip: int = 1) -> np.ndarray:
    """Halton sequence in ``[0,1)``, skipping the first ``skip`` indices."""
    idx = np.arange(skip, n + skip, dtype=np.int64)
    out = np.zeros(n, dtype=np.float64)
    f = 1.0
    i = idx.copy()
    while np.any(i > 0):
        f /= base
        out += f * (i % base)
        i //= base
    return out


def _inside(X: np.ndarray) -> np.ndarray:
    return np.all(np.abs(X) <= 1.0, axis=1)


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------

class Density:
    """Common interface: ``sample(n, seed)``, ``logpdf(X)``, ``test_sets(seed)``."""

    name: str = "density"
    is_sheet: bool = False

    def sample(self, n: int, seed: int = 0) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def logpdf(self, X: np.ndarray) -> np.ndarray | None:
        """Unnormalized log density on the cube, or ``None`` when there is no formula."""
        return None

    def dense_region_sample(self, n: int, seed: int = 0) -> np.ndarray:  # pragma: no cover
        """``n`` points from the densest part of the data, with training data all around."""
        raise NotImplementedError

    def dense_region_description(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def test_sets(self, seed: int = 10_000) -> dict[str, np.ndarray]:
        """The three fixed test sets: ``same_as_train``, ``uniform``, ``dense_region``."""
        n = test_set_size(self.d)
        rng = np.random.default_rng(seed + 991)
        return {"same_as_train": self.sample(n, seed=seed),
                "uniform": rng.uniform(-1.0, 1.0, size=(n, self.d)),
                "dense_region": self.dense_region_sample(n, seed=seed + 4242)}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(d={self.d})"


# ---------------------------------------------------------------------------
# even grid / uniform
# ---------------------------------------------------------------------------

@dataclass
class EvenGrid(Density):
    """Deterministic even coverage of the cube (``seed`` is ignored by design)."""

    d: int
    name: str = "even_grid"

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        if self.d == 1:
            return (-1.0 + (np.arange(n, dtype=np.float64) + 0.5) * (2.0 / n))[:, None]
        cols = [2.0 * halton(n, b) - 1.0 for b in HALTON_BASES[:self.d]]
        return np.stack(cols, axis=1)

    def dense_region_sample(self, n: int, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).uniform(-DENSE_CUBE, DENSE_CUBE, size=(n, self.d))

    def dense_region_description(self) -> str:
        return (f"uniform on [-{DENSE_CUBE},{DENSE_CUBE}]^{self.d}; margin = the outer band "
                f"of width {1 - DENSE_CUBE:g} on every face")


@dataclass
class Uniform(Density):
    d: int
    name: str = "uniform"

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, self.d))

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(np.asarray(X)), dtype=np.float64)

    def dense_region_sample(self, n: int, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).uniform(-DENSE_CUBE, DENSE_CUBE, size=(n, self.d))

    def dense_region_description(self) -> str:
        return (f"uniform on [-{DENSE_CUBE},{DENSE_CUBE}]^{self.d}; margin = the outer band "
                f"of width {1 - DENSE_CUBE:g} on every face")


# ---------------------------------------------------------------------------
# hotspot mixtures
# ---------------------------------------------------------------------------

def _mvn_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = len(mean)
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (X - mean).T)
    quad = np.sum(sol * sol, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (quad + logdet + d * np.log(2.0 * np.pi))


@dataclass
class _Mixture(Density):
    """``w_uniform * uniform(cube) + sum_i w_i * N_T(mu_i, Sigma_i)``, per-cluster rejection."""

    d: int
    w_uniform: float = 0.20
    means: list = field(default_factory=list)
    covs: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    name: str = "mixture"
    max_tries: int = 200

    def component_weights(self) -> np.ndarray:
        """Nominal weights in the order ``[uniform, cluster_1, cluster_2, ...]``."""
        return np.array([self.w_uniform] + list(self.weights), dtype=np.float64)

    def sample_with_labels(self, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        w = self.component_weights()
        labels = rng.choice(len(w), size=n, p=w / w.sum())
        X = np.empty((n, self.d), dtype=np.float64)
        X[labels == 0] = rng.uniform(-1.0, 1.0, size=((labels == 0).sum(), self.d))
        for ci in range(1, len(w)):
            idx = np.flatnonzero(labels == ci)
            if idx.size == 0:
                continue
            X[idx] = self._truncated_normal(idx.size, self.means[ci - 1], self.covs[ci - 1], rng)
        return X, labels

    def _truncated_normal(self, n, mean, cov, rng) -> np.ndarray:
        out = np.empty((n, self.d), dtype=np.float64)
        todo = np.arange(n)
        for _ in range(self.max_tries):
            cand = rng.multivariate_normal(mean, cov, size=todo.size, method="cholesky")
            ok = _inside(cand)
            out[todo[ok]] = cand[ok]
            todo = todo[~ok]
            if todo.size == 0:
                return out
        raise RuntimeError(f"truncated normal rejection did not converge ({todo.size} left)")

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        return self.sample_with_labels(n, seed=seed)[0]

    def densest_cluster(self) -> int:
        """Index (into ``means``/``covs``) of the cluster with the highest peak density
        ``w_i / sqrt(det Sigma_i)``; ties go to the first."""
        peaks = [w / np.sqrt(np.linalg.det(c)) for w, c in zip(self.weights, self.covs)]
        return int(np.argmax(peaks))

    def dense_region_sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw from the densest cluster, restricted to ``DENSE_SD`` standard deviations.

        The rest of that cluster (1-3 sd), the other clusters and the uniform background
        all lie outside this region, so the test points are surrounded by training data on
        every side. Points are also required to be inside the cube.
        """
        rng = np.random.default_rng(seed)
        i = self.densest_cluster()
        mean, cov = self.means[i], self.covs[i]
        L = np.linalg.cholesky(cov)
        out = np.empty((n, self.d), dtype=np.float64)
        got = 0
        for _ in range(self.max_tries * 10):
            cand = rng.multivariate_normal(mean, cov, size=max(n, 64), method="cholesky")
            sol = np.linalg.solve(L, (cand - mean).T)
            ok = (np.sum(sol * sol, axis=0) <= DENSE_SD ** 2) & _inside(cand)
            take = cand[ok][: n - got]
            out[got:got + len(take)] = take
            got += len(take)
            if got == n:
                return out
        raise RuntimeError("dense-region rejection sampling did not converge")

    def dense_region_description(self) -> str:
        i = self.densest_cluster()
        return (f"cluster {i} (mean {np.round(self.means[i], 3).tolist()}) within "
                f"{DENSE_SD:g} standard deviation; margin = its 1-3 sd shell, the other "
                f"clusters, and the uniform background")

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dens = self.w_uniform * np.full(len(X), 0.5 ** self.d)
        for w, mean, cov in zip(self.weights, self.means, self.covs):
            dens = dens + w * np.exp(_mvn_logpdf(X, mean, cov))
        return np.log(dens)


def Hotspots(d: int) -> _Mixture:
    """Three clusters plus a uniform background."""
    ones = np.ones(d)
    if d == 1:
        means = [np.array([0.45]), np.array([-0.45]), np.array([0.0])]
    else:
        u2 = u(d, 2)
        means = [0.45 * ones, -0.45 * ones, 0.35 * u2 / np.abs(u2).max()]
    covs = [(s ** 2) * np.eye(d) for s in (0.22, 0.28, 0.25)]
    return _Mixture(d=d, w_uniform=0.20, means=means, covs=covs,
                    weights=[0.40, 0.25, 0.15], name="hotspots")


def StretchedHotspots(d: int) -> _Mixture:
    """Two clusters stretched 3:1, so movement along ``u_2`` is nearly invisible."""
    Q = dct_basis(d)
    diag = np.array([0.25 ** 2, 0.083 ** 2] + [0.15 ** 2] * max(0, d - 2))[:d]
    if d == 1:
        diag = np.array([0.25 ** 2])
    Sigma = Q @ np.diag(diag) @ Q.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    ones = np.ones(d)
    return _Mixture(d=d, w_uniform=0.20, means=[0.35 * ones, -0.35 * ones],
                    covs=[Sigma, Sigma], weights=[0.40, 0.40], name="stretched_hotspots")


# ---------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------

@dataclass
class _Sheet(Density):
    d: int
    noisy: bool = False
    is_sheet: bool = True
    max_tries: int = 100
    n_redrawn: int = 0

    # -- subclass hooks --------------------------------------------------
    @property
    def sheet_dim(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def params(self, n: int, rng) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def perpendicular_basis(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Orthonormal basis of the directions perpendicular to the sheet at each point."""
        raise NotImplementedError

    # -- shared ----------------------------------------------------------
    def sheet_points(self, n: int, seed: int = 0) -> np.ndarray:
        """Points exactly on the sheet (no thickening)."""
        return self.embed(self.params(n, np.random.default_rng(seed)))

    def _displace(self, p, X0, mags, rng) -> np.ndarray:
        """Move each point perpendicular to the sheet by an amount drawn by ``mags``."""
        Nb = self.perpendicular_basis(p)               # [n, d, m]
        m = Nb.shape[2]
        coef = mags(rng, len(X0), m)                   # [n, m]
        return X0 + np.einsum("nij,nj->ni", Nb, coef)

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        p = self.params(n, rng)
        X0 = self.embed(p)
        bad = ~_inside(X0)
        if bad.any():
            raise AssertionError(
                f"{self.name}: {bad.sum()} points on the sheet left the cube "
                f"(max |x| = {np.abs(X0).max():.4f}); the constants should prevent this")
        if not self.noisy:
            return X0
        sd = self.noise_sd
        X = self._displace(p, X0, lambda r, nn, mm: r.normal(0.0, sd, size=(nn, mm)), rng)
        self.n_redrawn = 0
        for _ in range(self.max_tries):
            bad = ~_inside(X)
            if not bad.any():
                break
            self.n_redrawn += int(bad.sum())
            idx = np.flatnonzero(bad)
            X[idx] = self._displace(p[idx], X0[idx],
                                    lambda r, nn, mm: r.normal(0.0, sd, size=(nn, mm)), rng)
        else:
            raise RuntimeError(f"{self.name}: thickened points would not fall inside the cube")
        return X

    def dense_region_sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Points exactly on the sheet whose parameters lie in the inner
        ``DENSE_PARAM`` fraction of their range: away from the sheet's rim, and (for the
        thickened variants) in the middle of the slab."""
        rng = np.random.default_rng(seed)
        return self.embed(DENSE_PARAM * self.params(n, rng))

    def dense_region_description(self) -> str:
        return (f"points on the sheet with parameters scaled by {DENSE_PARAM:g} (the inner "
                f"{100 * DENSE_PARAM:.0f}% of the parameter range); margin = the outer "
                f"parameter band" + (" and the thickness of the slab" if self.noisy else ""))

    def offset_points(self, n: int, distance: float, seed: int = 0) -> np.ndarray:
        """Points at exactly ``distance`` perpendicular to the sheet, inside the cube.

        Points whose offset leaves the cube are dropped (never clipped), so the returned
        count can be smaller than ``n``.
        """
        rng = np.random.default_rng(seed)
        p = self.params(n, rng)
        X0 = self.embed(p)

        def unit(r, nn, mm):
            g = r.normal(size=(nn, mm))
            return distance * g / np.linalg.norm(g, axis=1, keepdims=True)

        X = self._displace(p, X0, unit, rng)
        return X[_inside(X)]

    @property
    def noise_sd(self) -> float:
        return SHEET_NOISE_SD

    def test_sets(self, seed: int = 10_000) -> dict[str, np.ndarray]:
        sets = super().test_sets(seed=seed)
        n = test_set_size(self.d)
        sets["on_sheet"] = self.sheet_points(n, seed=seed + 313)
        for r in SHEET_DISTANCES:
            sets[f"distance_{r:g}"] = self.offset_points(n // 4, r, seed=seed + int(1000 * r) + 7)
        return sets


@dataclass
class FlatSheet(_Sheet):
    """A flat sheet through the origin spanned by ``u_1`` (``d = 2``) or ``u_1, u_2``."""

    name: str = "flat_sheet"

    def __post_init__(self):
        self.name = "flat_sheet_noisy" if self.noisy else "flat_sheet"

    @property
    def sheet_dim(self) -> int:
        return 1 if self.d == 2 else 2

    def params(self, n, rng) -> np.ndarray:
        return rng.uniform(-FLAT_HALF_RANGE, FLAT_HALF_RANGE, size=(n, self.sheet_dim))

    def embed(self, p: np.ndarray) -> np.ndarray:
        y = np.zeros((len(p), self.d), dtype=np.float64)
        y[:, :self.sheet_dim] = p
        return y @ dct_basis(self.d).T          # x = Q y

    def perpendicular_basis(self, p: np.ndarray) -> np.ndarray:
        k = self.sheet_dim
        Nb = dct_basis(self.d)[:, k:]           # [d, d-k], orthonormal columns
        return np.broadcast_to(Nb, (len(p), self.d, self.d - k)).copy()

    def perpendicular_coords(self, X: np.ndarray) -> np.ndarray:
        """The exactly computable coordinates ``y_{k+1..d}`` perpendicular to the sheet."""
        y = np.asarray(X, dtype=np.float64) @ dct_basis(self.d)
        return y[:, self.sheet_dim:]

    def distance_to_sheet(self, X: np.ndarray) -> np.ndarray:
        """Exact distance from the sheet (it is a linear subspace)."""
        return np.linalg.norm(self.perpendicular_coords(X), axis=1)


@dataclass
class CurvedSheet(_Sheet):
    """A sinusoidally bent 1-D (``d = 2``) or 2-D (``d >= 3``) sheet, ``x = Q y(s,t)``."""

    name: str = "curved_sheet"

    def __post_init__(self):
        self.name = "curved_sheet_noisy" if self.noisy else "curved_sheet"

    @property
    def sheet_dim(self) -> int:
        return 1 if self.d == 2 else 2

    def params(self, n, rng) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(n, self.sheet_dim))

    def _y(self, p: np.ndarray) -> np.ndarray:
        if self.d == 2:
            t = p[:, 0]
            return np.stack([0.75 * t, 0.30 * np.sin(np.pi * t)], axis=1)
        s, t = p[:, 0], p[:, 1]
        full = np.stack([0.65 * s, 0.65 * t, 0.25 * np.sin(np.pi * s),
                         0.20 * np.sin(np.pi * t), 0.15 * np.sin(np.pi * (s + t))], axis=1)
        return full[:, :self.d]

    def _dy(self, p: np.ndarray) -> np.ndarray:
        """``dy/d(params)``, shape ``[n, d, k]`` (analytic)."""
        n = len(p)
        if self.d == 2:
            t = p[:, 0]
            Jy = np.zeros((n, 2, 1))
            Jy[:, 0, 0] = 0.75
            Jy[:, 1, 0] = 0.30 * np.pi * np.cos(np.pi * t)
            return Jy
        s, t = p[:, 0], p[:, 1]
        Jy = np.zeros((n, 5, 2))
        Jy[:, 0, 0] = 0.65
        Jy[:, 1, 1] = 0.65
        Jy[:, 2, 0] = 0.25 * np.pi * np.cos(np.pi * s)
        Jy[:, 3, 1] = 0.20 * np.pi * np.cos(np.pi * t)
        Jy[:, 4, 0] = 0.15 * np.pi * np.cos(np.pi * (s + t))
        Jy[:, 4, 1] = 0.15 * np.pi * np.cos(np.pi * (s + t))
        return Jy[:, :self.d, :]

    def embed(self, p: np.ndarray) -> np.ndarray:
        return self._y(p) @ dct_basis(self.d).T

    def tangent_jacobian(self, p: np.ndarray) -> np.ndarray:
        """``dx/d(params) = Q dy/d(params)``, shape ``[n, d, k]``."""
        return np.einsum("ij,njk->nik", dct_basis(self.d), self._dy(p))

    def perpendicular_basis(self, p: np.ndarray) -> np.ndarray:
        """Orthonormal perpendicular basis from the QR of the tangent Jacobian."""
        Jt = self.tangent_jacobian(p)                       # [n, d, k]
        k = Jt.shape[2]
        Qf, _ = np.linalg.qr(Jt, mode="complete")           # batched: [n, d, d]
        return np.ascontiguousarray(Qf[:, :, k:])

    def distance_to_sheet(self, X: np.ndarray, grid: int = 161) -> np.ndarray:
        """Approximate distance to the sheet: nearest point on a dense parameter grid.

        An upper bound (the exact projection has no closed form), meant for checks on
        modest point counts. The offset sets above have exact distances by construction.
        """
        X = np.asarray(X, dtype=np.float64)
        g = np.linspace(-1.0, 1.0, grid)
        if self.sheet_dim == 1:
            P = g[:, None]
        else:
            ss, tt = np.meshgrid(g, g, indexing="ij")
            P = np.stack([ss.ravel(), tt.ravel()], axis=1)
        M = self.embed(P)
        out = np.empty(len(X), dtype=np.float64)
        step = max(1, 2_000_000 // max(1, len(M)))
        for i in range(0, len(X), step):
            chunk = X[i:i + step]
            dist = np.linalg.norm(chunk[:, None, :] - M[None, :, :], axis=2)
            out[i:i + step] = dist.min(axis=1)
        return out


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def make_density(tag: str, d: int) -> Density:
    """Build a data geometry from its name."""
    if tag == "even_grid":
        return EvenGrid(d=d)
    if tag == "uniform":
        return Uniform(d=d)
    if tag == "hotspots":
        return Hotspots(d)
    if tag == "stretched_hotspots":
        return StretchedHotspots(d)
    if tag == "flat_sheet":
        return FlatSheet(d=d, noisy=False)
    if tag == "flat_sheet_noisy":
        return FlatSheet(d=d, noisy=True)
    if tag == "curved_sheet":
        return CurvedSheet(d=d, noisy=False)
    if tag == "curved_sheet_noisy":
        return CurvedSheet(d=d, noisy=True)
    raise KeyError(f"unknown data geometry {tag!r}")
