"""The two reference models the suite is exercised with.

``EvenGeometry`` is the no-adaptation reference: directions spread evenly over the
sphere, centers spaced evenly along each direction, nothing chosen from the data except
the output weights. Those come from one truncated-SVD least-squares solve with a bias
column at ``rcond = 1e-13``, the same solve the one-dimensional experiments use.

Geometry, given a first-layer budget ``B`` (the number of ``tanh`` units):

* ``n_per_direction = max(3, round(B^(1/d)))`` centers along each direction and
  ``n_directions = max(1, round(B / n_per_direction))`` directions, so the product is
  about ``B``.
* Directions. ``d=1``: the single direction ``[1]``. ``d=2``: ``n_directions`` equally
  spaced angles on ``[0, pi)``, offset by half a step so none lands on an axis.
  ``d>=3``: a deterministic, evenly spread set on the sphere -- spherical Fibonacci for
  ``d=3``, and for ``d>=4`` a fixed-seed Gaussian draw normalized to unit length.
  **The ``d>=4`` set is a placeholder**: it is evenly spread in distribution but is not
  an equal-weight cubature rule. Each direction is flipped to a canonical sign so that
  ``v`` and ``-v`` never both appear.
* Centers. The projection ``v . x`` runs over ``[-||v||_1, ||v||_1]`` on the cube, so the
  centers are ``n_per_direction`` evenly spaced points on ``[-T, T]`` with
  ``T = margin * ||v||_1``, ``margin = 1.25``. The extra 25% is what keeps the two ends
  of each direction from being starved: the ``tanh`` tail decays like
  ``exp(-2 lambda * (extra width))``, and 25% at ``lambda = 0.25`` puts that tail below
  the accuracy of the solve. It is load-bearing -- measured on the easiest 1-D task at
  ``B = 128``, relative ``L_2`` is about ``5e-14`` with the margin and ``1.4e-6`` without.
* Width. ``h = 2T/n_per_direction`` along each direction and ``gamma = lambda/h`` with
  ``lambda = 0.25`` (the value expC03 settled on).

``RandomFeatures`` is the external control: a random first layer (Xavier-normal) at the
same budget with the same solve. It is deliberately weak. It is here to show that the
frozen *geometry*, not the least-squares solve, is what buys accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["EvenGeometry", "RandomFeatures", "even_directions", "LAMBDA",
           "EDGE_MARGIN", "RCOND"]

LAMBDA = 0.25
EDGE_MARGIN = 1.25
RCOND = 1e-13
DIRECTION_SEED = 20260828


def _canonical_sign(V: np.ndarray) -> np.ndarray:
    """Flip each row so its entry of largest magnitude is positive."""
    V = np.asarray(V, dtype=np.float64).copy()
    for i in range(len(V)):
        j = int(np.argmax(np.abs(V[i])))
        if V[i, j] < 0:
            V[i] = -V[i]
    return V


def even_directions(d: int, n: int, seed: int = DIRECTION_SEED) -> np.ndarray:
    """A deterministic, evenly spread set of ``n`` directions, no two of them opposite."""
    if d == 1:
        return np.ones((1, 1), dtype=np.float64)
    if d == 2:
        th = np.arange(n) * np.pi / n
        return _canonical_sign(np.stack([np.cos(th), np.sin(th)], axis=1))
    if d == 3:
        # Spherical Fibonacci on the upper half sphere (opposite directions identified).
        i = np.arange(n, dtype=np.float64) + 0.5
        z = i / n                                   # (0, 1): upper half only
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        V = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
        return _canonical_sign(V)
    rng = np.random.default_rng(seed + 1000 * d + n)
    G = rng.normal(size=(n, d))
    return _canonical_sign(G / np.linalg.norm(G, axis=1, keepdims=True))


def _solve_svd(Phi: np.ndarray, y: np.ndarray, rcond: float) -> tuple[np.ndarray, float, dict]:
    """Truncated-SVD least squares with an appended bias column."""
    A = np.hstack([Phi, np.ones((len(Phi), 1), dtype=np.float64)])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > rcond * s[0]
    s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    sol = Vt.T @ (s_inv * (U.T @ np.asarray(y, dtype=np.float64).ravel()))
    info = {"rank": int(keep.sum()), "n_cols": A.shape[1],
            "largest_singular_value": float(s[0]), "smallest_singular_value": float(s[-1])}
    return sol[:-1], float(sol[-1]), info


@dataclass
class EvenGeometry:
    """Even directions, even centers, one least-squares solve. No adaptation anywhere."""

    d: int
    budget: int
    lam: float = LAMBDA
    margin: float = EDGE_MARGIN
    rcond: float = RCOND
    seed: int = DIRECTION_SEED
    name: str = "even_geometry"
    directions: np.ndarray = field(init=False)
    centers: np.ndarray = field(init=False)
    gammas: np.ndarray = field(init=False)
    weights: np.ndarray | None = field(init=False, default=None)
    bias: float = field(init=False, default=0.0)
    info: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        d, B = self.d, int(self.budget)
        n_per = max(3, int(round(B ** (1.0 / d))))
        n_dir = max(1, int(round(B / n_per)))
        self.n_per_direction, self.n_directions = n_per, n_dir
        V = even_directions(d, n_dir, seed=self.seed)
        dirs, cens, gams = [], [], []
        for v in V:
            T = self.margin * float(np.abs(v).sum())
            h = 2.0 * T / n_per
            t = -T + (np.arange(n_per) + 0.5) * h
            dirs.append(np.repeat(v[None, :], n_per, axis=0))
            cens.append(t)
            gams.append(np.full(n_per, self.lam / h))
        self.directions = np.vstack(dirs)
        self.centers = np.concatenate(cens)
        self.gammas = np.concatenate(gams)

    # -- interface -------------------------------------------------------
    def features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.tanh(self.gammas[None, :] * (X @ self.directions.T - self.centers[None, :]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EvenGeometry":
        self.weights, self.bias, self.info = _solve_svd(self.features(X), y, self.rcond)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("call fit() first")
        return self.features(X) @ self.weights + self.bias

    def geometry(self) -> dict:
        """Directions, centers and widths -- the shape every model reports."""
        return {"directions": self.directions, "centers": self.centers,
                "gammas": self.gammas, "n_directions": self.n_directions,
                "n_per_direction": self.n_per_direction,
                "unique_directions": even_directions(self.d, self.n_directions, seed=self.seed),
                "lambda": self.lam, "margin": self.margin}


@dataclass
class RandomFeatures:
    """External control: a random first layer at the same budget with the same solve."""

    d: int
    budget: int
    rcond: float = RCOND
    seed: int = 0
    name: str = "random_features"
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    weights: np.ndarray | None = field(init=False, default=None)
    bias: float = field(init=False, default=0.0)
    info: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed + 7717)
        sd = np.sqrt(2.0 / (self.d + self.budget))     # Xavier normal, gain 1
        self.W = rng.normal(0.0, sd, size=(self.budget, self.d))
        self.b = rng.normal(0.0, sd, size=self.budget)

    def features(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(X, dtype=np.float64) @ self.W.T + self.b[None, :])

    def fit(self, X, y):
        self.weights, self.bias, self.info = _solve_svd(self.features(X), y, self.rcond)
        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("call fit() first")
        return self.features(X) @ self.weights + self.bias

    def geometry(self) -> dict:
        gam = np.linalg.norm(self.W, axis=1)
        gam_safe = np.where(gam > 0, gam, 1.0)
        return {"directions": self.W / gam_safe[:, None],
                "centers": -self.b / gam_safe,
                "gammas": gam,
                "n_directions": int(self.budget), "n_per_direction": 1,
                "unique_directions": _canonical_sign(self.W / gam_safe[:, None])}
