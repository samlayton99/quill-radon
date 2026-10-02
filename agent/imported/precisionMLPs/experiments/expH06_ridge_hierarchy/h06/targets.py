"""Targets in any dimension, in absolute coordinates ``X`` on the cube ``[-1, 1]^d``.

Two families:

* the expH05 functions carried to ``d`` dimensions (radial ones use the suite's
  ``rho = ||x - a|| / sqrt(2)`` so the numbers stay comparable with 2-D);
* hidden-ridge sums ``sum_i g_i(u_i . x)`` with ``r`` random directions -- the known-answer
  family for atom discovery -- optionally plus a diffuse isotropic bump.

Anchors sit inside the data ball about ``X0[d]`` so the radial features are on the data.
"""

from __future__ import annotations

import numpy as np

from .core import X0

_RAD_SHIFT = np.array([-0.05, 0.05, -0.03, 0.04, -0.02])
_BUMP_SHIFT = np.array([-0.15, 0.35, -0.1, 0.2, 0.1])


def anchor_rad(d):
    return X0[d] + _RAD_SHIFT[:d]


def anchor_bump(d):
    return X0[d] + _BUMP_SHIFT[:d]


def _rho(X, a):
    dif = X - a[None, :]
    return np.sqrt(np.einsum("nk,nk->n", dif, dif) / 2.0)


def f_gauss_bump(X):
    d = X.shape[1]
    dif = X - anchor_bump(d)[None, :]
    return np.exp(-np.einsum("nk,nk->n", dif, dif) / 0.5 ** 2)


def f_radial_runge(X):
    return 1.0 / (1.0 + 16.0 * _rho(X, anchor_rad(X.shape[1])) ** 2)


def f_narrow_runge(X):
    return 1.0 / (1.0 + 144.0 * _rho(X, anchor_rad(X.shape[1])) ** 2)


def f_fast_waves(X):
    return np.cos(6 * np.pi * _rho(X, anchor_rad(X.shape[1])))


def f_slow_waves(X):
    return np.cos(np.pi * _rho(X, anchor_rad(X.shape[1])))


def f_composition(X):
    """Genuinely d-dimensional: exp(sin(pi x1) cos(pi x2) + 0.5 sum_{k>=3} cos(pi x_k + k/3))."""
    d = X.shape[1]
    s = np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])
    for k in range(2, d):
        s = s + 0.5 * np.cos(np.pi * X[:, k] + (k + 1) / 3.0)
    return np.exp(s)


def f_product_sines(X):
    """prod_k sin(2 pi x_k): in d dims an exact sum of 2^(d-1) ridges along (+-1, ..., +-1)."""
    return np.prod(np.sin(2 * np.pi * X), axis=1)


def f_spatial_packet(X):
    d = X.shape[1]
    rp = _rho(X, X0[d])
    packet = 0.8 * np.exp(-(rp / 0.18) ** 2) * np.cos(10 * np.pi * rp)
    bump = np.exp(-_rho(X, anchor_bump(d)) ** 2 / (2 * 0.5 ** 2))
    return packet + bump


def f_polynomial(X):
    x, y = X[:, 0], X[:, 1]
    out = x * x * y - x * y ** 3 + x * y
    if X.shape[1] > 2:
        z = X[:, 2]
        out = out + x * z * z - y * z + 0.5 * z ** 3
    return out


def f_off_packet(X):
    """A localized high-frequency feature OFF the ball center (at x0 + 0.6 r_ball along a
    fixed direction) on a smooth background: the concentrated-residual test case."""
    d = X.shape[1]
    u = np.ones(d) / np.sqrt(d)
    c = X0[d] + 0.18 * u                       # 0.6 * r at r = 0.3
    rc = _rho(X, c)
    return np.exp(-(rc / 0.06) ** 2) * np.cos(16 * np.pi * rc) + np.exp(-_rho(X, anchor_bump(d)) ** 2)


def f_off_packet2(X):
    """The off-center localized feature recalibrated so 2-4k-unit fits are in range:
    width 0.10, frequency 8 pi (the first version, width 0.06 / 16 pi, saturated every arm)."""
    d = X.shape[1]
    u = np.ones(d) / np.sqrt(d)
    c = X0[d] + 0.18 * u
    rc = _rho(X, c)
    return np.exp(-(rc / 0.10) ** 2) * np.cos(8 * np.pi * rc) + np.exp(-_rho(X, anchor_bump(d)) ** 2)


BASE_TARGETS = {
    "off_packet": f_off_packet,
    "off_packet2": f_off_packet2,
    "gauss_bump": f_gauss_bump,
    "radial_runge": f_radial_runge,
    "narrow_runge": f_narrow_runge,
    "fast_waves": f_fast_waves,
    "slow_waves": f_slow_waves,
    "composition": f_composition,
    "product_sines": f_product_sines,
    "spatial_packet": f_spatial_packet,
    "polynomial": f_polynomial,
}


# ---------------------------------------------------------------------------
# hidden-ridge sums
# ---------------------------------------------------------------------------

class RidgeSum:
    """``F(x) = sum_i g_i(u_i . (x - x0)) [+ bump]`` with ``r`` random unit directions.

    Profiles ``g_i(t) = sin(w_i t + phi_i)`` with ``w_i`` spread over ``[w_lo, w_hi]`` so
    every ridge carries several oscillations across a data ball of radius ~0.3; the
    optional diffuse term is ``0.3 exp(-||x - a||^2 / 0.5^2)`` (not a finite ridge sum).
    """

    def __init__(self, d, r, seed=0, w_lo=8.0, w_hi=20.0, bump=False):
        rng = np.random.default_rng(1000 * d + 17 * r + seed)
        U = rng.normal(size=(r, d))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        self.d, self.r, self.U = d, r, U
        self.w = np.linspace(w_lo, w_hi, r) if r > 1 else np.array([0.5 * (w_lo + w_hi)])
        self.phi = rng.uniform(0, 2 * np.pi, size=r)
        self.bump = bump
        self.x0 = X0[d]

    def profile(self, i, t):
        return np.sin(self.w[i] * t + self.phi[i])

    def __call__(self, X):
        Z = X - self.x0[None, :]
        P = Z @ self.U.T
        out = np.zeros(len(X))
        for i in range(self.r):
            out += self.profile(i, P[:, i])
        if self.bump:
            out += 0.3 * f_gauss_bump(X)
        return out

    def name(self):
        return f"ridge{self.r}" + ("_bump" if self.bump else "")


def get_target(name: str, d: int):
    """``name`` is a BASE_TARGETS key, or ``ridge<r>`` / ``ridge<r>_bump``."""
    if name in BASE_TARGETS:
        return BASE_TARGETS[name]
    if name.startswith("ridge"):
        core = name[len("ridge"):]
        bump = core.endswith("_bump")
        r = int(core.replace("_bump", ""))
        return RidgeSum(d, r, bump=bump)
    raise KeyError(name)
