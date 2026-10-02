"""expF17 1-D dictionary families for the dysts arm. W-exact column accounting.

Every family reports EXACTLY W columns (rbf_phs included -- its 2-term degree-1
tail comes out of the budget). Interface: rows(s, order) -> [len(s), W] for
d^order/ds^order, order in {0, 1} (the ODE arm needs u and u' only).

  qi_grid   W-1 uniform tanh centres (n_int + 2*halo, interior spans [-1,1],
            gamma = 0.25/h) + 1 bias.  In 1-D the Radon/tensor distinction
            collapses -- this is THE QI geometry.
  elm       W-1 iid tanh units, a,b ~ U(-R,R), + 1 bias
  spectral  T_0..T_{W-1} Chebyshev
  bwler     W CGL nodal values (value space, rows = cheb rows @ V^{-1})
  rbf_imq   W grid centres, (1+(eps r)^2)^{-1/2}, eps = epsh/h
  rbf_phs   W-2 grid centres |s-c|^3 + [1, s]

Seeds: qi/rbf grid-origin shift, elm fresh draw; spectral/bwler deterministic.
lambda = 0.25 baked for qi (17.3); halo default round(n_centres/8) capped 8,
the 1-D step-0 result (flat in N).
"""
from __future__ import annotations

import numpy as np

from .dicts import LAM_TANH, _cheb_basis, psi


class Ridge1D:
    def __init__(self, a, b, meta):
        self.a, self.b = a, b
        self.cols = len(a) + 1
        self.meta = meta

    def rows(self, s, order):
        t = np.tanh(s[:, None] * self.a[None, :] - self.b[None, :])
        A = (self.a ** order)[None, :] * psi(order, t)
        bias = np.ones((len(s), 1)) if order == 0 else np.zeros((len(s), 1))
        return np.hstack([A, bias])


def qi1d(W, halo, shift=0.0, lam=LAM_TANH):
    n_c = W - 1
    n_int = n_c - 2 * halo
    if n_int < 3:
        raise ValueError(f"qi1d: halo {halo} leaves n_int {n_int} < 3")
    h = 2.0 / (n_int - 1)
    c = -1.0 + (np.arange(n_c) - halo + shift) * h
    g = lam / h
    return Ridge1D(np.full(n_c, g), g * c,
                   dict(method="qi_grid", W=W, halo=halo, lam=lam, n_int=n_int,
                        shift=float(shift)))


def elm1d(W, R, rng):
    a = rng.uniform(-R, R, W - 1)
    b = rng.uniform(-R, R, W - 1)
    return Ridge1D(a, b, dict(method="elm", W=W, R=R))


class Cheb1D:
    def __init__(self, W):
        self.cols = W
        self.meta = dict(method="spectral", W=W, basis="chebyshev")

    def rows(self, s, order):
        return _cheb_basis(s, self.cols, order)


class BWLer1D:
    def __init__(self, W):
        j = np.arange(W)
        self.nodes = np.cos(j * np.pi / (W - 1))
        V = _cheb_basis(self.nodes, W, 0)
        self.Vinv = np.linalg.solve(V, np.eye(W))
        self.cols = W
        self.meta = dict(method="bwler", W=W, fd_k="spectral", nodes="cgl")

    def rows(self, s, order):
        return _cheb_basis(s, self.cols, order) @ self.Vinv


class RBF1D:
    def __init__(self, W, kind, epsh=1.0, shift=0.0):
        n_c = W if kind == "imq" else W - 2
        h = 2.0 / (n_c - 1)
        self.c = np.linspace(-1.0, 1.0, n_c) + shift * h
        self.kind = kind
        self.eps = epsh / h if kind == "imq" else None
        self.cols = W
        self.meta = dict(method=f"rbf_{kind}", W=W,
                         epsh=(epsh if kind == "imq" else None),
                         n_centres=n_c, shift=float(shift))

    def rows(self, s, order):
        dx = s[:, None] - self.c[None, :]
        if self.kind == "imq":
            e2 = self.eps ** 2
            q = 1.0 + e2 * dx * dx
            A = q ** -0.5 if order == 0 else -e2 * dx * q ** -1.5
            return A
        A = np.abs(dx) ** 3 if order == 0 else 3.0 * dx * np.abs(dx)
        tail = (np.stack([np.ones(len(s)), s], axis=1) if order == 0
                else np.stack([np.zeros(len(s)), np.ones(len(s))], axis=1))
        return np.hstack([A, tail])


METHODS_1D = ["qi_grid", "elm", "spectral", "bwler", "rbf_imq", "rbf_phs"]


def build_dictionary_1d(method, W, config, seed_rng):
    if method == "qi_grid":
        return qi1d(W, config["halo"], shift=seed_rng.uniform(-0.5, 0.5))
    if method == "elm":
        return elm1d(W, config["R"], seed_rng)
    if method == "spectral":
        return Cheb1D(W)
    if method == "bwler":
        return BWLer1D(W)
    if method == "rbf_imq":
        return RBF1D(W, "imq", epsh=config["epsh"], shift=seed_rng.uniform(-0.5, 0.5))
    if method == "rbf_phs":
        return RBF1D(W, "phs", shift=seed_rng.uniform(-0.5, 0.5))
    raise ValueError(method)
