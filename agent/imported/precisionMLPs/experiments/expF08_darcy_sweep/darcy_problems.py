"""Darcy problems for the expF08 precision sweep.

Two halves:
  1. A manufactured smooth-coefficient Darcy control posed natively on [-1,1]^2
     (u* = 0 on the boundary) where machine precision is verifiable. All
     hand-coded derivatives/forcing are FD-verified by verify_control().
  2. DarcyCoefficient: a cubic-spline surrogate (optional Gaussian pre-smooth)
     of a gridded benchmark coefficient, exposing the non-divergence-form
     operator terms on [-1,1]^2.

Benchmark convention: -div_s(a grad_s u) = 1 on [0,1]^2, u = 0 on the boundary.
On p = 2s - 1 this becomes  -a lap_p u - grad_p a . grad_p u = 1/4,
hence DARCY_FORCING = 0.25 and terms L = -a*lap - grad a . grad.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

PI = np.pi

# The FNO darcy_421 test set, JAX-exported to npz. Lives in the sibling
# continuous-mlps checkout; override with run_darcy.py --darcy-path.
DEFAULT_NPZ = "/scr/cdeng/continuous-mlps/data/fno_datasets_jax/darcy_test_421_jax.npz"

# ---------------------------------------------------------------------------
# Smooth control:  a = 3 + exp(sin(pi x) sin(pi y)),
#                  u* = sin(pi x) sin(pi y) + 0.5 sin(2 pi x) sin(pi y)
# ---------------------------------------------------------------------------


def control_a(P):
    x, y = P[:, 0], P[:, 1]
    return 3.0 + np.exp(np.sin(PI * x) * np.sin(PI * y))


def control_ax(P):
    x, y = P[:, 0], P[:, 1]
    return np.exp(np.sin(PI * x) * np.sin(PI * y)) * PI * np.cos(PI * x) * np.sin(PI * y)


def control_ay(P):
    x, y = P[:, 0], P[:, 1]
    return np.exp(np.sin(PI * x) * np.sin(PI * y)) * PI * np.sin(PI * x) * np.cos(PI * y)


def control_u(P):
    x, y = P[:, 0], P[:, 1]
    return np.sin(PI * x) * np.sin(PI * y) + 0.5 * np.sin(2 * PI * x) * np.sin(PI * y)


def control_ux(P):
    x, y = P[:, 0], P[:, 1]
    return PI * np.cos(PI * x) * np.sin(PI * y) + PI * np.cos(2 * PI * x) * np.sin(PI * y)


def control_uy(P):
    x, y = P[:, 0], P[:, 1]
    return PI * np.sin(PI * x) * np.cos(PI * y) + 0.5 * PI * np.sin(2 * PI * x) * np.cos(PI * y)


def control_lap(P):
    x, y = P[:, 0], P[:, 1]
    uxx = -PI**2 * np.sin(PI * x) * np.sin(PI * y) - 2 * PI**2 * np.sin(2 * PI * x) * np.sin(PI * y)
    uyy = -PI**2 * np.sin(PI * x) * np.sin(PI * y) - 0.5 * PI**2 * np.sin(2 * PI * x) * np.sin(PI * y)
    return uxx + uyy


def control_forcing(P):
    """f = -(a lap u* + grad a . grad u*)  so that L[u*] = f."""
    return -(control_a(P) * control_lap(P)
             + control_ax(P) * control_ux(P)
             + control_ay(P) * control_uy(P))


def control_terms():
    """L = -a lap - grad a . grad, as solver terms."""
    return [((2, 0), lambda P: -control_a(P)),
            ((0, 2), lambda P: -control_a(P)),
            ((1, 0), lambda P: -control_ax(P)),
            ((0, 1), lambda P: -control_ay(P))]


def _fd_partial(f, P, ax, ay, h=1e-5):
    """Central-difference d^ax_x d^ay_y f at P (f: [n,2] -> [n]); orders <= 2."""
    if ax == 0 and ay == 0:
        return f(P)
    if ax + ay == 1:
        col = 0 if ax else 1
        Pp, Pm = P.copy(), P.copy()
        Pp[:, col] += h
        Pm[:, col] -= h
        return (f(Pp) - f(Pm)) / (2 * h)
    if (ax, ay) == (2, 0) or (ax, ay) == (0, 2):
        col = 0 if ax else 1
        Pp, Pm = P.copy(), P.copy()
        Pp[:, col] += h
        Pm[:, col] -= h
        return (f(Pp) - 2 * f(P) + f(Pm)) / h**2
    raise ValueError((ax, ay))


def verify_control(tol=2e-4):
    """FD-check every hand-coded control derivative and the forcing identity."""
    rng = np.random.default_rng(7)
    P = rng.uniform(-0.9, 0.9, (60, 2))
    checks = [
        ("a_x", control_ax(P), _fd_partial(control_a, P, 1, 0)),
        ("a_y", control_ay(P), _fd_partial(control_a, P, 0, 1)),
        ("u_x", control_ux(P), _fd_partial(control_u, P, 1, 0)),
        ("u_y", control_uy(P), _fd_partial(control_u, P, 0, 1)),
        ("lap", control_lap(P),
         _fd_partial(control_u, P, 2, 0, h=1e-4) + _fd_partial(control_u, P, 0, 2, h=1e-4)),
        ("forcing", control_forcing(P),
         -(control_a(P) * (_fd_partial(control_u, P, 2, 0, h=1e-4)
                           + _fd_partial(control_u, P, 0, 2, h=1e-4))
           + _fd_partial(control_a, P, 1, 0) * _fd_partial(control_u, P, 1, 0)
           + _fd_partial(control_a, P, 0, 1) * _fd_partial(control_u, P, 0, 1))),
    ]
    for name, ours, fd in checks:
        scale = max(1.0, np.max(np.abs(fd)))
        err = np.max(np.abs(ours - fd)) / scale
        assert err < tol, f"control check '{name}' failed: rel err {err:.2e}"


# ---------------------------------------------------------------------------
# Real benchmark instances: spline surrogate of the gridded coefficient
# ---------------------------------------------------------------------------

DARCY_FORCING = 0.25  # -div_s(a grad_s u) = 1 on [0,1]^2, mapped to [-1,1]^2


def grid_1d(n, cell_centered=False):
    """The n dataset grid coordinates mapped to [-1,1]. cell_centered=True puts
    nodes at (k+1/2)h (the darcy_421 convention: stored boundary values are
    ~slope*h/2, not 0)."""
    if cell_centered:
        return -1.0 + (2.0 * np.arange(n) + 1.0) / n
    return np.linspace(-1.0, 1.0, n)


class DarcyCoefficient:
    """Cubic-spline surrogate of a gridded coefficient on [-1,1]^2.

    a_grid[i, j] = a(x_i, y_j) with x_i, y_j uniform over the domain
    (dataset [0,1] grid mapped to [-1,1]; node convention per grid_1d).
    sigma_px > 0 applies a Gaussian pre-smooth (in pixels) before fitting
    the spline.
    """

    def __init__(self, a_grid, sigma_px=0.0, cell_centered=False):
        a = np.asarray(a_grid, dtype=np.float64)
        if sigma_px > 0:
            a = gaussian_filter(a, sigma_px, mode="nearest")
        n0, n1 = a.shape
        self._sp = RectBivariateSpline(grid_1d(n0, cell_centered),
                                       grid_1d(n1, cell_centered), a, kx=3, ky=3)

    def a(self, P):
        return self._sp.ev(P[:, 0], P[:, 1])

    def ax(self, P):
        return self._sp.ev(P[:, 0], P[:, 1], dx=1)

    def ay(self, P):
        return self._sp.ev(P[:, 0], P[:, 1], dy=1)

    def terms(self):
        """L = -a lap - grad a . grad, as solver terms (use with DARCY_FORCING)."""
        return [((2, 0), lambda P: -self.a(P)),
                ((0, 2), lambda P: -self.a(P)),
                ((1, 0), lambda P: -self.ax(P)),
                ((0, 1), lambda P: -self.ay(P))]


def load_darcy_test(path=DEFAULT_NPZ, n_instances=16):
    """Load (a, u_ref) from the JAX-exported npz; tolerant to key naming and a
    leading channel dim. Returns float64 arrays [n, H, W]."""
    d = np.load(path)
    keys = set(d.keys())
    if {"x", "y"} <= keys:
        a, u = d["x"], d["y"]
    elif {"a", "u"} <= keys:
        a, u = d["a"], d["u"]
    else:
        raise KeyError(f"unrecognized darcy npz keys: {sorted(keys)}")
    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    a = a.reshape(a.shape[0], a.shape[-2], a.shape[-1])
    u = u.reshape(u.shape[0], u.shape[-2], u.shape[-1])
    return a[:n_instances], u[:n_instances]
