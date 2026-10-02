"""The drifting Taylor-Green vortex: an exact unsteady solution of 2-D incompressible
Navier-Stokes with a non-gradient advection term (expF18).

Physical problem on the box (x, y) in [-1, 1]^2, t in [0, T_END]:

    u_t + (u . grad) u + grad p - nu lap u = 0,     div u = 0.

Exact verifier: the Taylor-Green cell psi = cos(k x') cos(k y') / k, boosted by a uniform
drift (U, V) (Galilean invariance keeps it an exact NS solution):

    x' = x - U t,  y' = y - V t,  E(t) = exp(-2 nu k^2 t)
    u = U - cos(k x') sin(k y') E
    v = V + sin(k x') cos(k y') E
    p = -(1/4) (cos(2 k x') + cos(2 k y')) E^2

With k = pi/2 the box holds one vortex cell at t = 0; the drift slides it across the
box while the neighbouring (counter-rotating) cell enters. In the lab frame the advection
(u . grad) u is NOT a pure gradient (it is for the un-boosted cell), so the Stokes step of
Newton does not already solve the velocity -- the nonlinearity is exercised.

Solver coordinates are the scaled cube X = (x, y, tau) in [-1, 1]^3 with
t = (tau + 1) T_END / 2, so d/dt = (2 / T_END) d/dtau.  `verify()` checks the closed form
against the NS operator by finite differences.
"""
from __future__ import annotations

import numpy as np

K = np.pi / 2.0
NU = 0.01
DRIFT = (0.5, 0.25)
T_END = 2.0
DT_DTAU = 2.0 / T_END          # dt/dtau: t = (tau + 1) T_END / 2
GAUGE_POINT = (-0.7, -0.6)     # pressure fixed on this spatial point for all tau


def to_physical(X):
    """Scaled cube point(s) (x, y, tau) -> (x, y, t)."""
    X = np.asarray(X, dtype=np.float64)
    P = X.copy()
    P[:, 2] = (X[:, 2] + 1.0) * (T_END / 2.0)
    return P


def fields_physical(P, k=K, nu=NU, drift=DRIFT):
    """P [n, 3] = (x, y, t) -> [n, 3] stacked (u, v, p)."""
    P = np.asarray(P, dtype=np.float64)
    x, y, t = P[:, 0], P[:, 1], P[:, 2]
    U, V = drift
    xp, yp = x - U * t, y - V * t
    E = np.exp(-2.0 * nu * k * k * t)
    u = U - np.cos(k * xp) * np.sin(k * yp) * E
    v = V + np.sin(k * xp) * np.cos(k * yp) * E
    p = -0.25 * (np.cos(2 * k * xp) + np.cos(2 * k * yp)) * E * E
    return np.stack([u, v, p], axis=1)


def fields(X):
    """Scaled cube point(s) -> (u, v, p)."""
    return fields_physical(to_physical(X))


def vorticity_physical(P, k=K, nu=NU, drift=DRIFT):
    """omega = v_x - u_y of the exact field."""
    P = np.asarray(P, dtype=np.float64)
    x, y, t = P[:, 0], P[:, 1], P[:, 2]
    U, V = drift
    xp, yp = x - U * t, y - V * t
    E = np.exp(-2.0 * nu * k * k * t)
    return 2.0 * k * np.cos(k * xp) * np.cos(k * yp) * E


def _fd(f, P, axis, order, h):
    Pp, Pm = P.copy(), P.copy()
    Pp[:, axis] += h
    Pm[:, axis] -= h
    if order == 1:
        return (f(Pp) - f(Pm)) / (2 * h)
    return (f(Pp) - 2 * f(P) + f(Pm)) / (h * h)


def ns_residual_fd(P, h1=1e-5, h2=1e-4, k=K, nu=NU, drift=DRIFT):
    """Finite-difference NS residual of the closed form at physical points P:
    returns (max |momentum|, max |div|)."""
    f = lambda Q: fields_physical(Q, k, nu, drift)  # noqa: E731
    F = f(P)
    u, v = F[:, 0], F[:, 1]
    Ft = _fd(f, P, 2, 1, h1)
    Fx, Fy = _fd(f, P, 0, 1, h1), _fd(f, P, 1, 1, h1)
    Fxx, Fyy = _fd(f, P, 0, 2, h2), _fd(f, P, 1, 2, h2)
    mom_u = Ft[:, 0] + u * Fx[:, 0] + v * Fy[:, 0] + Fx[:, 2] - nu * (Fxx[:, 0] + Fyy[:, 0])
    mom_v = Ft[:, 1] + u * Fx[:, 1] + v * Fy[:, 1] + Fy[:, 2] - nu * (Fxx[:, 1] + Fyy[:, 1])
    div = Fx[:, 0] + Fy[:, 1]
    return float(max(np.abs(mom_u).max(), np.abs(mom_v).max())), float(np.abs(div).max())


def advection_is_not_a_gradient(P, h=1e-5, k=K, nu=NU, drift=DRIFT):
    """max |curl of (u.grad)u| at P -- zero for the un-boosted cell, nonzero with drift."""
    def adv(Q):
        F = fields_physical(Q, k, nu, drift)
        Fx, Fy = _fd(lambda R: fields_physical(R, k, nu, drift), Q, 0, 1, h), \
            _fd(lambda R: fields_physical(R, k, nu, drift), Q, 1, 1, h)
        u, v = F[:, 0:1], F[:, 1:2]
        return u * Fx[:, :2] + v * Fy[:, :2]
    ax_y = _fd(adv, P, 1, 1, 10 * h)[:, 0]
    ay_x = _fd(adv, P, 0, 1, 10 * h)[:, 1]
    return float(np.abs(ay_x - ax_y).max())


def verify(tol=1e-7, verbose=True):
    rng = np.random.default_rng(3)
    P = rng.uniform(-0.9, 0.9, (300, 3))
    P[:, 2] = rng.uniform(0.05, T_END - 0.05, 300)
    mom, div = ns_residual_fd(P)
    assert mom < tol and div < tol, f"drifting Taylor-Green fails NS: mom {mom:.1e} div {div:.1e}"
    curl = advection_is_not_a_gradient(P)
    assert curl > 1e-2, f"advection is a gradient (curl {curl:.1e}); drift missing?"
    if verbose:
        print(f"problem verified: NS residual (FD) mom {mom:.1e} div {div:.1e}; "
              f"curl of advection {curl:.2f} (non-gradient)")
    return mom, div, curl


if __name__ == "__main__":
    verify()
