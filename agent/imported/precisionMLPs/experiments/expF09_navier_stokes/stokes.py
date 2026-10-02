"""Stage-A manufactured Stokes problem on [-1,1]^2 (expF09).

Exact divergence-free field from the streamfunction psi = sin(pi x) sin(pi y):
  u* =  psi_y =  pi sin(pi x) cos(pi y)
  v* = -psi_x = -pi cos(pi x) sin(pi y)      (so u*_x + v*_y = 0 exactly)
  p* = cos(pi x) cos(pi y)
Stokes: -nu lap u + grad p = f, div u = 0. Forcing f = -nu lap u* + grad p*.
"""
from __future__ import annotations

import numpy as np

PI = np.pi
NU = 1.0  # Stokes is linear in nu; fixed for the manufactured case


def u_star(P):
    x, y = P[:, 0], P[:, 1]
    return PI * np.sin(PI * x) * np.cos(PI * y)


def v_star(P):
    x, y = P[:, 0], P[:, 1]
    return -PI * np.cos(PI * x) * np.sin(PI * y)


def p_star(P):
    x, y = P[:, 0], P[:, 1]
    return np.cos(PI * x) * np.cos(PI * y)


def u_star_x(P):
    x, y = P[:, 0], P[:, 1]
    return PI**2 * np.cos(PI * x) * np.cos(PI * y)


def v_star_y(P):
    x, y = P[:, 0], P[:, 1]
    return -PI**2 * np.cos(PI * x) * np.cos(PI * y)


def lap_u_star(P):
    x, y = P[:, 0], P[:, 1]
    return -2 * PI**3 * np.sin(PI * x) * np.cos(PI * y)


def lap_v_star(P):
    x, y = P[:, 0], P[:, 1]
    return 2 * PI**3 * np.cos(PI * x) * np.sin(PI * y)


def p_star_x(P):
    x, y = P[:, 0], P[:, 1]
    return -PI * np.sin(PI * x) * np.cos(PI * y)


def p_star_y(P):
    x, y = P[:, 0], P[:, 1]
    return -PI * np.cos(PI * x) * np.sin(PI * y)


def f_u(P):
    return -NU * lap_u_star(P) + p_star_x(P)


def f_v(P):
    return -NU * lap_v_star(P) + p_star_y(P)


def stokes_equations(interior_pts, boundary_pts, nu=NU):
    """The five equation blocks (+ pressure gauge) for solve_system."""
    return [
        dict(points=interior_pts,
             blocks={"u": [((2, 0), -nu), ((0, 2), -nu)], "p": [((1, 0), 1.0)]},
             rhs=f_u),
        dict(points=interior_pts,
             blocks={"v": [((2, 0), -nu), ((0, 2), -nu)], "p": [((0, 1), 1.0)]},
             rhs=f_v),
        dict(points=interior_pts,
             blocks={"u": [((1, 0), 1.0)], "v": [((0, 1), 1.0)]}, rhs=0.0),
        dict(points=boundary_pts, blocks={"u": [((0, 0), 1.0)]}, rhs=u_star),
        dict(points=boundary_pts, blocks={"v": [((0, 0), 1.0)]}, rhs=v_star),
        dict(points=np.zeros((1, 2)), blocks={"p": [((0, 0), 1.0)]}, rhs=p_star),
    ]


def _fd(f, P, ax, ay, h=1e-5):
    if ax == 0 and ay == 0:
        return f(P)
    col = 0 if ax else 1
    if ax + ay == 1:
        Pp, Pm = P.copy(), P.copy()
        Pp[:, col] += h
        Pm[:, col] -= h
        return (f(Pp) - f(Pm)) / (2 * h)
    Pp, Pm = P.copy(), P.copy()
    Pp[:, col] += h
    Pm[:, col] -= h
    return (f(Pp) - 2 * f(P) + f(Pm)) / h**2


def verify_stokes(tol=2e-4):
    """FD-check the hand-coded derivatives and the momentum forcing identity."""
    rng = np.random.default_rng(7)
    P = rng.uniform(-0.9, 0.9, (60, 2))
    checks = [
        ("u_x", u_star_x(P), _fd(u_star, P, 1, 0)),
        ("v_y", v_star_y(P), _fd(v_star, P, 0, 1)),
        ("p_x", p_star_x(P), _fd(p_star, P, 1, 0)),
        ("p_y", p_star_y(P), _fd(p_star, P, 0, 1)),
        ("lap_u", lap_u_star(P),
         _fd(u_star, P, 2, 0, h=1e-4) + _fd(u_star, P, 0, 2, h=1e-4)),
        ("lap_v", lap_v_star(P),
         _fd(v_star, P, 2, 0, h=1e-4) + _fd(v_star, P, 0, 2, h=1e-4)),
    ]
    for name, ours, fd in checks:
        scale = max(1.0, np.max(np.abs(fd)))
        err = np.max(np.abs(ours - fd)) / scale
        assert err < tol, f"stokes check '{name}': rel err {err:.2e}"
