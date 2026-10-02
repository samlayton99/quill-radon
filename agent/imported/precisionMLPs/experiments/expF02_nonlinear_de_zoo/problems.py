"""expF02 problem suite: 9 classical NONLINEAR differential problems (3 categories x orders 1-3).

Same categories/coordinates as expF01 (ode1d on [-1,1]; pde2d_steady on the unit
disk; pde2d_time on [-1,1]x[0,1] in scaled coords (x, tau), tau = 2t-1, so
d/dt = 2 d/dtau). The operator is split as

    residual(u) = sum_i s_i D_i u   (lin_terms, possibly empty)
                + N(D u)            (nl: pointwise nonlinearity of derivative fields)
                - f.

Each problem dict:
  key, category, order, title
  lin_terms    -- linear part, expF01 format (constant coeffs here)
  nl           -- dict(fields=[multi-indices], res=fn(vals, pts), jac=fn(vals, pts))
                  vals maps each field index -> array of that derivative of u at pts;
                  jac returns {field: dN/dfield array} (the Gauss-Newton diagonal)
  exact, forcing, conditions/exact_derivs (1D), bc_blocks (2D) -- as in expF01
  init         -- None (zero start) or dict(terms, rhs): a LINEAR problem solved once
                  with the same condition blocks to produce the Newton starting point
                  (harmonic extension for eikonal; Delta u = 2 sqrt(f) for Monge-Ampere).

verify_all() checks residual(u*) == f and all condition values by finite differences.
"""

from __future__ import annotations

import numpy as np

PI = np.pi


def _safe_exp(u):
    return np.exp(np.minimum(u, 40.0))


# ---------------------------------------------------------------------------
# Category A: 1D nonlinear ODEs on [-1, 1]
# ---------------------------------------------------------------------------

# A1 (order 1, IVP): logistic growth u' = 3u(1-u). Exact sigmoid, f = 0.
# (rate 3, not 4: at rate 4 the sigmoid tail at the IC is so flat that Newton
#  from zero lands on a shifted-sigmoid branch; rate 3 keeps the IC informative)
def _a1_u(x):
    return 1.0 / (1.0 + np.exp(-3.0 * x))

def _a1_u1(x):
    s = _a1_u(x)
    return 3.0 * s * (1.0 - s)

A1 = dict(
    key="ode1d_o1", category="ode1d", order=1,
    title="logistic  $u' = 3u(1-u)$  (IVP)",
    lin_terms=[(1, 1.0), (0, -3.0)],
    nl=dict(fields=[0],
            res=lambda v, p: 3.0 * v[0] ** 2,
            jac=lambda v, p: {0: 6.0 * v[0]}),
    exact=_a1_u, forcing=0.0,
    conditions=[(-1.0, 0)],
    exact_derivs={0: _a1_u, 1: _a1_u1},
    init=None,
)

# A2 (order 2, BVP): Bratu-type u'' + e^u = f (combustion). u* = sin(2x) + 0.3 x^2.
def _a2_u(x):
    return np.sin(2 * x) + 0.3 * x**2

def _a2_u1(x):
    return 2 * np.cos(2 * x) + 0.6 * x

def _a2_u2(x):
    return -4 * np.sin(2 * x) + 0.6

A2 = dict(
    key="ode1d_o2", category="ode1d", order=2,
    title="Bratu  $u'' + e^u = f$  (Dirichlet)",
    lin_terms=[(2, 1.0)],
    nl=dict(fields=[0],
            res=lambda v, p: _safe_exp(v[0]),
            jac=lambda v, p: {0: _safe_exp(v[0])}),
    exact=_a2_u,
    forcing=lambda x: _a2_u2(x) + np.exp(_a2_u(x)),
    conditions=[(-1.0, 0), (1.0, 0)],
    exact_derivs={0: _a2_u, 1: _a2_u1, 2: _a2_u2},
    init=None,
)

# A3 (order 3): Blasius operator u''' + 0.5 u u'' = f (boundary-layer similarity).
#   u* = x + 0.5 x^2 + 0.4 exp(-2 x^2); conditions u(-1), u'(-1), u(1).
def _a3_g(x):
    return np.exp(-2 * x**2)

def _a3_u(x):
    return x + 0.5 * x**2 + 0.4 * _a3_g(x)

def _a3_u1(x):
    return 1 + x - 1.6 * x * _a3_g(x)

def _a3_u2(x):
    return 1 + (6.4 * x**2 - 1.6) * _a3_g(x)

def _a3_u3(x):
    return (19.2 * x - 25.6 * x**3) * _a3_g(x)

A3 = dict(
    key="ode1d_o3", category="ode1d", order=3,
    title="Blasius  $u''' + \\frac{1}{2}uu'' = f$  (2 left + 1 right)",
    lin_terms=[(3, 1.0)],
    nl=dict(fields=[0, 2],
            res=lambda v, p: 0.5 * v[0] * v[2],
            jac=lambda v, p: {0: 0.5 * v[2], 2: 0.5 * v[0]}),
    exact=_a3_u,
    forcing=lambda x: _a3_u3(x) + 0.5 * _a3_u(x) * _a3_u2(x),
    conditions=[(-1.0, 0), (-1.0, 1), (1.0, 0)],
    exact_derivs={0: _a3_u, 1: _a3_u1, 2: _a3_u2, 3: _a3_u3},
    init=None,
)


# ---------------------------------------------------------------------------
# Category B: 2D steady nonlinear PDEs on the unit disk
# ---------------------------------------------------------------------------

# B1 (order 1, fully nonlinear): eikonal |grad u|^2 = f, Dirichlet on the circle.
#   u* = 2x + 0.8y + 0.4 sin(2x + y)  (monotone gradient: no interior caustics).
def _b1_u(P):
    x, y = P[:, 0], P[:, 1]
    return 2 * x + 0.8 * y + 0.4 * np.sin(2 * x + y)

def _b1_ux(P):
    x, y = P[:, 0], P[:, 1]
    return 2 + 0.8 * np.cos(2 * x + y)

def _b1_uy(P):
    x, y = P[:, 0], P[:, 1]
    return 0.8 + 0.4 * np.cos(2 * x + y)

def _b1_inflow_points(n=480):
    """Inflow arc of the unit circle: grad(u).n < 0 (outward normal n = the point).

    BC placement is NOT free. Newton linearizes the eikonal to
    2 grad(u_k) . grad(delta) = -r, a first-order TRANSPORT operator, which admits
    data only on the inflow boundary. Prescribing Dirichlet on the full circle
    over-determines the correction equation and stalls Newton (residual sticks at
    ~1e-3); the inflow arc alone gains ~2 orders. See expF02 writeup.
    """
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    B = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    grad = np.stack([_b1_ux(B), _b1_uy(B)], axis=1)
    return B[(grad * B).sum(axis=1) < 0]


B1 = dict(
    key="pde2d_steady_o1", category="pde2d_steady", order=1,
    title="eikonal  $u_x^2 + u_y^2 = f$  (inflow-arc Dirichlet)",
    lin_terms=[],
    nl=dict(fields=[(1, 0), (0, 1)],
            res=lambda v, p: v[(1, 0)] ** 2 + v[(0, 1)] ** 2,
            jac=lambda v, p: {(1, 0): 2 * v[(1, 0)], (0, 1): 2 * v[(0, 1)]}),
    exact=_b1_u,
    forcing=lambda P: _b1_ux(P) ** 2 + _b1_uy(P) ** 2,
    bc_blocks=[dict(where="inflow", points=_b1_inflow_points,
                    terms=[((0, 0), 1.0)], value=_b1_u)],
    # harmonic extension of the boundary data as the Newton start
    init=dict(terms=[((2, 0), 1.0), ((0, 2), 1.0)], rhs=0.0),
)

# B2 (order 2, fully nonlinear): Monge-Ampere det D^2 u = f, convex solution.
#   u* = exp((x^2 + y^2)/2); det D^2 u* = e^{x^2+y^2} (1 + x^2 + y^2) > 0.
def _b2_u(P):
    x, y = P[:, 0], P[:, 1]
    return np.exp(0.5 * (x**2 + y**2))

def _b2_f(P):
    x, y = P[:, 0], P[:, 1]
    return np.exp(x**2 + y**2) * (1 + x**2 + y**2)

B2 = dict(
    key="pde2d_steady_o2", category="pde2d_steady", order=2,
    title="Monge-Ampere  $u_{xx}u_{yy} - u_{xy}^2 = f$  (Dirichlet)",
    lin_terms=[],
    nl=dict(fields=[(2, 0), (0, 2), (1, 1)],
            res=lambda v, p: v[(2, 0)] * v[(0, 2)] - v[(1, 1)] ** 2,
            jac=lambda v, p: {(2, 0): v[(0, 2)], (0, 2): v[(2, 0)],
                              (1, 1): -2 * v[(1, 1)]}),
    exact=_b2_u,
    forcing=_b2_f,
    bc_blocks=[dict(where="circle", terms=[((0, 0), 1.0)], value=_b2_u)],
    # classical MA initializer: solve Delta u = 2 sqrt(f) with the same BCs
    init=dict(terms=[((2, 0), 1.0), ((0, 2), 1.0)],
              rhs=lambda P: 2.0 * np.sqrt(_b2_f(P))),
)

# B3 (order 3, stress): u_xxx + u_yyy + u u_x + u = f, Cauchy data on the circle.
#   u* = sin(2x) cos(2y)  (expF01 B3 solution with a Burgers advection term).
def _b3_u(P):
    x, y = P[:, 0], P[:, 1]
    return np.sin(2 * x) * np.cos(2 * y)

def _b3_ux(P):
    x, y = P[:, 0], P[:, 1]
    return 2 * np.cos(2 * x) * np.cos(2 * y)

def _b3_uy(P):
    x, y = P[:, 0], P[:, 1]
    return -2 * np.sin(2 * x) * np.sin(2 * y)

def _b3_f(P):
    x, y = P[:, 0], P[:, 1]
    uxxx = -8 * np.cos(2 * x) * np.cos(2 * y)
    uyyy = 8 * np.sin(2 * x) * np.sin(2 * y)
    return uxxx + uyyy + _b3_u(P) * _b3_ux(P) + _b3_u(P)

B3 = dict(
    key="pde2d_steady_o3", category="pde2d_steady", order=3,
    title="$u_{xxx} + u_{yyy} + uu_x + u = f$  (Cauchy data)",
    lin_terms=[((3, 0), 1.0), ((0, 3), 1.0), ((0, 0), 1.0)],
    nl=dict(fields=[(0, 0), (1, 0)],
            res=lambda v, p: v[(0, 0)] * v[(1, 0)],
            jac=lambda v, p: {(0, 0): v[(1, 0)], (1, 0): v[(0, 0)]}),
    exact=_b3_u,
    forcing=_b3_f,
    bc_blocks=[
        dict(where="circle", terms=[((0, 0), 1.0)], value=_b3_u),
        dict(where="circle",
             terms=[((1, 0), lambda P: P[:, 0]), ((0, 1), lambda P: P[:, 1])],
             value=lambda P: P[:, 0] * _b3_ux(P) + P[:, 1] * _b3_uy(P)),
    ],
    init=None,
)


# ---------------------------------------------------------------------------
# Category C: space-time nonlinear PDEs, scaled coords (x, tau), tau = 2t - 1
# ---------------------------------------------------------------------------

def _t_of(P):
    return 0.5 * (P[:, 1] + 1.0)

# C1 (order 1 in space): inviscid Burgers u_t + u u_x = f, manufactured pulse.
#   u* = 0.8 exp(-6 (x + 0.4 - t)^2)  (u* >= 0, so x = -1 is inflow).
def _c1_u(P):
    x, t = P[:, 0], _t_of(P)
    return 0.8 * np.exp(-6 * (x + 0.4 - t) ** 2)

def _c1_ux(P):
    x, t = P[:, 0], _t_of(P)
    return -9.6 * (x + 0.4 - t) * np.exp(-6 * (x + 0.4 - t) ** 2)

def _c1_f(P):
    # u*_t = -u*_x for a right-traveling profile, so f = u_x (u - 1)
    return _c1_ux(P) * (_c1_u(P) - 1.0)

C1 = dict(
    key="pde2d_time_o1", category="pde2d_time", order=1,
    title="inviscid Burgers  $u_t + uu_x = f$  (IC + inflow)",
    lin_terms=[((0, 1), 2.0)],
    nl=dict(fields=[(0, 0), (1, 0)],
            res=lambda v, p: v[(0, 0)] * v[(1, 0)],
            jac=lambda v, p: {(0, 0): v[(1, 0)], (1, 0): v[(0, 0)]}),
    exact=_c1_u,
    forcing=_c1_f,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_c1_u),
        dict(where="left", terms=[((0, 0), 1.0)], value=_c1_u),
    ],
    init=None,
)

# C2 (order 2 in space): viscous Burgers u_t + u u_x = nu u_xx, nu = 0.25.
#   Exact Cole-Hopf traveling front u = c1 - c2 tanh(c2 (x - c1 t)/(2 nu)),
#   c1 = 0.4, c2 = 0.6 (front moves right at speed 0.4; f = 0).
_NU, _C1F, _C2F = 0.25, 0.4, 0.6
_KF = _C2F / (2 * _NU)

def _c2_u(P):
    x, t = P[:, 0], _t_of(P)
    return _C1F - _C2F * np.tanh(_KF * (x - _C1F * t))

C2 = dict(
    key="pde2d_time_o2", category="pde2d_time", order=2,
    title="viscous Burgers  $u_t + uu_x = 0.25\\,u_{xx}$  (Cole-Hopf front)",
    lin_terms=[((0, 1), 2.0), ((2, 0), -_NU)],
    nl=dict(fields=[(0, 0), (1, 0)],
            res=lambda v, p: v[(0, 0)] * v[(1, 0)],
            jac=lambda v, p: {(0, 0): v[(1, 0)], (1, 0): v[(0, 0)]}),
    exact=_c2_u,
    forcing=0.0,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_c2_u),
        dict(where="left", terms=[((0, 0), 1.0)], value=_c2_u),
        dict(where="right", terms=[((0, 0), 1.0)], value=_c2_u),
    ],
    init=None,
)

# C3 (order 3 in space): KdV u_t + 6 u u_x + u_xxx = 0, single soliton (f = 0).
#   u = (c/2) sech^2(sqrt(c)/2 (x - x0 - c t)), c = 1.5, x0 = -0.4.
_CS, _X0 = 1.5, -0.4
_KS = np.sqrt(_CS) / 2

def _c3_theta(P):
    x, t = P[:, 0], _t_of(P)
    return _KS * (x - _X0 - _CS * t)

def _c3_u(P):
    th = _c3_theta(P)
    return (_CS / 2) / np.cosh(th) ** 2

def _c3_ux(P):
    th = _c3_theta(P)
    return -_CS * _KS * np.tanh(th) / np.cosh(th) ** 2

# Same dispersive well-posedness as expF01's Airy case: the u_xxx term gives
# leftward group velocity, so TWO conditions go on the RIGHT edge, one on the left.
C3 = dict(
    key="pde2d_time_o3", category="pde2d_time", order=3,
    title="KdV soliton  $u_t + 6uu_x + u_{xxx} = 0$  (IC + 2 right / 1 left)",
    lin_terms=[((0, 1), 2.0), ((3, 0), 1.0)],
    nl=dict(fields=[(0, 0), (1, 0)],
            res=lambda v, p: 6.0 * v[(0, 0)] * v[(1, 0)],
            jac=lambda v, p: {(0, 0): 6.0 * v[(1, 0)], (1, 0): 6.0 * v[(0, 0)]}),
    exact=_c3_u,
    forcing=0.0,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_c3_u),
        dict(where="left", terms=[((0, 0), 1.0)], value=_c3_u),
        dict(where="right", terms=[((0, 0), 1.0)], value=_c3_u),
        dict(where="right", terms=[((1, 0), 1.0)], value=_c3_ux),
    ],
    init=None,
)


PROBLEMS = {p["key"]: p for p in [A1, A2, A3, B1, B2, B3, C1, C2, C3]}
CATEGORIES = ["ode1d", "pde2d_steady", "pde2d_time"]


# ---------------------------------------------------------------------------
# Finite-difference verification (residual(u*) == f, condition values)
# ---------------------------------------------------------------------------

def _fd1(f, x, order, h):
    if order == 0:
        return f(x)
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    if order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    if order == 3:
        return (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h**3)
    raise ValueError(order)


def _fd2(f, P, ax, ay, h):
    def fx(xs):
        Q = P.copy(); Q[:, 0] = xs
        if ay == 0:
            return f(Q)
        def fy(ys):
            R = Q.copy(); R[:, 1] = ys
            return f(R)
        return _fd1(fy, Q[:, 1], ay, h)
    if ax == 0:
        return fx(P[:, 0])
    return _fd1(fx, P[:, 0], ax, h)


def _h_for(order):
    return {0: 1e-5, 1: 1e-5, 2: 1e-4, 3: 1e-3}[order]


def _residual_fd(prob, pts):
    """residual(u*) at pts by finite differences (linear part + nonlinearity)."""
    if prob["category"] == "ode1d":
        out = np.zeros_like(pts)
        for o, c in prob["lin_terms"]:
            out += c * _fd1(prob["exact"], pts, o, _h_for(o))
        vals = {o: _fd1(prob["exact"], pts, o, _h_for(o)) for o in prob["nl"]["fields"]}
    else:
        out = np.zeros(len(pts))
        for (ax, ay), c in prob["lin_terms"]:
            out += c * _fd2(prob["exact"], pts, ax, ay, _h_for(ax + ay))
        vals = {idx: _fd2(prob["exact"], pts, idx[0], idx[1], _h_for(sum(idx)))
                for idx in prob["nl"]["fields"]}
    return out + prob["nl"]["res"](vals, pts)


def verify_all(tol=5e-4, verbose=True):
    rng = np.random.default_rng(7)
    failures = []
    for key, prob in PROBLEMS.items():
        if prob["category"] == "ode1d":
            pts = rng.uniform(-0.9, 0.9, 40)
        else:
            pts = rng.uniform(-0.6, 0.6, (40, 2))
        f = prob["forcing"]
        f_true = f(pts) if callable(f) else np.full(len(np.atleast_1d(pts[..., 0] if pts.ndim > 1 else pts)), float(f))
        err = np.max(np.abs(_residual_fd(prob, pts) - f_true))
        scale = max(1.0, np.max(np.abs(f_true)))
        if err / scale > tol:
            failures.append((key, "residual", err / scale))
        if prob["category"] == "ode1d":
            for o, fn in prob["exact_derivs"].items():
                e = np.max(np.abs(_fd1(prob["exact"], pts, o, _h_for(o)) - fn(pts)))
                if e / max(1.0, np.max(np.abs(fn(pts)))) > tol:
                    failures.append((key, f"deriv{o}", e))
        else:
            for blk in prob["bc_blocks"]:
                out = np.zeros(len(pts))
                for (ax, ay), c in blk["terms"]:
                    cc = c(pts) if callable(c) else c
                    out += cc * _fd2(prob["exact"], pts, ax, ay, _h_for(ax + ay))
                e = np.max(np.abs(out - blk["value"](pts)))
                if e / max(1.0, np.max(np.abs(out))) > tol:
                    failures.append((key, f"bc:{blk['where']}", e))
        if verbose:
            print(f"  verified {key}")
    if failures:
        raise AssertionError(f"problem verification failed: {failures}")
    if verbose:
        print("all problem definitions verified against finite differences")


if __name__ == "__main__":
    verify_all()
