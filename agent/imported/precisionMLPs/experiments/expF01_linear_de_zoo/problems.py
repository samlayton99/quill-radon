"""expF01 problem suite: 9 linear differential problems (3 categories x orders 1-3).

Categories:
  ode1d        -- u(x) on [-1,1]
  pde2d_steady -- u(x,y) on the unit disk
  pde2d_time   -- u(x,t) on [-1,1] x [0,1], posed in scaled coords (x, tau) with
                  tau = 2t - 1, so d/dt = 2 d/dtau; operators are written in (x, tau).

Each problem is a dict:
  key, category, order, title
  terms        -- the linear operator L = sum_i s_i D_i:
                  1D [(deriv_order, coeff)], 2D [((ax, ay), coeff)];
                  coeff is a float or callable(points) -> [n] (variable coefficient)
  exact        -- exact solution: 1D exact(x), 2D exact(P[n,2])
  forcing      -- RHS f (same signature; 0.0 for the classical homogeneous cases)
  conditions   -- 1D: [(x0, deriv_order)], values pulled from exact_derivs
  exact_derivs -- 1D: {order: callable} for condition values + verification
  bc_blocks    -- 2D: [dict(where, terms, value)] with where in
                  {"circle", "inflow"} (steady) or {"ic", "left", "right"} (time)

All hand-coded derivatives / forcings are verified against finite differences in
verify_all(); run.py refuses to start if any check fails.
"""

from __future__ import annotations

import numpy as np

PI = np.pi


# ---------------------------------------------------------------------------
# Category A: 1D ODEs on [-1, 1]
# ---------------------------------------------------------------------------

# A1 (order 1, IVP): relaxation with a seasonally varying rate.
#   u' + (2 + sin(pi x)) u = f,   u(-1) given.   u* = exp(-x) cos(2 pi x)
def _a1_u(x):
    return np.exp(-x) * np.cos(2 * PI * x)

def _a1_u1(x):
    return -np.exp(-x) * (np.cos(2 * PI * x) + 2 * PI * np.sin(2 * PI * x))

A1 = dict(
    key="ode1d_o1", category="ode1d", order=1,
    title="u' + (2+sin$\\pi x$)u = f  (IVP)",
    terms=[(1, 1.0), (0, lambda x: 2.0 + np.sin(PI * x))],
    exact=_a1_u,
    forcing=lambda x: _a1_u1(x) + (2.0 + np.sin(PI * x)) * _a1_u(x),
    conditions=[(-1.0, 0)],
    exact_derivs={0: _a1_u, 1: _a1_u1},
)

# A2 (order 2, mixed BVP): damped driven oscillator, natural frequency 10.
#   u'' + 0.4 u' + 100 u = f,  u(-1) Dirichlet, u'(1) Neumann.
#   u* = exp(-x^2) sin(6x)
def _a2_u(x):
    return np.exp(-x**2) * np.sin(6 * x)

def _a2_u1(x):
    g = np.exp(-x**2)
    return g * (-2 * x * np.sin(6 * x) + 6 * np.cos(6 * x))

def _a2_u2(x):
    g = np.exp(-x**2)
    return g * ((4 * x**2 - 38) * np.sin(6 * x) - 24 * x * np.cos(6 * x))

A2 = dict(
    key="ode1d_o2", category="ode1d", order=2,
    title="u'' + 0.4u' + 100u = f  (Dirichlet+Neumann)",
    terms=[(2, 1.0), (1, 0.4), (0, 100.0)],
    exact=_a2_u,
    forcing=lambda x: _a2_u2(x) + 0.4 * _a2_u1(x) + 100.0 * _a2_u(x),
    conditions=[(-1.0, 0), (1.0, 1)],
    exact_derivs={0: _a2_u, 1: _a2_u1, 2: _a2_u2},
)

# A3 (order 3, two-point): steady linear dispersion.
#   u''' + 4 u' = f,  u(-1), u'(-1), u(1).   u* = sin(3x)(1 + 0.5x)
def _a3_u(x):
    return np.sin(3 * x) * (1 + 0.5 * x)

def _a3_u1(x):
    return 3 * np.cos(3 * x) * (1 + 0.5 * x) + 0.5 * np.sin(3 * x)

def _a3_u2(x):
    return -9 * np.sin(3 * x) * (1 + 0.5 * x) + 3 * np.cos(3 * x)

def _a3_u3(x):
    return -27 * np.cos(3 * x) * (1 + 0.5 * x) - 13.5 * np.sin(3 * x)

A3 = dict(
    key="ode1d_o3", category="ode1d", order=3,
    title="u''' + 4u' = f  (2 left + 1 right conds)",
    terms=[(3, 1.0), (1, 4.0)],
    exact=_a3_u,
    forcing=lambda x: _a3_u3(x) + 4.0 * _a3_u1(x),
    conditions=[(-1.0, 0), (-1.0, 1), (1.0, 0)],
    exact_derivs={0: _a3_u, 1: _a3_u1, 2: _a3_u2, 3: _a3_u3},
)


# ---------------------------------------------------------------------------
# Category B: 2D steady PDEs on the unit disk
# ---------------------------------------------------------------------------

_B = np.array([2.0, 1.0]) / np.sqrt(5.0)  # advection direction

# B1 (order 1, inflow BVP): steady advection-reaction (transport with decay).
#   b . grad u + u = f,  u given on the inflow part of the circle (b.n < 0).
#   u* = sin(2x - y) + 0.3 (x^2 + y^2)
def _b1_u(P):
    x, y = P[:, 0], P[:, 1]
    return np.sin(2 * x - y) + 0.3 * (x**2 + y**2)

def _b1_ux(P):
    x, y = P[:, 0], P[:, 1]
    return 2 * np.cos(2 * x - y) + 0.6 * x

def _b1_uy(P):
    x, y = P[:, 0], P[:, 1]
    return -np.cos(2 * x - y) + 0.6 * y

B1 = dict(
    key="pde2d_steady_o1", category="pde2d_steady", order=1,
    title="$b\\cdot\\nabla u + u = f$  (inflow BC)",
    terms=[((1, 0), _B[0]), ((0, 1), _B[1]), ((0, 0), 1.0)],
    exact=_b1_u,
    forcing=lambda P: _B[0] * _b1_ux(P) + _B[1] * _b1_uy(P) + _b1_u(P),
    bc_blocks=[dict(where="inflow", terms=[((0, 0), 1.0)], value=_b1_u)],
)

# B2 (order 2, Dirichlet BVP): screened Poisson (steady heat with cooling).
#   -Lap u + 4 u = f,  u given on the circle.
#   u* = sin(2x) exp(-y^2) + 0.5 cos(3y)
def _b2_u(P):
    x, y = P[:, 0], P[:, 1]
    return np.sin(2 * x) * np.exp(-y**2) + 0.5 * np.cos(3 * y)

def _b2_lap(P):
    x, y = P[:, 0], P[:, 1]
    g = np.exp(-y**2)
    uxx = -4 * np.sin(2 * x) * g
    uyy = np.sin(2 * x) * (4 * y**2 - 2) * g - 4.5 * np.cos(3 * y)
    return uxx + uyy

B2 = dict(
    key="pde2d_steady_o2", category="pde2d_steady", order=2,
    title="$-\\Delta u + 4u = f$  (Dirichlet)",
    terms=[((2, 0), -1.0), ((0, 2), -1.0), ((0, 0), 4.0)],
    exact=_b2_u,
    forcing=lambda P: -_b2_lap(P) + 4.0 * _b2_u(P),
    bc_blocks=[dict(where="circle", terms=[((0, 0), 1.0)], value=_b2_u)],
)

# B3 (order 3, Cauchy-data BVP): third-order stress test (not a classical BVP class;
#   pinned with full Cauchy data u and du/dn on the circle).
#   u_xxx + u_yyy + u = f.   u* = sin(2x) cos(2y)
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
    return uxxx + uyyy + _b3_u(P)

B3 = dict(
    key="pde2d_steady_o3", category="pde2d_steady", order=3,
    title="$u_{xxx} + u_{yyy} + u = f$  (Cauchy data)",
    terms=[((3, 0), 1.0), ((0, 3), 1.0), ((0, 0), 1.0)],
    exact=_b3_u,
    forcing=_b3_f,
    bc_blocks=[
        dict(where="circle", terms=[((0, 0), 1.0)], value=_b3_u),
        # du/dn on the unit circle: n = (x, y)
        dict(where="circle",
             terms=[((1, 0), lambda P: P[:, 0]), ((0, 1), lambda P: P[:, 1])],
             value=lambda P: P[:, 0] * _b3_ux(P) + P[:, 1] * _b3_uy(P)),
    ],
)


# ---------------------------------------------------------------------------
# Category C: space-time PDEs on [-1,1] x [0,1], scaled coords (x, tau)
# tau = 2t - 1  =>  d/dt = 2 d/dtau. Points P have columns (x, tau).
# ---------------------------------------------------------------------------

def _t_of(P):
    return 0.5 * (P[:, 1] + 1.0)

# C1 (order 1 in space, IVP): advection u_t + u_x = 0 (pulse transport).
#   Exact: traveling Gaussian u = exp(-12 (x + 0.5 - t)^2).
def _c1_u(P):
    x, t = P[:, 0], _t_of(P)
    return np.exp(-12 * (x + 0.5 - t) ** 2)

C1 = dict(
    key="pde2d_time_o1", category="pde2d_time", order=1,
    title="$u_t + u_x = 0$  (IC + inflow)",
    terms=[((0, 1), 2.0), ((1, 0), 1.0)],
    exact=_c1_u,
    forcing=0.0,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_c1_u),
        dict(where="left", terms=[((0, 0), 1.0)], value=_c1_u),
    ],
)

# C2 (order 2 in space, IBVP): heat equation u_t = nu u_xx, nu = 0.15.
#   Exact: two decaying sine modes (Dirichlet-0 compatible).
_NU = 0.15

def _c2_u(P):
    x, t = P[:, 0], _t_of(P)
    return (np.sin(PI * x) * np.exp(-_NU * PI**2 * t)
            + 0.4 * np.sin(2 * PI * x) * np.exp(-4 * _NU * PI**2 * t))

C2 = dict(
    key="pde2d_time_o2", category="pde2d_time", order=2,
    title="$u_t = 0.15\\,u_{xx}$  (IC + Dirichlet)",
    terms=[((0, 1), 2.0), ((2, 0), -_NU)],
    exact=_c2_u,
    forcing=0.0,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_c2_u),
        dict(where="left", terms=[((0, 0), 1.0)], value=_c2_u),
        dict(where="right", terms=[((0, 0), 1.0)], value=_c2_u),
    ],
)

# C3 (order 3 in space): Airy / linearized KdV u_t + u_xxx = 0 (dispersion).
#   Exact: two left-moving modes with different phase speeds k^2 (omega = k^3):
#   u = sin(1.5x + 3.375t) + 0.5 sin(2.2x + 10.648t).
_K1, _K2 = 1.5, 2.2

def _c3_u(P):
    x, t = P[:, 0], _t_of(P)
    return np.sin(_K1 * x + _K1**3 * t) + 0.5 * np.sin(_K2 * x + _K2**3 * t)

def _c3_ux(P):
    x, t = P[:, 0], _t_of(P)
    return (_K1 * np.cos(_K1 * x + _K1**3 * t)
            + 0.5 * _K2 * np.cos(_K2 * x + _K2**3 * t))

# BC placement is NOT free here. With u ~ exp(i(kx - wt)), u_t + u_xxx = 0 gives
# w = -k^3, so the group velocity dw/dk = -3k^2 < 0: energy travels LEFT. The
# well-posed split therefore puts TWO conditions at the RIGHT (inflow) edge and
# ONE at the left. Using 2-left/1-right instead costs ~5 orders of accuracy
# (3e-8 vs 2e-13 at W=2304) -- the operator still holds pointwise, but the
# discrete problem admits a near-null spurious mode. See expF01 writeup.
C3 = dict(
    key="pde2d_time_o3", category="pde2d_time", order=3,
    title="$u_t + u_{xxx} = 0$  (IC + 2 right / 1 left)",
    terms=[((0, 1), 2.0), ((3, 0), 1.0)],
    exact=_c3_u,
    forcing=0.0,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_c3_u),
        dict(where="left", terms=[((0, 0), 1.0)], value=_c3_u),
        dict(where="right", terms=[((0, 0), 1.0)], value=_c3_u),
        dict(where="right", terms=[((1, 0), 1.0)], value=_c3_ux),
    ],
)


PROBLEMS = {p["key"]: p for p in [A1, A2, A3, B1, B2, B3, C1, C2, C3]}
CATEGORIES = ["ode1d", "pde2d_steady", "pde2d_time"]


# ---------------------------------------------------------------------------
# Finite-difference verification of every hand-coded derivative / forcing
# ---------------------------------------------------------------------------

def _fd1(f, x, order, h):
    """Central-difference derivative of given order for scalar-arg f."""
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
    """FD partial d^ax_x d^ay_y for f(P[n,2]), by nesting 1D stencils."""
    def fx(xs):  # derivative in x at fixed rows
        Q = P.copy(); Q[:, 0] = xs
        return _fd1(lambda ys: _eval_shift(f, Q, 1, ys), Q[:, 1], ay, h) if ay else f(Q)
    return _fd1(fx, P[:, 0], ax, h) if ax else fx(P[:, 0])


def _eval_shift(f, P, col, vals):
    Q = P.copy(); Q[:, col] = vals
    return f(Q)


def _h_for(order):
    """FD step by derivative order: balance h^2 truncation vs eps/h^order roundoff."""
    return {0: 1e-5, 1: 1e-5, 2: 1e-4, 3: 1e-3}[order]


def _apply_L_fd(prob, pts, h=None):
    """Apply the operator to the exact solution by finite differences."""
    if prob["category"] == "ode1d":
        out = np.zeros_like(pts)
        for order, coeff in prob["terms"]:
            c = coeff(pts) if callable(coeff) else coeff
            out += c * _fd1(prob["exact"], pts, order, h or _h_for(order))
        return out
    out = np.zeros(len(pts))
    for (ax, ay), coeff in prob["terms"]:
        c = coeff(pts) if callable(coeff) else coeff
        out += c * _fd2(prob["exact"], pts, ax, ay, h or _h_for(ax + ay))
    return out


def verify_all(tol=2e-4, verbose=True):
    """Check L[u*] == f and all condition values against finite differences."""
    rng = np.random.default_rng(7)
    failures = []
    for key, prob in PROBLEMS.items():
        if prob["category"] == "ode1d":
            x = rng.uniform(-0.9, 0.9, 40)
            f_true = prob["forcing"](x) if callable(prob["forcing"]) else np.full_like(x, prob["forcing"])
            err = np.max(np.abs(_apply_L_fd(prob, x) - f_true))
            scale = max(1.0, np.max(np.abs(f_true)))
            if err / scale > tol:
                failures.append((key, "forcing", err / scale))
            for o, fn in prob["exact_derivs"].items():
                e = np.max(np.abs(_fd1(prob["exact"], x, o, _h_for(o)) - fn(x)))
                if e / max(1.0, np.max(np.abs(fn(x)))) > tol:
                    failures.append((key, f"deriv{o}", e))
        else:
            P = rng.uniform(-0.6, 0.6, (40, 2))
            f_true = prob["forcing"](P) if callable(prob["forcing"]) else np.full(len(P), prob["forcing"])
            err = np.max(np.abs(_apply_L_fd(prob, P) - f_true))
            scale = max(1.0, np.max(np.abs(f_true)))
            if err / scale > tol:
                failures.append((key, "forcing", err / scale))
            for blk in prob["bc_blocks"]:
                out = np.zeros(len(P))
                for (ax, ay), coeff in blk["terms"]:
                    c = coeff(P) if callable(coeff) else coeff
                    out += c * _fd2(prob["exact"], P, ax, ay, _h_for(ax + ay))
                e = np.max(np.abs(out - blk["value"](P)))
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
