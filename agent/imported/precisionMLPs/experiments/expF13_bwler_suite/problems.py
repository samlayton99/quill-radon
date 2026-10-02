"""expF13 problem suite: the BWLer PDE benchmarks (HazyResearch/bwler @ 7ff2e17).

Five benchmarks from "BWLer: Barycentric Weight Layer Elucidates a Precision-
Conditioning Tradeoff for PINNs" (arXiv 2506.23024), plus a manufactured-solution
control for the Poisson geometry. All problems are posed in SCALED coordinates
P = (xi, eta) in [-1,1]^2 with the physical map recorded per problem; derivative
scale factors are folded into the coefficients (as expF02 did with tau = 2t-1).

Physical setups (verbatim from bwler):
  convection  u_t + c u_x = 0,        (t,x) in [0,1]x[0,2pi], u0 = sin x, periodic. c in {40, 80}.
  reaction    u_t = rho u(1-u),       (t,x) in [0,1]x[0,2pi], u0 = exp(-(x-pi)^2/(2(pi/4)^2)),
              periodic in value (the Gaussian's derivative has a C^1 kink at the seam,
              so only VALUE periodicity is enforced -- matching bwler's penalty).
  wave        u_tt = c^2 u_xx,        (t,x) in [0,1]^2, c=2, u0 = sin(pi x)+0.5 sin(5 pi x),
              u_t(0,x)=0, Dirichlet u(t,0)=u(t,1)=0.
  burgers     u_t + u u_x = nu u_xx,  (t,x) in [0,1]x[-1,1], nu = 0.01/pi, u0 = -sin(pi x),
              Dirichlet 0. CLOSED FORM via Cole-Hopf (exact_colehopf below): a ratio of
              heat-kernel integrals under Gauss-Hermite quadrature (200 nodes; max
              deviation vs GH-600 on the 200x200 eval grid 2.9e-15, PDE residual ~1e-40
              at 40 dps, independent trapezoid agrees to 6.7e-16). The whole-line
              periodic solution satisfies u(+-1,t)=0 by odd symmetry, so it IS the
              Dirichlet solution. The legacy Chebfun pde15s reference (ref/*.mat) is
              only 6.0e-6 at its own nodes -- kept for comparison, never as an oracle.
  poisson_cg  Laplace on [-0.5,0.5]^2 minus four holes (centers (+-0.3,+-0.3), r=0.1),
              u=1 outer square, u=0 hole circles. COMSOL reference; nodal accuracy
              2.5e-5, measured against a converged method-of-fundamental-solutions
              solve (2026-08-31) -- the "~1e-2 ceiling" previously recorded here was
              wrong by 2.5 orders.
  poisson_man same geometry/BC structure, manufactured harmonic solution (true precision test).

Problem dict format follows expF02 (lin_terms / nl / bc_blocks), with one addition:
bc_blocks may carry where="periodic_x", meaning rows are built as
row(xi=-1, eta) - row(xi=+1, eta) for each listed derivative term, with value 0.

verify_all() checks every closed-form solution against its own residual operator by
finite differences, checks all IC/BC values, and sanity-checks both reference files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PI = np.pi
REF_DIR = Path(__file__).resolve().parent / "ref"

NU_BURGERS = 0.01 / PI


# ---------------------------------------------------------------------------
# Cole-Hopf closed form for Burgers (verified 2026-08-31; see expF17 SPEC 13.2/16.8):
#   u = -int sin(pi y) G dy / int G dy,  G = exp(-cos(pi y)/(2 pi nu)) * heat kernel,
# evaluated with Gauss-Hermite in zeta where y = x - 2 sqrt(nu t) zeta.
# ---------------------------------------------------------------------------

def exact_colehopf(x, t, nu=NU_BURGERS, n=200):
    """Exact Burgers solution for u0 = -sin(pi x), u(+-1, t) = 0, on PHYSICAL (x, t).

    n=200 Gauss-Hermite nodes: max deviation vs GH-600 on the 200x200 eval grid
    is < 3e-15 (n=75 is NOT enough at large t: 1.0e-12 near x=0, t=1)."""
    from scipy.special import roots_hermite
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    out = np.empty_like(x)
    z0 = t <= 0
    out[z0] = -np.sin(PI * x[z0])
    m = ~z0
    if m.any():
        zt, w = roots_hermite(n)
        with np.errstate(divide="ignore"):
            logw = np.log(w)[None, :]
        y = x[m][:, None] - 2.0 * np.sqrt(nu * t[m])[:, None] * zt[None, :]
        a = -np.cos(PI * y) / (2.0 * PI * nu) + logw
        a -= a.max(axis=1, keepdims=True)
        F = np.exp(a)
        out[m] = -np.sum(np.sin(PI * y) * F, axis=1) / np.sum(F, axis=1)
    return out


def _burgers_exact_scaled(nu):
    def u(P):
        return exact_colehopf(P[:, 0], 0.5 * (P[:, 1] + 1.0), nu=nu)
    return u


# ---------------------------------------------------------------------------
# convection: 2 u_eta + (c/pi) u_xi = 0    (x = pi(xi+1), t = (eta+1)/2)
# ---------------------------------------------------------------------------

def _conv_exact(c):
    def u(P):
        x, t = PI * (P[:, 0] + 1.0), 0.5 * (P[:, 1] + 1.0)
        return np.sin(x - c * t)
    return u


def make_convection(c):
    exact = _conv_exact(c)
    return dict(
        key=f"convection_c{int(c)}", category="time", order=1,
        title=f"convection  $u_t + {int(c)}\\,u_x = 0$  (IC + periodic)",
        lin_terms=[((0, 1), 2.0), ((1, 0), c / PI)],
        nl=dict(fields=[], res=lambda v, p: 0.0, jac=lambda v, p: {}),
        exact=exact, forcing=0.0,
        bc_blocks=[
            dict(where="ic", terms=[((0, 0), 1.0)], value=exact),
            dict(where="periodic_x", terms=[((0, 0), 1.0)], value=0.0),
            dict(where="periodic_x", terms=[((1, 0), 1.0)], value=0.0),
        ],
        bwler_ref=2.04e-13 if c == 40 else 1.10e-12,
    )


# ---------------------------------------------------------------------------
# reaction: 2 u_eta - rho u + rho u^2 = 0   (x = pi(xi+1), t = (eta+1)/2)
#   u0(xi) = exp(-8 xi^2) in scaled coords.
# ---------------------------------------------------------------------------

_RHO = 5.0


def _react_u0(xi):
    return np.exp(-8.0 * xi ** 2)


def _react_exact(P):
    a = _react_u0(P[:, 0])
    e = np.exp(_RHO * 0.5 * (P[:, 1] + 1.0))
    return a * e / (a * e + 1.0 - a)


REACTION = dict(
    key="reaction", category="time", order=1,
    title="reaction  $u_t = 5\\,u(1-u)$  (IC + periodic value)",
    lin_terms=[((0, 1), 2.0), ((0, 0), -_RHO)],
    nl=dict(fields=[(0, 0)],
            res=lambda v, p: _RHO * v[(0, 0)] ** 2,
            jac=lambda v, p: {(0, 0): 2.0 * _RHO * v[(0, 0)]}),
    exact=_react_exact, forcing=0.0,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_react_exact),
        dict(where="periodic_x", terms=[((0, 0), 1.0)], value=0.0),
    ],
    bwler_ref=6.94e-11,
)


# ---------------------------------------------------------------------------
# wave: 4 u_etaeta - 16 u_xixi = 0   (x = (xi+1)/2, t = (eta+1)/2, c = 2)
# ---------------------------------------------------------------------------

_BETA = 5.0


def _wave_exact(P):
    x, t = 0.5 * (P[:, 0] + 1.0), 0.5 * (P[:, 1] + 1.0)
    return (np.sin(PI * x) * np.cos(2 * PI * t)
            + 0.5 * np.sin(_BETA * PI * x) * np.cos(2 * _BETA * PI * t))


WAVE = dict(
    key="wave", category="time", order=2,
    title="wave  $u_{tt} = 4u_{xx}$, $\\beta=5$  (2 ICs + Dirichlet)",
    lin_terms=[((0, 2), 4.0), ((2, 0), -16.0)],
    nl=dict(fields=[], res=lambda v, p: 0.0, jac=lambda v, p: {}),
    exact=_wave_exact, forcing=0.0,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_wave_exact),
        dict(where="ic", terms=[((0, 1), 1.0)], value=0.0),   # u_t = 2 u_eta = 0
        dict(where="left", terms=[((0, 0), 1.0)], value=0.0),
        dict(where="right", terms=[((0, 0), 1.0)], value=0.0),
    ],
    bwler_ref=1.26e-11,
)


# ---------------------------------------------------------------------------
# burgers: 2 u_eta + u u_xi - nu u_xixi = 0   (x = xi, t = (eta+1)/2)
#   Parameterized by nu for the vanishing-viscosity continuation ladder.
# ---------------------------------------------------------------------------

def _burgers_ic(P):
    return -np.sin(PI * P[:, 0])


def make_burgers(nu):
    return dict(
        key="burgers", category="time", order=2,
        title=f"Burgers  $u_t + uu_x = \\nu u_{{xx}}$, $\\nu=0.01/\\pi$",
        lin_terms=[((0, 1), 2.0), ((2, 0), -nu)],
        nl=dict(fields=[(0, 0), (1, 0)],
                res=lambda v, p: v[(0, 0)] * v[(1, 0)],
                jac=lambda v, p: {(0, 0): v[(1, 0)], (1, 0): v[(0, 0)]}),
        exact=_burgers_exact_scaled(nu), forcing=0.0,
        bc_blocks=[
            dict(where="ic", terms=[((0, 0), 1.0)], value=_burgers_ic),
            dict(where="left", terms=[((0, 0), 1.0)], value=0.0),
            dict(where="right", terms=[((0, 0), 1.0)], value=0.0),
        ],
        bwler_ref=4.63e-3,
        nu=nu,
    )


def load_burgers_reference():
    """(u_ref [n_t, n_x], t [n_t], x [n_x]) from the Chebfun pde15s .mat."""
    import scipy.io
    mat = scipy.io.loadmat(REF_DIR / "burgers_1d_dirichlet.mat")
    return (np.asarray(mat["usol"], dtype=np.float64),
            np.asarray(mat["t"], dtype=np.float64).ravel(),
            np.asarray(mat["x"], dtype=np.float64).ravel())


# ---------------------------------------------------------------------------
# poisson_cg: u_xixi + u_etaeta = 0 on [-1,1]^2 minus 4 holes
#   (scaled: physical (x,y) = P/2; holes at (+-0.6,+-0.6), r = 0.2 scaled)
# ---------------------------------------------------------------------------

HOLE_CENTERS = np.array([(0.6, 0.6), (0.6, -0.6), (-0.6, 0.6), (-0.6, -0.6)])
HOLE_RADIUS = 0.2


def in_poisson_domain(P):
    ok = np.ones(len(P), dtype=bool)
    for cx, cy in HOLE_CENTERS:
        ok &= np.hypot(P[:, 0] - cx, P[:, 1] - cy) > HOLE_RADIUS
    return ok


POISSON_CG = dict(
    key="poisson_cg", category="steady", order=2,
    title="Poisson-CG  $\\Delta u = 0$, square minus 4 holes (COMSOL ref)",
    lin_terms=[((2, 0), 1.0), ((0, 2), 1.0)],
    nl=dict(fields=[], res=lambda v, p: 0.0, jac=lambda v, p: {}),
    exact=None, forcing=0.0,
    bc_blocks=[
        dict(where="square", terms=[((0, 0), 1.0)], value=1.0),
        dict(where="holes", terms=[((0, 0), 1.0)], value=0.0),
    ],
    bwler_ref=1.08e-2,
)


def load_poisson_reference():
    """(points_scaled [N,2], values [N]). COMSOL nodes are float32-quality."""
    data = np.loadtxt(REF_DIR / "poisson2d_cg_data.dat", comments="%")
    return 2.0 * data[:, :2], data[:, 2]


# manufactured control: harmonic away from the hole centers, Dirichlet from u*.
def _poisson_man_exact(P):
    out = 1.0 + 0.3 * (P[:, 0] ** 2 - P[:, 1] ** 2)
    for cx, cy in HOLE_CENTERS:
        out += 0.25 * np.log(np.hypot(P[:, 0] - cx, P[:, 1] - cy))
    return out


POISSON_MAN = dict(
    key="poisson_man", category="steady", order=2,
    title="Poisson control  $\\Delta u = 0$, same geometry, manufactured $u^*$",
    lin_terms=[((2, 0), 1.0), ((0, 2), 1.0)],
    nl=dict(fields=[], res=lambda v, p: 0.0, jac=lambda v, p: {}),
    exact=_poisson_man_exact, forcing=0.0,
    bc_blocks=[
        dict(where="square", terms=[((0, 0), 1.0)], value=_poisson_man_exact),
        dict(where="holes", terms=[((0, 0), 1.0)], value=_poisson_man_exact),
    ],
    bwler_ref=None,
)


PROBLEMS = {p["key"]: p for p in [
    make_convection(40.0), make_convection(80.0), REACTION, WAVE,
    make_burgers(NU_BURGERS), POISSON_CG, POISSON_MAN,
]}


# ---------------------------------------------------------------------------
# finite-difference verification
# ---------------------------------------------------------------------------

def _fd1(f, x, order, h):
    if order == 0:
        return f(x)
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    if order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2
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


_H = {0: 1e-5, 1: 1e-5, 2: 1e-4}


def _residual_fd(prob, P):
    out = np.zeros(len(P))
    for (ax, ay), c in prob["lin_terms"]:
        out += c * _fd2(prob["exact"], P, ax, ay, _H[max(ax, ay)])
    vals = {idx: _fd2(prob["exact"], P, idx[0], idx[1], _H[max(idx)])
            for idx in prob["nl"]["fields"]}
    return out + prob["nl"]["res"](vals, P)


def verify_all(tol=5e-4, verbose=True):
    rng = np.random.default_rng(7)
    failures = []

    # closed-form problems: residual(u*) == f, condition values
    for key in ["convection_c40", "convection_c80", "reaction", "wave", "poisson_man"]:
        prob = PROBLEMS[key]
        P = rng.uniform(-0.85, 0.85, (60, 2))
        if key == "poisson_man":
            P = P[in_poisson_domain(P)]
        r = _residual_fd(prob, P)
        f = prob["forcing"]
        fv = f(P) if callable(f) else np.full(len(P), float(f))
        scale = max(1.0, np.max(np.abs(_fd2(prob["exact"], P, *prob["lin_terms"][0][0],
                                            _H[max(prob["lin_terms"][0][0])])))
                    * abs(prob["lin_terms"][0][1]))
        err = np.max(np.abs(r - fv)) / scale
        if err > tol:
            failures.append((key, "residual", err))

        # condition rows: evaluate each block's terms on its points against value
        for blk in prob["bc_blocks"]:
            eta = np.linspace(-0.95, 0.95, 33)
            if blk["where"] == "ic":
                Pb = np.stack([eta, np.full_like(eta, -1.0)], axis=1)
            elif blk["where"] == "left":
                Pb = np.stack([np.full_like(eta, -1.0), eta], axis=1)
            elif blk["where"] == "right":
                Pb = np.stack([np.full_like(eta, 1.0), eta], axis=1)
            elif blk["where"] == "periodic_x":
                PL = np.stack([np.full_like(eta, -1.0), eta], axis=1)
                PR = np.stack([np.full_like(eta, 1.0), eta], axis=1)
                out = np.zeros(len(eta))
                for (ax, ay), c in blk["terms"]:
                    out += c * (_fd2(prob["exact"], PL, ax, ay, _H[max(ax, ay)])
                                - _fd2(prob["exact"], PR, ax, ay, _H[max(ax, ay)]))
                if np.max(np.abs(out)) > tol:
                    failures.append((key, "periodic", float(np.max(np.abs(out)))))
                continue
            elif blk["where"] in ("square", "holes"):
                continue  # value IS the exact fn for poisson_man; nothing to cross-check
            else:
                raise ValueError(blk["where"])
            out = np.zeros(len(Pb))
            for (ax, ay), c in blk["terms"]:
                out += c * _fd2(prob["exact"], Pb, ax, ay, _H[max(ax, ay)])
            val = blk["value"]
            vv = val(Pb) if callable(val) else np.full(len(Pb), float(val))
            e = np.max(np.abs(out - vv))
            if e > tol:
                failures.append((key, f"bc:{blk['where']}", float(e)))
        if verbose:
            print(f"  verified {key}")

    # burgers Cole-Hopf oracle: IC, BC, and agreement with the legacy .mat
    xs = np.linspace(-1.0, 1.0, 401)
    e_ic_ch = np.max(np.abs(exact_colehopf(xs, np.zeros_like(xs)) + np.sin(PI * xs)))
    tb = np.linspace(0.01, 1.0, 25)
    e_bc_ch = max(np.max(np.abs(exact_colehopf(np.full_like(tb, -1.0), tb))),
                  np.max(np.abs(exact_colehopf(np.full_like(tb, 1.0), tb))))
    if e_ic_ch > 1e-14 or e_bc_ch > 1e-14:
        failures.append(("burgers", "colehopf_ic_bc", (float(e_ic_ch), float(e_bc_ch))))
    elif verbose:
        print(f"  verified burgers Cole-Hopf oracle (IC {e_ic_ch:.1e}, BC {e_bc_ch:.1e})")

    # burgers reference: IC and BCs of the Chebfun solution
    u, t, x = load_burgers_reference()
    assert u.shape == (len(t), len(x)), "burgers ref orientation mismatch"
    e_ic = np.max(np.abs(u[0] + np.sin(PI * x)))
    e_bc = max(np.max(np.abs(u[:, 0])), np.max(np.abs(u[:, -1])))
    if e_ic > 1e-12 or e_bc > 1e-12:
        failures.append(("burgers", "reference", (e_ic, e_bc)))
    else:
        sub = slice(0, len(t), 4)
        XX, TT = np.meshgrid(x, t[sub], indexing="xy")
        d_mat = np.max(np.abs(u[sub] - exact_colehopf(XX.ravel(), TT.ravel())
                              .reshape(len(t[sub]), len(x))))
        if not 1e-7 < d_mat < 5e-5:  # the .mat carries ~6e-6 of its own error
            failures.append(("burgers", "mat_vs_colehopf", float(d_mat)))
        elif verbose:
            print(f"  verified burgers reference (IC {e_ic:.1e}, BC {e_bc:.1e}, "
                  f"grid {u.shape}; .mat vs Cole-Hopf {d_mat:.1e} -- the .mat's own error)")

    # poisson reference: node classification and BC values
    Pn, vn = load_poisson_reference()
    on_sq = np.max(np.abs(Pn), axis=1) > 1.0 - 1e-8
    d = np.min([np.hypot(Pn[:, 0] - cx, Pn[:, 1] - cy) for cx, cy in HOLE_CENTERS],
               axis=0)
    on_hole = np.abs(d - HOLE_RADIUS) < 1e-5
    inside_hole = d < HOLE_RADIUS - 1e-5
    if (inside_hole.sum() > 0
            or np.max(np.abs(vn[on_sq] - 1.0)) > 1e-5
            or np.max(np.abs(vn[on_hole])) > 1e-5):
        failures.append(("poisson_cg", "reference",
                         (int(inside_hole.sum()), float(np.max(np.abs(vn[on_sq] - 1.0))))))
    elif verbose:
        print(f"  verified poisson reference ({len(Pn)} nodes, {int(on_sq.sum())} on "
              f"square, {int(on_hole.sum())} on holes, none inside)")

    if failures:
        raise AssertionError(f"problem verification failed: {failures}")
    if verbose:
        print("all problem definitions verified")


if __name__ == "__main__":
    verify_all()
