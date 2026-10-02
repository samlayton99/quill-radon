"""expF03 held-out problem set for the model-selection test (part 2).

These three problems are NEVER used while developing the selection rule; the
frozen selector runs on them exactly once. Formats match expF01 (linear) /
expF02 (nonlinear) problem dicts so common.py solves them unchanged.

  heldout_wave     linear space-time: damped wave u_tt + 0.3 u_t = u_xx
                   (order 2 in TIME -- a condition type no zoo problem has:
                   two IC rows, u and u_t, on the t=0 edge)
  heldout_convdiff linear steady disk: variable-coefficient convection-diffusion
                   -div(kappa grad u) + b . grad u = f
  heldout_fisher   nonlinear space-time: Fisher-KPP u_t = u_xx + u(1-u), classical
                   Ablowitz-Zeppetella traveling front (f = 0)

verify_all() FD-checks all hand-coded forcings/derivatives via the expF01/expF02
verification machinery before any solve.
"""

from __future__ import annotations

import numpy as np

from common import load_problem_suite

PI = np.pi


def _t_of(P):
    return 0.5 * (P[:, 1] + 1.0)


# ---------------------------------------------------------------------------
# H1: damped wave (linear, space-time, order 2 in time)
#   u_tt + 0.3 u_t - u_xx = f  on [-1,1] x [0,1], scaled coords (x, tau).
#   u* = exp(-0.3 t) sin(pi x) cos(2 t) + 0.3 (1 - x^2) t
# ---------------------------------------------------------------------------

def _h1_u(P):
    x, t = P[:, 0], _t_of(P)
    return np.exp(-0.3 * t) * np.sin(PI * x) * np.cos(2 * t) + 0.3 * (1 - x**2) * t


def _h1_ut(P):
    x, t = P[:, 0], _t_of(P)
    e = np.exp(-0.3 * t)
    return (e * np.sin(PI * x) * (-0.3 * np.cos(2 * t) - 2 * np.sin(2 * t))
            + 0.3 * (1 - x**2))


def _h1_utt(P):
    x, t = P[:, 0], _t_of(P)
    e = np.exp(-0.3 * t)
    # d/dt of (-0.3 cos - 2 sin) e = e[(0.09 - 4) cos ... ]:
    return e * np.sin(PI * x) * ((0.09 - 4.0) * np.cos(2 * t) + 1.2 * np.sin(2 * t))


def _h1_uxx(P):
    x, t = P[:, 0], _t_of(P)
    return -PI**2 * np.exp(-0.3 * t) * np.sin(PI * x) * np.cos(2 * t) - 0.6 * t


def _h1_f(P):
    return _h1_utt(P) + 0.3 * _h1_ut(P) - _h1_uxx(P)


# in (x, tau): u_t = 2 u_tau, u_tt = 4 u_tautau
H1 = dict(
    key="heldout_wave", category="pde2d_time", order=2,
    title="damped wave  $u_{tt} + 0.3u_t = u_{xx}$  (2 ICs + Dirichlet)",
    terms=[((0, 2), 4.0), ((0, 1), 0.6), ((2, 0), -1.0)],
    exact=_h1_u,
    forcing=_h1_f,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_h1_u),
        dict(where="ic", terms=[((0, 1), 2.0)], value=_h1_ut),   # u_t = 2 u_tau
        dict(where="left", terms=[((0, 0), 1.0)], value=_h1_u),
        dict(where="right", terms=[((0, 0), 1.0)], value=_h1_u),
    ],
)


# ---------------------------------------------------------------------------
# H2: variable-coefficient convection-diffusion (linear, steady disk)
#   -div(kappa grad u) + b . grad u = f,  kappa = 1 + 0.3x + 0.2y^2, b = (1.5, -0.8)
#   expanded: -kappa (u_xx + u_yy) + (b1 - kappa_x) u_x + (b2 - kappa_y) u_y = f
#   u* = exp(0.3x) sin(2y) + 0.4 cos(x + 2y)
# ---------------------------------------------------------------------------

def _kap(P):
    x, y = P[:, 0], P[:, 1]
    return 1.0 + 0.3 * x + 0.2 * y**2


def _h2_u(P):
    x, y = P[:, 0], P[:, 1]
    return np.exp(0.3 * x) * np.sin(2 * y) + 0.4 * np.cos(x + 2 * y)


def _h2_ux(P):
    x, y = P[:, 0], P[:, 1]
    return 0.3 * np.exp(0.3 * x) * np.sin(2 * y) - 0.4 * np.sin(x + 2 * y)


def _h2_uy(P):
    x, y = P[:, 0], P[:, 1]
    return 2 * np.exp(0.3 * x) * np.cos(2 * y) - 0.8 * np.sin(x + 2 * y)


def _h2_uxx(P):
    x, y = P[:, 0], P[:, 1]
    return 0.09 * np.exp(0.3 * x) * np.sin(2 * y) - 0.4 * np.cos(x + 2 * y)


def _h2_uyy(P):
    x, y = P[:, 0], P[:, 1]
    return -4 * np.exp(0.3 * x) * np.sin(2 * y) - 1.6 * np.cos(x + 2 * y)


def _h2_f(P):
    x, y = P[:, 0], P[:, 1]
    kap = _kap(P)
    return (-kap * (_h2_uxx(P) + _h2_uyy(P))
            + (1.5 - 0.3) * _h2_ux(P)
            + (-0.8 - 0.4 * y) * _h2_uy(P))


H2 = dict(
    key="heldout_convdiff", category="pde2d_steady", order=2,
    title="$-\\nabla\\!\\cdot(\\kappa\\nabla u) + b\\cdot\\nabla u = f$  (Dirichlet)",
    terms=[((2, 0), lambda P: -_kap(P)), ((0, 2), lambda P: -_kap(P)),
           ((1, 0), 1.5 - 0.3), ((0, 1), lambda P: -0.8 - 0.4 * P[:, 1])],
    exact=_h2_u,
    forcing=_h2_f,
    bc_blocks=[dict(where="circle", terms=[((0, 0), 1.0)], value=_h2_u)],
)


# ---------------------------------------------------------------------------
# H3: Fisher-KPP (nonlinear, space-time)
#   u_t = u_xx + u(1-u), classical Ablowitz-Zeppetella traveling front (f = 0):
#   u = [1 + exp(x/sqrt(6) - 5t/6)]^{-2}, wave speed 5/sqrt(6).
#   Written as: 2 u_tau - u_xx - u + u^2 = 0.
# ---------------------------------------------------------------------------

_S6 = 1.0 / np.sqrt(6.0)


def _h3_u(P):
    x, t = P[:, 0], _t_of(P)
    return (1.0 + np.exp(_S6 * x - 5.0 * t / 6.0)) ** (-2)


H3 = dict(
    key="heldout_fisher", category="pde2d_time", order=2,
    title="Fisher-KPP  $u_t = u_{xx} + u(1-u)$  (traveling front)",
    lin_terms=[((0, 1), 2.0), ((2, 0), -1.0), ((0, 0), -1.0)],
    nl=dict(fields=[(0, 0)],
            res=lambda v, p: v[(0, 0)] ** 2,
            jac=lambda v, p: {(0, 0): 2.0 * v[(0, 0)]}),
    exact=_h3_u,
    forcing=0.0,
    bc_blocks=[
        dict(where="ic", terms=[((0, 0), 1.0)], value=_h3_u),
        dict(where="left", terms=[((0, 0), 1.0)], value=_h3_u),
        dict(where="right", terms=[((0, 0), 1.0)], value=_h3_u),
    ],
    init=None,
)

HELDOUT_LINEAR = {p["key"]: p for p in [H1, H2]}
HELDOUT_NONLINEAR = {p["key"]: p for p in [H3]}


def verify_all(tol=5e-4, verbose=True):
    """FD-verify the held-out problems using the zoo suites' machinery."""
    f01 = load_problem_suite("f01")
    f02 = load_problem_suite("f02")
    rng = np.random.default_rng(11)
    failures = []
    for key, prob in HELDOUT_LINEAR.items():
        P = rng.uniform(-0.6, 0.6, (40, 2))
        f_true = prob["forcing"](P)
        err = np.max(np.abs(f01._apply_L_fd(prob, P) - f_true))
        if err / max(1.0, np.max(np.abs(f_true))) > tol:
            failures.append((key, "forcing", err))
        for blk in prob["bc_blocks"]:
            out = np.zeros(len(P))
            for (ax, ay), c in blk["terms"]:
                cc = c(P) if callable(c) else c
                out += cc * f01._fd2(prob["exact"], P, ax, ay, f01._h_for(ax + ay))
            e = np.max(np.abs(out - blk["value"](P)))
            if e / max(1.0, np.max(np.abs(out))) > tol:
                failures.append((key, f"bc:{blk['where']}", e))
        if verbose:
            print(f"  verified {key}")
    for key, prob in HELDOUT_NONLINEAR.items():
        P = rng.uniform(-0.6, 0.6, (40, 2))
        f_true = np.full(len(P), 0.0)
        err = np.max(np.abs(f02._residual_fd(prob, P) - f_true))
        if err > tol:
            failures.append((key, "residual", err))
        if verbose:
            print(f"  verified {key}")
    if failures:
        raise AssertionError(f"held-out verification failed: {failures}")
    if verbose:
        print("held-out problems verified against finite differences")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    verify_all()
