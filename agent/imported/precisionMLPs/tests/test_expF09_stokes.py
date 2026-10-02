import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "experiments" / "expF08_darcy_sweep"))   # scalar ref
sys.path.append(str(REPO_ROOT / "experiments" / "expF09_navier_stokes"))

import core as scalar_core          # expF08 scalar solver, for the equivalence check
import core_system as cs


def test_solve_system_matches_scalar_on_poisson():
    """A one-field Poisson (-lap u = f, u=0 on the boundary) solved by
    solve_system must match the scalar solve_square to fp precision."""
    target = lambda P: np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])
    lap = lambda P: -2 * np.pi**2 * target(P)          # lap of the target
    forcing = lambda P: -lap(P)                         # -lap u = f  ->  f = -lap(target)
    Pb = scalar_core.boundary_points_square(200)
    # scalar reference
    ref = scalar_core.solve_square([((2, 0), -1.0), ((0, 2), -1.0)], forcing,
                                   [dict(points=Pb, terms=[((0, 0), 1.0)],
                                         values=target(Pb))], W=576, lam=0.25)
    rng = np.random.default_rng(1)
    Pi = rng.uniform(-1, 1, (3000, 2))
    eqs = [
        dict(points=Pi, blocks={"u": [((2, 0), -1.0), ((0, 2), -1.0)]}, rhs=forcing),
        dict(points=Pb, blocks={"u": [((0, 0), 1.0)]}, rhs=target),
    ]
    m = cs.solve_system(["u"], eqs, W=576, lam=0.25, seed=42, interior_ref=Pi)
    P = rng.uniform(-1, 1, (400, 2))
    a = scalar_core.eval_model(ref, P)
    b = cs.eval_field(m, "u", P)
    # both are min-norm fits of the same problem; agree to solver precision
    assert scalar_core.rel_l2(b, target(P)) < 1e-9
    assert scalar_core.rel_l2(a, target(P)) < 1e-9


import stokes


def test_target_is_divergence_free():
    """u*_x + v*_y == 0 analytically at random points."""
    rng = np.random.default_rng(3)
    P = rng.uniform(-1, 1, (500, 2))
    div = stokes.u_star_x(P) + stokes.v_star_y(P)
    assert np.max(np.abs(div)) < 1e-12


def test_verify_stokes_forcing():
    """All hand-coded derivatives and the momentum forcing match finite
    differences (raises on mismatch)."""
    stokes.verify_stokes()


import run_stokes as rs


def test_stokes_reaches_floor():
    """At W=1600 the manufactured Stokes solve hits the velocity floor and the
    divergence residual is tiny (verified prototype: vel ~8e-13, div ~2e-9)."""
    rec = rs.evaluate_cell(W=1600, lam=0.25)
    assert rec["vel_rel_l2"] < 1e-10
    assert rec["max_div"] < 1e-7
