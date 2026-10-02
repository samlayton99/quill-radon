import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF08_darcy_sweep"))

import core
import darcy_problems as dp


# --- core solver ----------------------------------------------------------

def test_psi_matches_fd_derivatives_of_tanh():
    """psi(o, tanh(z)) must equal d^o/dz^o tanh(z) for o = 1..3."""
    rng = np.random.default_rng(0)
    z = rng.uniform(-2, 2, 200)
    h = {1: 1e-6, 2: 1e-5, 3: 1e-3}  # order-3 roundoff ~ eps/h^3: h=1e-3 keeps it ~1e-7
    fd = {
        1: (np.tanh(z + h[1]) - np.tanh(z - h[1])) / (2 * h[1]),
        2: (np.tanh(z + h[2]) - 2 * np.tanh(z) + np.tanh(z - h[2])) / h[2] ** 2,
        3: (np.tanh(z + 2 * h[3]) - 2 * np.tanh(z + h[3])
            + 2 * np.tanh(z - h[3]) - np.tanh(z - 2 * h[3])) / (2 * h[3] ** 3),
    }
    for o in (1, 2, 3):
        assert np.max(np.abs(core.psi(o, np.tanh(z)) - fd[o])) < 1e-4


def test_rows_2d_first_derivative_matches_fd():
    """rows_2d with term ((1,0),1) applied to random coefficients must equal
    the FD x-derivative of the evaluated model."""
    rng = np.random.default_rng(1)
    dirs, offs, gamma = core.radon_geometry(64, 0.25)
    sol = rng.normal(size=len(offs) + len(core.MONO_2D))
    model = dict(dirs=dirs, offs=offs, gamma=gamma, sol=sol, W=len(offs))
    P = rng.uniform(-0.8, 0.8, (50, 2))
    dx = core.eval_model(model, P, terms=[((1, 0), 1.0)])
    h = 1e-6
    Pp, Pm = P.copy(), P.copy()
    Pp[:, 0] += h
    Pm[:, 0] -= h
    fd = (core.eval_model(model, Pp) - core.eval_model(model, Pm)) / (2 * h)
    assert np.max(np.abs(dx - fd)) < 1e-4 * max(1.0, np.max(np.abs(fd)))


def test_rows_2d_variable_coefficient_scales_rows():
    """A callable coefficient must multiply rows pointwise: rows with coeff c(P)
    equal diag(c) @ rows with coeff 1."""
    rng = np.random.default_rng(2)
    dirs, offs, gamma = core.radon_geometry(36, 0.25)
    P = rng.uniform(-1, 1, (20, 2))
    c = lambda Q: 1.0 + Q[:, 0] ** 2
    r_var = core.rows_2d(P, dirs, offs, gamma, [((2, 0), c)])
    r_one = core.rows_2d(P, dirs, offs, gamma, [((2, 0), 1.0)])
    assert np.allclose(r_var, c(P)[:, None] * r_one)


def test_boundary_points_square_on_edges():
    B = core.boundary_points_square(100)
    assert B.shape == (400, 2)
    on_edge = (np.abs(np.abs(B[:, 0]) - 1) < 1e-12) | (np.abs(np.abs(B[:, 1]) - 1) < 1e-12)
    assert on_edge.all()


def test_solve_square_interpolates_polynomial_exactly():
    """L = identity, forcing = a cubic: the poly tail alone must fit to fp precision."""
    target = lambda P: 1.0 + P[:, 0] - 2 * P[:, 1] + 0.5 * P[:, 0] * P[:, 1] ** 2
    Pb = core.boundary_points_square(64)
    bc = [dict(points=Pb, terms=[((0, 0), 1.0)], values=target(Pb))]
    model = core.solve_square([((0, 0), 1.0)], target, bc, W=64, lam=0.25)
    rng = np.random.default_rng(3)
    P = rng.uniform(-1, 1, (200, 2))
    assert core.rel_l2(core.eval_model(model, P), target(P)) < 1e-10


# --- Darcy problems -------------------------------------------------------

def test_verify_control_passes():
    """All hand-coded control derivatives/forcing must match finite differences."""
    dp.verify_control()  # raises AssertionError on any mismatch


def test_control_u_vanishes_on_boundary():
    s = np.linspace(-1, 1, 50)
    for edge in [np.stack([s, np.full_like(s, -1.0)], 1),
                 np.stack([s, np.full_like(s, 1.0)], 1),
                 np.stack([np.full_like(s, -1.0), s], 1),
                 np.stack([np.full_like(s, 1.0), s], 1)]:
        assert np.max(np.abs(dp.control_u(edge))) < 1e-12


def test_control_a_strictly_positive():
    rng = np.random.default_rng(0)
    P = rng.uniform(-1, 1, (500, 2))
    assert dp.control_a(P).min() > 1.0


def test_darcy_coefficient_spline_reproduces_smooth_grid():
    """Spline surrogate of a smooth gridded field must reproduce values and
    first derivatives away from the grid edges."""
    n = 101
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    a_grid = 3.0 + np.sin(2 * X) * np.cos(Y)
    coeff = dp.DarcyCoefficient(a_grid, sigma_px=0.0)
    rng = np.random.default_rng(4)
    P = rng.uniform(-0.9, 0.9, (200, 2))
    a_true = 3.0 + np.sin(2 * P[:, 0]) * np.cos(P[:, 1])
    ax_true = 2 * np.cos(2 * P[:, 0]) * np.cos(P[:, 1])
    ay_true = -np.sin(2 * P[:, 0]) * np.sin(P[:, 1])
    assert np.max(np.abs(coeff.a(P) - a_true)) < 1e-5
    assert np.max(np.abs(coeff.ax(P) - ax_true)) < 1e-3
    assert np.max(np.abs(coeff.ay(P) - ay_true)) < 1e-3


def test_darcy_coefficient_smoothing_reduces_gradients():
    """Pre-smoothing a two-valued (piecewise-constant) field must shrink the
    max spline gradient."""
    n = 101
    g = np.linspace(-1, 1, n)
    X, _ = np.meshgrid(g, g, indexing="ij")
    a_grid = np.where(X > 0.0, 12.0, 3.0)
    rng = np.random.default_rng(5)
    P = rng.uniform(-0.9, 0.9, (500, 2))
    raw = dp.DarcyCoefficient(a_grid, sigma_px=0.0)
    smooth = dp.DarcyCoefficient(a_grid, sigma_px=4.0)
    assert np.max(np.abs(smooth.ax(P))) < np.max(np.abs(raw.ax(P)))


def test_darcy_terms_shape_and_sign():
    """terms() must be the non-divergence form L = -a lap - grad a . grad."""
    n = 51
    g = np.linspace(-1, 1, n)
    a_grid = np.full((n, n), 2.0)  # constant a: L = -2 lap, grad terms ~ 0
    coeff = dp.DarcyCoefficient(a_grid, sigma_px=0.0)
    terms = coeff.terms()
    assert [t[0] for t in terms] == [(2, 0), (0, 2), (1, 0), (0, 1)]
    P = np.zeros((3, 2))
    assert np.allclose(terms[0][1](P), -2.0, atol=1e-8)
    assert np.allclose(terms[2][1](P), 0.0, atol=1e-6)


def test_grid_1d_cell_centered():
    """Cell-centered nodes sit at (k+1/2)h in [0,1] mapped to [-1,1]: first node
    -1 + 1/n, last node 1 - 1/n, uniform spacing 2/n."""
    n = 10
    g = dp.grid_1d(n, cell_centered=True)
    assert np.isclose(g[0], -1.0 + 1.0 / n)
    assert np.isclose(g[-1], 1.0 - 1.0 / n)
    assert np.allclose(np.diff(g), 2.0 / n)
    assert np.allclose(dp.grid_1d(n, cell_centered=False), np.linspace(-1, 1, n))


def test_darcy_coefficient_cell_centered_alignment():
    """A linear field sampled at cell centers must be reproduced exactly at
    those centers only under the cell-centered convention."""
    n = 41
    g = dp.grid_1d(n, cell_centered=True)
    X, Y = np.meshgrid(g, g, indexing="ij")
    a_grid = 2.0 + 0.5 * X + 0.25 * Y
    coeff = dp.DarcyCoefficient(a_grid, sigma_px=0.0, cell_centered=True)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)[::37]
    assert np.max(np.abs(coeff.a(P) - (2.0 + 0.5 * P[:, 0] + 0.25 * P[:, 1]))) < 1e-10


# --- end-to-end control solve --------------------------------------------

def test_control_solve_reaches_high_precision():
    """The full pipeline on the smooth control must reach <=1e-6 rel L2 by
    W=1024 (the loose tolerance keeps the test robust to seed/lambda wiggle)."""
    Pb = core.boundary_points_square(480)
    bc = [dict(points=Pb, terms=[((0, 0), 1.0)], values=np.zeros(len(Pb)))]
    model = core.solve_square(dp.control_terms(), dp.control_forcing, bc,
                              W=1024, lam=0.25)
    g = np.linspace(-1, 1, 241)
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    err = core.rel_l2(core.eval_model(model, P), dp.control_u(P))
    assert err < 1e-6, f"control rel L2 {err:.2e}"
