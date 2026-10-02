"""Tests for expH04: the mesh-finding pipeline (monitor -> grading -> placement -> fit)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "experiments" / "expH01_highdim_suite"))
sys.path.append(str(REPO / "experiments" / "expH04_mesh_finding"))

from h01suite.baseline import EvenGeometry                       # noqa: E402
from h01suite.tasks import get_task                               # noqa: E402
from mesh import (AdaptiveGeometry, Monitors, grade_spacing,      # noqa: E402
                  oracle_derivatives, place_by_density, surrogate_derivatives)


def test_grading_bounds_the_neighbor_ratio_and_keeps_the_count():
    grid = np.linspace(-1, 1, 2001)
    dt = grid[1] - grid[0]
    rho = 0.05 + np.exp(-((grid - 0.3) / 0.01) ** 2)      # a spike the grid can resolve
    n = 64
    h = grade_spacing(1.0 / (n * rho / np.trapezoid(rho, dx=dt)), dt, 0.15, n)
    assert abs(np.trapezoid(1.0 / h, dx=dt) - n) < 1e-6
    assert np.max(np.abs(np.diff(h))) <= 0.15 * dt * (1 + 1e-9)


def test_placement_neighbor_ratio_under_the_grade():
    grid = np.linspace(-1, 1, 2001)
    m = np.exp(-((grid - 0.3) / 0.02) ** 2)
    for n in (32, 128, 512):
        c, h, info = place_by_density(grid, m, n, 2.0 / 3.0, g=0.15)
        assert np.all(np.diff(c) > 0)
        assert info["max_neighbor_ratio"] < 1.2
        assert len(c) == n
        # the floor: s = 2/3 keeps a third of the centers even, so no spacing is wider than
        # about 3x even (the grading step rescales, so allow some slack)
        assert info["max_spacing"] <= 4.0 * info["even_spacing"]


@pytest.mark.parametrize("d,B", [(1, 128), (2, 256)])
def test_floor_zero_reproduces_the_even_reference(d, B):
    task = get_task("1.14" if d == 1 else "2.14")
    X, _ = task.train_set(8 * B)
    ev = EvenGeometry(d=d, budget=B)
    ad = AdaptiveGeometry(d=d, budget=B, s=0.0).build(X, Monitors("even"))
    assert np.max(np.abs(ev.centers - ad.centers)) < 1e-12
    assert np.max(np.abs(ev.gammas - ad.gammas) / ev.gammas) < 1e-12
    assert np.allclose(ev.directions, ad.directions)


def test_gamma_times_local_spacing_is_lambda():
    task = get_task("1.16")
    X, _ = task.train_set(1024)
    g = AdaptiveGeometry(d=1, budget=128).build(X, Monitors("data", beta=1.0))
    c = g.centers
    h = np.empty_like(c)
    h[1:-1] = 0.5 * (c[2:] - c[:-2]); h[0] = c[1] - c[0]; h[-1] = c[-1] - c[-2]
    assert np.allclose(g.gammas * h, 0.25)


def test_surrogate_derivatives_match_finite_differences():
    task = get_task("2.1")
    X, y = task.train_set(2048)
    m = EvenGeometry(d=2, budget=256).fit(X, y)
    V = np.array([[1.0, 0.0], [0.6, 0.8]])
    Xs = X[:50]
    d1 = surrogate_derivatives(m, Xs, V, 1)
    d2 = surrogate_derivatives(m, Xs, V, 2)
    # the fitted weights are large (~1e7), so the finite difference is the noisy side:
    # eps = 1e-3 balances truncation against roundoff in the prediction
    eps = 1e-3
    for i, v in enumerate(V):
        fp, fm, f0 = m.predict(Xs + eps * v), m.predict(Xs - eps * v), m.predict(Xs)
        fd1, fd2 = (fp - fm) / (2 * eps), (fp - 2 * f0 + fm) / eps ** 2
        assert np.max(np.abs(d1[i] - fd1)) < 1e-4 * np.max(np.abs(fd1))
        assert np.max(np.abs(d2[i] - fd2)) < 1e-2 * np.max(np.abs(fd2))


def test_oracle_second_derivative_of_a_known_function():
    # task 1.1 in d=1 is smooth; compare the FD second derivative with an FD of grad_F
    task = get_task("1.1")
    X = np.linspace(-0.9, 0.9, 41)[:, None]
    V = np.array([[1.0]])
    d2 = oracle_derivatives(task, X, V, 2)
    eps = 1e-3
    ref = (task.F(X + eps) - 2 * task.F(X) + task.F(X - eps)) / eps ** 2
    assert np.allclose(d2[0], ref, rtol=1e-3, atol=1e-3)


def test_data_monitor_puts_centers_where_the_data_is():
    task = get_task("1.16")             # hotspot data, densest cluster at +0.45
    X, _ = task.train_set(4096)
    g = AdaptiveGeometry(d=1, budget=128).build(X, Monitors("data", beta=1.0))
    near = np.mean(np.abs(g.centers - 0.45) < 0.22)
    even = EvenGeometry(d=1, budget=128)
    near_even = np.mean(np.abs(even.centers - 0.45) < 0.22)
    assert near > 1.5 * near_even


def test_estimated_slope_mesh_reaches_the_floor_on_the_spike_at_the_hotspot():
    """1.16: even mesh ~1e-6 on the dense region at B = 128; the two-solve mesh ~1e-13."""
    task = get_task("1.16")
    B = 128
    X, y = task.train_set(8 * B)
    sets = task.test_sets()
    Xd, yd = sets["dense_region"], task.F(sets["dense_region"])
    ev = EvenGeometry(d=1, budget=B).fit(X, y)
    e_even = np.linalg.norm(ev.predict(Xd) - yd) / np.linalg.norm(yd)
    D = surrogate_derivatives(ev, X, np.array([[1.0]]), 1)
    ad = AdaptiveGeometry(d=1, budget=B).build(X, Monitors("roughness", r=1, deriv=D)).fit(X, y)
    e_ad = np.linalg.norm(ad.predict(Xd) - yd) / np.linalg.norm(yd)
    assert e_even > 1e-8
    assert e_ad < 1e-11


def test_active_subspace_geometry_lives_in_the_subspace_and_keeps_the_budget():
    from mesh import active_subspace_geometry, gradient_covariance, active_dimension
    task = get_task("5.5")               # composition: depends on z_1, z_2 only
    X, _ = task.train_set(2048)
    evals, W = gradient_covariance(task.grad_F(X))
    m = active_dimension(evals)
    assert m == 2
    assert evals[2] / evals[0] < 1e-12
    geo = active_subspace_geometry(5, 4096, W, m)
    assert abs(int(geo.per_direction.sum()) - 4096) < 0.02 * 4096   # rounding, as in the reference
    V = geo.unique_directions
    n_a = geo.mesh_info["active"]["n_active_dirs"]
    P = W[:, :2] @ W[:, :2].T
    assert np.allclose(V[:n_a] @ P, V[:n_a], atol=1e-12)          # inside the subspace
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0)
    # the active mesh is split like a 2-D problem: sqrt(0.8 * 4096) ~ 57 per direction
    assert 50 <= geo.per_direction[0] <= 64


def test_active_subspace_mesh_beats_even_on_composition_in_3d():
    """3.5 at B=1024: even mesh ~1e-4, active subspace (true gradients) ~1e-9 or better."""
    from mesh import active_subspace_geometry, gradient_covariance, active_dimension
    task = get_task("3.5")
    B = 1024
    X, y = task.train_set(8 * B)
    sets = task.test_sets()
    Xd, yd = sets["dense_region"], task.F(sets["dense_region"])
    ev = EvenGeometry(d=3, budget=B).fit(X, y)
    e_even = np.linalg.norm(ev.predict(Xd) - yd) / np.linalg.norm(yd)
    evals, W = gradient_covariance(task.grad_F(X))
    geo = active_subspace_geometry(3, B, W, active_dimension(evals))
    D = oracle_derivatives(task, X, geo.unique_directions, 1)
    ad = geo.build(X, Monitors("roughness", r=1, deriv=D)).fit(X, y)
    e_ad = np.linalg.norm(ad.predict(Xd) - yd) / np.linalg.norm(yd)
    assert e_ad < 1e-3 * e_even
