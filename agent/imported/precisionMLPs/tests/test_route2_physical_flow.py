"""Boundary-driven benchmarks expose physical data only and preserve NS math."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_physical_flow import (ORDERS, boundary_velocity, momentum_and_divergence,
    make_problem, coordinate_keys, transfer_coefficients, load_native, audit)
from solver.general_residual import ResidualSolution


def test_zero_force_ns_operator_matches_hand_derived_nonlinear_field():
    points = np.random.default_rng(3).uniform(-1., 1., (17, 2))
    x, y = points.T
    # u=x², v=-2xy is divergence free; p=x+3y fixes both pressure gradients.
    zeros, ones = np.zeros_like(x), np.ones_like(x)
    jets = {(0, 0): np.column_stack((x*x, -2*x*y, x+3*y)),
            (1, 0): np.column_stack((2*x, -2*y, ones)),
            (0, 1): np.column_stack((zeros, -2*x, 3*ones)),
            (2, 0): np.column_stack((2*ones, zeros, zeros)),
            (0, 2): np.zeros((len(x), 3))}
    expected = np.column_stack((2*x**3-.2+1, 2*x*x*y+3, zeros))
    for representation in (jets, {a: torch.tensor(value) for a, value in jets.items()}):
        np.testing.assert_allclose(momentum_and_divergence(representation, .1), expected,
                                   rtol=4e-15, atol=4e-15)


@pytest.mark.parametrize('case', ['cavity', 'disk_stirring', 'disk_counterrotating'])
def test_physics_contains_no_reference_force_or_interior_observations(case):
    problem = make_problem(case, degree=3, centers=33, interior=19, boundary_per_side=9)
    assert problem.fields == 3
    assert problem.parameter_initial.size == 0
    assert [b.name for b in problem.blocks] == ['zero_force_momentum_and_divergence',
                                               'prescribed_wall_velocity', 'pressure_gauge']
    for block in problem.blocks:
        points = torch.tensor(block.points)
        jets = {a: torch.zeros((len(points), 3), dtype=torch.float64) for a in ORDERS}
        parameters = torch.empty((len(points), 0), dtype=torch.float64)
        residual = block.function(points, jets, parameters)
        if block.name != 'prescribed_wall_velocity':
            np.testing.assert_array_equal(residual, 0.)
        else:
            # A zero network fails only prescribed boundary data initially.
            assert np.linalg.norm(residual) > 1.
            ids = np.arange(len(points))[::3]
            batched = block.batch_function(points[ids], {a: v[ids] for a, v in jets.items()},
                                             parameters[ids], ids)
            np.testing.assert_array_equal(batched, residual[ids])
            if case.startswith('disk_'):
                prescribed = boundary_velocity(block.points, case)
                np.testing.assert_allclose(np.sum(prescribed*block.points, axis=1), 0., atol=2e-16)
            else:
                trace = boundary_velocity(block.points, case)
                np.testing.assert_array_equal(trace[np.abs(block.points[:, 0]) == 1.], 0.)
                np.testing.assert_array_equal(trace[block.points[:, 1] == -1.], 0.)


@pytest.mark.parametrize('case', ['cavity', 'disk_stirring', 'disk_counterrotating'])
def test_native_continuation_maps_coordinates_without_fitting(case, tmp_path):
    old = make_problem(case, degree=2, centers=33, interior=8, boundary_per_side=5).features
    new = make_problem(case, degree=4, centers=33, interior=8, boundary_per_side=5).features
    coefficients = np.random.default_rng(41).normal(size=(old.size, 3))
    keys = coordinate_keys(old)
    transferred = transfer_coefficients(coefficients, keys, new)
    lookup = {key: i for i, key in enumerate(coordinate_keys(new))}
    for key, expected in zip(keys, coefficients):
        np.testing.assert_array_equal(transferred[lookup[key]], expected)
    np.testing.assert_array_equal(transferred[[i for k, i in lookup.items() if k not in keys]], 0.)
    metadata = dict(kind='native_boundary_driven_ns', case=case, lid_speed=1., viscosity=.1)
    path = tmp_path/'native.npz'
    np.savez(path, coefficients=coefficients, coordinate_keys=np.asarray(keys), metadata=json.dumps(metadata))
    actual, actual_keys, saved = load_native(path, case, 1.)
    np.testing.assert_array_equal(actual, coefficients)
    assert actual_keys == keys and saved['viscosity'] == .1
    with pytest.raises(ValueError, match='boundary velocity'):
        load_native(path, case, 2.)
    metadata['kind'] = 'external_reference'
    np.savez(path, coefficients=coefficients, coordinate_keys=np.asarray(keys), metadata=json.dumps(metadata))
    with pytest.raises(ValueError, match='native boundary-driven'):
        load_native(path, case, 1.)


@pytest.mark.parametrize('case', ['cavity', 'disk_stirring', 'disk_counterrotating'])
def test_zero_physics_residual_cannot_hide_unmet_driving_boundary(case):
    problem = make_problem(case, degree=2, centers=33, interior=8, boundary_per_side=5)
    solution = ResidualSolution(problem, np.zeros((problem.features.size, 3)), np.empty(0),
                                dict(status='test_only_zero_state', execution='streamed'))
    _, result = audit(solution, case, .1, count=17)
    assert result['momentum_rms'] == 0.
    assert result['divergence_rms'] == 0.
    assert result['wall_velocity_rms'] > .1
    assert result['sampled_residual_reaches_requested_tolerance'] is False
    assert result['continuum_accuracy_certified'] is False
    assert result['truth_error_available'] is False


def test_enclosing_disk_chart_preserves_square_physics_and_covers_corners():
    from route2_disk_profiles import DiskProfileOperator
    box = make_problem('cavity', degree=4, centers=65, interior=19, boundary_per_side=9)
    disk = make_problem('cavity', degree=4, centers=65, interior=19, boundary_per_side=9,
                        coordinate_basis='disk')
    assert isinstance(disk.features, DiskProfileOperator)
    for original, transformed in zip(box.blocks, disk.blocks):
        np.testing.assert_array_equal(original.points, transformed.points)
        jets = {a: torch.randn((len(original.points), 3), dtype=torch.float64) for a in ORDERS}
        points = torch.tensor(original.points)
        params = torch.empty((len(points), 0), dtype=torch.float64)
        np.testing.assert_array_equal(original.function(points, jets, params),
                                      transformed.function(points, jets, params))
    corners = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    np.testing.assert_allclose(np.sum((corners*disk.features.scale)**2, axis=1), 1., atol=3e-16)
    # Same geometry-only polynomial coordinates; no PDE solve or target fit.
    coefficients = np.zeros((disk.features.size, 3))
    coefficients[0] = [1., 2., 3.]
    jets = disk.features.selected_ideal_columns(corners, np.arange(disk.features.size), ORDERS)
    np.testing.assert_allclose(jets[(0, 0)]@coefficients, np.tile([1., 2., 3.], (4, 1)), atol=1e-14)


def test_native_physical_run_rejects_changed_coordinate_chart(tmp_path):
    from route2_physical_flow import run
    features = make_problem('cavity', degree=2, centers=33, interior=8, boundary_per_side=5).features
    path = tmp_path/'box.npz'
    metadata = dict(kind='native_boundary_driven_ns', case='cavity', lid_speed=1., viscosity=.1,
                    degree=2, centers=33, lam=.2, coordinate_basis='box')
    np.savez(path, coefficients=np.zeros((features.size, 3)),
             coordinate_keys=np.asarray(coordinate_keys(features)), metadata=json.dumps(metadata))
    with pytest.raises(ValueError, match='coordinate basis and chart'):
        run(case='cavity', degrees=(4,), coordinate_basis='disk', resume=str(path))
