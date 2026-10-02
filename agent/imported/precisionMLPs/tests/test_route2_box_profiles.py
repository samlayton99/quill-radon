"""Independent checks for the streamed ball-moment factorization."""
from pathlib import Path
import sys
import tracemalloc

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import legvander

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))

from route2_box_profiles import BoxProfileOperator
from route2_scaling_features import fourth_order_sphere_rule
from solver.ball_ridge_features import analytic_profile_map, gegenbauer_profiles
from solver.general_residual import ResidualBlock, ResidualProblem, linearize_residual


@pytest.mark.parametrize('dimension,degree', [(2, 0), (2, 4), (3, 4), (4, 3)])
def test_entire_small_map_matches_original_exact_ball_moments(dimension, degree):
    operator = BoxProfileOperator([[-1., 1.]]*dimension, degree, centers=33,
                                  direction_block_size=3)
    reference = analytic_profile_map(dimension, degree, operator.directions,
                                     operator.angular_weights, operator.multiindices)
    for index in range(operator.size):
        column = np.zeros(operator.size); column[index] = 1.
        actual = operator.to_directional(column)
        expected = operator.base.pack(reference[:, 0, index].sum(), reference[:, 1:, index])
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=4e-13)
    assert not hasattr(operator, 'profile_map')
    assert operator.metrics['geometry_map_bytes'] == 0


def test_coordinate_transpose_actual_jets_selected_columns_and_batched_adjoint():
    rng = np.random.default_rng(831)
    operator = BoxProfileOperator([[-.7, .9], [-1.2, .4], [-.5, 1.]], 4,
                                  centers=129, block_size=5, direction_block_size=4)
    points = rng.uniform(operator.bounds[:, 0], operator.bounds[:, 1], size=(17, 3))
    coefficients = rng.normal(size=operator.size)*.03
    cotangent = rng.normal(size=operator.base.size)
    np.testing.assert_allclose(operator.to_directional(coefficients) @ cotangent,
        coefficients @ operator.directional_transpose(cotangent), atol=4e-14, rtol=3e-13)
    orders = ((0, 0, 0), (1, 0, 0), (0, 0, 2), (1, 1, 0))
    values = operator.forward_jets(coefficients, points, orders)
    indices = np.array([0, 4, operator.size-1, 4])
    selected = operator.selected_columns(points, indices, orders)
    for position, index in enumerate(indices):
        unit = np.zeros(operator.size); unit[index] = 1.
        expected = operator.forward_jets(unit, points, orders)
        for order in orders:
            np.testing.assert_allclose(selected[order][:, position], expected[order],
                                       atol=3e-13, rtol=5e-13)
    cotangents = {order: rng.normal(size=len(points)) for order in orders}
    transpose = operator.adjoint_jets(cotangents, points)
    np.testing.assert_allclose(coefficients @ transpose,
        sum(values[order] @ cotangents[order] for order in orders), atol=4e-12, rtol=4e-13)
    accumulator = operator.prepare_adjoint()
    for start in range(0, len(points), 4):
        operator.accumulate_adjoint_jets({order: z[start:start+4] for order, z in cotangents.items()},
                                         points[start:start+4], accumulator)
    np.testing.assert_allclose(operator.finish_adjoint(accumulator), transpose, atol=5e-12, rtol=5e-13)


def test_ordinary_export_and_autograd_match_actual_tanh_operator():
    rng = np.random.default_rng(812)
    operator = BoxProfileOperator([[-1., 1.]]*4, 3, centers=129)
    coefficients = rng.normal(size=operator.size)*.01
    points = rng.uniform(-.7, .7, size=(11, 4))
    model = operator.torch_model(coefficients)
    assert [type(layer) for layer in model] == [torch.nn.Linear, torch.nn.Tanh, torch.nn.Linear]
    x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    y = model(x)[:, 0]
    first = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    mixed = torch.autograd.grad(first[:, 0].sum(), x)[0][:, 2]
    jets = operator.forward_jets(coefficients, points, ((0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)))
    np.testing.assert_allclose(y.detach(), jets[(0, 0, 0, 0)], atol=5e-14, rtol=3e-13)
    np.testing.assert_allclose(first[:, 0].detach(), jets[(1, 0, 0, 0)], atol=5e-13, rtol=4e-13)
    np.testing.assert_allclose(mixed.detach(), jets[(1, 0, 1, 0)], atol=3e-12, rtol=5e-13)


def test_signed_sparse_cubature_reconstructs_known_quadratics():
    dimension = 6
    directions, weights = fourth_order_sphere_rule(dimension)
    operator = BoxProfileOperator([[-1., 1.]]*dimension, 2, centers=129,
                                  directions=directions, angular_weights=weights)
    assert operator.metrics['angular_weight_total_variation'] > 1.
    points = np.random.default_rng(451).uniform(-.8, .8, size=(9, dimension))
    coefficients = np.random.default_rng(512).normal(size=operator.size)*.01
    truth = np.ones((len(points), operator.size))
    for axis in range(dimension):
        truth *= legvander(points[:, axis], 2)[:, operator.multiindices[:, axis]]
    np.testing.assert_allclose(operator.forward(coefficients, points), truth@coefficients,
                               atol=3e-14, rtol=3e-13)


def test_high_degree_conversion_uses_bounded_storage_and_audits_cancellation():
    # Old global map here would contain M*13*1820 entries (>400 MB).
    # Peak below includes constructor, forward coordinate map and its adjoint.
    tracemalloc.start()
    operator = BoxProfileOperator([[-1., 1.]]*4, 12, centers=33, direction_block_size=8)
    rng = np.random.default_rng(751)
    coefficients = rng.normal(size=operator.size)/operator.size
    directional = operator.to_directional(coefficients)
    probe = rng.normal(size=operator.base.size)
    transpose = operator.directional_transpose(probe)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 16*1024**2
    assert operator.metrics['coordinate_workspace_bytes'] == 8*13*operator.size
    np.testing.assert_allclose(directional @ probe, coefficients @ transpose, atol=2e-11, rtol=2e-11)

    # An independent arbitrary-precision moment construction is used only
    # for selected audit rows/columns, never inside the new operator.
    columns = np.array([operator.size//2, operator.size-1])
    rows = np.array([0, operator.direction_count//2, operator.direction_count-1])
    reference = analytic_profile_map(4, 12, operator.directions[rows],
                                     operator.angular_weights[rows], operator.multiindices[columns])
    for position, column in enumerate(columns):
        unit = np.zeros(operator.size); unit[column] = 1.
        _, profiles = operator.base.unpack(operator.to_directional(unit))
        scale = max(1., np.max(np.abs(reference[:, 1:, position])))
        np.testing.assert_allclose(profiles[rows]/scale, reference[:, 1:, position]/scale,
                                   atol=3e-12, rtol=3e-12)

    # Degree twelve is explicitly NOT certified to machine precision: this
    # checks a realistic finite cancellation envelope for ideal profiles,
    # independent of the coarsely encoded N=33 tanh bank above.
    points = rng.uniform(-.6, .6, size=(7, 4))
    bias, profiles = operator.base.unpack(directional)
    ideal = bias+np.einsum('qmn,mn->q', gegenbauer_profiles(
        (points*operator.scale)@operator.directions.T, 4, 12)[:, :, 1:], profiles)
    truth = np.ones((len(points), operator.size))
    for axis in range(4):
        truth *= legvander(points[:, axis], 12)[:, operator.multiindices[:, axis]]
    np.testing.assert_allclose(ideal, truth@coefficients, atol=2e-10, rtol=2e-10)


def test_streamed_nonlinear_engine_never_requests_dense_features(monkeypatch):
    operator = BoxProfileOperator([[-1., 1.]]*3, 2, centers=65, block_size=4)
    rng = np.random.default_rng(163)
    points = rng.uniform(-.8, .8, size=(13, 3))
    zero, xx, yy, zz = (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)
    block = ResidualBlock('reaction', points,
        lambda x, j, p: -j[xx][:, 0]-j[yy][:, 0]-j[zz][:, 0]+j[zero][:, 0]**3,
        (zero, xx, yy, zz))
    problem = ResidualProblem(operator, [block], execution='streamed', batch_size=5)
    for name in ('evaluate', 'evaluate_many'):
        monkeypatch.setattr(operator, name, lambda *a, **kw: (_ for _ in ()).throw(AssertionError('dense features')))
    coefficients = rng.normal(size=(operator.size, 1))*.03
    linear = linearize_residual(problem, coefficients)
    vector, cotangent = rng.normal(size=operator.size), rng.normal(size=len(points))
    np.testing.assert_allclose(cotangent@(linear.operator@vector),
                               vector@linear.operator.rmatvec(cotangent), atol=3e-12, rtol=3e-13)
    assert linear.metrics['basis_cache_bytes'] == 0


def test_no_fitting_and_input_validation(monkeypatch):
    operator = BoxProfileOperator([[-1., 1.]]*3, 3, centers=33)
    def forbidden(*args, **kwargs):
        raise AssertionError('No fitted coordinate construction')
    for name in ('lstsq', 'svd', 'pinv', 'solve'):
        monkeypatch.setattr(np.linalg, name, forbidden)
    coefficients = np.ones(operator.size)*.01
    assert np.all(np.isfinite(operator.forward(coefficients, np.zeros((2, 3)))))
    assert np.all(np.isfinite(operator.directional_transpose(np.ones(operator.base.size))))
    with pytest.raises(ValueError, match='bounds'):
        BoxProfileOperator([[0., 0.], [-1., 1.]], 2)
    with pytest.raises(ValueError, match='together'):
        BoxProfileOperator([[-1., 1.]]*3, 2, directions=np.eye(3))
    with pytest.raises(ValueError, match='indices'):
        operator.selected_columns(np.zeros((1, 3)), np.array([-1]), [(0, 0, 0)])


def test_selected_columns_share_basis_panels_and_keep_only_requested_profiles(monkeypatch):
    operator = BoxProfileOperator([[-1., 1.]]*3, 3, centers=65, block_size=4)
    points = np.random.default_rng(99).uniform(-.7, .7, size=(11, 3))
    orders = ((0, 0, 0), (1, 0, 0), (2, 0, 0))
    calls = []
    original = operator.base._basis_many
    def counted(z, orders):
        calls.append((z.shape, orders))
        return original(z, orders)
    monkeypatch.setattr(operator.base, '_basis_many', counted)
    # A full prepared multi-column neuron state B*k would defeat the panel
    # memory bound. selected_columns must construct only one H*k bank panel.
    monkeypatch.setattr(operator.base, 'prepare_forward_fields',
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('B*k state')))
    counts = []
    for indices in (np.array([1]), np.array([0, 1, 5, 5])):
        calls.clear()
        selected = operator.selected_columns(points, indices, orders)
        counts.append(len(calls))
        for position, index in enumerate(indices):
            unit = np.zeros(operator.size); unit[index] = 1.
            scalar = operator.forward_jets(unit, points, orders)
            for order in orders:
                np.testing.assert_allclose(selected[order][:, position], scalar[order],
                                           atol=2e-13, rtol=5e-13)
    expected = operator.direction_count*3
    assert counts == [expected, expected]
    calls.clear()
    empty = operator.selected_columns(points, np.array([], dtype=int), orders)
    assert not calls
    assert all(value.shape == (len(points), 0) for value in empty.values())
    assert operator.metrics['selected_column_profile_panel_bytes_per_column'] == 8*operator.base.size
    assert operator.metrics['selected_column_neuron_panel_bytes_per_column'] == 8*operator.base.centers_per_direction
