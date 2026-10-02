"""Disk angular coordinates: exact identity, normalization, native tanh products."""
import math
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_disk_profiles import DiskProfileOperator
from route2_directional_operator import gegenbauer_profiles
from solver.general_residual import ResidualBlock, ResidualProblem, linearize_residual


def disk_quadrature(order):
    nodes, weights = leggauss(order)
    r = np.sqrt((nodes+1)/2)
    angles = 2*np.pi*np.arange(4*order)/(4*order)
    points = np.column_stack(((r[:, None]*np.cos(angles)).ravel(),
                              (r[:, None]*np.sin(angles)).ravel()))
    return points, np.repeat(weights/2/(4*order), 4*order)


def ideal_disk_modes(operator, points):
    radius = np.linalg.norm(points, axis=1)
    angle = np.arctan2(points[:, 1], points[:, 0])
    columns = []
    for n, k, sine, normalization in zip(operator.degrees, operator.harmonics,
                                         operator.is_sine, operator.normalizations):
        radial = np.zeros(len(points))
        for s in range((n-k)//2+1):
            factor = ((-1)**s*math.factorial(int(n-s))/
                      (math.factorial(s)*math.factorial(int((n+k)//2-s))*
                       math.factorial(int((n-k)//2-s))))
            radial += factor*radius**(n-2*s)
        columns.append(normalization*radial*(np.sin(k*angle) if sine else np.cos(k*angle)))
    return np.column_stack(columns)


@pytest.mark.parametrize('degree,extra_directions', [(0, 0), (3, 0), (6, 0), (6, 3)])
def test_analytical_angular_identity_and_ideal_disk_orthonormality(degree, extra_directions):
    operator = DiskProfileOperator(degree, centers=65, direction_count=degree+1+extra_directions)
    points, weights = disk_quadrature(degree+3)
    ideal = ideal_disk_modes(operator, points)
    reconstructed = np.empty_like(ideal)
    reconstructed[:, 0] = 1.
    profiles = gegenbauer_profiles(points @ operator.directions.T, 2, degree)
    for n in range(1, degree+1):
        indices = np.arange(operator.degree_slices[n].start, operator.degree_slices[n].stop)
        reconstructed[:, indices] = profiles[:, :, n] @ operator._angular_columns(indices)
    np.testing.assert_allclose(reconstructed, ideal, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(ideal.T @ (weights[:, None]*ideal), np.eye(operator.size),
                               atol=2e-13, rtol=2e-13)
    assert operator.size == (degree+1)*(degree+2)//2
    assert operator.metrics['geometry_map_bytes'] == 0
    assert operator.angular_table.shape == (operator.direction_count, degree+1)
    assert not hasattr(operator, 'multiindices')


def test_canonical_conversion_transpose_and_actual_tanh_selected_columns():
    random = np.random.default_rng(713)
    operator = DiskProfileOperator(5, centers=129, block_size=7)
    points = random.uniform(-.6, .6, size=(19, 2))
    coefficients = random.normal(size=operator.size)*.1
    raw_cotangent = random.normal(size=operator.base.size)
    np.testing.assert_allclose(operator.to_directional(coefficients) @ raw_cotangent,
        coefficients @ operator.directional_transpose(raw_cotangent), rtol=2e-14, atol=2e-14)
    orders = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1))
    values = operator.forward_jets(coefficients, points, orders)
    base_values = operator.base.forward_jets(operator.to_directional(coefficients), points, orders)
    matrix = operator.evaluate_many(points, orders)
    indices = np.array([5, 0, 11, 5, 3])
    selected = operator.selected_columns(points, indices, orders)
    for d in orders:
        np.testing.assert_allclose(values[d], base_values[d], atol=1e-14)
        np.testing.assert_allclose(matrix[d] @ coefficients, values[d], atol=5e-13, rtol=3e-13)
        np.testing.assert_allclose(selected[d], matrix[d][:, indices], atol=2e-13, rtol=3e-13)
    cotangents = {d: random.normal(size=len(points)) for d in orders}
    pullback = operator.adjoint_jets(cotangents, points)
    np.testing.assert_allclose(coefficients @ pullback,
        sum(values[d] @ cotangents[d] for d in orders), rtol=3e-13, atol=3e-13)
    accumulator = operator.prepare_adjoint()
    for start in range(0, len(points), 4):
        operator.accumulate_adjoint_jets({d: y[start:start+4] for d, y in cotangents.items()},
                                         points[start:start+4], accumulator)
    np.testing.assert_allclose(operator.finish_adjoint(accumulator), pullback, atol=4e-12, rtol=4e-13)
    # Actual encoding accuracy is checked independently of the angular identity.
    # At N=129 the fifth-degree profile has a finite halo-encoding error
    # around 6e-12; orthonormality above concerns the exact ideal profiles.
    np.testing.assert_allclose(matrix[(0, 0)], ideal_disk_modes(operator, points), atol=1e-11, rtol=1e-11)


def test_export_is_ordinary_tanh_and_matches_direct_derivatives():
    rng = np.random.default_rng(73)
    operator = DiskProfileOperator(4, centers=129, block_size=9)
    coefficients = rng.normal(size=operator.size)*.05
    points = rng.uniform(-.6, .6, size=(17, 2))
    model = operator.torch_model(coefficients)
    assert [type(layer) for layer in model] == [torch.nn.Linear, torch.nn.Tanh, torch.nn.Linear]
    inputs = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    output = model(inputs)[:, 0]
    gradient = torch.autograd.grad(output.sum(), inputs, create_graph=True)[0]
    xx = torch.autograd.grad(gradient[:, 0].sum(), inputs)[0][:, 0]
    native = operator.forward_jets(coefficients, points, ((0, 0), (1, 0), (0, 1), (2, 0)))
    np.testing.assert_allclose(output.detach(), native[(0, 0)], atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(gradient[:, 0].detach(), native[(1, 0)], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(gradient[:, 1].detach(), native[(0, 1)], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(xx.detach(), native[(2, 0)], atol=2e-11, rtol=2e-11)


def test_streamed_engine_accepts_nonredundant_modes_without_dense_evaluation(monkeypatch):
    operator = DiskProfileOperator(3, centers=65, block_size=5)
    rng = np.random.default_rng(902)
    points = rng.uniform(-.6, .6, size=(11, 2))
    block = ResidualBlock('nonlinear', points,
        lambda x,j,p:-j[(2, 0)][:, 0]-j[(0, 2)][:, 0]+j[(0, 0)][:, 0]**3,
        ((0, 0), (2, 0), (0, 2)))
    problem = ResidualProblem(operator, [block], execution='streamed', batch_size=4)
    for name in ('evaluate', 'evaluate_many'):
        monkeypatch.setattr(operator, name, lambda *a, **kw: (_ for _ in ()).throw(AssertionError('dense feature request')))
    coefficients = rng.normal(size=(operator.size, 1))*.1
    linear = linearize_residual(problem, coefficients)
    v, z = rng.normal(size=operator.size), rng.normal(size=len(points))
    np.testing.assert_allclose(z @ (linear.operator @ v), v @ linear.operator.rmatvec(z), atol=5e-12, rtol=2e-13)
    indices = np.array([0, 1, 4, 6])
    gram = linear.operator.column_gram(indices, np.ones(len(indices)))
    columns = np.column_stack([linear.operator @ np.eye(operator.size)[:, i] for i in indices])
    np.testing.assert_allclose(gram, columns.T @ columns, atol=2e-12, rtol=2e-13)
    assert linear.metrics['basis_cache_bytes'] == 0


def test_no_factorization_in_coordinate_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Coordinate construction must be analytical')
    # Build the shared encoder before forbidding unrelated quadrature setup.
    operator = DiskProfileOperator(4, centers=65)
    for name in ('lstsq', 'svd', 'pinv', 'solve'):
        monkeypatch.setattr(np.linalg, name, forbidden)
    coefficients = np.arange(operator.size, dtype=float)*.01
    assert np.all(np.isfinite(operator.to_directional(coefficients)))
    assert np.all(np.isfinite(operator.forward(coefficients, np.array([[.2, .1]]))))
    with pytest.raises(ValueError, match='at least'):
        DiskProfileOperator(4, direction_count=4)
