"""Shared tanh panels for several fields preserve scalar operators/adjoints."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_directional_operator import DirectionalProfileOperator, jet_batch_plan
from route2_disk_profiles import DiskProfileOperator
from route2_box_profiles import BoxProfileOperator


def make_operator(kind):
    if kind == 'direct':
        directions = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                               [1., 1., 1.]])/np.array([1., 1., 1., np.sqrt(3)])[:, None]
        return DirectionalProfileOperator(directions, 3, centers=65, block_size=4)
    if kind == 'disk':
        return DiskProfileOperator(4, centers=65, block_size=4)
    return BoxProfileOperator([[-1., 1.]]*3, 3, centers=65, block_size=4)


@pytest.mark.parametrize('kind', ['direct', 'disk', 'box'])
@pytest.mark.parametrize('fields', [1, 4])
def test_multifield_matches_scalar_jets_adjoint_and_batches(kind, fields):
    operator = make_operator(kind)
    dimension = operator.dimension
    rng = np.random.default_rng(476)
    points = rng.uniform(-.4, .4, size=(13, dimension))
    coefficients = rng.normal(size=(operator.size, fields))*.03
    orders = ((0,)*dimension, (1,)+(0,)*(dimension-1), (2,)+(0,)*(dimension-1),
              (1, 1)+(0,)*(dimension-2))
    prepared = operator.prepare_forward_fields(coefficients)
    actual = operator.forward_prepared_fields_jets(prepared, points, orders)
    for field in range(fields):
        scalar = operator.forward_jets(coefficients[:, field], points, orders)
        for order in orders:
            np.testing.assert_allclose(actual[order][:, field], scalar[order], atol=3e-13, rtol=4e-13)
    # Reusing compiled neurons for multiple point batches must be exact to
    # arithmetic tolerance; no batch is allowed to rebuild field geometry.
    for start in range(0, len(points), 3):
        batch = operator.forward_prepared_fields_jets(prepared, points[start:start+3], orders)
        for order in orders:
            np.testing.assert_allclose(batch[order], actual[order][start:start+3], atol=3e-13, rtol=4e-13)
    cotangents = {order: rng.normal(size=(len(points), fields)) for order in orders}
    accumulator = operator.prepare_adjoint_fields(fields)
    for start in range(0, len(points), 3):
        operator.accumulate_adjoint_fields_jets(
            {order: y[start:start+3] for order, y in cotangents.items()},
            points[start:start+3], accumulator)
    transpose = operator.finish_adjoint_fields(accumulator)
    for field in range(fields):
        scalar = operator.adjoint_jets({order: y[:, field] for order, y in cotangents.items()}, points)
        np.testing.assert_allclose(transpose[:, field], scalar, atol=3e-12, rtol=5e-13)
    np.testing.assert_allclose(np.sum(coefficients*transpose),
        sum(np.sum(actual[order]*cotangents[order]) for order in orders), atol=3e-12, rtol=5e-13)


@pytest.mark.parametrize('kind', ['direct', 'disk', 'box'])
def test_basis_evaluations_are_independent_of_field_count(kind, monkeypatch):
    operator = make_operator(kind)
    base = getattr(operator, 'base', operator)
    original = base._basis_many
    calls = []
    def counted(z, orders):
        calls.append((z.shape, frozenset(orders)))
        return original(z, orders)
    monkeypatch.setattr(base, '_basis_many', counted)
    points = np.random.default_rng(23).uniform(-.4, .4, size=(11, operator.dimension))
    zero = (0,)*operator.dimension
    first = (1,)+(0,)*(operator.dimension-1)
    second = (2,)+(0,)*(operator.dimension-1)
    counts = []
    for fields in (1, 4):
        coefficients = np.ones((operator.size, fields))*.01
        prepared = operator.prepare_forward_fields(coefficients)
        calls.clear()
        operator.forward_prepared_fields_jets(prepared, points, (zero, first, second))
        forward = len(calls)
        accumulator = operator.prepare_adjoint_fields(fields)
        calls.clear()
        operator.accumulate_adjoint_fields_jets(
            {order: np.ones((len(points), fields)) for order in (zero, first, second)}, points, accumulator)
        counts.append((forward, len(calls)))
        assert prepared['readouts'].shape == (base.direction_count, base.centers_per_direction, fields)
        assert all(shape[0] <= 4 and shape[1] <= 4 and shape[2] == base.centers_per_direction
                   and orders == frozenset((0, 1, 2)) for shape, orders in calls)
    point_batch, direction_batch, _ = jet_batch_plan(
        base.centers_per_direction, base.block_size, 3, 4, base.direction_count)
    # All three derivative orders and every field share the same panel.
    expected = ((base.direction_count+direction_batch-1)//direction_batch
                *((len(points)+point_batch-1)//point_batch))
    assert counts == [(expected, expected), (expected, expected)]


@pytest.mark.parametrize('kind', ['direct', 'disk', 'box'])
def test_multifield_input_validation(kind):
    operator = make_operator(kind)
    with pytest.raises(ValueError, match='matrix'):
        operator.prepare_forward_fields(np.zeros(operator.size))
    with pytest.raises(ValueError, match='matrix'):
        operator.prepare_forward_fields(np.zeros((operator.size, 0)))
    with pytest.raises(ValueError, match='fields'):
        operator.prepare_adjoint_fields(0)
    accumulator = operator.prepare_adjoint_fields(2)
    with pytest.raises(ValueError, match='Q-by-F'):
        operator.accumulate_adjoint_fields_jets({(0,)*operator.dimension: np.ones(3)},
                                                np.zeros((3, operator.dimension)), accumulator)
