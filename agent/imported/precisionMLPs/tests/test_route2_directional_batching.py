"""Fused bounded tanh panels retain the original operator and its transpose."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_directional_operator import (
    DirectionalProfileOperator, JET_PANEL_BUDGET_BYTES, jet_batch_plan)


def make_operator(centers=65):
    directions = np.random.default_rng(163).normal(size=(9, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return DirectionalProfileOperator(directions, 3, centers=centers,
                                      block_size=128, scale=1/np.sqrt(3))


@pytest.mark.parametrize('shape', [(17,), (3, 17)])
def test_fused_basis_matches_stable_scalar_basis_including_saturated_tails(shape):
    operator = make_operator()
    h = operator.centers_per_direction
    z = np.random.default_rng(929).uniform(-80., 80., size=shape+(h,))
    fused = operator._basis_many(z, set(range(5)))
    for order, values in fused.items():
        reference = operator._basis(z, order)
        np.testing.assert_allclose(values, reference, rtol=4e-14, atol=0.)
    # tanh itself has rounded to -1, but changes in its tail remain nonzero.
    z = operator.anchor_z[None, :]+.001
    values = operator._basis_many(z, {0})[0]
    mask = operator.anchor_z < -20
    assert np.all(values[:, mask] != 0.)
    assert np.all((np.tanh(z)-np.tanh(operator.anchor_z))[:, mask] == 0.)
    np.testing.assert_allclose(values[:, mask], operator._basis(z, 0)[:, mask], rtol=2e-12)


def test_direction_batches_match_legacy_operator_adjoint_and_export(monkeypatch):
    operator = make_operator()
    rng = np.random.default_rng(22)
    points = rng.uniform(-1., 1., size=(73, 3))
    coefficients = rng.normal(size=(operator.size, 4))*.01
    orders = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 2, 0), (0, 0, 3))
    prepared = operator.prepare_forward_fields(coefficients)
    cotangents = {a: rng.normal(size=(len(points), 4)) for a in orders}

    # Independent legacy oracle: one direction and one total order at a time.
    reference = {a: np.broadcast_to(prepared['beta'], (len(points), 4)).copy()
                 if sum(a) == 0 else np.zeros((len(points), 4)) for a in orders}
    accumulator = operator.prepare_adjoint_fields(4)
    accumulator['constant_sum'] += cotangents[(0, 0, 0)].sum(axis=0)
    for m, w in enumerate(operator.physical_directions):
        projection = (points-operator.midpoint)@w
        z = operator.encoding.gamma*(projection[:, None]-operator.encoding.centers)
        for a in orders:
            basis = operator._basis(z, sum(a))
            factor = np.prod(w**np.asarray(a))
            reference[a] += factor*(basis@prepared['readouts'][m])
            if sum(a) == 0:
                reference[a] += prepared['anchors'][m]
            accumulator['neuron_gradient'][m] += factor*(basis.T@cotangents[a])
    reference_adjoint = operator.finish_adjoint_fields(accumulator)

    panels = []
    original = operator._basis_many
    def observed(z, total_orders):
        panels.append(z.shape)
        return original(z, total_orders)
    monkeypatch.setattr(operator, '_basis_many', observed)
    actual = operator.forward_prepared_fields_jets(prepared, points, orders)
    actual_accumulator = operator.prepare_adjoint_fields(4)
    operator.accumulate_adjoint_fields_jets(cotangents, points, actual_accumulator)
    actual_adjoint = operator.finish_adjoint_fields(actual_accumulator)
    for a in orders:
        np.testing.assert_allclose(actual[a], reference[a], rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(actual_adjoint, reference_adjoint, rtol=3e-12, atol=3e-11)
    np.testing.assert_allclose(np.sum(coefficients*actual_adjoint),
                              sum(np.sum(actual[a]*cotangents[a]) for a in orders),
                              rtol=2e-13, atol=2e-12)
    assert all(directions <= 4 and rows <= 64 and h == operator.centers_per_direction
               for directions, rows, h in panels)
    assert any(directions == 4 and rows == 64 for directions, rows, _ in panels)

    # Export is still the ordinary dense tanh network, without fused helpers.
    import torch
    x = torch.tensor(points[:11], dtype=torch.float64, requires_grad=True)
    model = operator.torch_model(coefficients[:, 0])
    value = model(x)[:, 0]
    derivative = torch.autograd.grad(value.sum(), x, create_graph=True)[0][:, 0]
    mixed = torch.autograd.grad(derivative.sum(), x)[0][:, 1]
    for a, output in [((0, 0, 0), value), ((1, 0, 0), derivative), ((1, 1, 0), mixed)]:
        np.testing.assert_allclose(output.detach().numpy(), actual[a][:11, 0],
                                   rtol=2e-11, atol=2e-12)


def test_batch_planner_bounds_local_panels_independent_of_query_count():
    points, directions, workspace = jet_batch_plan(291, 128, 3, 4, 169)
    assert (points, directions, workspace) == (64, 4, 13_119_488)
    for centers in (291, 2048, 8192, 65_536):
        points, directions, workspace = jet_batch_plan(centers, 512, 3, 4, 169)
        assert 1 <= points <= 64 and 1 <= directions <= 4
        assert workspace <= JET_PANEL_BUDGET_BYTES
    # An encoding whose single row exceeds the target must be reported honestly.
    points, directions, workspace = jet_batch_plan(1_000_000, 512, 3, 4, 169)
    assert (points, directions) == (1, 1)
    assert workspace > JET_PANEL_BUDGET_BYTES
