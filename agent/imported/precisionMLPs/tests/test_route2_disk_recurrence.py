"""Opt-in recurrence preserves ideal jets and leaves the actual MLP alone."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/expF19_radon_direct_pde'))
from route2_disk_profiles import DiskProfileOperator
from route2_affine_disk_profiles import AffineDiskProfileOperator
from route2_disk_recurrence import (RecurrenceDiskProfileOperator,
    RecurrenceAffineDiskProfileOperator, BatchedRecurrenceDiskProfileOperator,
    BatchedRecurrenceAffineDiskProfileOperator, WORKSPACE_LIMIT)


ORDERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))


def disk_points(seed=36, count=47):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2*np.pi, count)
    radius = np.sqrt(rng.uniform(0, 1, count))
    edge = np.linspace(0, 2*np.pi, 25)[:-1]
    return np.vstack(([[0., 0.]], np.c_[radius*np.cos(angle), radius*np.sin(angle)],
                      np.c_[np.cos(edge), np.sin(edge)]))


@pytest.mark.parametrize('degree', [0, 1, 2, 16, 32, 64])
def test_recurrence_matches_independent_scipy_jets_with_boundaries(degree):
    features = RecurrenceDiskProfileOperator(degree, centers=33,
        midpoint=[.4, -.2], scale=[1.3, .7])
    points = disk_points()/features.scale + features.midpoint
    indices = np.arange(features.size)
    prepared = features.prepare_ideal_columns(points, ORDERS)
    actual = features.selected_ideal_columns(points, indices, ORDERS, prepared=prepared)
    expected = DiskProfileOperator.selected_ideal_columns(features, points, indices, ORDERS)
    for d in ORDERS:
        scale = max(1., np.max(abs(expected[d])))
        np.testing.assert_allclose(actual[d], expected[d], rtol=3e-12, atol=scale*5e-14)
        assert np.linalg.norm(actual[d]-expected[d]) <= max(1., np.linalg.norm(expected[d]))*4e-14
    assert prepared['retained_bytes'] <= prepared['plan']['retained_workspace_bound'] <= WORKSPACE_LIMIT
    assert 'radial' not in features.__dict__  # no dataset-wide state on operator


def test_affine_corners_subset_order_and_duplicate_indices():
    features = RecurrenceAffineDiskProfileOperator([[-2., 3.], [.2, .5]], 16, centers=33)
    points = np.array([[-2., .2], [-2., .5], [3., .2], [3., .5], [.5, .35]])
    indices = np.array([features.size-1, 0, 18, 18, 3, 52])
    derivatives = ((0, 2), (0, 0), (1, 1), (0, 2))
    expected = AffineDiskProfileOperator.selected_ideal_columns(features, points, indices, derivatives)
    actual = features.selected_ideal_columns(points, indices, derivatives,
        prepared=features.prepare_ideal_columns(points, derivatives))
    for d in expected:
        np.testing.assert_allclose(actual[d], expected[d], rtol=5e-12,
                                  atol=max(1., np.max(abs(expected[d])))*5e-14)


def test_prepared_panels_are_reusable_and_forward_adjoint_are_consistent():
    features = RecurrenceDiskProfileOperator(32, centers=33)
    rng = np.random.default_rng(831)
    points = disk_points(count=15)
    coefficients = rng.normal(size=(features.size, 3))
    cotangents = {d: rng.normal(size=(len(points), 3)) for d in ORDERS}
    prepared = features.prepare_ideal_columns(points, ORDERS)
    forward = {d: np.zeros((len(points), 3)) for d in ORDERS}
    adjoint = np.zeros_like(coefficients)
    for start in range(0, features.size, 23):
        indices = np.arange(start, min(start+23, features.size))
        jets = features.selected_ideal_columns(points, indices, ORDERS, prepared=prepared)
        for d in ORDERS:
            forward[d] += jets[d] @ coefficients[indices]
            adjoint[indices] += jets[d].T @ cotangents[d]
    lhs = sum(np.sum(forward[d]*cotangents[d]) for d in ORDERS)
    rhs = np.sum(coefficients*adjoint)
    np.testing.assert_allclose(lhs, rhs, atol=1e-8, rtol=2e-14)
    old = DiskProfileOperator.selected_ideal_columns(features, points, np.arange(features.size), ORDERS)
    for d in ORDERS:
        np.testing.assert_allclose(forward[d], old[d] @ coefficients,
                                  atol=max(1., np.max(abs(forward[d])))*2e-13, rtol=2e-12)


def test_prepared_state_rejects_other_geometry_derivatives_or_mutated_points():
    features = RecurrenceDiskProfileOperator(4, centers=33)
    other = RecurrenceDiskProfileOperator(4, centers=33)
    points = disk_points(count=2)
    prepared = features.prepare_ideal_columns(points, ORDERS)
    with pytest.raises(ValueError, match='same geometry'):
        other.selected_ideal_columns(points, np.array([1]), ORDERS, prepared=prepared)
    with pytest.raises(ValueError, match='same geometry'):
        features.selected_ideal_columns(points, np.array([1]), [(0, 0)], prepared=prepared)
    points[0, 0] += .01
    with pytest.raises(ValueError, match='same geometry'):
        features.selected_ideal_columns(points, np.array([1]), ORDERS, prepared=prepared)


def test_memory_refusal_precedes_table_allocation(monkeypatch):
    import route2_disk_recurrence as implementation
    features = RecurrenceDiskProfileOperator(64, centers=33)
    plan = features.ideal_recurrence_plan(128, ORDERS)
    assert plan['construction_workspace_bound'] < WORKSPACE_LIMIT
    monkeypatch.setattr(implementation, '_radial_family',
                        lambda *args: pytest.fail('allocated after failed workspace preflight'))
    with pytest.raises(MemoryError, match='16-MiB'):
        features.prepare_ideal_columns(np.zeros((plan['maximum_rows']+1, 2)), ORDERS)


def test_one_off_gram_columns_keep_original_path_without_full_table_setup(monkeypatch):
    import route2_disk_recurrence as implementation
    features = RecurrenceDiskProfileOperator(32, centers=33)
    points = disk_points(count=3)
    indices = np.array([1, 52, 136])
    expected = DiskProfileOperator.selected_ideal_columns(features, points, indices, ORDERS)
    monkeypatch.setattr(implementation, '_radial_family',
                        lambda *args: pytest.fail('one-off Gram panel unnecessarily builds all modes'))
    actual = features.selected_ideal_columns(points, indices, ORDERS)
    for d in ORDERS:
        np.testing.assert_array_equal(actual[d], expected[d])


def test_empty_and_higher_derivative_fallback():
    features = RecurrenceDiskProfileOperator(6, centers=33)
    points = disk_points(count=2)
    assert features.selected_ideal_columns(points, np.array([], int), ORDERS)[(0, 0)].shape == (len(points), 0)
    assert features.selected_ideal_columns(points, np.array([0]), []) == {}
    assert features.selected_ideal_columns(np.empty((0, 2)), np.array([0]), ORDERS)[(0, 0)].shape == (0, 1)
    orders = ((3, 0), (1, 2))
    indices = np.arange(features.size)
    prepared = features.prepare_ideal_columns(points, orders)
    assert prepared['fallback']
    actual = features.selected_ideal_columns(points, indices, orders, prepared=prepared)
    expected = features._selected_ideal_columns_ridge(points, indices, orders)
    for d in orders:
        np.testing.assert_array_equal(actual[d], expected[d])


def test_actual_tanh_paths_and_export_are_bitwise_unchanged():
    old = DiskProfileOperator(4, centers=33)
    new = RecurrenceDiskProfileOperator(4, centers=33)
    rng = np.random.default_rng(893)
    coefficients = rng.normal(size=old.size)*.01
    points = disk_points(count=3)
    for d in ORDERS:
        np.testing.assert_array_equal(old.forward_jets(coefficients, points, [d])[d],
                                      new.forward_jets(coefficients, points, [d])[d])
    original, changed = old.torch_model(coefficients), new.torch_model(coefficients)
    for a, b in zip(original.parameters(), changed.parameters()):
        np.testing.assert_array_equal(a.detach().numpy(), b.detach().numpy())


@pytest.mark.parametrize('feature_class', [RecurrenceDiskProfileOperator, BatchedRecurrenceDiskProfileOperator])
def test_existing_streamed_correction_protocol_uses_prepared_recurrence(feature_class):
    from solver.general_residual import ResidualBlock, ResidualProblem, linearize_residual
    rng = np.random.default_rng(9)
    points = disk_points(count=3)
    def residual(x, jets, parameters):
        return jets[(2, 0)] + jets[(0, 2)] + jets[(0, 0)]**2 + .2*jets[(1, 0)]
    block = ResidualBlock('physics', points, residual, ((0, 0), (1, 0), (2, 0), (0, 2)))
    features = feature_class(5, centers=33)
    old = DiskProfileOperator(5, centers=33)
    coefficients = rng.normal(size=(features.size, 2))*.03
    a = linearize_residual(ResidualProblem(features, [block], 2, execution='streamed', batch_size=9), coefficients)
    b = linearize_residual(ResidualProblem(old, [block], 2, execution='streamed', batch_size=9), coefficients)
    delta = rng.normal(size=coefficients.size)
    cotangent = rng.normal(size=len(a.residual))
    actual = a.operator.ideal_operator
    expected = b.operator.ideal_operator
    np.testing.assert_allclose(actual.matvec(delta), expected.matvec(delta), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual.rmatvec(cotangent), expected.rmatvec(cotangent), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(cotangent @ actual.matvec(delta), delta @ actual.rmatvec(cotangent),
                               atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(a.residual, b.residual)
    counts = a.metrics['product_counts']
    assert counts.get('ideal_correction_table_bytes_max', 0) > 0 or counts.get('ideal_tensor_forward_batches', 0) > 0


@pytest.mark.parametrize('degree', [16, 32, 64])
def test_complete_row_products_stay_bounded_and_obey_adjoint(degree):
    import tracemalloc
    features = BatchedRecurrenceDiskProfileOperator(degree, centers=33)
    rng = np.random.default_rng(940)
    points = disk_points(count=103)
    coefficients = rng.normal(size=(features.size, 3))
    cotangents = {d: rng.normal(size=(len(points), 3)) for d in ORDERS}
    plan = features.ideal_tensor_plan(3, len(points), ORDERS)
    assert plan['enabled'] and plan['workspace_bytes'] <= WORKSPACE_LIMIT
    assert not plan['complete_dataset_feature_cache']
    forward_state = features.prepare_ideal_tensor_forward(coefficients)
    forward = {d: np.empty((len(points), 3)) for d in ORDERS}
    adjoint_state = features.prepare_ideal_tensor_adjoint(3)
    for start in range(0, len(points), plan['rows']):
        stop = min(start+plan['rows'], len(points))
        batch = points[start:stop]
        tracemalloc.start()
        jets = features.forward_ideal_tensor_jets(forward_state, batch, ORDERS)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Include fixed input and output coefficient states omitted by tracing.
        assert peak + 2*coefficients.nbytes <= plan['workspace_bytes']
        for d in ORDERS:
            forward[d][start:stop] = jets[d]
        features.accumulate_ideal_tensor_jets({d: v[start:stop] for d, v in cotangents.items()}, batch, adjoint_state)
    adjoint = features.finish_ideal_tensor_adjoint(adjoint_state)
    lhs = sum(np.sum(forward[d]*cotangents[d]) for d in ORDERS)
    rhs = np.sum(coefficients*adjoint)
    np.testing.assert_allclose(lhs, rhs, rtol=5e-14, atol=1e-8)
    expected = DiskProfileOperator.selected_ideal_columns(features, points, np.arange(features.size), ORDERS)
    for d in ORDERS:
        np.testing.assert_allclose(forward[d], expected[d] @ coefficients,
                                  atol=max(1., np.max(abs(forward[d])))*2e-13, rtol=2e-12)
    if plan['rows'] < len(points):
        with pytest.raises(MemoryError, match='row count'):
            features.forward_ideal_tensor_jets(forward_state, points, ORDERS)


def test_affine_complete_rows_use_the_same_physical_chart():
    features = BatchedRecurrenceAffineDiskProfileOperator([[-2., 3.], [.2, .5]], 12, centers=33)
    coefficients = np.linspace(-.1, .2, features.size)[:, None]
    points = np.array([[-2., .2], [3., .5], [-2., .5], [3., .2]])
    state = features.prepare_ideal_tensor_forward(coefficients)
    actual = features.forward_ideal_tensor_jets(state, points, ORDERS)
    expected = DiskProfileOperator.selected_ideal_columns(features, points, np.arange(features.size), ORDERS)
    for d in ORDERS:
        np.testing.assert_allclose(actual[d], expected[d] @ coefficients, rtol=2e-12, atol=1e-8)
    assert not features.ideal_tensor_plan(1, 4, [(3, 0)])['enabled']
