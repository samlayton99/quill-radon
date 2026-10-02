"""Shared field execution must preserve parameter and aggregation derivatives."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_box_profiles import BoxProfileOperator
from route2_disk_profiles import DiskProfileOperator
from solver.general_residual import ResidualBlock, ResidualProblem, linearize_residual


@pytest.mark.parametrize('kind', ['box', 'disk'])
@pytest.mark.parametrize('aggregation_kind', ['none', 'dense', 'sparse'])
@pytest.mark.parametrize('relation', ['eq', 'le'])
def test_shared_fields_parameters_aggregation_and_selected_gram(
        kind, aggregation_kind, relation, monkeypatch):
    rng = np.random.default_rng(874)
    features = (BoxProfileOperator([[-1., 1.]]*3, 3, centers=33, block_size=3)
                if kind == 'box' else DiskProfileOperator(3, centers=33, block_size=3))
    dimension = features.dimension
    zero, first = (0,)*dimension, (1,)+(0,)*(dimension-1)
    points = rng.uniform(-.4, .4, (11, dimension))

    def coupled_residual(x, jets, parameters):
        u, v, w = jets[zero].unbind(1)
        du, dv, dw = jets[first].unbind(1)
        return torch.stack((u*v+parameters[:, 0]*dw-w,
                            w*w+parameters[:, 1]*du+v,
                            u+v+w+parameters[:, 0]*parameters[:, 1]*dv), dim=1)

    aggregation = (None if aggregation_kind == 'none' else rng.normal(size=(7, len(points))))
    if aggregation_kind == 'sparse':
        aggregation = sparse.csr_matrix(aggregation)
    block = ResidualBlock('coupled', points, coupled_residual, (zero, first),
                          aggregation=aggregation, relation=relation, scale=[.8, 1.1, .7])
    problem = ResidualProblem(features, [block], fields=3, parameter_initial=[.7, 1.2])
    coefficients = rng.normal(size=(features.size, 3))*.02
    cached = linearize_residual(problem, coefficients)

    def forbidden_dense_features(*args, **kwargs):
        raise AssertionError('Shared streamed execution requested a full feature matrix')

    for name in ('evaluate', 'evaluate_many'):
        monkeypatch.setattr(features, name, forbidden_dense_features)
    streamed = linearize_residual(problem, coefficients, execution='streamed', batch_size=3)
    assert streamed.metrics['shared_multifield_active']
    assert streamed.metrics['basis_cache_bytes'] == 0
    np.testing.assert_allclose(streamed.residual, cached.residual, atol=3e-14, rtol=3e-13)

    direction = rng.normal(size=features.size*3+2)
    cotangent = rng.normal(size=len(cached.residual))
    forward = streamed.operator @ direction
    transpose = streamed.operator.rmatvec(cotangent)
    np.testing.assert_allclose(forward, cached.operator @ direction, atol=3e-12, rtol=5e-13)
    np.testing.assert_allclose(transpose, cached.operator.rmatvec(cotangent), atol=3e-12, rtol=5e-13)
    np.testing.assert_allclose(cotangent @ forward, direction @ transpose, atol=3e-12, rtol=5e-13)
    # The last two entries are physical parameters, not field coefficients.
    # This explicitly exercises their accumulation across point batches.
    assert np.linalg.norm(transpose[-2:]) > 1e-4

    # Repeated coefficient indices, different output fields, and both scalar
    # parameters must use the same flattening convention in the block Gram.
    indices = np.array([0, 1, 4, features.size*3, features.size*3+1, 1])
    scales = np.array([1., .2, .6, 2., 1., .3])
    actual_gram = streamed.operator.column_gram(indices, scales)
    expected_gram = cached.operator.column_gram(indices, scales)
    np.testing.assert_allclose(actual_gram, expected_gram, atol=3e-12, rtol=5e-13)
    columns = []
    for index, scale in zip(indices, scales):
        unit = np.zeros(len(direction))
        unit[index] = scale
        columns.append(streamed.operator @ unit)
    panel = np.column_stack(columns)
    np.testing.assert_allclose(actual_gram, panel.T @ panel, atol=3e-12, rtol=5e-13)

    counts = streamed.metrics['product_counts']
    assert counts['callback_rows_max'] <= 3
    assert counts['feature_rows_max'] <= 3
    assert counts['column_panel_bytes_max'] <= 3*3*len(indices)*8
    assert counts['prepared_forward_bytes_max'] > 0
    assert counts['prepared_adjoint_bytes_max'] > 0
