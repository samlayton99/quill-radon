"""The depth compiler exports ordinary affine/tanh graphs and their derivatives."""
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial import legendre
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.pure_mlp import PureMLPFeatures, analytic_jet, autograd_jet, architecture_metrics

torch.set_num_threads(1)


def polynomial_features(features, points, derivative):
    columns = []
    for mode in features.multiindices:
        result = np.ones(len(points))
        for axis, degree in enumerate(mode):
            coefficient = np.zeros(degree+1)
            coefficient[-1] = 1.
            coefficient = legendre.legder(coefficient, m=derivative[axis])
            result *= legendre.legval(points[:, axis], coefficient)
        columns.append(result)
    return np.array(columns).T


def test_forward_is_serializable_standard_mlp_and_readout(tmp_path):
    features = PureMLPFeatures([[-1,1]]*2, 3, cells=128, lam=.25)
    points = np.random.default_rng(113).uniform(-1,1,(17,2))
    coefficients = np.random.default_rng(117).normal(size=(features.size,2))
    network = features.export_readout(coefficients)
    assert all(type(layer) in (nn.Linear, nn.Tanh) for layer in network)
    traced = torch.jit.trace(network, torch.tensor(points))
    operators = {node.kind() for node in traced.inlined_graph.nodes()}
    assert operators <= {'prim::GetAttr', 'aten::linear', 'aten::tanh'}
    assert {'aten::linear', 'aten::tanh'} <= operators
    path = tmp_path/'standard_mlp.pt'
    torch.jit.save(traced, str(path))
    loaded = torch.jit.load(str(path))
    np.testing.assert_allclose(loaded(torch.tensor(points)).detach().numpy(),
                               features.evaluate(points)@coefficients, atol=2e-14, rtol=2e-14)
    # Actual differentiation of the serialized graph remains available.
    differentiable = torch.tensor(points, requires_grad=True)
    gradient = torch.autograd.grad(loaded(differentiable)[:,0].sum(), differentiable)[0]
    expected = analytic_jet(network, points, (1,0))[:,0]
    torch.testing.assert_close(gradient[:,0], expected, atol=1e-11, rtol=1e-11)
    metrics = architecture_metrics(network)
    assert metrics['product_gates'] is False
    assert metrics['stored_parameters'] > metrics['nonzero_parameters']


@pytest.mark.parametrize('dimension', [2,4])
def test_composed_products_values_first_second_and_mixed_derivatives(dimension):
    features = PureMLPFeatures([[-1,1]]*dimension, dimension, cells=256, lam=.2,
                               multiindices=np.array([[0]*dimension,[1]*dimension]))
    points = np.r_[np.random.default_rng(739).uniform(-1,1,(17,dimension)),
                   np.ones((1,dimension)), -np.ones((1,dimension))]
    derivatives = [(0,)*dimension, (1,)+(0,)*(dimension-1),
                   (2,)+(0,)*(dimension-1), (1,1)+(0,)*(dimension-2)]
    for derivative in derivatives:
        actual = features.evaluate(points, derivative)
        expected = polynomial_features(features, points, derivative)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        torch_derivative = autograd_jet(features.network, points[:7], derivative).numpy()
        np.testing.assert_allclose(torch_derivative, actual[:7], atol=2e-11, rtol=2e-11)
    assert features.metrics['product_levels'] == (1 if dimension == 2 else 2)


def test_affine_fusion_preserves_function_and_derivatives_up_to_roundoff():
    features = PureMLPFeatures([[-1,1]]*2, 2, cells=128, lam=.25)
    coefficients = np.random.default_rng(337).normal(size=(features.size,1))
    compact = features.export_readout(coefficients)
    fused = features.export_readout(coefficients, fuse_affines=True)
    assert all(type(layer) is (nn.Linear if i % 2 == 0 else nn.Tanh)
               for i,layer in enumerate(fused))
    points = np.random.default_rng(991).uniform(-.9,.9,(13,2))
    for derivative in ((0,0), (1,0), (2,0), (1,1)):
        np.testing.assert_allclose(analytic_jet(compact, points, derivative),
                                   analytic_jet(fused, points, derivative), atol=3e-11, rtol=3e-11)


def test_physical_scaling_and_invalid_contracts():
    features = PureMLPFeatures([[2,4],[-3,1]], 2, cells=256, lam=.2)
    assert features.metrics['interior_centers'] == 257
    assert features.leaf_encoding.halo_per_side == 17
    assert features.square_encoding.halo_per_side == 17
    points = np.array([[2.3,-2.2],[3.1,.2]])
    normalized = (points-features.bounds.mean(axis=1))*2/np.diff(features.bounds,axis=1).ravel()
    expected = polynomial_features(features, normalized, (0,1))*.5
    np.testing.assert_allclose(features.evaluate(points,(0,1)), expected, atol=2e-12, rtol=2e-12)
    with pytest.raises(ValueError, match='order'):
        features.evaluate(points, (3,0))
    with pytest.raises(ValueError, match='multiindices'):
        PureMLPFeatures([[-1,1]], 2, multiindices=[[3]])
    with pytest.raises(ValueError, match='product_bound'):
        PureMLPFeatures([[-1,1]], 2, product_bound=1.)
