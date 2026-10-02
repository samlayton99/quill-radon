"""Inverse NS keeps prescribed forcing fixed and differentiates viscosity."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_hardening_inverse_ns import make_problem, OBSERVATION_POINTS
from route2_hardening_disk import exact, ORDERS
from solver.general_residual import linearize_residual, residual_vector


def exact_jets(points):
    x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    values = exact(x)
    gradients = [torch.autograd.grad(values[:, field].sum(), x, create_graph=True)[0]
                 for field in range(3)]
    jets = {(0, 0): values,
            (1, 0): torch.stack([g[:, 0] for g in gradients], dim=1),
            (0, 1): torch.stack([g[:, 1] for g in gradients], dim=1)}
    for axis, order in ((0, (2, 0)), (1, (0, 2))):
        jets[order] = torch.stack([
            torch.autograd.grad(g[:, axis].sum(), x, retain_graph=True)[0][:, axis]
            for g in gradients], dim=1)
    return jets


def test_exact_state_and_explicit_observations_obey_fixed_inverse_problem():
    problem = make_problem(degree=3, centers=33, interior=13, boundary=11, batch_size=4)
    assert len(OBSERVATION_POINTS) == 5
    assert len(problem.blocks[-1].points)*2 == 10
    np.testing.assert_array_equal(problem.parameter_initial, [.15])
    np.testing.assert_array_equal(problem.parameter_lower, [.05])
    np.testing.assert_array_equal(problem.parameter_upper, [1.])
    for block in problem.blocks:
        jets = exact_jets(block.points)
        parameter = torch.full((len(block.points), 1), .3, dtype=torch.float64)
        full = block.function(torch.tensor(block.points), jets, parameter)
        np.testing.assert_allclose(full.detach(), 0., atol=3e-14)
        if block.batch_function is not None:
            rows = np.arange(len(block.points))[::2]
            batched = block.batch_function(torch.tensor(block.points[rows]),
                {order: values[rows] for order, values in jets.items()}, parameter[rows], rows)
            np.testing.assert_allclose(batched.detach(), full[rows].detach(), atol=0., rtol=0.)
    block = problem.blocks[0]
    jets = exact_jets(block.points)
    wrong = block.function(torch.tensor(block.points), jets,
                           torch.full((len(block.points), 1), .15, dtype=torch.float64))
    laplacian = jets[(2, 0)][:, :2]+jets[(0, 2)][:, :2]
    np.testing.assert_allclose(wrong[:, :2].detach(), (.15*laplacian).detach(), atol=4e-14)
    assert torch.linalg.norm(wrong[:, :2]) > 1.


def test_streamed_inverse_parameter_column_matches_finite_difference(monkeypatch):
    problem = make_problem(degree=3, centers=33, interior=13, boundary=11, batch_size=4)
    for name in ('evaluate', 'evaluate_many'):
        monkeypatch.setattr(problem.features, name,
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError('No dense feature matrix')))
    coefficients = np.random.default_rng(193).normal(size=(problem.features.size, 3))*.03
    linear = linearize_residual(problem, coefficients)
    direction = np.zeros(problem.features.size*3+1)
    direction[-1] = 1.
    h = 1e-6
    finite = (residual_vector(problem, coefficients, np.array([.15+h]))-
              residual_vector(problem, coefficients, np.array([.15-h])))/(2*h)
    np.testing.assert_allclose(linear.operator @ direction, finite, atol=4e-10, rtol=2e-8)
    assert np.linalg.norm(finite) > .01
    cotangent = np.random.default_rng(143).normal(size=len(finite))
    np.testing.assert_allclose(cotangent @ (linear.operator @ direction),
        direction @ linear.operator.rmatvec(cotangent), atol=4e-13, rtol=4e-13)
    assert linear.metrics['basis_cache_bytes'] == 0
    assert linear.metrics['shared_multifield_active']
