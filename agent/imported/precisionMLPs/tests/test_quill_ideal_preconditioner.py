"""Ideal coordinates accelerate ONLY preconditioner setup, never the PDE."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_box_profiles import BoxProfileOperator
from route2_disk_profiles import DiskProfileOperator
from solver.general_residual import (ResidualBlock, ResidualProblem, linearize_residual,
                                    _RightPreconditioner, solve_residual)


@pytest.mark.parametrize('kind', ['box', 'disk'])
def test_ideal_jets_match_resolved_low_degree_tanh_and_skip_neuron_basis(kind, monkeypatch):
    operator = (BoxProfileOperator([[-.8, 1.2], [-1.5, .5]], 4, centers=257)
                if kind == 'box' else DiskProfileOperator(4, centers=257,
                    midpoint=[.2, -.5], scale=[1., 1.]))
    points = np.random.default_rng(382).uniform(-.4, .4, size=(13, 2))+np.array([.2, -.5])
    indices = np.array([0, 1, 4, operator.size-1, 4])
    orders = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 3))
    actual = operator.selected_columns(points, indices, orders)
    def forbidden(*args, **kwargs):
        raise AssertionError('Ideal preconditioner panel may not evaluate neurons')
    monkeypatch.setattr(operator.base, '_basis', forbidden)
    ideal = operator.selected_ideal_columns(points, indices, orders)
    for order in orders:
        np.testing.assert_allclose(actual[order], ideal[order], atol=2e-10, rtol=3e-11)
    empty = operator.selected_ideal_columns(points, np.array([], int), orders)
    assert all(value.shape == (len(points), 0) for value in empty.values())


def test_box_ideal_jets_have_correct_physical_chain_factors():
    operator = BoxProfileOperator([[-2., 4.], [-.5, 1.5], [1., 2.]], 4, centers=33)
    index = operator._lookup[(2, 1, 1)]
    points = np.array([[.3, .4, 1.2], [1.2, -.1, 1.8]])
    orders = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 1), (0, 0, 2))
    ideal = operator.selected_ideal_columns(points, np.array([index]), orders)
    for order in orders:
        x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
        z = (x-torch.tensor(operator.midpoint))*torch.tensor(operator.scale*np.sqrt(3))
        value = (.5*(3*z[:, 0]**2-1))*z[:, 1]*z[:, 2]
        for axis, count in enumerate(order):
            for _ in range(count):
                value = torch.autograd.grad(value.sum(), x, create_graph=True)[0][:, axis]
        np.testing.assert_allclose(ideal[order][:, 0], value.detach(), atol=2e-14, rtol=2e-14)


def make_problem(kind='box', execution='streamed', centers=129, aggregation=False):
    features = (BoxProfileOperator([[-1., 1.]]*2, 4, centers=centers, block_size=4)
                if kind == 'box' else DiskProfileOperator(4, centers=centers, block_size=4))
    rng = np.random.default_rng(146)
    points = rng.uniform(-.5, .5, size=(13, 2))
    zero, dx, xx = (0, 0), (1, 0), (2, 0)
    def callback(x, jets, parameters):
        u, v = jets[zero][:, 0], jets[zero][:, 1]
        return torch.stack((parameters[:, 0]*jets[xx][:, 0]+u*v+jets[dx][:, 1],
                            v**3+jets[xx][:, 1]+x[:, 0]*u), dim=1)
    aggregate = rng.normal(size=(5, len(points))) if aggregation else None
    block = ResidualBlock('coupled', points, callback, (zero, dx, xx),
                          aggregation=aggregate, scale=np.array([1.2, .8]))
    problem = ResidualProblem(features, [block], fields=2, parameter_initial=[.7],
                              execution=execution, batch_size=4)
    coefficients = rng.normal(size=(features.size, 2))*.1
    return problem, coefficients


@pytest.mark.parametrize('kind', ['box', 'disk'])
@pytest.mark.parametrize('execution', ['cached', 'streamed'])
def test_resolved_ideal_gram_approximates_actual_gram(kind, execution):
    problem, coefficients = make_problem(kind, execution)
    linear = linearize_residual(problem, coefficients)
    indices = np.array([0, 3, 7, 14, 19, problem.features.size*2])
    scales = np.linspace(.8, 1.3, len(indices))
    actual = linear.operator.column_gram(indices, scales)
    ideal = linear.operator.approximate_column_gram(indices, scales)
    np.testing.assert_allclose(ideal, actual, atol=2e-11, rtol=2e-10)
    assert linear.metrics['product_counts']['approximate_column_gram_calls'] == 1


@pytest.mark.parametrize('execution', ['cached', 'streamed'])
def test_approximate_gram_uses_actual_nonlinear_state_and_aggregation(execution):
    # Deliberately coarse encoding makes actual and ideal states different.
    # The preconditioner must linearize physics at ACTUAL state regardless.
    problem, coefficients = make_problem('box', execution, centers=9, aggregation=True)
    block = problem.blocks[0]
    linear = linearize_residual(problem, coefficients)
    indices = np.array([0, 3, 9, problem.features.size*2])
    scales = np.array([.7, 1.1, .9, 1.3])
    actual = {order: np.column_stack([problem.features.forward(coefficients[:, field], block.points, order)
                                      for field in range(2)]) for order in block.derivatives}
    cp = indices < problem.features.size*2
    ideal_columns = problem.features.selected_ideal_columns(block.points, indices[cp]//2, block.derivatives)
    points = torch.tensor(block.points)
    def local_model(delta):
        jets = {order: torch.tensor(value) for order, value in actual.items()}
        parameters = torch.full((len(points), 1), .7, dtype=torch.float64)
        selected_position = 0
        for position, index in enumerate(indices):
            if cp[position]:
                for order in block.derivatives:
                    field_mask = torch.zeros(2, dtype=torch.float64); field_mask[index % 2] = 1.
                    jets[order] = jets[order]+torch.tensor(ideal_columns[order][:, selected_position, None])*field_mask*delta[position]*scales[position]
                selected_position += 1
            else:
                parameters = parameters+delta[position]*scales[position]
        residual = block.function(points, jets, parameters)
        residual = torch.tensor(block.aggregation) @ residual
        return residual*torch.tensor(np.sqrt(block.weight/len(block.aggregation))/block.scale)
    jacobian = torch.autograd.functional.jacobian(local_model, torch.zeros(len(indices), dtype=torch.float64)).numpy().reshape(-1, len(indices))
    ideal_gram = linear.operator.approximate_column_gram(indices, scales)
    np.testing.assert_allclose(ideal_gram, jacobian.T@jacobian, atol=3e-12, rtol=4e-13)


def test_ideal_panels_are_never_called_by_residual_jvp_adjoint_or_export(monkeypatch):
    problem, coefficients = make_problem(centers=33)
    features = problem.features
    original = features.selected_ideal_columns
    def forbidden(*args, **kwargs):
        raise AssertionError('Ideal features used outside explicit preconditioner construction')
    monkeypatch.setattr(features, 'selected_ideal_columns', forbidden)
    linear = linearize_residual(problem, coefficients)
    rng = np.random.default_rng(742)
    vector, cotangent = rng.normal(size=linear.operator.shape[1]), rng.normal(size=len(linear.residual))
    forward = linear.operator @ vector
    transpose = linear.operator.rmatvec(cotangent)
    features.compile(coefficients[:, 0])
    # Enable only while assembling the approximate blocks and record sizes.
    calls = []
    def counted(points, indices, derivatives):
        calls.append((len(points), len(indices)))
        return original(points, indices, derivatives)
    monkeypatch.setattr(features, 'selected_ideal_columns', counted)
    monkeypatch.setattr(features, 'selected_columns', forbidden)
    preconditioner = _RightPreconditioner(problem, linear.operator,
        np.ones(linear.operator.shape[1]), block_size=8, basis='ideal')
    assert calls and max(rows for rows, _ in calls) <= problem.batch_size
    assert max(columns for _, columns in calls) <= 8
    assert preconditioner.metrics['basis'] == 'ideal'
    monkeypatch.setattr(features, 'selected_ideal_columns', forbidden)
    np.testing.assert_array_equal(linear.operator@vector, forward)
    np.testing.assert_array_equal(linear.operator.rmatvec(cotangent), transpose)
    features.compile(coefficients[:, 0])


def test_preconditioner_basis_validation():
    problem, _ = make_problem()
    with pytest.raises(ValueError, match='preconditioner_basis'):
        solve_residual(problem, max_iterations=0, preconditioner_basis='hidden')
