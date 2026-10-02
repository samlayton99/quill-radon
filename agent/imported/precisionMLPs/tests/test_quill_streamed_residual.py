"""Streaming must preserve the cached objective without materializing its basis."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.general_residual import (ResidualBlock, ResidualProblem,
    linearize_residual, residual_vector, solve_residual, _RightPreconditioner)


class OperatorMonomials:
    dimension = 1
    metrics = {'test_fixture': 'operator_monomials'}

    def __init__(self, degree, forbid_dense=False):
        self.size, self.forbid_dense = degree+1, forbid_dense
        self.forward_rows, self.adjoint_rows, self.column_shapes = [], [], []

    def columns(self, points, indices, derivative=None):
        x = np.asarray(points)[:, 0]
        d = 0 if derivative is None else derivative[0]
        return np.column_stack([np.zeros(len(x)) if n < d else
            math.factorial(int(n))/math.factorial(int(n)-d)*x**(n-d) for n in indices])

    def evaluate(self, points, derivative=None):
        if self.forbid_dense:
            raise AssertionError('Dense feature evaluation is forbidden')
        return self.columns(points, np.arange(self.size), derivative)

    def forward(self, c, points, derivative=None):
        self.forward_rows.append(len(points))
        return self.columns(points, np.arange(self.size), derivative) @ c

    def forward_jets(self, c, points, derivatives):
        return {d: self.forward(c, points, d) for d in derivatives}

    def adjoint_jets(self, cotangents, points):
        self.adjoint_rows.append(len(points))
        return sum(self.columns(points, np.arange(self.size), d).T @ values
                   for d, values in cotangents.items())

    def selected_columns(self, points, indices, derivatives):
        self.column_shapes.append((len(points), len(indices)))
        return {d: self.columns(points, indices, d) for d in derivatives}


@pytest.mark.parametrize('aggregate', ['none', 'dense', 'sparse'])
@pytest.mark.parametrize('relation', ['eq', 'le', 'ge'])
def test_streamed_cached_products_fd_adjoint_and_small_gram(aggregate, relation):
    rng = np.random.default_rng(29)
    points = np.linspace(-.9, .9, 13)[:, None]
    def callback(x, jets, p):
        u, v = jets[(0,)].unbind(1)
        ux, vx = jets[(1,)].unbind(1)
        return torch.stack((u*u+.4*v+p[:, 0]*ux+.1,
                            torch.sin(v)+vx+p[:, 1]*u+x[:, 0]), dim=1)
    aggregation = None if aggregate == 'none' else rng.normal(size=(5, len(points)))
    if aggregate == 'sparse':
        aggregation = sparse.csr_matrix(aggregation)
    block = ResidualBlock('mixed', points, callback, ((0,), (1,)), weight=2.3,
                          scale=[.8, 1.7], aggregation=aggregation, relation=relation)
    features = OperatorMonomials(4)
    problem = ResidualProblem(features, [block], fields=2, parameter_initial=[.7, 1.1])
    c = rng.normal(size=(5, 2))*.2
    cached = linearize_residual(problem, c)
    features.forbid_dense = True
    streamed = linearize_residual(problem, c, execution='streamed', batch_size=3)
    v, z = rng.normal(size=12), rng.normal(size=len(cached.residual))
    np.testing.assert_allclose(streamed.residual, cached.residual, atol=3e-15)
    np.testing.assert_allclose(streamed.operator @ v, cached.operator @ v, atol=3e-14)
    np.testing.assert_allclose(streamed.operator.rmatvec(z), cached.operator.rmatvec(z), atol=3e-14)
    np.testing.assert_allclose(z @ (streamed.operator @ v), v @ streamed.operator.rmatvec(z), atol=3e-14)
    h = 1e-6
    dc, dp = v[:10].reshape(5, 2), v[10:]
    finite = (residual_vector(problem, c+h*dc, problem.parameter_initial+h*dp,
                               execution='streamed', batch_size=3)-
              residual_vector(problem, c-h*dc, problem.parameter_initial-h*dp,
                               execution='streamed', batch_size=3))/(2*h)
    np.testing.assert_allclose(streamed.operator @ v, finite, rtol=2e-8, atol=3e-9)
    # Duplicate feature indices from different fields and physical parameters.
    indices, scales = np.array([0, 1, 4, 10, 11]), np.array([1., .4, 2., 3., .7])
    expected = cached.operator.column_gram(indices, scales)
    actual = streamed.operator.column_gram(indices, scales)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    assert max(features.forward_rows+features.adjoint_rows) <= 3
    assert max(rows for rows, cols in features.column_shapes) <= 3
    assert streamed.metrics['basis_cache_bytes'] == 0
    assert streamed.metrics['local_derivative_bytes'] == cached.metrics['local_derivative_bytes']
    assert streamed.metrics['input_point_bytes'] == points.nbytes
    assert streamed.metrics['product_counts']['callback_rows_max'] <= 3


@pytest.mark.parametrize('preconditioner', ['diagonal', 'block'])
def test_streamed_inverse_solve_and_prediction_never_call_dense_features(preconditioner):
    points = np.linspace(0, 1, 19)[:, None]
    features = OperatorMonomials(2, forbid_dense=True)
    blocks = [ResidualBlock('pde', points, lambda x,j,p:-p[:, 0]*j[(2,)][:, 0]-1.4, ((2,),)),
              ResidualBlock('boundary', np.array([[0.], [1.]]), lambda x,j,p:j[(0,)][:, 0], ((0,),), weight=5.),
              ResidualBlock('data', np.array([[.2], [.5], [.8]]),
                  lambda x,j,p:j[(0,)][:, 0]-x[:, 0]*(1-x[:, 0]), ((0,),), weight=5.)]
    problem = ResidualProblem(features, blocks, parameter_initial=[.3],
                              parameter_bounds=([.1], [2.]), execution='streamed', batch_size=4)
    result = solve_residual(problem, tolerance=2e-12, high_accuracy=True,
                            preconditioner=preconditioner, preconditioner_block_size=2)
    assert result.status == 'converged', result.metrics
    np.testing.assert_allclose(result.parameters, [.7], atol=1e-11)
    np.testing.assert_allclose(result.coefficients[:, 0], [0, 1, -1], atol=1e-11)
    test = np.linspace(0, 1, 37)[:, None]
    np.testing.assert_allclose(result.evaluate(test)[:, 0], test[:, 0]*(1-test[:, 0]), atol=1e-11)
    np.testing.assert_allclose(result.evaluate(test, derivative=(1,))[:, 0], 1-2*test[:, 0], atol=1e-11)
    assert result.metrics['execution'] == 'streamed'
    assert max(features.forward_rows+features.adjoint_rows) <= 4
    assert result.metrics['basis_cache_bytes'] == 0
    if preconditioner == 'block':
        assert result.metrics['preconditioner_builds'][0]['column_row_chunk_size'] == 4


def test_execution_override_survives_into_solution_evaluation_and_data_row_indices():
    points = np.linspace(-1, 1, 11)[:, None]
    target = torch.tensor(np.cos(points[:, 0]), dtype=torch.float64)
    def original(x, jets, p):
        return jets[(0,)][:, 0]-target
    def batched(x, jets, p, indices):
        return jets[(0,)][:, 0]-target[indices]
    features = OperatorMonomials(3, forbid_dense=True)
    problem = ResidualProblem(features, [ResidualBlock('data', points, original, ((0,),), batch_function=batched)])
    c = np.zeros((4, 1))
    np.testing.assert_allclose(residual_vector(problem, c, execution='streamed', batch_size=3),
                               -target.numpy()/np.sqrt(len(points)))
    result = solve_residual(problem, execution='streamed', batch_size=3, tolerance=1e-3)
    assert result.evaluate(points).shape == (11, 1)
    assert max(features.forward_rows) <= 3


@pytest.mark.parametrize('batch_size', [1, 3])
def test_row_coupling_rejected_in_streamed_backend(batch_size):
    block = ResidualBlock('bad', np.arange(7.)[:, None],
                          lambda x,j,p:j[(0,)][:, 0].mean().expand(len(x)), ((0,),))
    problem = ResidualProblem(OperatorMonomials(1, True), [block], execution='streamed', batch_size=batch_size)
    with pytest.raises(ValueError, match='couples rows'):
        linearize_residual(problem, np.ones((2, 1)))


def test_streamed_preconditioner_uses_bounded_selected_columns_only():
    features = OperatorMonomials(12, True)
    points = np.linspace(-1, 1, 31)[:, None]
    block = ResidualBlock('data', points, lambda x,j,p:j[(0,)][:, 0], ((0,),))
    problem = ResidualProblem(features, [block], execution='streamed', batch_size=5)
    linear = linearize_residual(problem, np.zeros((features.size, 1)))
    preconditioner = _RightPreconditioner(problem, linear.operator, np.ones(features.size), block_size=3)
    assert preconditioner.metrics['setup_matvec_calls'] == 0
    assert max(rows for rows, cols in features.column_shapes) <= 5
    assert max(cols for rows, cols in features.column_shapes) <= 3
    assert preconditioner.metrics['factor_bytes'] <= 8*features.size*3
    assert linear.metrics['basis_cache_bytes'] == 0


def test_invalid_execution_and_missing_protocol_are_explicit():
    block = ResidualBlock('data', np.ones((2, 1)), lambda x,j,p:j[(0,)][:, 0], ((0,),))
    with pytest.raises(ValueError, match='execution'):
        ResidualProblem(OperatorMonomials(1), [block], execution='silent_dense_fallback')
    with pytest.raises(ValueError, match='batch_size'):
        ResidualProblem(OperatorMonomials(1), [block], batch_size=0)
    features = OperatorMonomials(1)
    features.selected_columns = None
    with pytest.raises(TypeError, match='selected_columns'):
        linearize_residual(ResidualProblem(features, [block], execution='streamed'), np.zeros((2, 1)))


class PreparedMonomials(OperatorMonomials):
    """Prepared-path spy; fallback products fail if invoked by the engine."""
    def __init__(self, degree):
        super().__init__(degree, forbid_dense=True)
        self.prepare_forward_count = self.prepare_adjoint_count = self.finish_adjoint_count = 0

    def prepare_forward(self, coefficients):
        self.prepare_forward_count += 1
        return {'readouts': np.asarray(coefficients).copy()}

    def forward_prepared_jets(self, state, points, derivatives):
        self.forward_rows.append(len(points))
        return {d: self.columns(points, np.arange(self.size), d) @ state['readouts']
                for d in derivatives}

    def prepare_adjoint(self):
        self.prepare_adjoint_count += 1
        return {'neuron_gradient': np.zeros(self.size)}

    def accumulate_adjoint_jets(self, cotangents, points, accumulator):
        self.adjoint_rows.append(len(points))
        accumulator['neuron_gradient'] += sum(
            self.columns(points, np.arange(self.size), d).T @ v for d, v in cotangents.items())

    def finish_adjoint(self, accumulator):
        self.finish_adjoint_count += 1
        return accumulator['neuron_gradient'].copy()

    def forward(self, *args, **kwargs):
        raise AssertionError('Prepared forward path must be reused across batches')

    def forward_jets(self, *args, **kwargs):
        raise AssertionError('Prepared forward path must be reused across batches')

    def adjoint_jets(self, *args, **kwargs):
        raise AssertionError('Prepared adjoint path must accumulate before conversion')


def test_prepared_states_convert_once_per_field_across_batches_and_blocks():
    from solver.general_residual import ResidualSolution
    rng = np.random.default_rng(932)
    x = np.linspace(-.8, .8, 15)[:, None]
    def callback(x, j, p):
        u, v = j[(0,)].unbind(1)
        return torch.stack((u*u+v+p[:, 0]*j[(1,)][:, 0],
                            torch.sin(v)+j[(1,)][:, 1]), dim=1)
    blocks = [ResidualBlock('first', x, callback, ((0,), (1,))),
              ResidualBlock('second', x[::2].copy(), callback, ((0,), (1,)),
                            aggregation=rng.normal(size=(3, len(x[::2]))))]
    c, parameters = rng.normal(size=(4, 2))*.1, np.array([.8])
    baseline = ResidualProblem(OperatorMonomials(3), blocks, fields=2, parameter_initial=parameters)
    expected = linearize_residual(baseline, c)
    features = PreparedMonomials(3)
    problem = ResidualProblem(features, blocks, fields=2, parameter_initial=parameters,
                              execution='streamed', batch_size=4)
    actual = linearize_residual(problem, c)
    assert features.prepare_forward_count == 2
    np.testing.assert_allclose(actual.residual, expected.residual, atol=1e-15)
    v, z = rng.normal(size=9), rng.normal(size=len(actual.residual))
    np.testing.assert_allclose(actual.operator @ v, expected.operator @ v, atol=3e-14)
    assert features.prepare_forward_count == 4
    np.testing.assert_allclose(actual.operator.rmatvec(z), expected.operator.rmatvec(z), atol=3e-14)
    assert features.prepare_adjoint_count == features.finish_adjoint_count == 2
    counts = actual.metrics['product_counts']
    assert counts['prepared_forward_calls'] == 4
    assert counts['prepared_adjoint_calls'] == 2
    assert counts['prepared_forward_bytes_max'] == counts['prepared_adjoint_bytes_max'] == c.nbytes
    assert max(features.forward_rows+features.adjoint_rows) <= 4
    solution = ResidualSolution(problem, c, parameters, {'execution': 'streamed', 'batch_size': 4})
    np.testing.assert_allclose(solution.evaluate(x), baseline.features.evaluate(x) @ c, atol=1e-15)
    assert features.prepare_forward_count == 6
