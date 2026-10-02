"""Operator and nonlinear/inverse regressions for the generic residual engine."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.general_residual import (ResidualBlock, ResidualProblem,
                                     linearize_residual, residual_vector, solve_residual,
                                     _RightPreconditioner)
from solver.general_features import ConstructedFeatures


class Monomials:
    """Independent tiny basis keeps operator tests independent of QUILL code."""
    dimension = 1
    metrics = {'test_fixture': 'monomials'}

    def __init__(self, degree):
        self.size = degree+1

    def evaluate(self, points, derivative=None):
        x = np.asarray(points)[:, 0]
        d = 0 if derivative is None else derivative[0]
        return np.column_stack([np.zeros(len(x)) if n < d else
                                math.factorial(n)/math.factorial(n-d)*x**(n-d)
                                for n in range(self.size)])


@pytest.mark.parametrize('aggregate', ['identity', 'dense', 'sparse'])
def test_matrix_free_adjoint_and_directional_derivative(aggregate):
    rng = np.random.default_rng(924)
    points = np.linspace(-.9, .9, 9)[:, None]
    def callback(x, jet, p):
        u, v = jet[(0,)].unbind(1)
        ux, vx = jet[(1,)].unbind(1)
        return torch.stack((u*u+p[:, 0]*ux+v, vx+p[:, 1]*torch.sin(u)+x[:, 0]), dim=1)
    matrix = None if aggregate == 'identity' else rng.normal(size=(4, len(points)))
    if aggregate == 'sparse':
        matrix = sparse.csr_matrix(matrix)
    block = ResidualBlock('mixed', points, callback, ((0,), (1,)),
                          weight=2.3, scale=[.7, 1.4], aggregation=matrix)
    problem = ResidualProblem(Monomials(3), [block], fields=2, parameter_initial=[.7, 1.1])
    c = rng.normal(size=(4, 2))*.2
    p = np.array([.7, 1.1])
    linear = linearize_residual(problem, c, p)
    direction = rng.normal(size=10)
    cotangent = rng.normal(size=len(linear.residual))
    left = np.dot(linear.operator @ direction, cotangent)
    right = np.dot(direction, linear.operator.rmatvec(cotangent))
    np.testing.assert_allclose(left, right, rtol=3e-13, atol=3e-13)
    h = 1e-6
    dc, dp = direction[:8].reshape(4, 2), direction[8:]
    finite_difference = (residual_vector(problem, c+h*dc, p+h*dp)-
                         residual_vector(problem, c-h*dc, p-h*dp))/(2*h)
    np.testing.assert_allclose(linear.operator @ direction, finite_difference, rtol=3e-8, atol=3e-9)
    assert linear.metrics['local_derivative_bytes'] > 0
    assert linear.metrics['unassembled_dense_jacobian_bytes'] == 8*linear.operator.shape[0]*linear.operator.shape[1]


def test_nonlinear_actual_quill_field():
    features = ConstructedFeatures([[-1., 1.]], degree=3, backend='quill', max_derivative=2)
    points = np.linspace(-.96, .96, 25)[:, None]
    def equation(x, jet, parameters):
        exact = 1+.2*x[:, 0]+.1*x[:, 0]**2
        return -jet[(2,)][:, 0]+jet[(0,)][:, 0]**3-(-.2+exact**3)
    def boundary(x, jet, parameters):
        return jet[(0,)][:, 0]-(1+.2*x[:, 0]+.1*x[:, 0]**2)
    problem = ResidualProblem(features, [
        ResidualBlock('equation', points, equation, ((0,), (2,))),
        ResidualBlock('boundary', np.array([[-1.], [1.]]), boundary, ((0,),), weight=4.)])
    result = solve_residual(problem, tolerance=2e-9, max_iterations=25)
    assert result.status == 'converged', result.metrics
    test_points = np.linspace(-1, 1, 101)[:, None]
    np.testing.assert_allclose(result.evaluate(test_points)[:, 0],
                               1+.2*test_points[:, 0]+.1*test_points[:, 0]**2, atol=2e-8)
    assert result.metrics['lsmr_iterations'] > 0
    assert result.metrics['basis_cache_bytes'] > 0


@pytest.mark.parametrize('preconditioner', ['diagonal', 'block'])
def test_inverse_scalar_parameter_with_observation_and_bounds(preconditioner):
    points = np.linspace(0, 1, 13)[:, None]
    def equation(x, jet, p):
        return -p[:, 0]*jet[(2,)][:, 0]-1.4
    def boundary(x, jet, p):
        return jet[(0,)][:, 0]
    def observations(x, jet, p):
        return jet[(0,)][:, 0]-x[:, 0]*(1-x[:, 0])
    problem = ResidualProblem(Monomials(2), [
        ResidualBlock('equation', points, equation, ((2,),)),
        ResidualBlock('boundary', np.array([[0.], [1.]]), boundary, ((0,),), weight=5.),
        ResidualBlock('observations', np.array([[.2], [.5], [.8]]), observations, ((0,),), weight=5.)],
        parameter_initial=[.3], parameter_bounds=([.1], [2.]))
    result = solve_residual(problem, tolerance=1e-9, preconditioner=preconditioner)
    assert result.status == 'converged', result.metrics
    np.testing.assert_allclose(result.parameters, [.7], atol=2e-8)
    np.testing.assert_allclose(result.coefficients[:, 0], [0, 1, -1], atol=2e-8)


def test_integral_gauge_is_an_explicit_aggregated_block():
    points = np.linspace(-1, 1, 9)[:, None]
    blocks = [ResidualBlock('slope', points, lambda x,j,p:j[(1,)][:, 0]-2., ((1,),)),
              ResidualBlock('mean', points, lambda x,j,p:j[(0,)][:, 0]-1., ((0,),),
                            aggregation=np.ones((1, len(points)))/len(points))]
    result = solve_residual(ResidualProblem(Monomials(1), blocks), tolerance=1e-9)
    assert result.status == 'converged'
    np.testing.assert_allclose(result.coefficients[:, 0], [1, 2], atol=2e-9)


def test_nonlocal_callback_requires_explicit_aggregation():
    points = np.linspace(-1, 1, 7)[:, None]
    block = ResidualBlock('illegal_mean', points,
                          lambda x,j,p:j[(0,)][:, 0].mean().expand(len(x)), ((0,),))
    with pytest.raises(ValueError, match='couples rows'):
        linearize_residual(ResidualProblem(Monomials(1), [block]), np.ones((2, 1)))


@pytest.mark.parametrize('relation', ['le', 'ge'])
def test_aggregated_inequality_active_mask_adjoint_and_fd(relation):
    points = np.linspace(-1, 1, 9)[:, None]
    aggregation = np.stack((np.ones(9)/9, -np.ones(9)/9, points[:, 0]/9))
    block = ResidualBlock('weak_inequality', points,
                          lambda x,j,p:j[(0,)][:, 0]**2-.2, ((0,),),
                          aggregation=aggregation, relation=relation)
    problem = ResidualProblem(Monomials(2), [block])
    coefficients = np.array([[1.1], [.2], [-.1]])
    linear = linearize_residual(problem, coefficients)
    assert 0 < linear.block_metrics[0]['active_entries'] < 3
    direction, cotangent = np.array([.3, -.4, .2]), np.array([.7, -.2, .4])
    np.testing.assert_allclose((linear.operator @ direction) @ cotangent,
                               direction @ linear.operator.rmatvec(cotangent), atol=2e-13)
    h = 1e-6
    finite_difference = (residual_vector(problem, coefficients+h*direction[:, None])-
                         residual_vector(problem, coefficients-h*direction[:, None]))/(2*h)
    np.testing.assert_allclose(linear.operator @ direction, finite_difference, atol=2e-9)


def test_inequality_feasibility_and_incompatible_soft_constraint_optimum():
    points = np.array([[0.]])
    inequality = ResidualBlock('upper_bound', points, lambda x,j,p:j[(0,)][:, 0]-1,
                               ((0,),), relation='le', weight=4.)
    feasible = solve_residual(ResidualProblem(Monomials(0), [inequality]),
                              initial_coefficients=np.array([[2.]]), tolerance=1e-9)
    assert feasible.status == 'converged'
    assert feasible.coefficients[0, 0] <= 1+1e-9
    fit = ResidualBlock('incompatible_fit', points, lambda x,j,p:j[(0,)][:, 0]-2, ((0,),))
    conflicting = solve_residual(ResidualProblem(Monomials(0), [fit, inequality]), tolerance=1e-10)
    assert conflicting.status == 'stationary'
    assert not conflicting.metrics['converged']
    np.testing.assert_allclose(conflicting.coefficients[0, 0], 1.2, atol=1e-8)
    assert conflicting.metrics['blocks'][1]['maximum_scaled_rms'] > .19
    assert conflicting.metrics['product_counts']['matvec_calls'] > 0
    assert conflicting.metrics['product_counts']['rmatvec_calls'] > 0


def test_iteration_budget_and_nonfinite_statuses_are_not_success():
    points = np.linspace(-1, 1, 7)[:, None]
    block = ResidualBlock('fit', points, lambda x,j,p:j[(0,)][:, 0]-1., ((0,),))
    problem = ResidualProblem(Monomials(1), [block])
    assert solve_residual(problem, max_iterations=0).status == 'iteration_limit'
    assert solve_residual(problem, max_residual_evaluations=1).status == 'budget_exhausted'
    accepted_then_budget = solve_residual(problem, max_residual_evaluations=2)
    assert accepted_then_budget.status == 'budget_exhausted'
    actual = residual_vector(problem, accepted_then_budget.coefficients)
    np.testing.assert_allclose(accepted_then_budget.metrics['objective'], .5*(actual @ actual), atol=1e-20)
    bad = ResidualBlock('nonfinite', points, lambda x,j,p:torch.log(j[(0,)][:, 0]), ((0,),))
    assert solve_residual(ResidualProblem(Monomials(1), [bad])).status == 'nonfinite_residual'


def test_block_preconditioner_transpose_and_bounded_factors():
    rng = np.random.default_rng(63)
    points = rng.uniform(-1, 1, (61, 2))
    features = ConstructedFeatures([[-1, 1], [-1, 1]], 5, backend='polynomial')
    block = ResidualBlock('values', points, lambda x,j,p:j[(0,0)][:, 0], ((0,0),))
    problem = ResidualProblem(features, [block])
    linear = linearize_residual(problem, np.zeros((features.size, 1)))
    pre = _RightPreconditioner(problem, linear.operator, np.ones(features.size), block_size=5)
    u, v = rng.normal(size=features.size), rng.normal(size=features.size)
    np.testing.assert_allclose(pre.apply(u) @ v, u @ pre.transpose(v), rtol=2e-13, atol=2e-13)
    assert pre.metrics['maximum_block_size'] <= 5
    assert pre.metrics['factor_bytes'] < features.size**2*8


def test_block_preconditioning_fourth_derivative_problem():
    features = ConstructedFeatures([[-1, 1]], 12, backend='polynomial', max_derivative=4)
    x = np.linspace(-.99, .99, 81)[:, None]
    ends = np.array([[-1.], [1.]])
    problem = ResidualProblem(features, [
        ResidualBlock('fourth_derivative', x, lambda x,j,p:j[(4,)][:, 0]-24., ((4,),)),
        ResidualBlock('endpoint_value', ends, lambda x,j,p:j[(0,)][:, 0], ((0,),)),
        ResidualBlock('endpoint_slope', ends, lambda x,j,p:j[(1,)][:, 0], ((1,),))])
    result = solve_residual(problem, preconditioner='block', tolerance=1e-8, max_iterations=20)
    assert result.status == 'converged', result.metrics
    np.testing.assert_allclose(result.evaluate(x)[:, 0], (1-x[:, 0]**2)**2, atol=1e-8)
    assert result.metrics['preconditioner_setup_seconds'] > 0
    assert result.metrics['product_counts']['column_gram_calls'] > 0
    assert result.metrics['preconditioner_builds'][0]['setup_matvec_calls'] == 0


def test_auto_switches_on_expensive_krylov_work_without_equation_branches():
    features = ConstructedFeatures([[-1, 1]], 12, backend='polynomial', max_derivative=4)
    x = np.linspace(-.99, .99, 81)[:, None]
    ends = np.array([[-1.], [1.]])
    problem = ResidualProblem(features, [
        ResidualBlock('interior', x, lambda x,j,p:j[(4,)][:, 0]-24., ((4,),)),
        ResidualBlock('endpoint_value', ends, lambda x,j,p:j[(0,)][:, 0], ((0,),)),
        ResidualBlock('endpoint_slope', ends, lambda x,j,p:j[(1,)][:, 0], ((1,),))])
    result = solve_residual(problem, preconditioner='auto', lsmr_max_iterations=8,
                            tolerance=1e-8, max_iterations=20)
    assert result.status == 'converged', result.metrics
    assert result.metrics['active_preconditioner'] == 'block'
    assert result.metrics['preconditioner_switches'][0]['reason'] == 'krylov_iteration_cost'
    np.testing.assert_allclose(result.evaluate(x)[:, 0], (1-x[:, 0]**2)**2, atol=1e-8)


def test_higher_derivative_order_is_delegated_to_feature_backend():
    points = np.linspace(-1, 1, 5)[:, None]
    block = ResidualBlock('fifth', points, lambda x,j,p:j[(5,)][:, 0]-120., ((5.,),))
    problem = ResidualProblem(Monomials(5), [block])
    result = solve_residual(problem, tolerance=1e-8)
    assert result.status == 'converged'
    np.testing.assert_allclose(result.evaluate(points, (5,)), 120., atol=1e-8)


def test_high_accuracy_accepts_small_coefficient_correction_with_large_field_effect():
    class ScaledFeatures:
        dimension, size = 1, 2

        def evaluate(self, points, derivative=None):
            return np.column_stack((np.ones(len(points)), 1e10*points[:, 0]))

    points = np.linspace(-1, 1, 39)[:, None]
    block = ResidualBlock('values', points,
                          lambda x,j,p:j[(0,)][:, 0]-(1+1e-8*x[:, 0]), ((0,),))
    problem = ResidualProblem(ScaledFeatures(), [block])
    initial = np.array([[1.], [0.]])
    baseline = solve_residual(problem, initial_coefficients=initial, tolerance=1e-15)
    precise = solve_residual(problem, initial_coefficients=initial, tolerance=1e-15,
                             high_accuracy=True)
    assert baseline.status == 'stagnated'
    assert baseline.metrics['blocks'][0]['rms'] > 1e-9
    assert precise.status == 'converged', precise.metrics
    np.testing.assert_allclose(precise.coefficients[1, 0], 1e-18, rtol=2e-8, atol=0.)
    assert precise.metrics['blocks'][0]['rms'] < 1e-15


def test_high_accuracy_recomputes_linear_defects_in_ill_conditioned_space():
    rng = np.random.default_rng(104)
    left = np.linalg.qr(rng.normal(size=(60, 14)))[0]
    right = np.linalg.qr(rng.normal(size=(14, 14)))[0]
    matrix = (left*np.geomspace(1., 1e-7, 14)) @ right.T
    exact = rng.normal(size=14)
    target = torch.tensor(matrix @ exact)

    class MatrixFeatures:
        dimension, size = 1, 14

        def evaluate(self, points, derivative=None):
            return matrix[np.asarray(points[:, 0], int)]

    block = ResidualBlock('linear', np.arange(60.)[:, None],
                          lambda x,j,p:j[(0,)][:, 0]-target, ((0,),))
    problem = ResidualProblem(MatrixFeatures(), [block])
    baseline = solve_residual(problem, tolerance=1e-15, lsmr_max_iterations=500)
    precise = solve_residual(problem, tolerance=1e-15, lsmr_max_iterations=500,
                             high_accuracy=True)
    assert baseline.metrics['blocks'][0]['rms'] > 1e-10
    assert precise.status == 'converged', precise.metrics
    assert precise.metrics['blocks'][0]['rms'] < 1e-15
    # A tiny residual cannot certify coefficients in ill-conditioned spaces.
    assert np.linalg.norm(precise.coefficients[:, 0]-exact) < 2e-7
    assert any(item.get('linear_defect_corrections') for item in precise.history)
    assert precise.metrics['settings']['lsmr_condition_limit'] == 0.
    assert not precise.metrics['arithmetic']['tolerance_was_relaxed']
    linear_iterations = sum(
        sum(s['iterations'] for s in item.get('linear_solves', []))+
        sum(s['iterations'] for s in item.get('linear_defect_corrections', []))
        for item in precise.history)
    assert precise.metrics['lsmr_iterations'] == linear_iterations


def test_high_accuracy_nonlinear_root_and_unattainable_target_are_distinguished():
    points = np.linspace(-1, 1, 41)[:, None]
    block = ResidualBlock('cubic', points,
                          lambda x,j,p:j[(0,)][:, 0]**3-(1+.1*x[:, 0])**3, ((0,),))
    result = solve_residual(ResidualProblem(Monomials(1), [block]),
                            initial_coefficients=np.array([[.8], [0.]]),
                            tolerance=5e-16, high_accuracy=True)
    assert result.status == 'converged', result.metrics
    assert result.metrics['blocks'][0]['rms'] <= 5e-16
    impossible = ResidualBlock('unrepresentable', points,
                               lambda x,j,p:j[(0,)][:, 0]-x[:, 0], ((0,),))
    failed = solve_residual(ResidualProblem(Monomials(0), [impossible]),
                            tolerance=1e-20, high_accuracy=True)
    assert not failed.metrics['converged']
    assert failed.metrics['blocks'][0]['rms'] > .5
    assert failed.metrics['arithmetic']['requested_tolerance_below_machine_epsilon']
    assert failed.metrics['tolerance'] == 1e-20


@pytest.mark.parametrize('options', [dict(high_accuracy='yes'),
                                   dict(lsmr_condition_limit=-1.),
                                   dict(linear_refinement_steps=-1)])
def test_invalid_precision_controls_rejected(options):
    points = np.zeros((1, 1))
    problem = ResidualProblem(Monomials(0), [ResidualBlock(
        'fit', points, lambda x,j,p:j[(0,)][:, 0]-1, ((0,),))])
    with pytest.raises(ValueError):
        solve_residual(problem, **options)


@pytest.mark.parametrize('aggregate', ['identity', 'dense', 'sparse'])
@pytest.mark.parametrize('relation', ['eq', 'le'])
def test_local_column_gram_equals_operator_products_with_row_chunks(aggregate, relation):
    rng = np.random.default_rng(218)
    points = np.linspace(-1, 1, 17)[:, None]
    def callback(x, jet, p):
        u, v = jet[(0,)].unbind(1)
        ux, vx = jet[(1,)].unbind(1)
        return torch.stack((u*u+p[:, 0]*ux+v, vx+p[:, 1]*u-.4), dim=1)
    aggregation = None if aggregate == 'identity' else rng.normal(size=(11, len(points)))
    if aggregate == 'sparse':
        aggregation = sparse.csr_matrix(aggregation)
    block = ResidualBlock('mixed', points, callback, ((0,), (1,)),
                          scale=[.4, 2.], weight=1.3, aggregation=aggregation, relation=relation)
    problem = ResidualProblem(Monomials(3), [block], fields=2, parameter_initial=[.7, 1.2])
    linear = linearize_residual(problem, rng.normal(size=(4, 2)))
    indices, scales = np.array([9, 1, 6, 8, 3]), rng.uniform(.2, 2., 5)
    columns = []
    for index, scale in zip(indices, scales):
        delta = np.zeros(10)
        delta[index] = scale
        columns.append(linear.operator.matvec(delta))
    panel = np.column_stack(columns)
    gram = linear.operator.column_gram(indices, scales, chunk_rows=4)
    np.testing.assert_allclose(gram, panel.T @ panel, rtol=2e-14, atol=2e-13)
    assert linear.metrics['product_counts']['column_panel_bytes_max'] <= 4*2*5*8


def test_iteration_callback_receives_independent_current_state():
    points = np.linspace(-1, 1, 7)[:, None]
    problem = ResidualProblem(Monomials(1), [ResidualBlock(
        'fit', points, lambda x,j,p:j[(0,)][:, 0]-(1+.2*x[:, 0]), ((0,),))])
    seen = []
    def checkpoint(coefficients, parameters, record):
        seen.append((coefficients.copy(), record.copy()))
        coefficients[:] = 999.
        record['objective'] = 999.
    result = solve_residual(problem, high_accuracy=True, tolerance=1e-14,
                            iteration_callback=checkpoint)
    assert result.status == 'converged', result.metrics
    assert len(seen) == len(result.history)
    np.testing.assert_array_equal(seen[-1][0], result.coefficients)
    assert all(item['objective'] != 999. for item in result.history)
    np.testing.assert_allclose(result.coefficients[:, 0], [1., .2], atol=1e-14)


def test_completed_krylov_direction_is_tried_after_soft_time_budget(monkeypatch):
    import solver.general_residual as engine_module
    clock = [0.]
    calls = []
    monkeypatch.setattr(engine_module.time, 'perf_counter', lambda:clock[0])
    def slow_lsmr(operator, rhs, **kwargs):
        calls.append(kwargs)
        # This one-column problem has A=1 and target coefficient1.
        clock[0] = 2.
        return (np.ones(operator.shape[1]), 7, 1, 0., 0., 1., 1., 1.)
    monkeypatch.setattr(engine_module, 'lsmr', slow_lsmr)
    problem = ResidualProblem(Monomials(0), [ResidualBlock(
        'fit', np.zeros((1, 1)), lambda x,j,p:j[(0,)][:, 0]-1., ((0,),))])
    result = solve_residual(problem, high_accuracy=True, preconditioner='auto',
                            max_seconds=1., max_residual_evaluations=2)
    assert result.status == 'budget_exhausted'
    np.testing.assert_array_equal(result.coefficients, [[1.]])
    assert result.metrics['objective'] == 0.
    assert result.metrics['blocks'][0]['rms'] == 0.
    assert len(calls) == 1  # No extra refinement or expensive auto block build.
    assert not result.metrics['preconditioner_builds']
    assert result.metrics['residual_evaluations'] == 2
    assert result.metrics['budget']['time_overshoot_seconds'] == 1.
    assert result.metrics['budget']['completed_direction_trials_after_time_budget'] == 1
    assert not result.metrics['budget']['residual_evaluation_limit_exceeded']
    assert result.history[0]['linear_refinement_stopped_by_budget']


def test_soft_budget_never_overrides_hard_residual_evaluation_limit(monkeypatch):
    import solver.general_residual as engine_module
    def unexpected_lsmr(*args, **kwargs):
        raise AssertionError('No solve is useful when no objective evaluation remains')
    monkeypatch.setattr(engine_module, 'lsmr', unexpected_lsmr)
    problem = ResidualProblem(Monomials(0), [ResidualBlock(
        'fit', np.zeros((1, 1)), lambda x,j,p:j[(0,)][:, 0]-1., ((0,),))])
    result = solve_residual(problem, high_accuracy=True, max_seconds=1., max_residual_evaluations=1)
    assert result.status == 'budget_exhausted'
    np.testing.assert_array_equal(result.coefficients, [[0.]])
    assert result.metrics['residual_evaluations'] == 1
    assert result.metrics['budget']['completed_direction_trials_after_time_budget'] == 0


@pytest.mark.parametrize('high_accuracy, requested, expected', [
    (False, None, 0), (True, None, 2), (True, 0, 0), (True, 1, 1)])
def test_explicit_linear_refinement_count_overrides_accuracy_default(
        monkeypatch, high_accuracy, requested, expected):
    import solver.general_residual as engine_module
    calls = []
    def incomplete_lsmr(operator, rhs, **kwargs):
        calls.append(kwargs)
        # Leave a correctable defect after every call to exercise the loop.
        direction = .5*operator.rmatvec(rhs)
        return (direction, 7, 1, 1., 1., 1., 1., float(np.linalg.norm(direction)))
    monkeypatch.setattr(engine_module, 'lsmr', incomplete_lsmr)
    problem = ResidualProblem(Monomials(0), [ResidualBlock(
        'fit', np.zeros((1, 1)), lambda x,j,p:j[(0,)][:, 0]-1., ((0,),))])
    result = solve_residual(problem, high_accuracy=high_accuracy,
                            linear_refinement_steps=requested, max_iterations=1)
    assert result.metrics['settings']['linear_refinement_steps'] == expected
    assert len(calls) == 1+expected
    assert len(result.history[0].get('linear_defect_corrections', [])) == expected
