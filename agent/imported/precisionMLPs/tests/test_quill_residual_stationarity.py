"""A least-squares remainder is not a failed Krylov solve or a solved PDE."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual


class MatrixFeatures:
    dimension = 1
    metrics = {'test_fixture': 'matrix_operator'}

    def __init__(self, matrix, streamed):
        self.matrix, self.size = matrix, matrix.shape[1]
        self.streamed, self.maximum_rows = streamed, 0

    def rows(self, points):
        self.maximum_rows = max(self.maximum_rows, len(points))
        return self.matrix[np.asarray(points[:, 0], dtype=int)]

    def evaluate(self, points, derivative=None):
        assert not self.streamed, 'Streamed solve must not materialize the full basis'
        return self.rows(points)

    def forward_jets(self, coefficients, points, derivatives):
        values = self.rows(points) @ coefficients
        return {d: values for d in derivatives}

    def forward(self, coefficients, points, derivative=None):
        return self.rows(points) @ coefficients

    def adjoint_jets(self, cotangents, points):
        return self.rows(points).T @ sum(cotangents.values())

    def selected_columns(self, points, indices, derivatives):
        columns = self.rows(points)[:, indices]
        return {d: columns for d in derivatives}


def linear_problem(matrix, target, execution):
    features = MatrixFeatures(matrix, execution == 'streamed')
    values = torch.tensor(target, dtype=torch.float64)
    block = ResidualBlock('linear', np.arange(len(target), dtype=float)[:, None],
        lambda x, jets, p: jets[(0,)][:, 0]-values[x[:, 0].long()], ((0,),))
    return ResidualProblem(features, [block], execution=execution, batch_size=5)


@pytest.mark.parametrize('execution', ['cached', 'streamed'])
def test_compatible_linear_system_reaches_requested_target(execution):
    rng = np.random.default_rng(177)
    matrix = rng.normal(size=(31, 5))
    exact = rng.normal(size=5)
    problem = linear_problem(matrix, matrix @ exact, execution)
    result = solve_residual(problem, high_accuracy=True, tolerance=2e-15,
                            lsmr_max_iterations=100, max_iterations=12)
    assert result.status == 'converged', result.metrics
    assert result.metrics['blocks'][0]['rms'] <= 2e-15
    np.testing.assert_allclose(result.coefficients[:, 0], exact, rtol=0., atol=2e-15)
    if execution == 'streamed':
        assert problem.features.maximum_rows <= 5
        assert result.metrics['basis_cache_bytes'] == 0


@pytest.mark.parametrize('execution', ['cached', 'streamed'])
def test_inconsistent_system_stops_without_futile_damping_retries(execution):
    rng = np.random.default_rng(818)
    orthogonal = np.linalg.qr(rng.normal(size=(29, 6)))[0]
    matrix, remainder = orthogonal[:, :5], 1e-8*orthogonal[:, 5]
    exact = rng.normal(size=5)
    target = matrix @ exact+remainder
    problem = linear_problem(matrix, target, execution)
    result = solve_residual(problem, high_accuracy=True, tolerance=1e-15,
                            lsmr_max_iterations=100, max_iterations=20)
    assert result.status == 'linearized_stationary', result.metrics
    assert not result.metrics['converged']
    assert not result.metrics['optimization_converged']
    assert result.metrics['tolerance'] == 1e-15
    assert not result.metrics['arithmetic']['tolerance_was_relaxed']
    np.testing.assert_allclose(result.coefficients[:, 0], exact, rtol=0., atol=2e-15)
    np.testing.assert_allclose(result.metrics['blocks'][0]['rms'],
                               np.linalg.norm(remainder)/np.sqrt(len(target)), rtol=1e-7)
    final = result.history[-1]
    assert len(final['linear_solves']) == 1
    evidence = final['linearized_stationarity']
    assert evidence['lsmr_completed'] and evidence['orthogonal_to_working_precision']
    assert evidence['no_accepted_objective_step']
    assert final['linear_refinement_stopped_by_orthogonality']
    assert not final['linear_defect_corrections']
    if execution == 'streamed':
        assert problem.features.maximum_rows <= 5
        assert result.metrics['basis_cache_bytes'] == 0


@pytest.mark.parametrize('execution', ['cached', 'streamed'])
def test_iteration_limited_krylov_keeps_useful_directions(execution):
    rng = np.random.default_rng(926)
    matrix = rng.normal(size=(27, 8))*np.geomspace(1., .03, 8)
    target = matrix @ np.arange(1., 9.)
    problem = linear_problem(matrix, target, execution)
    result = solve_residual(problem, high_accuracy=True, tolerance=1e-15,
        lsmr_max_iterations=1, linear_refinement_steps=0, max_iterations=3)
    assert result.status == 'iteration_limit', result.metrics
    assert not result.metrics['converged']
    for record in result.history[:-1]:
        assert record['lsmr_stop'] == 7
        assert not record['linearized_stationarity']['lsmr_completed']
        assert record['trial_objective'] < record['objective']
        assert record['step_norm'] > 0.
    assert result.history[-1]['objective'] < .2*result.history[0]['objective']
    assert result.metrics['lsmr_iterations'] == 3
    if execution == 'streamed':
        assert problem.features.maximum_rows <= 5
        assert result.metrics['basis_cache_bytes'] == 0


@pytest.mark.parametrize('execution', ['cached', 'streamed'])
def test_inexact_inner_schedule_reaches_same_floor_with_less_early_krylov_work(execution):
    rng = np.random.default_rng(818)
    left = np.linalg.qr(rng.normal(size=(61, 12)))[0]
    right = np.linalg.qr(rng.normal(size=(12, 12)))[0]
    matrix = (left*np.geomspace(1., .02, 12)) @ right.T
    exact = rng.normal(size=12)
    values = matrix @ exact
    target = torch.tensor(values+.15*values**3)
    block = ResidualBlock('nonlinear', np.arange(61, dtype=float)[:, None],
        lambda x, j, p: j[(0,)][:, 0]+.15*j[(0,)][:, 0]**3-target[x[:, 0].long()], ((0,),))
    results = []
    for scheduled in (False, True):
        problem = ResidualProblem(MatrixFeatures(matrix, execution == 'streamed'), [block],
                                  execution=execution, batch_size=5)
        results.append(solve_residual(problem, high_accuracy=True, tolerance=2e-15,
            lsmr_max_iterations=300, max_iterations=25, inexact_newton=scheduled,
            linear_refinement_steps=0))
    baseline, scheduled = results
    assert baseline.status == scheduled.status == 'converged'
    assert scheduled.metrics['blocks'][0]['rms'] <= 2e-15
    np.testing.assert_allclose(scheduled.coefficients, baseline.coefficients, atol=3e-14, rtol=3e-14)
    assert scheduled.metrics['lsmr_iterations'] < baseline.metrics['lsmr_iterations']
    first = scheduled.history[0]['linear_solves'][0]
    assert .001 < first['tolerance'] <= .05
    assert not first['accurate_tolerance']
    linear_records = [inner for outer in scheduled.history for inner in outer.get('linear_solves', [])]
    assert linear_records[-1]['accurate_tolerance']
    assert linear_records[-1]['tolerance'] == scheduled.metrics['settings']['effective_lsmr_tolerance']
    assert scheduled.metrics['settings']['inexact_newton']
    assert not scheduled.metrics['arithmetic']['tolerance_was_relaxed']


@pytest.mark.parametrize('execution', ['cached', 'streamed'])
def test_large_irreducible_residual_requires_tight_retry_before_stationarity(execution):
    problem = linear_problem(np.ones((8, 1)), np.arange(8.)-3.5, execution)
    result = solve_residual(problem, high_accuracy=True, tolerance=1e-15,
                            max_iterations=8, inexact_newton=True)
    assert result.status == 'linearized_stationary'
    assert not result.metrics['converged']
    record = result.history[-1]
    assert len(record['linear_solves']) == 2
    assert not record['linear_solves'][0]['accurate_tolerance']
    assert record['linear_solves'][1]['accurate_tolerance']
    assert record['inner_tolerance_retries'][0]['reason'] == 'tiny_coarse_direction'
    assert record['linearized_stationarity']['inner_solve_accurate']
    assert record['linear_refinement_skipped_for_inexact_step']
    assert result.metrics['tolerance'] == 1e-15
    assert result.metrics['blocks'][0]['rms'] > 1.


def test_rejected_coarse_direction_cannot_trigger_precision_limit(monkeypatch):
    import solver.general_residual as module
    original = module.lsmr
    def zero_coarse_direction(operator, rhs, **kwargs):
        if kwargs['atol'] > .001:
            return (np.zeros(operator.shape[1]), 7, 1, np.linalg.norm(rhs),
                    1., 1., 1., 0.)
        return original(operator, rhs, **kwargs)
    monkeypatch.setattr(module, 'lsmr', zero_coarse_direction)
    rng = np.random.default_rng(163)
    matrix = rng.normal(size=(27, 4))
    problem = linear_problem(matrix, matrix @ np.ones(4), 'cached')
    result = solve_residual(problem, high_accuracy=True, tolerance=3e-15,
                            inexact_newton=True, max_iterations=12)
    assert result.status == 'converged'
    record = result.history[0]
    assert record['inner_tolerance_retries'][0]['reason'] == 'tiny_coarse_direction'
    assert record['linear_solves'][1]['accurate_tolerance']


def test_inexact_schedule_disabled_by_default_and_validated():
    matrix = np.array([[1., 0.], [0., 1.], [1., -1.]])
    problem = linear_problem(matrix, matrix @ np.array([.3, -.2]), 'cached')
    default = solve_residual(problem, high_accuracy=True, tolerance=2e-15)
    explicit = solve_residual(problem, high_accuracy=True, tolerance=2e-15, inexact_newton=False)
    np.testing.assert_array_equal(default.coefficients, explicit.coefficients)
    assert default.metrics['lsmr_iterations'] == explicit.metrics['lsmr_iterations']
    assert not default.metrics['settings']['inexact_newton']
    with pytest.raises(ValueError, match='inexact_newton'):
        solve_residual(problem, inexact_newton='auto')


@pytest.mark.parametrize('execution',['cached','streamed'])
def test_accepted_negligible_inexact_step_requests_one_tight_actual_check(monkeypatch,execution):
    import solver.general_residual as module
    original=module.lsmr
    def tiny_accepted_coarse_step(operator,rhs,**kwargs):
        if kwargs['atol']>.001:
            return (np.full(operator.shape[1],1e-6),1,1,np.linalg.norm(rhs),1.,1.,1.,1e-6)
        return original(operator,rhs,**kwargs)
    monkeypatch.setattr(module,'lsmr',tiny_accepted_coarse_step)
    target=.2+np.tile([-1.,1.],8)
    problem=linear_problem(np.ones((len(target),1)),target,execution)
    result=solve_residual(problem,initial_coefficients=np.array([[.2-1e-4]]),
        high_accuracy=True,tolerance=1e-14,inexact_newton=True,
        linear_refinement_steps=0,damping=1e-30,max_iterations=6)
    assert result.status=='linearized_stationary',result.metrics
    assert result.metrics['iterations']==1
    assert not result.metrics['converged'] and not result.metrics['optimization_converged']
    assert result.metrics['blocks'][0]['rms']>.9
    assert result.metrics['tolerance']==1e-14 and not result.metrics['arithmetic']['tolerance_was_relaxed']
    first,check=result.history
    assert first['relative_objective_decrease']>0
    assert first['requested_tight_actual_model_check']=='accepted_negligible_inexact_step'
    assert not first['linear_solves'][0]['accurate_tolerance']
    assert check['tight_actual_model_check']=='previous_accepted_inexact_step_negligible'
    assert check['linearization_basis']=='actual'
    assert all(step['accurate_tolerance'] and step['linearization_basis']=='actual' for step in check['linear_solves'])
    evidence=check['linearized_stationarity']
    assert evidence['actual_jacobian'] and evidence['inner_solve_accurate']
    assert evidence['lsmr_completed'] and evidence['orthogonal_to_working_precision']
    np.testing.assert_allclose(result.coefficients,.2,atol=1e-14,rtol=0.)
