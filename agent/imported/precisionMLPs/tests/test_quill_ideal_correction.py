"""Approximate correction operators never replace actual neural residuals."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_box_profiles import BoxProfileOperator
from route2_disk_profiles import DiskProfileOperator
from solver.general_residual import ResidualBlock, ResidualProblem, linearize_residual, solve_residual


def coupled_problem(kind, execution):
    features = (BoxProfileOperator([[-1., 1.]]*2, 4, centers=65)
                if kind == 'box' else DiskProfileOperator(4, centers=65))
    rng = np.random.default_rng(551)
    points = rng.uniform(-.5, .5, size=(13, 2))
    zero, dx, xx = (0, 0), (1, 0), (2, 0)
    def callback(x, j, p):
        u, v = j[zero][:, 0], j[zero][:, 1]
        return torch.stack((p[:, 0]*j[xx][:, 0]+u*v+j[dx][:, 1],
                            v**3+j[xx][:, 1]+x[:, 0]*u), dim=1)
    block = ResidualBlock('coupled', points, callback, (zero, dx, xx),
        aggregation=rng.normal(size=(7, len(points))), scale=[1.2, .8])
    return ResidualProblem(features, [block], fields=2, parameter_initial=[.7],
                           execution=execution, batch_size=4), rng.normal(size=(features.size, 2))*.1


@pytest.mark.parametrize('kind', ['box', 'disk'])
@pytest.mark.parametrize('execution', ['cached', 'streamed'])
def test_ideal_correction_adjoint_gram_and_actual_parameter_columns(kind, execution):
    problem, coefficients = coupled_problem(kind, execution)
    linear = linearize_residual(problem, coefficients)
    ideal = linear.operator.ideal_operator
    rng = np.random.default_rng(572)
    vector, cotangent = rng.normal(size=ideal.shape[1]), rng.normal(size=ideal.shape[0])
    np.testing.assert_allclose(cotangent@(ideal@vector), vector@ideal.rmatvec(cotangent),
                               atol=3e-12, rtol=3e-13)
    indices = np.array([0, 3, 7, 14, ideal.shape[1]-1])
    scales = np.linspace(.8, 1.3, len(indices))
    columns = []
    for index, scale in zip(indices, scales):
        unit = np.zeros(ideal.shape[1]); unit[index] = scale
        columns.append(ideal@unit)
    columns = np.column_stack(columns)
    np.testing.assert_allclose(columns.T@columns, linear.operator.approximate_column_gram(indices, scales),
                               atol=3e-11, rtol=4e-13)
    parameter = np.zeros(ideal.shape[1]); parameter[-1] = 1.
    np.testing.assert_allclose(ideal@parameter, linear.operator@parameter, atol=2e-15, rtol=5e-15)


def test_ideal_inner_products_bound_both_axes_and_never_evaluate_neurons(monkeypatch):
    features = BoxProfileOperator([[-1., 1.]]*2, 12, centers=33)
    points = np.random.default_rng(91).uniform(-.4, .4, size=(11, 2))
    block = ResidualBlock('reaction', points, lambda x, j, p: j[(0, 0)][:, 0]**3+j[(2, 0)][:, 0],
                          ((0, 0), (2, 0)))
    problem = ResidualProblem(features, [block], execution='streamed', batch_size=4)
    monkeypatch.setattr(features,'ideal_tensor_plan',lambda *a,**k:dict(enabled=False))
    coefficients = np.random.default_rng(79).normal(size=(features.size, 1))*.01
    linear = linearize_residual(problem, coefficients)
    def forbidden(*args, **kwargs):
        raise AssertionError('Actual neuron work inside ideal correction products')
    monkeypatch.setattr(features.base, '_basis_many', forbidden)
    monkeypatch.setattr(features, 'evaluate_many', forbidden)
    calls = []
    original = features.selected_ideal_columns
    def observed(points, indices, derivatives, **kwargs):
        calls.append((len(points), len(indices)))
        return original(points, indices, derivatives, **kwargs)
    monkeypatch.setattr(features, 'selected_ideal_columns', observed)
    ideal = linear.operator.ideal_operator
    assert np.all(np.isfinite(ideal@np.ones(ideal.shape[1])))
    assert np.all(np.isfinite(ideal.rmatvec(np.ones(ideal.shape[0]))))
    assert max(rows for rows, _ in calls) <= 4
    assert max(columns for _, columns in calls) == 64 < features.size
    assert linear.metrics['basis_cache_bytes'] == 0
    assert linear.metrics['product_counts']['ideal_correction_columns_max'] == 64


def simple_reaction_problem(execution='streamed'):
    features = BoxProfileOperator([[-1., 1.]]*2, 2, centers=129)
    points = np.random.default_rng(793).uniform(-.8, .8, size=(37, 2))
    def callback(x, jets, parameters):
        target = .3+.2*x[:, 0]-.1*x[:, 1]+.08*x[:, 0]*x[:, 1]
        value = jets[(0, 0)][:, 0]
        return value+.2*value**3-(target+.2*target**3)
    block = ResidualBlock('reaction', points, callback, ((0, 0),))
    return ResidualProblem(features, [block], execution=execution, batch_size=8)


@pytest.mark.parametrize('execution', ['cached', 'streamed'])
def test_ideal_correction_converges_against_actual_neural_residual(execution):
    problem = simple_reaction_problem(execution)
    result = solve_residual(problem, high_accuracy=True, tolerance=2e-14,
        linearization_basis='ideal', preconditioner='block', preconditioner_basis='ideal',
        linear_refinement_steps=0, max_iterations=15)
    assert result.status == 'converged', result.metrics
    actual = problem.features.forward(result.coefficients[:, 0], problem.blocks[0].points)
    residual = problem.blocks[0].function(torch.tensor(problem.blocks[0].points),
        {(0, 0): torch.tensor(actual[:, None])}, torch.empty((len(actual), 0))).numpy()
    assert np.sqrt(np.mean(residual**2)) <= 2e-14
    assert result.metrics['product_counts']['ideal_matvec_calls'] > 0
    assert result.metrics['product_counts']['rmatvec_calls'] > 0  # Actual objective gradient.
    assert result.metrics['settings']['linearization_basis'] == 'ideal'
    assert all(inner['linearization_basis'] in ('actual', 'ideal')
               for row in result.history for inner in row.get('linear_solves', []))


@pytest.mark.parametrize('failure', ['zero', 'nonfinite'])
def test_invalid_or_stalled_ideal_correction_falls_back_before_precision_claims(monkeypatch, failure):
    problem = simple_reaction_problem()
    monkeypatch.setattr(problem.features,'ideal_tensor_plan',lambda *a,**k:dict(enabled=False))
    original = problem.features.selected_ideal_columns
    def broken(points, indices, derivatives, **kwargs):
        result = original(points, indices, derivatives, **kwargs)
        return {d: np.full_like(v, 0. if failure == 'zero' else np.nan) for d, v in result.items()}
    monkeypatch.setattr(problem.features, 'selected_ideal_columns', broken)
    result = solve_residual(problem, high_accuracy=True, tolerance=2e-14,
        linearization_basis='ideal', preconditioner='diagonal', linear_refinement_steps=0,
        max_iterations=15)
    assert result.status == 'converged', result.metrics
    assert any(row.get('linearization_fallbacks') for row in result.history)
    assert any(inner['linearization_basis'] == 'actual'
               for row in result.history for inner in row.get('linear_solves', []))
    for row in result.history:
        if row.get('linearized_stationarity', {}).get('no_accepted_objective_step'):
            assert row['linearized_stationarity']['actual_jacobian']


def test_actual_linearization_default_is_unchanged_and_option_validated():
    problem = simple_reaction_problem()
    first = solve_residual(problem, high_accuracy=True, tolerance=2e-14)
    second = solve_residual(problem, high_accuracy=True, tolerance=2e-14, linearization_basis='actual')
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    with pytest.raises(ValueError, match='linearization_basis'):
        solve_residual(problem, linearization_basis='secret')


def test_accepted_but_negligible_ideal_step_switches_to_actual_model(monkeypatch):
    problem = simple_reaction_problem()
    monkeypatch.setattr(problem.features,'ideal_tensor_plan',lambda *a,**k:dict(enabled=False))
    original = problem.features.selected_ideal_columns
    def too_large(points, indices, derivatives, **kwargs):
        return {d: 1e10*v for d, v in original(points, indices, derivatives, **kwargs).items()}
    monkeypatch.setattr(problem.features, 'selected_ideal_columns', too_large)
    result = solve_residual(problem, high_accuracy=True, tolerance=2e-14,
        linearization_basis='ideal', preconditioner='diagonal', linear_refinement_steps=0,
        max_iterations=10)
    assert result.status == 'converged', result.metrics
    assert result.history[0]['linearization_fallbacks'] == ['accepted_negligible_ideal_step']
    assert result.history[1]['linearization_basis'] == 'actual'
    assert all(inner['linearization_basis'] == 'actual'
               for row in result.history[1:] for inner in row.get('linear_solves', []))


@pytest.mark.parametrize('multiplier', [0., 1e10])
def test_optional_ideal_stall_requests_refinement_without_a_false_certificate(monkeypatch,multiplier):
    problem=simple_reaction_problem()
    monkeypatch.setattr(problem.features,'ideal_tensor_plan',lambda *a,**k:dict(enabled=False))
    original=problem.features.selected_ideal_columns
    def inaccurate(points,indices,derivatives,**kwargs):
        return {d:multiplier*v for d,v in original(points,indices,derivatives,**kwargs).items()}
    monkeypatch.setattr(problem.features,'selected_ideal_columns',inaccurate)
    result=solve_residual(problem,high_accuracy=True,tolerance=2e-14,
        linearization_basis='ideal',preconditioner='diagonal',linear_refinement_steps=0,
        refine_on_ideal_stall=True,max_iterations=10)
    assert result.status=='approximate_model_stalled'
    assert not result.metrics['converged']
    assert not result.metrics['optimization_converged']
    assert max(b['maximum_scaled_rms'] for b in result.metrics['blocks'])>2e-14
    assert result.history[-1]['uncertified_refinement_request']
    assert all(inner['linearization_basis']=='ideal'
               for row in result.history for inner in row.get('linear_solves',[]))
    # Returning early preserves only an actually accepted neural state.
    final=linearize_residual(problem,result.coefficients).residual
    initial=linearize_residual(problem,np.zeros_like(result.coefficients)).residual
    assert np.linalg.norm(final)<=np.linalg.norm(initial)
    np.testing.assert_allclose(.5*float(final@final),result.metrics['objective'],rtol=2e-14)


def test_ideal_refinement_option_validation():
    with pytest.raises(ValueError,match='refine_on_ideal_stall'):
        solve_residual(simple_reaction_problem(),refine_on_ideal_stall='yes')


def test_box_ideal_tables_reused_only_within_bounded_row_product(monkeypatch):
    import route2_box_profiles as box
    features = BoxProfileOperator([[-1., 1.]]*2, 12, centers=33)
    monkeypatch.setattr(features,'ideal_tensor_plan',lambda *a,**k:dict(enabled=False))
    points = np.random.default_rng(9).uniform(-.4, .4, size=(11, 2))
    orders = ((0, 0), (1, 0), (2, 0), (0, 1))
    block = ResidualBlock('linear', points,
        lambda x, j, p: sum(j[d] for d in orders), orders)
    linear = linearize_residual(ResidualProblem(features, [block], execution='streamed', batch_size=4),
                               np.zeros((features.size, 1)))
    calls = []
    original = box.eval_jacobi
    def counted(*args, **kwargs):
        calls.append(1); return original(*args, **kwargs)
    monkeypatch.setattr(box, 'eval_jacobi', counted)
    ideal = linear.operator.ideal_operator
    ideal @ np.ones(ideal.shape[1])
    assert len(calls) == 3*5  # 3 row batches, 5 distinct axis orders, independent of coordinate panels.
    ideal.rmatvec(np.ones(ideal.shape[0]))
    assert len(calls) == 2*3*5
    assert linear.metrics['product_counts']['ideal_correction_table_bytes_max'] == 8*4*13*5
    state = features.prepare_ideal_columns(points[:4], orders)
    with pytest.raises(ValueError, match='same geometry, points'):
        features.selected_ideal_columns(points[1:5], np.array([0]), orders, prepared=state)


@pytest.mark.parametrize('linearization', ['ideal','actual'])
def test_invalid_ideal_preconditioner_never_allocates_actual_column_panels(monkeypatch,linearization):
    problem=simple_reaction_problem()
    original=problem.features.selected_ideal_columns
    def broken(points,indices,derivatives,**kwargs):
        return {d:np.full_like(v,np.nan) for d,v in original(points,indices,derivatives,**kwargs).items()}
    def forbidden(*args,**kwargs):
        raise AssertionError('Unbudgeted actual column panels during ideal preconditioner fallback')
    monkeypatch.setattr(problem.features,'selected_ideal_columns',broken)
    monkeypatch.setattr(problem.features,'selected_columns',forbidden)
    result=solve_residual(problem,high_accuracy=True,tolerance=2e-14,
        linearization_basis=linearization,preconditioner='block',preconditioner_basis='ideal',
        linear_refinement_steps=0,max_iterations=15)
    assert result.status=='converged',result.metrics
    assert result.metrics['active_preconditioner']=='diagonal'
    assert any(row.get('preconditioner_fallbacks') for row in result.history)


def test_preconditioner_refresh_releases_previous_factor_set_before_allocation(monkeypatch):
    import weakref
    import solver.general_residual as solver
    original=solver._RightPreconditioner
    references=[]
    def observed(*args,**kwargs):
        if references:
            assert references[-1]() is None, 'Previous factors remain alive during refreshed allocation'
        result=original(*args,**kwargs)
        references.append(weakref.ref(result))
        return result
    monkeypatch.setattr(solver,'_RightPreconditioner',observed)
    result=solve_residual(simple_reaction_problem(),high_accuracy=True,tolerance=2e-14,
        preconditioner='block',preconditioner_basis='ideal',preconditioner_refresh=1,
        linearization_basis='ideal',linear_refinement_steps=0)
    assert result.status=='converged'
    assert len(references)>1
    assert all(item['linearization_basis']=='ideal' and item['local_sensitivity_basis']=='actual_neural'
               for item in result.metrics['preconditioner_builds'])


def test_expired_after_linearization_does_not_start_gradient_or_preconditioning(monkeypatch):
    import solver.general_residual as solver
    now=[0.]
    monkeypatch.setattr(solver.time,'perf_counter',lambda:now[0])
    def expired(*args):now[0]=2.
    def forbidden(*args,**kwargs):raise AssertionError('Preconditioning after exhausted budget')
    monkeypatch.setattr(solver,'_RightPreconditioner',forbidden)
    result=solve_residual(simple_reaction_problem(),max_seconds=1.,iteration_callback=expired,
        linearization_basis='ideal',preconditioner='block',preconditioner_basis='ideal')
    assert result.status=='budget_exhausted'
    assert result.metrics['product_counts']['rmatvec_calls']==0
    assert result.metrics['lsmr_iterations']==0


@pytest.mark.parametrize('broken_direction',['forward','transpose'])
def test_invalid_tensor_products_trigger_guarded_actual_fallback(monkeypatch,broken_direction):
    problem=simple_reaction_problem()
    if broken_direction=='forward':
        def broken(state,points,orders):
            return {d:np.full((len(points),1),np.nan) for d in orders}
        monkeypatch.setattr(problem.features,'forward_ideal_tensor_jets',broken)
    else:
        monkeypatch.setattr(problem.features,'accumulate_ideal_tensor_jets',
                            lambda cotangents,points,state:state.fill(np.nan))
    result=solve_residual(problem,high_accuracy=True,tolerance=2e-14,
        linearization_basis='ideal',preconditioner='block',preconditioner_basis='ideal',
        linear_refinement_steps=0,max_iterations=15)
    assert result.status=='converged',result.metrics
    assert any(row.get('linearization_fallbacks') for row in result.history)
    assert any(inner['linearization_basis']=='actual'
               for row in result.history for inner in row.get('linear_solves',[]))
