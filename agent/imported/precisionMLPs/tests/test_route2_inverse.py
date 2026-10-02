from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.general_adaptive import ResidualDeclaration
from solver.general_domains import ResidualDomain
from solver.general_residual import ResidualBlock
from solver.route2_inverse import SensorObservations,freeze_physics_parameters,solve_route2_inverse


def diffusion(initial=1.1):
    domain=ResidualDomain([[-1.,1.],[-1.,1.]])
    def blocks(count,seed):
        x=domain.interior(count,seed);wall=domain.boundary(64,seed+1).points
        def pde(x,j,p):return -p[:,:1]*(j[(2,0)]+j[(0,2)])-2*(2-x[:,:1]**2-x[:,1:2]**2)
        return [ResidualBlock('diffusion',x,pde,((2,0),(0,2))),
                ResidualBlock('zero_wall',wall,lambda x,j,p:j[(0,0)],((0,0),))]
    return ResidualDeclaration(domain,blocks,parameter_initial=[initial],parameter_bounds=([.2],[2.]))


def test_parameter_freezing_copies_values_and_removes_inner_unknown():
    source=diffusion();eta=np.array([.7]);frozen=freeze_physics_parameters(source,eta);eta[0]=1.8
    assert len(frozen.parameter_initial)==0 and frozen.parameter_bounds is None
    block=frozen.make_blocks(5,71)[0];x=torch.tensor(block.points)
    jets={(2,0):torch.ones((5,1)),(0,2):torch.ones((5,1))}
    actual=block.function(x,jets,torch.empty((5,0)))
    expected=-1.4-2*(2-x[:,:1]**2-x[:,1:2]**2)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
    assert source.parameter_initial==[1.1]


def test_unresolved_forward_never_queries_sensor_values(monkeypatch):
    import solver.route2_inverse as module
    from types import SimpleNamespace
    failed=SimpleNamespace(status='resolution_limit',metrics={'validated':False},parameters=np.array([]),
        problem=SimpleNamespace(features=SimpleNamespace(degree=2)),history=[])
    monkeypatch.setattr(module,'solve_route2_native',lambda *a,**k:failed)
    monkeypatch.setattr(module,'_sensor_residual',lambda *a,**k:pytest.fail('Unresolved field was scored'))
    result=module.solve_route2_inverse(diffusion(),[SensorObservations([[0.,0.]],[1.])])
    assert result.status=='unresolved_initial_forward' and result.solution is None
    assert result.unresolved_solution is failed
    assert 'sensor_objective' not in result.metrics['forward_runs'][0]


@pytest.mark.parametrize('noise',[0.,.002])
def test_native_reduced_diffusion_inverse_matches_exact_statistical_optimum(noise):
    rng=np.random.default_rng(892);x=rng.uniform(-.8,.8,(9,2));g=(1-x[:,0]**2)*(1-x[:,1]**2)
    values=g/.7+noise*rng.normal(size=9)
    # Analytic least-squares parameter is used only after native fitting.
    expected=(g@g)/(g@values)
    observations=[SensorObservations(x,values,scale=noise if noise else 1.)]
    result=solve_route2_inverse(diffusion(),observations,max_seconds=90,max_iterations=8,
        inner_options=dict(coordinates='affine_disk',degrees=(2,4,6),centers=129,tolerance=2e-9,
                           max_seconds=15,check_points=31,export_check_points=33,max_iterations=12),
        gradient_tolerance=1e-8,step_tolerance=1e-9)
    assert result.status in ('data_stationary','data_tolerance'),result.metrics
    assert abs(result.parameters[0]-expected)<3e-7
    assert result.metrics['validated_forward']
    assert len(result.solution.parameters)==0
    assert all(row['validated'] for row in result.metrics['forward_runs'])
    assert all(row['inner_parameter_count']==0 for row in result.metrics['forward_runs'])
    assert [type(layer) for layer in result.export()]==[torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
    assert result.history[-1]['local_response_rank']==1
    for row in result.history:
        np.testing.assert_allclose(row['normalized_sensor_rms']**2,2*row['sensor_objective'],atol=1e-25)


def test_observation_validation_and_immutability():
    q=np.array([[0.,0.]]);y=np.array([1.]);obs=SensorObservations(q,y)
    q[0,0]=2;y[0]=3
    assert obs.points[0,0]==0 and obs.values[0,0]==1
    with pytest.raises(ValueError,match='positive'):SensorObservations([[0,0]],[1],scale=0)


def test_unresolved_sensitivity_preserves_validated_current_and_is_not_scored(monkeypatch):
    import solver.route2_inverse as module
    from types import SimpleNamespace
    good=SimpleNamespace(status='sampled_residual_checks_passed',metrics={'validated':True},
        parameters=np.array([]),problem=SimpleNamespace(features=SimpleNamespace(degree=4)),history=[])
    bad=SimpleNamespace(status='time_budget',metrics={'validated':False},
        parameters=np.array([]),problem=SimpleNamespace(features=SimpleNamespace(degree=4)),history=[])
    calls=[];scored=[]
    def forward(*args,**kwargs):
        calls.append(1);return good if len(calls)==1 else bad
    def sensor(solution,observations):
        scored.append(solution);return np.array([1.])
    monkeypatch.setattr(module,'solve_route2_native',forward)
    monkeypatch.setattr(module,'_sensor_residual',sensor)
    result=module.solve_route2_inverse(diffusion(),[SensorObservations([[0.,0.]],[1.])])
    assert result.status=='unresolved_sensitivity_forward'
    assert result.solution is good and result.unresolved_solution is bad
    assert scored==[good] and result.parameters.tolist()==[1.1]
    assert not result.metrics['outer_converged']


def test_optional_halved_step_guard_detects_unresolved_parameter_response(monkeypatch):
    import solver.route2_inverse as module
    from types import SimpleNamespace
    values=iter([1.,1.+1e-6,1.-1e-6,1.+2e-6,1.-2e-6])
    def forward(*args,**kwargs):
        return SimpleNamespace(status='sampled_residual_checks_passed',metrics={'validated':True},
            parameters=np.array([]),problem=SimpleNamespace(features=SimpleNamespace(degree=4)),history=[],
            value=next(values))
    monkeypatch.setattr(module,'solve_route2_native',forward)
    monkeypatch.setattr(module,'_sensor_residual',lambda solution,obs:np.array([solution.value]))
    result=module.solve_route2_inverse(diffusion(),[SensorObservations([[0.,0.]],[1.])],check_difference_step=True)
    assert result.status=='unresolved_finite_difference'
    assert result.history[0]['halved_step_response_relative_changes'][0]>.7
    assert not result.metrics['outer_converged']
    assert 'does not establish' in result.metrics['numerical_rank_scope']


def test_sensor_likelihood_is_invariant_to_arbitrary_group_partition():
    from solver.route2_inverse import _sensor_residual
    from types import SimpleNamespace
    class Value(torch.nn.Module):
        def forward(self,x):return x[:,:1]
    solution=SimpleNamespace(export=lambda:Value())
    x=np.arange(12,dtype=float).reshape(6,2);y=np.linspace(-1,1,6)
    all_data=[SensorObservations(x,y,scale=.2)]
    split=[SensorObservations(x[:2],y[:2],scale=.2),SensorObservations(x[2:],y[2:],scale=.2)]
    np.testing.assert_array_equal(_sensor_residual(solution,all_data),_sensor_residual(solution,split))


@pytest.mark.parametrize('family',['disk','affine_disk','box'])
def test_native_truncation_preserves_named_modes_chart_and_parameter_isolation(family):
    from solver.route2_inverse import _truncated_native_guess
    from solver.general_residual import ResidualProblem,ResidualSolution
    from route2_disk_profiles import DiskProfileOperator
    from route2_affine_disk_profiles import AffineDiskProfileOperator
    from route2_box_profiles import BoxProfileOperator
    source=diffusion();oldfixed=freeze_physics_parameters(source,[.7]);newfixed=freeze_physics_parameters(source,[1.2])
    bounds=np.array([[-2.,4.],[1.,5.]])
    if family=='disk':features=DiskProfileOperator(6,centers=33,midpoint=[1.,3.],scale=[.2,.3])
    elif family=='affine_disk':features=AffineDiskProfileOperator(bounds,6,centers=33)
    else:features=BoxProfileOperator(bounds,6,centers=33)
    problem=ResidualProblem(features,oldfixed.make_blocks(9,1),execution='streamed')
    values=np.arange(features.size,dtype=float)[:,None]/features.size
    previous=ResidualSolution(problem,values,np.empty(0),{'status':'sampled_residual_checks_passed','validated':True})
    guess,provenance=_truncated_native_guess(previous,newfixed,(2,4,6,8))
    assert guess.problem.features.degree==4 and provenance['source_degree']==6
    assert provenance['target_or_sensor_fit'] is False and not guess.metrics['validated']
    np.testing.assert_array_equal(guess.coefficients,values[:guess.problem.features.size])
    np.testing.assert_array_equal(guess.problem.features.midpoint,features.midpoint)
    np.testing.assert_array_equal(guess.problem.features.scale,features.scale)
    np.testing.assert_array_equal(guess.problem.features.directions,features.directions)
    np.testing.assert_array_equal(guess.problem.features.base.encoding.gamma,features.base.encoding.gamma)
    assert len(guess.parameters)==len(guess.problem.parameter_initial)==0
    # New physics closes over 1.2, not the previous .7 or declaration's 1.1.
    block=guess.problem.blocks[0];x=torch.tensor(block.points)
    j={(2,0):torch.ones((len(x),1)),(0,2):torch.ones((len(x),1))}
    expected=-2.4-2*(2-x[:,:1]**2-x[:,1:2]**2)
    np.testing.assert_allclose(block.function(x,j,torch.empty((len(x),0))),expected)
    original=values.copy();original[guess.problem.features.size:]=0
    points=np.array([[1.,3.],[1.2,3.1]])
    for order in ((0,0),(1,0),(0,2)):
        np.testing.assert_allclose(guess.problem.features.forward(guess.coefficients[:,0],points,order),
            features.forward(original[:,0],points,order),atol=3e-12,rtol=3e-12)
    values[:]=0
    assert np.any(guess.coefficients!=0)


def test_truncation_unsupported_family_and_first_ladder_degree_start_cold():
    from solver.route2_inverse import _truncated_native_guess
    from types import SimpleNamespace
    previous=SimpleNamespace(problem=SimpleNamespace(features=SimpleNamespace(degree=6)))
    assert _truncated_native_guess(previous,freeze_physics_parameters(diffusion(),[.7]),(2,4,6))[0] is None
    previous.problem.features.degree=2
    unused,info=_truncated_native_guess(previous,freeze_physics_parameters(diffusion(),[.7]),(2,4,6))
    assert 'no strictly coarser' in info['reason']


def test_reduced_inverse_truncated_warm_starts_recheck_same_resolution():
    x=np.array([[-.6,.3],[.2,.5],[.5,-.4]])
    values=(1-x[:,0]**2)*(1-x[:,1]**2)/.7
    result=solve_route2_inverse(diffusion(),[SensorObservations(x,values)],max_seconds=60,
        max_iterations=8,truncate_warm_starts=True,
        inner_options=dict(coordinates='affine_disk',degrees=(2,4,6),centers=129,tolerance=2e-9,
            max_seconds=15,check_points=31,export_check_points=33,max_iterations=12))
    assert result.status in ('data_stationary','data_tolerance')
    runs=result.metrics['forward_runs']
    assert runs[0]['warm_start_provenance']['kind']=='cold'
    assert all(row['warm_start_provenance']['degree']==4 for row in runs[1:])
    assert all(row['warm_start_provenance']['candidate_parameters']==row['parameters'] for row in runs[1:])
    assert runs[1]['warm_start_provenance']['source_parameters']==[1.1]
    assert all(row['degree']==6 for row in runs)
    assert abs(result.parameters[0]-.7)<3e-7


def test_forward_work_counts_unresolved_stages_and_flags_incomplete_archives():
    from solver.route2_inverse import _native_forward_work
    from types import SimpleNamespace
    history=[dict(degree=2,solver_status='iteration_limit',solver_work=dict(lsmr_iterations=13,iterations=4,
        residual_evaluations=10,preconditioner_setup_seconds=.2)),
        dict(degree=4,solver_status='converged',solver_work=dict(lsmr_iterations=7,iterations=2,
        residual_evaluations=5,preconditioner_setup_seconds=.1))]
    record=_native_forward_work(SimpleNamespace(history=history))
    assert record['complete'] and record['totals']['lsmr_iterations']==20
    assert record['totals']['iterations']==6 and record['totals']['residual_evaluations']==15
    assert not _native_forward_work(SimpleNamespace(history=[]))['complete']


def test_product_accounting_sums_calls_but_takes_workspace_maxima():
    from solver.route2_inverse import _combine_product_counts
    assert _combine_product_counts([{'ideal_matvec_calls':5,'feature_rows_max':64},
                                    {'ideal_matvec_calls':7,'feature_rows_max':32}])==dict(ideal_matvec_calls=12,feature_rows_max=64)
