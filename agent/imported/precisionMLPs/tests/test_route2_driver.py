"""Route2 must solve and validate the neural model, including captured data."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.route2 import solve_route2,raw_residual_audit,footprint
from solver.general_domains import ResidualDomain
from solver.general_adaptive import ResidualDeclaration
from solver.general_residual import ResidualBlock,ResidualProblem,solve_residual
from solver.ball_ridge_features import BallRidgeFeatures


def declaration():
    domain=ResidualDomain([[-1.,1.]]*2)
    def truth(x):return .1+.05*x[:,0:1]+.03*x[:,1:2]
    def blocks(count,seed):
        x=domain.interior(count,seed);b=domain.boundary(max(80,count//2),seed+1).points
        source=torch.tensor(truth(x)**3);boundary=torch.tensor(truth(b))
        return [ResidualBlock('PDE',x,lambda x,j,p:-j[(2,0)]-j[(0,2)]+j[(0,0)]**3-source,((0,0),(2,0),(0,2))),
                ResidualBlock('BC',b,lambda x,j,p:j[(0,0)]-boundary,((0,0),))]
    return ResidualDeclaration(domain,blocks),truth


def test_driver_uses_neural_continuation_and_raw_export():
    spec,truth=declaration()
    s=solve_route2(spec,degrees=(1,2),check_points=83,tolerance=1e-9,max_seconds=20)
    assert s.status=='sampled_residual_checks_passed'
    assert s.history[0]['initialization']=='zero'
    assert s.history[1]['initialization']=='previous actual neural PDE solution'
    model=s.export();assert [type(layer) for layer in model]==[torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
    x=spec.domain.interior(51,317)
    np.testing.assert_allclose(model(torch.tensor(x)).detach(),truth(x),atol=1e-9)
    assert s.metrics['reference_solution_used'] is False


def test_resource_refusal_happens_before_large_construction():
    spec,_=declaration()
    factory=spec.make_blocks;calls=[]
    def counted(count,seed):
        calls.append(count);return factory(count,seed)
    spec.make_blocks=counted
    with pytest.raises(RuntimeError,match='affordable'):
        solve_route2(spec,degrees=(4,),maximum_neurons=10)
    assert calls==[257,257]  # Two independent validation sets, before large construction.
    assert footprint(4,4,129)['neurons']==19125
    assert footprint(20,2,129)['neurons']==61200
    assert footprint(10,3,129)['directions']==580
    assert footprint(10,2,129,angular_rule='tensor')['directions']==3**9


def test_sparse_constructor_is_reachable_through_general_driver():
    domain=ResidualDomain([[-1.,1.]]*4)
    order=(0,)*4
    def blocks(count,seed):
        x=domain.interior(count,seed)
        return [ResidualBlock('algebraic_physics',x,lambda x,j,p:j[order]+j[order]**3-.101,(order,))]
    s=solve_route2(ResidualDeclaration(domain,blocks),degrees=(1,2),angular_rule='sparse',
                   centers=65,tolerance=1e-8,max_seconds=20)
    assert s.status=='sampled_residual_checks_passed'
    assert s.problem.features.metrics['directions']==16
    np.testing.assert_allclose(s.export()(torch.zeros((3,4),dtype=torch.float64)).detach(),.1,atol=1e-8)


def test_parameter_family_boundaries_cover_entire_parameter_interval():
    from route2_parametric_study import blocks
    traces=blocks(257,113)[1].points
    for side in (-1.,1.):
        values=traces[traces[:,0]==side,1]
        assert values.min()<1.05 and values.max()>1.95


def test_raw_audit_keeps_captured_source_and_integral_rows():
    spec,_=declaration();f=BallRidgeFeatures(spec.domain.bounds,1)
    blocks=spec.make_blocks(93,521)
    x=spec.domain.interior(77,823)
    # Integral row uses all 77 locations, no subsampling the aggregation.
    blocks.append(ResidualBlock('mean_zero_derivative',x,lambda x,j,p:j[(2,0)],((2,0),),aggregation=np.ones((1,77))/77))
    s=solve_residual(ResidualProblem(f,blocks),tolerance=1e-10,high_accuracy=True,max_seconds=10)
    checks=raw_residual_audit(s,blocks)
    assert checks[0]['points']==93 and checks[2]['points']==77
    assert max(r['maximum_scaled_rms'] for r in checks)<1e-9


def test_nonfinite_raw_export_is_not_treated_as_a_pass(monkeypatch):
    import solver.route2 as driver
    original=driver.ordinary_jets;calls=[]
    def nonfinite_second_block(model,points,orders,**options):
        jets=original(model,points,orders,**options);calls.append(1)
        if len(calls)%2==0:
            for values in jets.values():values.fill_(float('nan'))
        return jets
    monkeypatch.setattr(driver,'ordinary_jets',nonfinite_second_block)
    spec,_=declaration()
    s=solve_route2(spec,degrees=(1,2),tolerance=1e-9,max_seconds=20)
    assert s.status!='sampled_residual_checks_passed'
    assert s.history[-1]['ordinary_export_validation'][1]['finite'] is False
    assert np.isinf(s.history[-1]['ordinary_export_validation'][1]['maximum_scaled_rms'])


def test_failed_later_known_profile_gate_preserves_solved_model():
    domain=ResidualDomain([[-1,1],[-1,1]],levelset=lambda x:(x*x).sum(axis=1)-1)
    def blocks(count,seed):
        return [ResidualBlock('constant',domain.interior(count,seed),lambda x,j,p:j[(0,0)]-.1,((0,0),))]
    s=solve_route2(ResidualDeclaration(domain,blocks),coordinates='disk',degrees=(0,1),
                   centers=3,tolerance=1e-8,max_seconds=20)
    assert s.status=='construction_limit'
    assert len(s.history)==1 and s.problem.features.degree==0
    np.testing.assert_allclose(s.evaluate([[0,0]]),.1,atol=1e-8)


def test_streamed_disk_driver_coupled_fields_continuation_and_native_export(monkeypatch):
    from route2_disk_profiles import DiskProfileOperator
    domain=ResidualDomain([[-1.,1.]]*2,levelset=lambda x:(x*x).sum(axis=1)-1)
    callback_rows=[]
    def truth(x):
        return np.column_stack((.1+.05*x[:,0]+.03*x[:,1],-.2+.02*x[:,0]-.04*x[:,1]))
    def blocks(count,seed):
        x=domain.interior(count,seed)
        angles=np.linspace(0,2*np.pi,max(24,count//8),endpoint=False)
        boundary=np.column_stack((np.cos(angles),np.sin(angles)))
        captured=torch.tensor(truth(boundary),dtype=torch.float64)
        def pde(x,j,p):
            callback_rows.append(len(x))
            u,v=j[(0,0)].unbind(1)
            u_star=.1+.05*x[:,0]+.03*x[:,1]
            v_star=-.2+.02*x[:,0]-.04*x[:,1]
            lap=j[(2,0)]+j[(0,2)]
            return torch.stack((-lap[:,0]+u**3+.2*v-(u_star**3+.2*v_star),
                                 -lap[:,1]+v+.3*u-(v_star+.3*u_star)),dim=1)
        def boundary_full(x,j,p):
            return j[(0,0)]-captured
        def boundary_batch(x,j,p,indices):
            callback_rows.append(len(x))
            return j[(0,0)]-captured[indices]
        return [ResidualBlock('coupled_pde',x,pde,((0,0),(2,0),(0,2))),
                ResidualBlock('observed_boundary',boundary,boundary_full,((0,0),),
                              weight=3.,batch_function=boundary_batch)]
    for name in ('evaluate','evaluate_many'):
        monkeypatch.setattr(DiskProfileOperator,name,
            lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Dense feature fallback forbidden')))
    iterates=[]
    def snapshot(c,p,record):iterates.append((c.copy(),record['iteration']))
    solution=solve_route2(ResidualDeclaration(domain,blocks,fields=2),coordinates='disk',
        execution='streamed',batch_size=17,degrees=(1,2),centers=129,tolerance=2e-10,
        check_points=37,max_seconds=25,solver_options={'iteration_callback':snapshot})
    assert solution.status=='sampled_residual_checks_passed',solution.history
    assert solution.metrics['execution']=='streamed'
    assert solution.problem.execution=='streamed'
    assert solution.problem.fields==2
    assert len(solution.history)==2
    assert all(row['estimated_jet_cache_bytes']==0 for row in solution.history)
    assert all(row['cost']['profile_map_bytes']==0 for row in solution.history)
    assert max(callback_rows)<=17
    old=[c for c,it in iterates if c.shape==(3,2)][-1]
    continued=[c for c,it in iterates if c.shape==(6,2) and it==0][0]
    np.testing.assert_allclose(continued[:3],old,atol=0,rtol=0)
    np.testing.assert_array_equal(continued[3:],0)
    points=domain.interior(31,741)
    expected=truth(points)
    np.testing.assert_allclose(solution.evaluate(points),expected,atol=2e-10)
    model=solution.export()
    assert [type(layer) for layer in model]==[torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
    assert model[2].out_features==2
    np.testing.assert_allclose(model(torch.tensor(points)).detach(),expected,atol=2e-10)
    checks=raw_residual_audit(solution.solution,blocks(23,117))
    assert max(row['maximum_scaled_rms'] for row in checks)<2e-10
    assert max(callback_rows)<=17


def test_streamed_driver_rejects_invalid_settings_before_sampling_and_counts_no_map():
    spec,_=declaration()
    calls=[]
    spec.make_blocks=lambda *args:calls.append(args)
    with pytest.raises(ValueError,match='coordinates must'):
        solve_route2(spec,execution='streamed',coordinates='invalid')
    assert not calls
    with pytest.raises(ValueError,match='conflicting solver_options'):
        solve_route2(spec,execution='streamed',coordinates='disk',solver_options={'execution':'cached'})
    assert not calls
    with pytest.raises(ValueError,match='batch_size'):
        solve_route2(spec,execution='streamed',coordinates='disk',batch_size=0)
    estimate=footprint(2,16,257,fields=2,coordinates='disk',execution='streamed',batch_size=31)
    assert estimate['profile_map_bytes']==estimate['map_workspace_bytes']==0
    assert estimate['readout_coordinates']==153
    assert estimate['prepared_state_bytes']==8*2*(estimate['neurons']+17)
    assert estimate['scalar_basis_panel_bytes']==8*31*(257+34)
    assert footprint(2,16,257,coordinates='disk')['profile_map_bytes']>0


def test_streamed_box_driver_uses_factored_conversion_and_ordinary_export(monkeypatch):
    from route2_box_profiles import BoxProfileOperator
    domain=ResidualDomain([[-1.,1.]]*3)
    order=(0,0,0)
    def truth(x):return .2+.03*x[:,0]-.02*x[:,1]*x[:,2]
    def blocks(count,seed):
        x=domain.interior(count,seed)
        def equation(x,j,p):
            u=j[order][:,0];target=truth(x)
            return u+u**3-(target+target**3)
        return [ResidualBlock('nonlinear_algebraic_control',x,equation,(order,))]
    for name in ('evaluate','evaluate_many'):
        monkeypatch.setattr(BoxProfileOperator,name,
            lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Dense fallback forbidden')))
    solved=solve_route2(ResidualDeclaration(domain,blocks),execution='streamed',coordinates='box',
        degrees=(1,2),centers=65,tolerance=2e-8,max_seconds=30,check_points=31,
        export_check_points=29,batch_size=19,
        solver_options={'preconditioner':'diagonal','lsmr_max_iterations':120})
    assert all(r['cost']['profile_map_bytes']==0 and r['estimated_jet_cache_bytes']==0 for r in solved.history)
    x=domain.interior(37,910)
    np.testing.assert_allclose(solved.export()(torch.tensor(x)).detach().numpy()[:,0],truth(x),atol=2e-8)
    assert solved.metrics['export_check_points']==29
    assert solved.problem.features.dimension==3
    assert [type(v) for v in solved.export()]==[torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
    estimate=footprint(4,8,129,fields=4,coordinates='box',execution='streamed')
    assert estimate['profile_map_bytes']==estimate['map_workspace_bytes']==0
    assert estimate['coordinate_workspace_bytes']==8*9*estimate['readout_coordinates']


def test_preflight_includes_large_contour_workspace_before_feature_construction(monkeypatch):
    import solver.route2 as driver
    from route2_disk_profiles import DiskProfileOperator
    domain=ResidualDomain([[-1.,1.]]*2,levelset=lambda x:(x*x).sum(axis=1)-1)
    def blocks(count,seed):
        return [ResidualBlock('control',domain.interior(count,seed),lambda x,j,p:j[(0,0)],((0,0),))]
    monkeypatch.setattr(driver,'DiskProfileOperator',
        lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Preflight must reject before encode')))
    cost=footprint(2,1,100001,coordinates='disk',execution='streamed',lam=.2)
    assert cost['encoder_peak_array_estimate_bytes']>512*1024**2
    with pytest.raises(RuntimeError,match='affordable'):
        solve_route2(ResidualDeclaration(domain,blocks),degrees=(1,),centers=100001,
            maximum_neurons=1000000,maximum_working_array_mb=512,
            coordinates='disk',execution='streamed',check_points=17)


def test_ideal_preconditioner_preflight_does_not_charge_directional_panels(monkeypatch):
    import solver.route2 as driver
    domain=ResidualDomain([[-1.,1.]]*4)
    orders=((0,0,0,0),(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),
            (2,0,0,0),(0,2,0,0),(0,0,2,0))
    def blocks(count,seed):
        return [ResidualBlock('four_field_physics',domain.interior(count,seed),
                lambda x,j,p:j[orders[0]],orders)]
    spec=ResidualDeclaration(domain,blocks,fields=4)
    cost=footprint(4,16,129,fields=4,execution='streamed')
    settings=dict(preconditioner_block_size=640,linearization_basis='ideal')
    actual=driver._streamed_solver_workspace(cost,blocks(17,3),4,0,128,129,'box',settings)
    ideal=driver._streamed_solver_workspace(cost,blocks(17,3),4,0,128,129,'box',
                                         dict(settings,preconditioner_basis='ideal'))
    parts=ideal['preconditioner_components']
    assert parts['directional_panel_bytes']==0
    assert parts['ideal_coordinate_panel_bytes']>0 and parts['factor_bytes']>0
    assert ideal['ideal_correction_workspace_bytes']>0
    assert actual['preconditioner_workspace_bytes']-ideal['preconditioner_workspace_bytes']>390*1024**2
    # Direction count changes actual profile storage, never ideal-coordinate storage.
    enlarged=dict(cost,directions=3*cost['directions'])
    assert driver._streamed_solver_workspace(enlarged,blocks(17,3),4,0,128,129,'box',
        dict(settings,preconditioner_basis='ideal'))==ideal
    class ReachedConstruction(Exception):pass
    def constructor(*args,**kwargs):raise ReachedConstruction
    monkeypatch.setattr(driver,'BoxProfileOperator',constructor)
    common=dict(degrees=(16,),centers=129,maximum_neurons=1000000,
        maximum_working_array_mb=512,coordinates='box',execution='streamed',check_points=17)
    with pytest.raises(RuntimeError,match='affordable'):
        solve_route2(spec,solver_options=settings,**common)
    with pytest.raises(ReachedConstruction):
        solve_route2(spec,solver_options=dict(settings,preconditioner_basis='ideal'),**common)


def test_memory_adaptive_preconditioner_selects_largest_affordable_block():
    import solver.route2 as driver
    spec,_=declaration();blocks=spec.make_blocks(31,9)
    cost=footprint(2,12,129,fields=2,execution='streamed')
    options=dict(preconditioner='block',preconditioner_basis='ideal',
                 linearization_basis='ideal',preconditioner_block_size=64)
    exact=driver._streamed_solver_workspace(cost,blocks,2,1,16,129,'box',
                                           dict(options,preconditioner_block_size=23))
    budget=exact['preconditioner_workspace_bytes']+exact['ideal_correction_workspace_bytes']
    chosen,workspace=driver._memory_bounded_preconditioner(cost,blocks,2,1,16,129,'box',options,budget)
    assert chosen['preconditioner_block_size']==23
    assert workspace['block_budget_selection']['changed']
    assert options['preconditioner_block_size']==64  # Caller's dictionary remains unchanged.
    refused,smallest=driver._memory_bounded_preconditioner(cost,blocks,2,1,16,129,'box',options,0)
    assert refused['preconditioner_block_size']==1
    assert smallest['preconditioner_workspace_bytes']>0  # Driver must still refuse the allocation.


def test_native_preset_is_opt_in_and_forwards_explicit_overrides(monkeypatch):
    import solver.route2 as driver
    calls=[]
    def capture(declaration,**kwargs):calls.append((declaration,kwargs));return 'sentinel'
    monkeypatch.setattr(driver,'solve_route2',capture)
    spec,_=declaration()
    assert driver.solve_route2_native(spec,max_seconds=23,solver_options={'preconditioner_block_size':128})=='sentinel'
    _,options=calls[0]
    assert options['execution']=='streamed' and options['adaptive_preconditioner_blocks']
    assert options['degrees']==(4,8,12,16,24,32,40,48)
    assert options['max_seconds']==23 and np.isinf(options['maximum_neurons'])
    assert options['solver_options']['linearization_basis']=='ideal'
    assert options['solver_options']['preconditioner_basis']=='ideal'
    assert options['solver_options']['preconditioner_refresh']==1
    assert options['solver_options']['preconditioner_field_parity']=='auto'
    assert options['solver_options']['preconditioner_block_size']==128


def test_native_preset_solves_and_validates_actual_nonlinear_neural_problem():
    from solver.route2 import solve_route2_native
    # This callback reads x directly, so it supports bounded row evaluation.
    domain=ResidualDomain([[-1.,1.]]*2)
    def blocks(count,seed):
        x=domain.interior(count,seed)
        return [ResidualBlock('nonlinear',x,
            lambda x,j,p:j[(0,0)]**3+j[(0,0)]-(.2+.1*x[:,:1])**3-(.2+.1*x[:,:1]),((0,0),))]
    solved=solve_route2_native(ResidualDeclaration(domain,blocks),degrees=(1,2),centers=65,
        max_seconds=20,check_points=23,tolerance=1e-10)
    assert solved.status=='sampled_residual_checks_passed'
    assert solved.solution.metrics['settings']['linearization_basis']=='ideal'
    assert solved.history[-1]['ordinary_export_validation'][0]['maximum_scaled_rms']<1e-10


def test_underresolved_native_parametric_stage_requests_refinement_after_tight_check():
    from solver.route2 import solve_route2_native
    from route2_parametric_study import DOMAIN,blocks
    solved=solve_route2_native(ResidualDeclaration(DOMAIN,blocks),degrees=(6,),centers=257,
        tolerance=1e-13,max_seconds=15,check_points=17)
    assert solved.solution.status=='linearized_stationary'
    assert solved.solution.metrics['iterations']<20
    assert solved.history[0]['refinement_reason']=='sampled_stationarity_above_tolerance'
    assert solved.history[0]['normalized_validation_score']>1
    assert not solved.metrics['validated']
    final=solved.solution.history[-1]
    assert final['tight_actual_model_check']=='previous_accepted_inexact_step_negligible'
    assert final['linearized_stationarity']['actual_jacobian']
    assert final['linearized_stationarity']['inner_solve_accurate']


def test_driver_preserves_evaluation_budget_provenance():
    spec,_=declaration()
    solved=solve_route2(spec,degrees=(1,2),centers=65,max_seconds=20,check_points=17,
                       solver_options={'max_residual_evaluations':1})
    assert solved.solution.status=='budget_exhausted'
    assert solved.status=='evaluation_budget'
    assert solved.metrics['validated'] is False


def test_native_solution_resume_preserves_coordinates_without_a_fit():
    spec,_=declaration()
    first=solve_route2(spec,degrees=(1,),centers=65,max_seconds=20,check_points=17,
        tolerance=1e-8,coordinates='box',execution='streamed',batch_size=256)
    snapshots=[]
    resumed=solve_route2(spec,degrees=(2,),centers=65,max_seconds=20,check_points=17,
        tolerance=1e-8,coordinates='box',execution='streamed',initial_solution=first,batch_size=256,
        solver_options={'iteration_callback':lambda c,p,row:snapshots.append(c.copy())})
    np.testing.assert_array_equal(snapshots[0][:len(first.coefficients)],first.coefficients)
    np.testing.assert_array_equal(snapshots[0][len(first.coefficients):],0)
    assert resumed.metrics['initial_solution_supplied']
    assert resumed.history[0]['initialization']=='previous actual neural PDE solution'
    with pytest.raises(ValueError,match='include all'):
        solve_route2(spec,degrees=(0,),initial_solution=resumed)


def test_same_degree_resume_is_not_a_resolution_agreement_certificate():
    spec,_=declaration()
    first=solve_route2(spec,degrees=(1,),check_points=17,tolerance=1e-8,max_seconds=20)
    resumed=solve_route2(spec,degrees=(1,),check_points=17,tolerance=1e-8,
                         max_seconds=20,initial_solution=first)
    assert resumed.history[0]['successive_field_difference']<1e-8
    assert resumed.history[0]['comparison_uses_richer_resolution'] is False
    assert resumed.status=='resolution_limit'


def test_disk_resume_rejects_different_coordinate_bases():
    domain=ResidualDomain([[-1.,1.]]*2,levelset=lambda x:(x*x).sum(axis=1)-1)
    def blocks(count,seed):
        return [ResidualBlock('constant',domain.interior(count,seed),lambda x,j,p:j[(0,0)]-.1,((0,0),))]
    spec=ResidualDeclaration(domain,blocks)
    first=solve_route2(spec,degrees=(1,),coordinates='disk',check_points=17,
                       tolerance=1e-8,max_seconds=20)
    with pytest.raises(ValueError,match='same coordinate basis'):
        solve_route2(spec,degrees=(2,),coordinates='disk',execution='streamed',
                     initial_solution=first)


@pytest.mark.parametrize('coordinates', ['box','disk'])
@pytest.mark.parametrize('execution', ['cached','streamed'])
def test_resume_rejects_changed_affine_chart_before_constructing_features(coordinates,execution):
    domain=ResidualDomain([[-1.,1.]]*2,
        levelset=(lambda x:(x*x).sum(axis=1)-1) if coordinates=='disk' else None)
    def blocks(count,seed):
        return [ResidualBlock('constant',domain.interior(count,seed),
                lambda x,j,p:j[(0,0)]-.1,((0,0),))]
    spec=ResidualDeclaration(domain,blocks)
    first=solve_route2(spec,degrees=(1,),centers=65,coordinates=coordinates,
        execution=execution,check_points=17,tolerance=1e-8,max_seconds=20)
    if coordinates=='box':
        changed=ResidualDeclaration(ResidualDomain([[-2.,2.],[-1.,1.]]),blocks)
        with pytest.raises(ValueError,match='affine coordinate chart'):
            solve_route2(changed,degrees=(2,),coordinates=coordinates,execution=execution,initial_solution=first)
    else:
        for change in ({'disk_radius':2.},{'disk_midpoint':[.1,0.]}):
            with pytest.raises(ValueError,match='affine coordinate chart'):
                solve_route2(spec,degrees=(2,),coordinates=coordinates,execution=execution,
                             initial_solution=first,**change)
