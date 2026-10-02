"""Meaningful checks of public PDE interfaces, scaling, reconstruction and limits."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver import PeriodicEvolution,ReactionDiffusion1D,Resolution,Conservation1D,BoxProblem,solve,calibrate


def test_scaled_domain_derivatives_and_export():
    p=PeriodicEvolution(lambda x:np.cos(2*np.pi*x),lambda t,x,u,ux,uxx,p:.1*uxx,domain=(0.,1.))
    s=solve(p,t_end=.05,resolution=Resolution(modes=8),rtol=1e-10,atol=1e-12)
    x=np.linspace(0,1,51);damp=np.exp(-.1*(2*np.pi)**2*.05)
    assert np.max(abs(s.evaluate(x,.05)-damp*np.cos(2*np.pi*x)))<1e-8
    assert np.max(abs(s.evaluate(x,.05,1)+2*np.pi*damp*np.sin(2*np.pi*x)))<1e-7
    a=s.readouts(.05)
    raw=np.tanh(a['gamma']*(x[:,None]-a['centers']))@a['weights']+a['bias']
    assert np.max(abs(raw-s.evaluate(x,.05)))<1e-12
    with pytest.raises(ValueError):s.evaluate(x,.06)
    with pytest.raises(ValueError):s.evaluate([-1.],.03)


def test_refinement_and_initial_condition_rejection():
    p=PeriodicEvolution(lambda x:np.sin(3*x),lambda t,x,u,ux,uxx,p:-ux)
    s=solve(p,t_end=.1,adaptive=True,tolerance=1e-7,initial_tolerance=1e-8,starting_modes=4,maximum_modes=16)
    assert s.metrics['spatial_tolerance_met_empirically']
    unresolved=PeriodicEvolution(lambda x:np.sin(17*x),p.rhs)
    with pytest.raises(ValueError,match='underresolved'):
        solve(unresolved,t_end=.1,resolution=Resolution(modes=4))
    with pytest.raises(NotImplementedError):solve(object())


def test_conservative_dispatch_and_neural_primitive():
    s=solve(Conservation1D(lambda x:np.sin(np.pi*x)),t_end=.1,cells=64)
    assert abs(s['metrics']['mass_balance_residual'])<1e-12
    nn=s['neural_primitive'];x=s['x']
    derivative=(x[:,None]>nn['knots'])@nn['weights']
    assert np.max(abs(derivative-s['values']))<1e-14
    with pytest.raises(NotImplementedError):
        solve(Conservation1D(lambda x:x,flux='unimplemented'),t_end=.1)


def test_inverse_rank_and_noise_scaling():
    x=np.arange(1.,6.)
    full=calibrate(lambda z:z[0]*x+z[1],2*x+3,[1.,1.],sigma=.1)
    assert np.max(abs(full['parameters']-[2,3]))<1e-8
    assert full['rank']==2 and full['covariance'].shape==(2,2)
    deficient=calibrate(lambda z:z[0]*x,2*x,[1.,8.],sigma=.1)
    assert deficient['rank']==1 and deficient['covariance'] is None
    bounded=calibrate(lambda z:z[0]*x,2*x,[.5],bounds=(0,1),sigma=.1)
    assert bounded['active_mask'][0]==1 and bounded['covariance'] is None
    wrong_model=calibrate(lambda z:z[0]*x,2*x+3,[1.],sigma=.01)
    assert wrong_model['observation_consistency']=='inconsistent_with_assumed_noise'


def test_projected_residual_is_not_physical_convergence():
    p=BoxProblem(lambda xy:np.sin(xy[:,0])*np.sin(xy[:,1])+np.sin(5*xy[:,0])*np.sin(5*xy[:,1]),beta=0)
    s=solve(p,modes=4)
    assert s.metrics['converged']
    assert s.status=='underresolved'
    assert s.metrics['independent_relative_residual']>.5
    refined=solve(p,adaptive=True,tolerance=1e-6,starting_modes=4,maximum_modes=12)
    assert refined.status=='residual_checks_passed'


def test_input_validation_and_stiff_reaction_dispatch():
    with pytest.raises(ValueError,match='integer'):Resolution(modes=4.5)
    with pytest.raises(ValueError,match='integer'):Resolution(centers=64.5)
    with pytest.raises(ValueError,match='positive integers'):
        solve(BoxProblem(lambda xy:np.ones(len(xy))),max_iterations=0)
    p=ReactionDiffusion1D(lambda x:np.ones_like(x)*.5,reaction_rate=1.)
    s=solve(p,t_end=.01,modes=4,cells=32,rtol=1e-10,atol=1e-12)
    exact=1/np.sqrt(1+3*np.exp(-.02))
    assert np.max(abs(s['values']-exact))<1e-8
