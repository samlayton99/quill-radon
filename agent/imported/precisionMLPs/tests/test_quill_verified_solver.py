import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver import PeriodicEvolution,solve_verified


def test_agreeing_truncations_do_not_hide_unresolved_forcing():
    p=PeriodicEvolution(lambda x:np.zeros_like(x),lambda t,x,u,ux,uxx,p:np.sin(x)+np.sin(17*x))
    limited=solve_verified(p,t_end=.1,starting_modes=4,maximum_modes=16,tail_limit=None)
    assert limited.status=='spatial_resolution_limit'
    resolved=solve_verified(p,t_end=.1,starting_modes=4,maximum_modes=64,tail_limit=None)
    assert resolved.status=='checks_passed'
    x=np.linspace(-np.pi,np.pi,157)
    assert np.max(abs(resolved.evaluate(x,.1)-.1*(np.sin(x)+np.sin(17*x))))<1e-8
    assert resolved.metrics['verification']['time_history'][-1]['field_difference']<1e-7


def test_temporally_oscillating_force_is_not_aliased_by_validation_times():
    # Six equally spaced checks on [0,1] all see a zero RHS, despite the
    # omitted force accumulating .5*sin(17*x) over the full trajectory.
    p=PeriodicEvolution(lambda x:np.ones_like(x),
        lambda t,x,u,ux,uxx,p:np.sin(5*np.pi*t)**2*np.sin(17*x))
    limited=solve_verified(p,t_end=1.,starting_modes=4,maximum_modes=8,tail_limit=None)
    assert limited.status=='spatial_resolution_limit'
    checks=limited.metrics['verification']['spatial_history']
    assert all(not row['residual_pass'] for row in checks)
