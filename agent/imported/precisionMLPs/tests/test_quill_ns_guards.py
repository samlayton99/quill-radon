"""Regressions for spurious success in a truncated incompressible PDE solve."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver import PeriodicNS


def constant_velocity(points):
    return np.column_stack([np.zeros(len(points)),np.ones(len(points)),np.zeros(len(points))])


def test_ns_rejects_an_initial_mode_aliased_into_the_retained_basis():
    model=PeriodicNS(cutoff=3,n_centers=129)
    initial=lambda x:np.column_stack([np.sin(9*x[:,1]),np.zeros((len(x),2))])
    result=model.solve(initial=initial,final_time=.01,dt=.005)
    assert result.status=='initial_underresolved'
    assert result.time==0.
    assert result.metrics['initial_condition']['relative_error']>.5


@pytest.mark.parametrize('amplitude',[
    lambda t:t*(1-t),
    lambda t:np.sin(10*np.pi*t)**2,
])
def test_ns_rejects_unresolved_force_even_when_final_or_periodic_checks_see_zero(amplitude):
    model=PeriodicNS(cutoff=3,n_centers=129,viscosity=.001)
    force=lambda t,x:np.column_stack([np.zeros((len(x),2)),amplitude(t)*np.sin(30*x[:,0])])
    result=model.solve(initial=constant_velocity,forcing=force,final_time=1.,dt=.05)
    assert result.status=='underresolved'
    # Time refinement alone agrees on the same incorrectly truncated problem.
    assert result.metrics['time_refinement_relative']<1e-8


def test_ns_accepts_force_absorbed_entirely_by_pressure():
    model=PeriodicNS(cutoff=3,n_centers=129)
    gradient=lambda t,x:np.column_stack([np.cos(x[:,0]),np.zeros((len(x),2))])
    result=model.solve(initial=constant_velocity,forcing=gradient,final_time=.02,dt=.005)
    assert result.status=='converged_sampled'
    points=np.array([[.2,.5,.4],[-.7,1.,-.1]])
    np.testing.assert_allclose(result.evaluate(points),constant_velocity(points),atol=1e-10)
