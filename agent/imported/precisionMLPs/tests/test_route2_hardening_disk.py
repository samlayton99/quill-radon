"""Nonpolynomial no-slip Navier--Stokes forcing/constraint checks."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_hardening_disk import exact,source,make_problem,ORDERS
from route2_streamed_study import disk_points,boundary_points


def test_manufactured_flow_is_solenoidal_nonlinear_and_zero_on_wall():
    x=disk_points(127,315)
    force=source(x,.3)
    np.testing.assert_allclose(force[:,2],0,atol=2e-15)
    edge=exact(torch.tensor(boundary_points(83),dtype=torch.float64)).numpy()
    np.testing.assert_allclose(edge[:,:2],0,atol=2e-15)
    # Changing viscosity changes diffusion while retaining the same nonlinear
    # advection and pressure. Zero-viscosity forcing is substantially nonzero.
    assert np.linalg.norm(source(x,0)[:,:2])>5
    assert np.linalg.norm(exact(torch.tensor(x,dtype=torch.float64)).numpy()[:,:2])>5


def test_prescribed_sources_batch_correctly_and_constraints_do_not_use_interior_truth():
    problem=make_problem(2,centers=65,interior=19,boundary=13)
    block=problem.blocks[0]
    jets={o:torch.zeros((19,3),dtype=torch.float64) for o in ORDERS}
    full=block.function(torch.tensor(block.points),jets,torch.empty((19,0)))
    ids=np.array([1,5,9])
    part=block.batch_function(torch.tensor(block.points[ids]),{o:j[ids] for o,j in jets.items()},
                             torch.empty((3,0)),ids)
    torch.testing.assert_close(part,full[ids],atol=0,rtol=0)
    assert [b.name for b in problem.blocks]==['momentum_and_divergence','no_slip','pressure_gauge']
    assert problem.fields==3 and problem.execution=='streamed'


def test_reflection_balancing_preserves_domain_and_cancels_odd_moments():
    problem=make_problem(2,centers=65,interior=128,reflection_balanced=True)
    points=problem.blocks[0].points
    assert len(points)==128
    assert np.max(np.sum(points**2,axis=1))<=1
    for a,b in [(1,0),(0,1),(3,2),(2,3),(3,3)]:
        assert abs(np.sum(points[:,0]**a*points[:,1]**b))<1e-14
