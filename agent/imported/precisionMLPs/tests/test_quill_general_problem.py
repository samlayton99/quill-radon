"""End-to-end equation-only declarations, without manual jets or samplers."""
import sys
from pathlib import Path
import numpy as np
import sympy as sp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.general_domains import ResidualDomain
from solver.general_symbolic import DifferentialResidual
from solver.general_problem import GeneralProblem,Condition,trace_sample_count


def test_symbolic_nonlinear_boundary_problem_end_to_end():
    x,y=sp.symbols('x y');u=sp.Function('u')(x,y)
    truth=x*x+y*y
    equation=DifferentialResidual((x,y),(u,),-sp.diff(u,x,2)-sp.diff(u,y,2)+u**3+4-truth**3)
    boundary=DifferentialResidual((x,y),(u,),u-truth)
    problem=GeneralProblem(ResidualDomain([[-1,1],[-1,1]]),equation,
                           [Condition('boundary',boundary)])
    solution=problem.solve(degrees=(2,4,6),tolerance=1e-6)
    assert solution.status=='sampled_checks_passed'
    points=problem.domain.interior(137,371)
    assert np.max(abs(solution.evaluate(points)[:,0]-(points*points).sum(axis=1)))<1e-6


def test_initial_slice_and_boundary_selection_exclude_future_labels():
    x,t=sp.symbols('x t');u=sp.Function('u')(x,t)
    equation=DifferentialResidual((x,t),(u,),sp.diff(u,t)-sp.diff(u,x,2))
    initial=DifferentialResidual((x,t),(u,),u-sp.sin(sp.pi*x))
    boundary=DifferentialResidual((x,t),(u,),u)
    domain=ResidualDomain([[-1,1],[0,1]])
    p=GeneralProblem(domain,equation,[Condition('initial',initial,location='slice',axis=1,value=0.),
                     Condition('walls',boundary,selector=lambda z:np.isclose(abs(z[:,0]),1.))])
    blocks=p.blocks(400,21)
    assert np.all(blocks[1].points[:,1]==0.)
    assert np.all(np.isclose(abs(blocks[2].points[:,0]),1.))
    assert not np.any((blocks[2].points[:,1]==1)&(abs(blocks[2].points[:,0])<1))


def test_boundary_sampling_scales_with_trace_dimension():
    # Every 3D face of a four-input degree12 polynomial has455 trace modes.
    assert trace_sample_count(12,4)==910
    assert trace_sample_count(24,4)>4*trace_sample_count(12,4)
    assert trace_sample_count(24,2)==50
