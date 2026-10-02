"""Independent checks of the generic declaration/validation layer."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.general_domains import ResidualDomain
from solver.general_residual import ResidualBlock
from solver.general_adaptive import ResidualDeclaration,solve_declared


def test_constructed_poisson_continuation_and_fresh_validation():
    domain=ResidualDomain([[-1,1],[-1,1]])
    def blocks(n,seed):
        x=domain.interior(n,seed);b=domain.boundary(64,seed+1).points
        return [ResidualBlock('laplacian',x,lambda x,j,p:j[(2,0)]+j[(0,2)]-4,((2,0),(0,2))),
                ResidualBlock('boundary',b,lambda x,j,p:j[(0,0)]-(x*x).sum(axis=1)[:,None],((0,0),))]
    s=solve_declared(ResidualDeclaration(domain,blocks),degrees=(2,4,6),tolerance=1e-6)
    assert s.status=='sampled_checks_passed'
    x=domain.interior(173,815)
    assert np.max(abs(s.evaluate(x)[:,0]-(x*x).sum(axis=1)))<1e-6
    assert len(s.history)>=2
    assert all(not b['reuses_training_locations'] for b in s.history[-1]['validation'])


def test_unresolved_declared_problem_is_not_a_success():
    domain=ResidualDomain([[-1,1]])
    def blocks(n,seed):
        x=domain.interior(n,seed)
        return [ResidualBlock('oscillatory_source',x,
            lambda x,j,p:j[(2,)]-torch.sin(40*x),((2,),)),
            ResidualBlock('boundary',np.array([[-1.],[1.]]),lambda x,j,p:j[(0,)],((0,),))]
    s=solve_declared(ResidualDeclaration(domain,blocks),degrees=(2,4),backend='polynomial',
                     tolerance=1e-6,max_iterations=6)
    assert not s.metrics['validated']
    assert s.status=='resolution_limit'
    assert s.history[-1]['maximum_validation_rms']>.1


def test_clipped_implicit_domain_includes_box_and_zero_surface():
    domain=ResidualDomain([[-1,1],[-1,1]],lambda x:x[:,0])
    b=domain.boundary(512,517)
    assert np.all(b.points[:,0]<=1e-9)
    assert np.any(abs(b.points[:,0]+1)<1e-12)
    assert np.any(abs(b.points[:,1]-1)<1e-12)
    assert np.any(abs(b.points[:,1]+1)<1e-12)
    zero=abs(b.points[:,0])<1e-12
    assert zero.sum()>100
    assert np.allclose(b.normals[zero],[1.,0.])
    allbox=ResidualDomain([[-1,1],[-1,1]],lambda x:-torch.ones(len(x)))
    points=allbox.boundary(80,2).points
    assert np.all(np.max(abs(points),axis=1)==1)


def test_memory_budget_tries_an_affordable_intermediate_degree():
    domain=ResidualDomain([[-1,1]])
    def blocks(n,seed):
        return [ResidualBlock('equation',domain.interior(n,seed),
                              lambda x,j,p:j[(2,)]-2,((2,),)),
                ResidualBlock('boundary',np.array([[-1.],[1.]]),
                              lambda x,j,p:j[(0,)]-1,((0,),))]
    s=solve_declared(ResidualDeclaration(domain,blocks),degrees=(2,6),
                     backend='polynomial',maximum_basis_cache_mb=.011)
    assert s.status=='sampled_checks_passed'
    assert s.metrics['attempted_degrees']==[2,6,4]
    assert [row['degree'] for row in s.history]==[2,4]
    x=domain.interior(51,527)
    assert np.max(abs(s.evaluate(x)[:,0]-x[:,0]**2))<1e-7
