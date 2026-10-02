from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.residual_scaling import local_equation_scales,normalize_equations
from solver.general_residual import ResidualBlock
from solver.general_domains import ResidualDomain
from solver.general_adaptive import ResidualDeclaration


def test_operator_scaling_excludes_forcing_and_uses_units():
    x=np.array([[.1,.2],[.4,.8]])
    def make(forcing):
        return ResidualBlock('PDE',x,lambda x,j,p:3*j[(0,1)]+(2+x[:,:1])*j[(2,0)]-forcing,
                             ((0,1),(2,0)))
    expected=np.sqrt((3*5/4)**2+((2+x[:,0])*5/2**2)**2)
    for forcing in (0.,1e12):
        actual=local_equation_scales(make(forcing),1,[],[2.,4.],[5.])
        np.testing.assert_allclose(actual[:,0],expected)


def test_fixed_zero_jet_scaling_and_batched_declaration_preserve_roots():
    domain=ResidualDomain([[0.,1.],[0.,1.]])
    def blocks(count,seed):
        x=domain.interior(count,seed)
        return [ResidualBlock('PDE',x,lambda x,j,p:j[(0,1)]+5*(j[(0,0)]**3-j[(0,0)]),((0,0),(0,1))),
                ResidualBlock('boundary',x,lambda x,j,p:j[(0,0)]-1,((0,0),))]
    original=ResidualDeclaration(domain,blocks)
    normalized=normalize_equations(original,['PDE'])
    a=original.make_blocks(7,81);b=normalized.make_blocks(7,81)
    assert b[1].weight==a[1].weight
    rows=np.array([1,4,6]);x=torch.tensor(a[0].points[rows]);v=torch.tensor([[.1],[.4],[.7]])
    jets={(0,0):v,(0,1):-5*(v**3-v)}
    np.testing.assert_allclose(b[0].batch_function(x,jets,torch.empty((3,0)),rows),0.,atol=1e-15)
    jets[(0,1)]+=2
    np.testing.assert_allclose(b[0].batch_function(x,jets,torch.empty((3,0)),rows),2/np.sqrt(26))
