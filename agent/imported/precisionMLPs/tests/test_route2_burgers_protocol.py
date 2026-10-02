"""Reference and physical-coordinate checks for the bounded Burgers study."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.ball_ridge_features import BallRidgeFeatures
from route2_burgers_study import physical_variant,raw_values,initial_data,reference,NU,AMPLITUDE,K


def test_reference_satisfies_unforced_burgers_and_prescribed_initial_condition():
    rng=np.random.default_rng(981)
    x=np.column_stack([rng.uniform(-.5,.5,31),rng.uniform(0,1,31)])
    tx=torch.tensor(x,dtype=torch.float64,requires_grad=True)
    a=AMPLITUDE*torch.exp(-NU*K*K*tx[:,1:2])
    u=2*NU*K*a*torch.sin(K*tx[:,0:1])/(1+a*torch.cos(K*tx[:,0:1]))
    first=torch.autograd.grad(u.sum(),tx,create_graph=True)[0]
    dxx=torch.autograd.grad(first[:,0].sum(),tx)[0][:,0:1]
    r=first[:,1:2]+u*first[:,0:1]-NU*dxx
    assert torch.max(torch.abs(r)).item()<8e-16
    x[:,1]=0.
    np.testing.assert_allclose(initial_data(x[:,0:1]),reference(x),rtol=0,atol=0)


def test_time_slab_mapping_and_plain_export_preserve_physical_derivatives():
    base=BallRidgeFeatures([[-.5,.5],[0.,1.]],3,257,.2)
    f=physical_variant(base,.375,.5)
    x=np.array([[-.2,.4],[.13,.45],[.39,.49]])
    c=np.zeros((f.size,1));index=np.flatnonzero(np.all(f.multiindices==(0,2),axis=1))[0];c[index]=1.
    z=(x[:,1]-.4375)*16
    np.testing.assert_allclose(f.evaluate(x)@c,(.5*(3*z*z-1))[:,None],atol=2e-13)
    np.testing.assert_allclose(f.evaluate(x,(0,1))@c,(48*z)[:,None],atol=2e-11)
    np.testing.assert_allclose(f.evaluate(x,(0,2))@c,np.full((len(x),1),768.),atol=2e-9)
    np.testing.assert_allclose(raw_values(f.compile(c),x),f.evaluate(x)@c,atol=3e-13)
    np.testing.assert_array_equal(base.bounds,[[-.5,.5],[0.,1.]])
