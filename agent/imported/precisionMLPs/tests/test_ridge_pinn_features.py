"""Independent angular, architecture, derivative and PDE checks for flat ridges."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.special import eval_chebyu
from numpy.polynomial.legendre import leggauss

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.ridge_pinn_features import RidgePINNFeatures
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual


def points(n=31,seed=121):
    q=np.random.default_rng(seed).random((n,2))
    return np.sqrt(q[:,0,None])*np.column_stack([np.cos(2*np.pi*q[:,1]),np.sin(2*np.pi*q[:,1])])


def test_common_direction_formula_reproduces_all_ridge_polynomials():
    f=RidgePINNFeatures(8,directions=9,backend='polynomial')
    x=points(151)
    exact=np.column_stack([eval_chebyu(n,x@np.array([np.cos(k*np.pi/(n+1)),np.sin(k*np.pi/(n+1))]))/np.sqrt(np.pi)
                           for n,k in f.mode_pairs])
    np.testing.assert_allclose(f.evaluate(x),exact,atol=2e-13,rtol=2e-13)


def test_disk_orthogonality_and_underresolved_direction_negative_control():
    z,w=leggauss(14); r=np.sqrt((z+1)/2)
    a=np.arange(48)*2*np.pi/48
    x=(r[:,None,None]*np.column_stack([np.cos(a),np.sin(a)])[None,:,:]).reshape(-1,2)
    weights=np.repeat(w/4,48)*2*np.pi/48
    f=RidgePINNFeatures(6,backend='polynomial')
    a=f.evaluate(x)
    np.testing.assert_allclose(a.T@(weights[:,None]*a),np.eye(f.size),atol=2e-13)
    bad=RidgePINNFeatures(6,directions=2,backend='polynomial',allow_angular_underresolution=True)
    assert np.linalg.matrix_rank(bad.evaluate(x),tol=1e-10)<bad.size


def test_flat_dense_tanh_export_with_torch_derivatives():
    f=RidgePINNFeatures(5,centers=129,radius=1.7,midpoint=(.2,-.3))
    x=points(17)*1.7+[.2,-.3]
    c=np.random.default_rng(98).normal(size=(f.size,1))*.1
    net=f.compile(c)
    dense1=torch.nn.Linear(2,len(net['first_bias']),dtype=torch.float64)
    dense2=torch.nn.Linear(len(net['first_bias']),1,dtype=torch.float64)
    with torch.no_grad():
        dense1.weight.copy_(torch.tensor(net['first_weights']))
        dense1.bias.copy_(torch.tensor(net['first_bias']))
        dense2.weight.copy_(torch.tensor(net['output_weights'].T))
        dense2.bias.copy_(torch.tensor(net['output_bias']))
    model=torch.nn.Sequential(dense1,torch.nn.Tanh(),dense2)
    tx=torch.tensor(x,requires_grad=True,dtype=torch.float64)
    y=model(tx)
    first=torch.autograd.grad(y.sum(),tx,create_graph=True)[0]
    second=torch.autograd.grad(first[:,0].sum(),tx)[0]
    for derivative,actual in [((0,0),y.detach().numpy()),((1,0),first[:,0:1].detach().numpy()),
                              ((2,0),second[:,0:1].detach().numpy()),((1,1),second[:,1:2].detach().numpy())]:
        np.testing.assert_allclose(f.evaluate(x,derivative)@c,actual,atol=3e-12,rtol=3e-12)
        np.testing.assert_allclose(f.evaluate_flat(x,c,derivative),actual,atol=3e-12,rtol=3e-12)
    assert f.metrics['product_gates'] is False


def test_domain_guard():
    f=RidgePINNFeatures(2)
    with pytest.raises(ValueError,match='outside'):
        f.evaluate([[1.,1.]])


def test_nonlinear_disk_pde_from_zero_with_no_interior_solution_labels():
    f=RidgePINNFeatures(2,centers=257)
    x=points(140);a=np.arange(41)*2*np.pi/41
    b=np.column_stack([np.cos(a),np.sin(a)])
    def exact(x):return .2+.1*x[:,0:1]+.15*x[:,1:2]+.05*x[:,0:1]*x[:,1:2]
    source=torch.tensor(exact(x)**3)
    boundary=torch.tensor(exact(b))
    blocks=[ResidualBlock('PDE',x,lambda x,j,p:-j[(2,0)]-j[(0,2)]+j[(0,0)]**3-source,((0,0),(2,0),(0,2))),
            ResidualBlock('boundary',b,lambda x,j,p:j[(0,0)]-boundary,((0,0),))]
    s=solve_residual(ResidualProblem(f,blocks),tolerance=1e-11,max_iterations=15,
                     high_accuracy=True,linear_refinement_steps=0,max_seconds=30.)
    assert s.status=='converged'
    held=points(151,99)
    assert np.max(np.abs(s.evaluate(held)-exact(held)))<1e-10
