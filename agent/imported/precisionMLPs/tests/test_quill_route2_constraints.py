"""Physical declaration, ambiguity and ordinary-network integration checks."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
import route2_constraints_common as c


def truth_jets(points):
    x = torch.tensor(points,dtype=torch.float64,requires_grad=True)
    values = c.truth(x)
    gradient = torch.autograd.grad(values.sum(),x,create_graph=True)[0]
    xx = torch.autograd.grad(gradient[:,0].sum(),x,retain_graph=True)[0][:,0]
    yy = torch.autograd.grad(gradient[:,1].sum(),x)[0][:,1]
    return {c.ZERO:values.detach().numpy(),c.DX:gradient[:,0].detach().numpy(),
            c.DY:gradient[:,1].detach().numpy(),c.DXX:xx.numpy(),c.DYY:yy.numpy()}


def test_manufactured_forcing_matches_independent_autograd_and_scale_ambiguity():
    points = c.annulus_points(47,823)
    jets = truth_jets(points)
    declaration = c.declarations(3,False,'robin')
    residual = c.strong_residual(points,jets,c.KAPPA,3,declaration['force'])
    np.testing.assert_allclose(residual,0,atol=3e-14)
    linear = c.declarations(0,True,'dirichlet')
    for kappa in (.3,.6,1.,2.7,4.):
        rescaled = {d:v*c.KAPPA/kappa for d,v in jets.items()}
        residual = c.strong_residual(points,rescaled,kappa,0,linear['force'])
        np.testing.assert_allclose(residual,0,atol=3e-14)
    assert np.max(abs(c.truth(c.circle_points(41,c.R0)))) < 1e-16
    assert np.max(abs(c.truth(c.circle_points(41,1.)))) < 1e-15


def test_annulus_integral_and_mixed_flux_signs():
    points,weights = c.annulus_quadrature()
    np.testing.assert_allclose(weights.sum(),1.,atol=2e-15)
    np.testing.assert_allclose(weights@c.truth(points),.8*(1-c.R0**2)**2/6,atol=2e-15)
    declaration = c.declarations(3,False,'robin')
    for name,radius in [('inner',c.R0),('outer',1.)]:
        points = c.circle_points(17,radius)
        jets = {d:torch.tensor(v[:,None]) for d,v in truth_jets(points).items()}
        residual = declaration[name+'_flux'].function(torch.tensor(points),jets,torch.empty((len(points),0)))
        torch.testing.assert_close(residual,torch.zeros_like(residual),atol=2e-14,rtol=0.)


def test_actual_flat_network_inverse_and_unidentifiable_status_audit():
    torch.set_num_threads(1)
    features = c.new_features()
    points = c.annulus_points(12,55109)
    observed = (points,c.truth(points))
    identified,declaration = c.build_problem(features,beta=0,inverse=True,boundary_kind='dirichlet',anchors=observed)
    solution = c.solve_residual(identified,**c.SOLVER_SETTINGS)
    assert solution.status == 'converged'
    assert abs(solution.parameters[0]-c.KAPPA) < 1e-8
    audit = c.local_identifiability(identified,solution)
    assert audit['parameter_identifiable'] and audit['full_column_rank']
    omitted,_ = c.build_problem(features,beta=0,inverse=True,boundary_kind='dirichlet')
    ambiguity = c.local_identifiability(omitted,solution)
    assert not ambiguity['parameter_identifiable']
    assert not ambiguity['full_column_rank']
    model = c.torch_model(features,solution.coefficients)
    assert [type(layer) for layer in model] == [torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
    held = c.annulus_points(19,331)
    with torch.no_grad():
        predicted = model(torch.tensor(held)).numpy()[:,0]
    np.testing.assert_allclose(predicted,c.truth(held),atol=3e-10)
    raw = c.raw_jets(model,held)
    residual = c.strong_residual(held,raw,float(solution.parameters[0]),0,declaration['force'])
    np.testing.assert_allclose(residual,0,atol=2e-9)
