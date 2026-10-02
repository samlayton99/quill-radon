"""Independent moment and ordinary-MLP checks for dimensional scaling work."""
import sys
from pathlib import Path
from itertools import combinations_with_replacement
import numpy as np
import pytest
import torch
from scipy.special import gamma
from numpy.polynomial.legendre import legvander

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_scaling_features import fourth_order_sphere_rule, sixth_order_sphere_rule, LowDegreeSparseRidgeFeatures, ActiveAxisRidgeFeatures


@pytest.mark.parametrize('dimension',[2,4,6,10])
def test_sparse_spherical_rule_matches_every_even_monomial_through_four(dimension):
    directions, weights = fourth_order_sphere_rule(dimension)
    assert len(weights) == dimension**2
    np.testing.assert_allclose(weights.sum(),1,atol=2e-15)
    assert abs(weights).sum() < 3
    for degree in (2,4):
        for axes in combinations_with_replacement(range(dimension),degree):
            counts=np.bincount(axes,minlength=dimension)
            if np.any(counts%2):
                truth=0.
            else:
                truth=gamma(dimension/2)/gamma((dimension+degree)/2)*np.prod(gamma((counts+1)/2)/gamma(.5))
            measured=weights@np.prod(directions[:,axes],axis=1)
            np.testing.assert_allclose(measured,truth,rtol=2e-14,atol=2e-15)


@pytest.mark.parametrize('dimension',[2,5,8])
def test_sparse_sixth_order_rule_all_moments(dimension):
    from math import comb
    directions,weights=sixth_order_sphere_rule(dimension)
    assert len(weights)==sum(2**(k-1)*comb(dimension,k) for k in range(1,min(dimension,3)+1))
    np.testing.assert_allclose(weights.sum(),1,atol=3e-15)
    assert abs(weights).sum()<9
    for degree in (2,4,6):
        for axes in combinations_with_replacement(range(dimension),degree):
            counts=np.bincount(axes,minlength=dimension)
            truth=(0. if np.any(counts%2) else gamma(dimension/2)/gamma((dimension+degree)/2)*np.prod(gamma((counts+1)/2)/gamma(.5)))
            np.testing.assert_allclose(weights@np.prod(directions[:,axes],axis=1),truth,rtol=4e-14,atol=3e-15)


def test_sparse_angular_map_represents_all_quadratic_modes_and_raw_ad():
    f=LowDegreeSparseRidgeFeatures([[-1,1]]*7,2,centers=129)
    points=np.random.default_rng(613).uniform(-.8,.8,(11,7))
    exact=np.ones((len(points),f.size))
    for axis in range(7):
        exact*=legvander(points[:,axis],2)[:,f.multiindices[:,axis]]
    np.testing.assert_allclose(f.evaluate(points),exact,atol=5e-13,rtol=5e-13)
    coefficients=np.random.default_rng(82).normal(size=f.size)*.03
    model=f.torch_model(coefficients)
    assert [type(layer) for layer in model] == [torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
    x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    values=model(x)[:,0]
    first=torch.autograd.grad(values.sum(),x,create_graph=True)[0]
    mixed=torch.autograd.grad(first[:,0].sum(),x)[0][:,1]
    np.testing.assert_allclose(values.detach(),exact@coefficients,atol=5e-13,rtol=5e-13)
    np.testing.assert_allclose(mixed.detach(),f.evaluate(points,(1,1,0,0,0,0,0))@coefficients,atol=2e-12)


def test_active_axis_prior_stays_literal_mlp_and_cannot_depend_on_inactive_axes():
    f=ActiveAxisRidgeFeatures([[-1,1]]*8,(2,6),3,centers=129)
    coefficients=np.random.default_rng(5).normal(size=f.size)*.1
    points=np.random.default_rng(93).uniform(-.8,.8,(5,8))
    model=f.torch_model(coefficients)
    assert model[0].in_features == 8
    inactive=[0,1,3,4,5,7]
    np.testing.assert_array_equal(model[0].weight.detach().numpy()[:,inactive],0)
    zero=(1,0,0,0,0,0,0,0)
    np.testing.assert_array_equal(f.evaluate(points,zero),0)
    np.testing.assert_allclose(model(torch.tensor(points)).detach().numpy()[:,0],f.evaluate(points)@coefficients,atol=2e-13)


def test_boundary_sampling_does_not_alias_sobol_bits_with_face_selection():
    from route2_scaling_study import sample_boundary
    points=sample_boundary(10,(0,1),128)
    for face,block in enumerate(np.split(points,4)):
        axis=face//2
        np.testing.assert_array_equal(block[:,axis],-1. if face%2==0 else 1.)
        free=1-axis
        assert block[:,free].min()<-.9 and block[:,free].max()>.9
