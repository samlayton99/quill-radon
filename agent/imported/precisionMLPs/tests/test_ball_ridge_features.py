"""Check the new analytic map independently of any PDE reference solution."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.special import eval_gegenbauer
from numpy.polynomial.legendre import legvander

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from ridge_frame_dimension_study import sphere_rule,ball_points
from solver.ball_ridge_features import BallRidgeFeatures,gegenbauer_profiles


@pytest.mark.parametrize('dimension',[2,3,4])
def test_ball_frame_identity_all_degrees(dimension):
    omega,q=sphere_rule(dimension,6)
    nu=np.arange(1,dimension+1,dtype=float);nu/=np.linalg.norm(nu)
    x=ball_points(dimension,31,991)
    for degree in range(7):
        prediction=eval_gegenbauer(degree,dimension/2,x@omega.T)@(q*eval_gegenbauer(degree,dimension/2,omega@nu))
        exact=eval_gegenbauer(degree,dimension/2,x@nu)
        np.testing.assert_allclose(prediction,exact,atol=4e-12,rtol=4e-13)


def test_insufficient_angular_resolution_is_not_fixed_by_more_centers():
    d=4;n=6;omega,q=sphere_rule(d,3)
    nu=np.arange(1,d+1,dtype=float);nu/=np.linalg.norm(nu)
    x=np.r_[ball_points(d,17,213),nu[None,:]]
    actual=eval_gegenbauer(n,d/2,x@omega.T)@(q*eval_gegenbauer(n,d/2,omega@nu))
    truth=eval_gegenbauer(n,d/2,x@nu)
    assert np.max(abs(actual-truth))/eval_gegenbauer(n,d/2,1.)>.1


def test_exact_moments_construct_every_coordinate_without_fitting():
    f=BallRidgeFeatures([[-.5,.5]]*4,4)
    x=np.random.default_rng(163).uniform(-.5,.5,(31,4));z=x*f.scale
    truth=np.ones((len(x),f.size))
    for axis in range(4):truth*=legvander(2*x[:,axis],4)[:,f.multiindices[:,axis]]
    profiles=gegenbauer_profiles(z@f.directions.T,4,4)
    angular=np.einsum('qmn,mnk->qk',profiles,f.profile_map,optimize=True)
    np.testing.assert_allclose(angular,truth,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(f.evaluate(x),truth,atol=1e-12,rtol=1e-12)


def test_plain_mlp_export_jets_and_serialization(tmp_path):
    f=BallRidgeFeatures([[-.7,.5],[-.2,.6],[-1.,1.]],3)
    coefficients=np.random.default_rng(31).normal(size=(f.size,2))*.03
    model=f.torch_model(coefficients)
    assert [type(layer) for layer in model]==[torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
    x=torch.tensor([[.1,.2,.3],[-.3,.1,-.4]],dtype=torch.float64,requires_grad=True)
    y=model(x);gradient=torch.autograd.grad(y[:,0].sum(),x,create_graph=True)[0]
    mixed=torch.autograd.grad(gradient[:,0].sum(),x)[0][:,1]
    points=x.detach().numpy()
    np.testing.assert_allclose(y.detach(),f.evaluate(points)@coefficients,atol=2e-13)
    np.testing.assert_allclose(mixed.detach(),f.evaluate(points,(1,1,0))@coefficients[:,0],atol=2e-12)
    path=tmp_path/'plain.pt';torch.save(model.state_dict(),path)
    reloaded=torch.nn.Sequential(torch.nn.Linear(3,model[0].out_features,dtype=torch.float64),torch.nn.Tanh(),torch.nn.Linear(model[0].out_features,2,dtype=torch.float64))
    reloaded.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(reloaded(x),y,rtol=0,atol=0)


def test_domain_and_derivative_validation():
    f=BallRidgeFeatures([[-1,1]]*2,2)
    with pytest.raises(ValueError,match='ball'):f.evaluate([[3.,3.]])
    with pytest.raises(ValueError,match='Derivative'):f.evaluate([[0.,0.]],(-1,0))


def test_high_degree_moment_construction_avoids_overflow_and_cancellation():
    # factorial(20)*factorial(3) already overflows an int64 intermediate.
    # Moreover ARM longdouble has only float64 precision. The analytic
    # construction must resolve alternating moments BEFORE exporting floats.
    f=BallRidgeFeatures([[-1,1]]*2,24)
    x=np.array([[.1,.2],[-.3,.4],[.6,-.2]])
    profiles=gegenbauer_profiles((x*f.scale)@f.directions.T,2,24)
    constructed=np.einsum('qmn,mnk->qk',profiles,f.profile_map,optimize=True)
    truth=legvander(x[:,0],24)[:,f.multiindices[:,0]]*legvander(x[:,1],24)[:,f.multiindices[:,1]]
    np.testing.assert_allclose(constructed,truth,atol=2e-6,rtol=2e-6)
    # Exact degree/parity zeros protect simple modes against high-degree leakage.
    np.testing.assert_array_equal(f.profile_map[:,1:,0],0)
    assert f.metrics['map_construction_decimal_digits']>=50
def test_shared_derivative_profiles_preserve_values_and_reduce_evaluations(monkeypatch):
    import solver.ball_ridge_features as module
    from route2_scaling_features import LowDegreeSparseRidgeFeatures
    f=LowDegreeSparseRidgeFeatures([[-1,1]]*6,2,centers=65)
    points=np.random.default_rng(701).uniform(-.7,.7,(17,6))
    orders=[(0,)*6]+[tuple(2 if j==axis else 0 for j in range(6)) for axis in range(6)]
    separate={order:f.evaluate(points,order) for order in orders}
    original=module._evaluate_encoding;calls=[]
    def counted(*args):
        calls.append(args[2]);return original(*args)
    monkeypatch.setattr(module,'_evaluate_encoding',counted)
    together=f.evaluate_many(points,orders)
    assert calls==[0,2]
    for order in orders:
        np.testing.assert_array_equal(together[order],separate[order])
