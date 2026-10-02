"""Frame normalization, differential identity and genuine neural PDE checks."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import eval_gegenbauer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
import route2_frame_iteration as frame
from route2_directional_operator import DirectionalProfileOperator, gegenbauer_profiles


def directions(p):
    angle=np.pi*np.arange(p+1)/(p+1)
    return np.column_stack([np.cos(angle),np.sin(angle)])


@pytest.mark.parametrize('p',[4,8,12])
def test_direct_frame_reproduces_every_monomial(p):
    points,w=frame.quadrature(p+2,4*p+4)
    op=SimpleNamespace(directions=directions(p),degree=p)
    rng=np.random.default_rng(867)
    held=rng.uniform(-.6,.6,(21,2))
    profiles=gegenbauer_profiles(held@op.directions.T,2,p)[:,:,1:]
    for i in range(p+1):
        for j in range(p+1-i):
            f=points[:,0]**i*points[:,1]**j
            mean,b=frame.analyze(f,points,w,op)
            got=mean+np.einsum('qmn,mn->q',profiles,b)
            np.testing.assert_allclose(got,held[:,0]**i*held[:,1]**j,atol=2e-13,rtol=2e-13)


@pytest.mark.parametrize('d',[2,3,5])
def test_ball_operator_ridge_eigenvalue(d):
    s=np.linspace(-.98,.98,71)
    a=d/2
    for n in range(1,13):
        f=eval_gegenbauer(n,a,s)
        first=2*a*eval_gegenbauer(n-1,a+1,s)
        second=4*a*(a+1)*eval_gegenbauer(n-2,a+2,s) if n>=2 else np.zeros_like(s)
        applied=-(1-s*s)*second+(d+1)*s*first+frame.ALPHA*f
        np.testing.assert_allclose(applied,(n*(n+d)+frame.ALPHA)*f,atol=2e-9,rtol=1e-12)


def test_zero_start_without_reference_or_linear_solve(monkeypatch):
    p=12
    op=DirectionalProfileOperator(directions(p),p,centers=257,lam=.2)
    points,w=frame.quadrature(12,40)
    rhs=frame.source(points)
    held=np.random.default_rng(457).uniform(-.65,.65,(40,2))
    expected=frame.target_jets(held)[(0,0)]
    expected_source=frame.source(held)

    def forbidden(*args,**kwargs):
        raise AssertionError('The iteration must not solve a global system or consult the solution.')

    for name in ['solve','lstsq','pinv','svd']:
        monkeypatch.setattr(np.linalg,name,forbidden)
    monkeypatch.setattr(frame,'target_jets',forbidden)
    monkeypatch.setattr(frame,'source',forbidden)
    theta,history=frame.iterate(op,points,w,rhs)
    model=op.torch_model(theta)
    jets=frame.ordinary_jets(model,held)
    assert len(history)<50
    assert np.linalg.norm(jets[(0,0)]-expected)/np.linalg.norm(expected)<1e-10
    residual=frame.linear_part(held,jets)+np.tanh(jets[(0,0)])-expected_source
    assert np.max(abs(residual))<2e-9
    # The early transient contracts at the predicted resolution-independent rate.
    corrections=np.array([h['coordinate_correction_norm'] for h in history])
    assert np.max(corrections[6:20]/corrections[5:19])<.5


def test_scalar_ridge_operator_matches_ordinary_full_hessian():
    op=DirectionalProfileOperator(directions(5),5,centers=129)
    theta=np.random.default_rng(243).normal(size=op.size)*.1
    points=np.random.default_rng(52).uniform(-.6,.6,(25,2))
    u,lu=frame.neural_field_and_operator(op,theta,points)
    j=frame.ordinary_jets(op.torch_model(theta),points)
    np.testing.assert_allclose(u,j[(0,0)],atol=3e-12,rtol=3e-12)
    np.testing.assert_allclose(lu,frame.linear_part(points,j),atol=5e-11,rtol=3e-12)
