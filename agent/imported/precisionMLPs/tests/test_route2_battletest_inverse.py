from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
import route2_battletest_inverse as experiment


def test_reference_satisfies_shared_initial_and_wall_data():
    x=np.linspace(-1,1,61)
    for nu in (.1,.02):
        np.testing.assert_allclose(experiment.oracle(np.column_stack([x,np.zeros_like(x)]),nu),-np.sin(np.pi*x)[:,None],atol=5e-16)
        walls=np.array([(x,t) for x in (-1.,1.) for t in np.linspace(0,1,21)])
        np.testing.assert_allclose(experiment.oracle(walls,nu),0,atol=2e-15)
        q=experiment.points(257,919)
        np.testing.assert_allclose(experiment.oracle(q,nu,256),experiment.oracle(q,nu,512),atol=2e-12,rtol=0)


def test_reference_obeys_nonlinear_pde_independent_torch_autograd():
    from scipy.special import roots_hermite
    nodes,weights=roots_hermite(256)
    q=experiment.points(23,592);q[:,1]=.05+.9*q[:,1]
    for nu in (.1,.02):
        x=torch.tensor(q,dtype=torch.float64,requires_grad=True)
        y=x[:,:1]+torch.sqrt(4*nu*x[:,1:2])*torch.tensor(nodes)
        phi=torch.exp(-torch.cos(np.pi*y)/(2*np.pi*nu))*torch.tensor(weights)
        u=-(phi*torch.sin(np.pi*y)).sum(1,keepdim=True)/phi.sum(1,keepdim=True)
        first=torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        second=torch.autograd.grad(first[:,0].sum(),x)[0][:,:1]
        residual=first[:,1:2]+u*first[:,:1]-nu*second
        assert residual.abs().max()<2e-12
        np.testing.assert_allclose(u.detach().numpy(),experiment.oracle(q,nu),atol=1e-14)


def test_declaration_never_calls_oracle_or_contains_truth(monkeypatch):
    data=experiment.generate_measurements(.1,.001,1)
    def forbidden(*args,**kwargs):raise AssertionError('oracle entered fitting declaration')
    monkeypatch.setattr(experiment,'oracle',forbidden)
    decl=experiment.make_declaration(data,.04)
    blocks=decl.make_blocks(71,109)
    assert decl.parameter_initial==[.04]
    for block in blocks:
        n=len(block.points);j={d:torch.zeros((n,1),dtype=torch.float64) for d in block.derivatives}
        p=torch.full((n,1),.08,dtype=torch.float64)
        value=block.function(torch.tensor(block.points),j,p)
        if block.batch_function:
            rows=np.array([1,3,7]);got=block.batch_function(torch.tensor(block.points[rows]),{d:v[rows] for d,v in j.items()},p[rows],rows)
            np.testing.assert_array_equal(got,value[rows])
    assert len(blocks[-1].points)==32


def test_freeze_is_immutable_and_records_all_combinations(tmp_path):
    first=experiment.freeze(tmp_path);assert first==experiment.freeze(tmp_path)
    assert len(experiment.protocol()['cases'])==14
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Frozen'):experiment.freeze(tmp_path)


def test_reduced_burgers_gate_matches_existing_physics_and_never_uses_oracle(monkeypatch):
    import route2_battletest_inverse_burgers_gate as gate
    data=experiment.Measurements(np.zeros((1,2)),np.zeros((1,1)),0.)
    expected=experiment.make_declaration(data,.04).make_blocks(37,721)[:3]
    def forbidden(*a,**k):raise AssertionError('Oracle entered blind forward')
    monkeypatch.setattr(gate,'oracle',forbidden)
    actual=gate.physics_declaration().make_blocks(37,721)
    rng=np.random.default_rng(9)
    for old,new in zip(expected,actual):
        np.testing.assert_array_equal(old.points,new.points)
        j={d:torch.tensor(rng.normal(size=(len(old.points),1))) for d in old.derivatives}
        p=torch.full((len(old.points),1),.04,dtype=torch.float64);x=torch.tensor(old.points)
        np.testing.assert_array_equal(old.function(x,j,p),new.function(x,j,p))
