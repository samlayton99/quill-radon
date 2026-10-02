"""Structural regression checks for the reusable compact model."""
import torch
import pytest
from fixed_qi_mlp import FixedQIMLP


@pytest.fixture(autouse=True)
def fp64():
    original=torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(original)


def test_dense_equivalence_and_calibration():
    x=torch.randn(128,5,generator=torch.Generator().manual_seed(72))
    model=FixedQIMLP(5,channels=(4,3,2),centers=(7,9,11))
    diagnostics=model.initialize_(x,seed=3)
    torch.testing.assert_close(model(x),model.expanded_mlp()(x),rtol=1e-13,atol=1e-14)
    for d in diagnostics:
        torch.testing.assert_close(torch.tensor(d["abs_quantile"]),torch.ones(len(d["abs_quantile"])),rtol=1e-13,atol=1e-14)


def test_geometry_is_immutable_under_training():
    x=torch.randn(64,3,generator=torch.Generator().manual_seed(12))
    model=FixedQIMLP(3,channels=(4,3),centers=(9,11))
    model.initialize_(x,seed=4)
    before={name:b.clone() for name,b in model.named_buffers()}
    for bank in model.banks:
        assert abs(bank.gamma.item()*bank.h-.25)<1e-15
        torch.testing.assert_close(bank.centers[1:]-bank.centers[:-1],torch.full((bank.n-1,),bank.h),rtol=1e-13,atol=1e-14)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for _ in range(5):
        opt.zero_grad();loss=(model(x).squeeze()-torch.sin(x[:,0])).square().mean();loss.backward()
        assert all(p.weight.grad.norm()>0 for p in model.projections)
        opt.step()
    assert all(torch.equal(before[name],b) for name,b in model.named_buffers())
    torch.testing.assert_close(model(x),model.expanded_mlp()(x),rtol=1e-13,atol=1e-14)
