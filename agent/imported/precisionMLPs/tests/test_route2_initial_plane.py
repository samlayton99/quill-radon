from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_initial_plane_profiles import InitialPlaneProfileOperator,concatenate_scalar_mlps
from route2_affine_disk_profiles import AffineDiskProfileOperator
from solver.route2 import ordinary_jets
from solver.general_residual import ResidualBlock,ResidualProblem,linearize_residual


def test_sparse_multiplication_and_all_derivatives_match_product_rule():
    f=InitialPlaneProfileOperator([[-2.,3.],[1.,4.]],8,centers=65,initial_time=1.3)
    source=AffineDiskProfileOperator(f.bounds,7,centers=65)
    x=np.random.default_rng(33).uniform([-2.,1.],[3.,4.],(17,2))
    orders=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))
    q=source.selected_ideal_columns(x,np.arange(source.size),orders)
    actual=f.full.selected_ideal_columns(x,np.arange(f.full.size),orders)
    factor=(x[:,1]-f.initial_time)*f.scale[1]
    for order in orders:
        expected=factor[:,None]*q[order]
        if order[1]:
            previous=(order[0],order[1]-1)
            expected+=order[1]*f.scale[1]*q[previous]
        np.testing.assert_allclose(actual[order]@f.multiplication.toarray(),expected,atol=3e-12,rtol=2e-12)
    assert f.multiplication.nnz<=7*f.size
    assert np.linalg.matrix_rank(f.multiplication.toarray())==f.size


def test_actual_initial_trace_adjoint_and_flat_export():
    f=InitialPlaneProfileOperator([[-1.,1.],[0.,1.]],6,centers=129)
    rng=np.random.default_rng(535)
    c=rng.normal(size=f.size)*.02
    x=rng.uniform([-1.,0.],[1.,1.],(13,2))
    orders=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))
    values=f.forward_jets(c,x,orders)
    cot={d:rng.normal(size=len(x)) for d in orders}
    lhs=sum(v@cot[d] for d,v in values.items())
    rhs=c@f.adjoint_jets(cot,x)
    np.testing.assert_allclose(lhs,rhs,rtol=2e-12,atol=2e-12)
    columns=f.selected_columns(x,np.arange(f.size),orders)
    for d in orders:np.testing.assert_allclose(columns[d]@c,values[d],atol=5e-12,rtol=5e-12)
    initial=x.copy();initial[:,1]=0.
    trace=f.forward_jets(c,initial,orders)
    for d in orders:
        if d[1]==0:np.testing.assert_array_equal(trace[d],0.)
    model=f.torch_model(c)
    actual=ordinary_jets(model,x,orders)
    for d in orders:np.testing.assert_allclose(actual[d].numpy().ravel(),values[d],rtol=1e-10,atol=1e-10)
    assert [type(layer) for layer in model]==[torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
    assert float(model(torch.from_numpy(initial)).abs().max())<1e-13


def test_multifield_adjoint_and_streamed_nonlinear_jacobian():
    f=InitialPlaneProfileOperator([[-1.,1.],[0.,1.]],4,centers=65)
    rng=np.random.default_rng(172)
    c=rng.normal(size=(f.size,2))*.01;x=rng.uniform([-1.,0.],[1.,1.],(11,2))
    orders=((0,0),(1,0),(0,1),(2,0))
    vals=f.forward_prepared_fields_jets(f.prepare_forward_fields(c),x,orders)
    cot={d:rng.normal(size=(len(x),2)) for d in orders}
    acc=f.prepare_adjoint_fields(2);f.accumulate_adjoint_fields_jets(cot,x,acc)
    grad=f.finish_adjoint_fields(acc)
    np.testing.assert_allclose(sum(np.sum(vals[d]*cot[d]) for d in orders),np.sum(c*grad),atol=1e-11)
    block=ResidualBlock('physics',x,lambda x,j,p:j[(0,1)]+j[(0,0)]**3-.1*j[(2,0)],orders)
    problem=ResidualProblem(f,[block],2,execution='streamed',batch_size=4)
    lin=linearize_residual(problem,c)
    a=rng.normal(size=c.size);b=rng.normal(size=len(lin.residual))
    np.testing.assert_allclose(b@lin.operator.matvec(a),a@lin.operator.rmatvec(b),rtol=1e-11,atol=1e-11)


def test_concatenation_preserves_value_and_derivatives():
    f=InitialPlaneProfileOperator([[-1.,1.],[0.,1.]],3,centers=65)
    c=np.arange(f.size)*.01
    first=f.torch_model(c);second=f.torch_model(c[::-1].copy())
    result=concatenate_scalar_mlps(first,second)
    x=np.random.default_rng(82).uniform([-1.,0.],[1.,1.],(9,2))
    orders=((0,0),(0,1),(2,0))
    combined=ordinary_jets(result,x,orders)
    a=ordinary_jets(first,x,orders);b=ordinary_jets(second,x,orders)
    for order in orders:np.testing.assert_allclose(combined[order],a[order]+b[order],atol=1e-12,rtol=1e-12)
