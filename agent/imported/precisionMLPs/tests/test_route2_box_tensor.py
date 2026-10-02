"""Sum factorization changes ideal correction evaluation, never the primal."""
from pathlib import Path
import sys
import tracemalloc
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_box_profiles import BoxProfileOperator
from route2_box_tensor import tensor_product_plan
from solver.general_residual import ResidualProblem,ResidualBlock,linearize_residual


def orders_for(d):
    zero=(0,)*d
    first=tuple(tuple(int(a==axis) for a in range(d)) for axis in range(d))
    second=tuple(tuple(2*int(a==axis) for a in range(d)) for axis in range(d-1))
    return (zero,)+first+second


@pytest.mark.parametrize('dimension',[2,3,4])
def test_tensor_fields_match_selected_coordinates_and_exact_transpose(dimension):
    rng=np.random.default_rng(46+dimension)
    bounds=np.array([[-2.,1.+axis*.3] for axis in range(dimension)])
    features=BoxProfileOperator(bounds,5,centers=33)
    points=bounds[:,0]+rng.uniform(.2,.8,(13,dimension))*np.diff(bounds,axis=1)[:,0]
    orders=orders_for(dimension)
    coefficients=rng.normal(size=(features.size,4))
    panels=features.selected_ideal_columns(points,np.arange(features.size),orders)
    state=features.prepare_ideal_tensor_forward(coefficients)
    actual=features.forward_ideal_tensor_jets(state,points,orders)
    cotangents={order:rng.normal(size=(len(points),4)) for order in orders}
    for order in orders:
        np.testing.assert_allclose(actual[order],panels[order]@coefficients,atol=2e-11,rtol=2e-13)
    gradient=features.prepare_ideal_tensor_adjoint(4)
    # Accumulation across point batches must equal one global transpose.
    for sl in (slice(0,6),slice(6,None)):
        features.accumulate_ideal_tensor_jets({d:v[sl] for d,v in cotangents.items()},points[sl],gradient)
    result=features.finish_ideal_tensor_adjoint(gradient)
    expected=sum(panels[d].T@cotangents[d] for d in orders)
    np.testing.assert_allclose(result,expected,atol=2e-11,rtol=2e-13)
    np.testing.assert_allclose(sum(np.sum(cotangents[d]*actual[d]) for d in orders),
                               np.sum(result*coefficients),atol=2e-10,rtol=3e-13)


def test_tensor_allocation_cap_and_measured_workspace():
    orders=orders_for(4)
    plan=tensor_product_plan(4,16,4,128,orders)
    assert plan['enabled'] and plan['rows']<128
    assert plan['workspace_bytes']<=16*1024**2
    assert not tensor_product_plan(7,8,4,128,orders_for(7))['enabled']
    features=BoxProfileOperator([[-1.,1.]]*4,16,centers=33)
    rng=np.random.default_rng(44)
    points=rng.uniform(-.5,.5,(plan['rows'],4));coefficients=rng.normal(size=(features.size,4))
    tracemalloc.start()
    state=features.prepare_ideal_tensor_forward(coefficients)
    values=features.forward_ideal_tensor_jets(state,points,orders)
    _,forward_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert forward_peak<plan['workspace_bytes']
    del state,values
    cotangents={d:rng.normal(size=(len(points),4)) for d in orders}
    tracemalloc.start()
    state=features.prepare_ideal_tensor_adjoint(4)
    features.accumulate_ideal_tensor_jets(cotangents,points,state)
    _,adjoint_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert adjoint_peak<plan['workspace_bytes']
    with pytest.raises(ValueError,match='point batch'):
        features.forward_ideal_tensor_jets(state,np.zeros((128,4)),orders)


def test_ideal_engine_tensor_path_never_calls_neurons_or_selected_panels(monkeypatch):
    features=BoxProfileOperator([[-1.,1.]]*3,5,centers=33)
    rng=np.random.default_rng(102)
    points=rng.uniform(-.5,.5,(17,3));orders=orders_for(3)
    block=ResidualBlock('coupled',points,
        lambda x,j,p:j[orders[0]]**2+sum(j[d] for d in orders[1:]),orders)
    problem=ResidualProblem(features,[block],fields=3,execution='streamed',batch_size=7)
    linear=linearize_residual(problem,rng.normal(size=(features.size,3))*.01)
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected basis evaluation in tensor correction')
    monkeypatch.setattr(features,'selected_ideal_columns',forbidden)
    monkeypatch.setattr(features.base,'_basis_many',forbidden)
    operator=linear.operator.ideal_operator
    delta=rng.normal(size=operator.shape[1]);cotangent=rng.normal(size=operator.shape[0])
    np.testing.assert_allclose(cotangent@(operator@delta),delta@operator.rmatvec(cotangent),
                               atol=2e-11,rtol=2e-13)
    counts=linear.metrics['product_counts']
    assert counts['ideal_tensor_forward_batches']==counts['ideal_tensor_adjoint_batches']==3
    assert counts['ideal_tensor_workspace_plan']['enabled']
    assert 'ideal_correction_columns_max' not in counts
