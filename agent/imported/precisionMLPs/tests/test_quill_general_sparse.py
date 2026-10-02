"""Sparse product banks preserve actual constructed fields and derivatives."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.general_features import ConstructedFeatures
from solver.general_sparse import (selected_feature_bank,coefficient_support,
                                  embed_coefficients,extend_support)


@pytest.mark.parametrize('backend',['polynomial','quill'])
def test_restriction_preserves_constructed_field_and_physical_derivatives(backend):
    full=ConstructedFeatures([[-2,3],[-1,1]],6,backend=backend)
    subset=np.array([(0,0),(2,0),(1,1),(0,6),(4,2)])
    reduced=selected_feature_bank(full.bounds,subset,backend=backend)
    co=np.zeros((full.size,2));lookup={tuple(a):i for i,a in enumerate(full.multiindices)}
    for k,a in enumerate(subset):co[lookup[tuple(a)]]=[.2*k,-.1*k*k]
    short=embed_coefficients(full,co,reduced)
    points=np.random.default_rng(431).uniform(full.bounds[:,0],full.bounds[:,1],(131,2))
    for d in [(0,0),(1,0),(0,2),(1,1)]:
        np.testing.assert_allclose(full.evaluate(points,d)@co,reduced.evaluate(points,d)@short,
                                   atol=3e-13,rtol=3e-14)


def test_coefficient_discard_bound_and_frontier_include_missing_interactions():
    full=ConstructedFeatures([[-1,1]]*3,4,backend='polynomial')
    co=np.random.default_rng(872).normal(size=(full.size,2))*1e-5
    co[0]=[1.,2.]
    support,metrics=coefficient_support(full,co,discard_budget=1e-4)
    bank=selected_feature_bank(full.bounds,support,backend='polynomial')
    points=np.random.default_rng(878).uniform(-1,1,(251,3))
    delta=full.evaluate(points)@co-bank.evaluate(points)@embed_coefficients(full,co,bank)
    assert abs(delta).max()<=metrics['discarded_coefficient_bound']+1e-15
    grown=extend_support(np.array([[0,0,0],[2,0,0],[0,2,0]]),layers=2)
    assert (1,1,1) in {tuple(a) for a in grown}
    assert (4,0,0) in {tuple(a) for a in grown}
    assert (0,0,2) in {tuple(a) for a in grown}


def test_derivative_aware_discard_bound_covers_physical_second_derivatives():
    full=ConstructedFeatures([[0,.5],[-1,1]],7,backend='polynomial')
    co=np.random.default_rng(951).normal(size=(full.size,1))*1e-8
    co[0]=1
    orders=[(0,0),(2,0),(1,1)]
    support,metrics=coefficient_support(full,co,discard_budget=1e-4,derivatives=orders)
    bank=selected_feature_bank(full.bounds,support,backend='polynomial')
    points=np.random.default_rng(982).uniform(full.bounds[:,0],full.bounds[:,1],(183,2))
    for order in orders:
        delta=full.evaluate(points,order)@co-bank.evaluate(points,order)@embed_coefficients(full,co,bank)
        assert abs(delta).max()<=metrics['discarded_coefficient_bound']+1e-14


def test_axis_growth_adds_every_direction_and_preserves_parent_closure():
    grown=extend_support(np.array([[0,0],[1,1]]),layers=3,mode='axis')
    support={tuple(a) for a in grown}
    assert (4,1) in support and (1,4) in support
    assert (2,2) not in support
    for a in support:
        for axis in range(2):
            if a[axis]:
                parent=list(a);parent[axis]-=1
                assert tuple(parent) in support
