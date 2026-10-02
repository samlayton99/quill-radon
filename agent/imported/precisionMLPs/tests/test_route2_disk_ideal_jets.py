"""Fast Cartesian disk jets agree with the independent ridge identity."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_disk_profiles import DiskProfileOperator


@pytest.mark.parametrize('degree',[4,12,24])
def test_cartesian_jacobi_jets_match_angular_identity_at_origin_axes_and_edge(degree):
    op=DiskProfileOperator(degree,centers=65,midpoint=[.3,-.2],scale=[.7,1.3])
    rng=np.random.default_rng(839)
    angle=rng.uniform(0,2*np.pi,15)
    radius=np.sqrt(rng.uniform(0,1,15))
    reference=np.vstack(([[0,0],[1,0],[0,1],[-1,0],[0,-1]],
        np.column_stack((radius*np.cos(angle),radius*np.sin(angle)))))
    points=reference/op.scale+op.midpoint
    indices=np.r_[0,np.arange(max(0,op.size-2*degree),op.size),op.size-1].astype(int)
    derivatives=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))
    fast=op.selected_ideal_columns(points,indices,derivatives)
    oracle=op._selected_ideal_columns_ridge(points,indices,derivatives)
    for d in derivatives:
        scale=max(1,np.max(abs(oracle[d])))
        np.testing.assert_allclose(fast[d],oracle[d],atol=scale*5e-13,rtol=2e-12)


def test_low_degree_cartesian_formulas_and_empty_panels():
    op=DiskProfileOperator(2,centers=33)
    points=np.array([[0.,0.],[.2,-.3]])
    idx=int(np.flatnonzero((op.degrees==2)&(op.harmonics==0))[0])
    jets=op.selected_ideal_columns(points,np.array([idx]),[(0,0),(1,0),(1,1),(2,0)])
    np.testing.assert_allclose(jets[(0,0)][:,0],np.sqrt(3)*(2*np.sum(points**2,axis=1)-1))
    np.testing.assert_allclose(jets[(1,0)][:,0],4*np.sqrt(3)*points[:,0])
    np.testing.assert_allclose(jets[(2,0)][:,0],4*np.sqrt(3))
    np.testing.assert_array_equal(jets[(1,1)],0.)
    assert op.selected_ideal_columns(points,np.array([],int),[(0,0)])[(0,0)].shape==(2,0)
