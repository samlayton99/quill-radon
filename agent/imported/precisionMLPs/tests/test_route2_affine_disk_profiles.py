"""A stable reference-disk chart can cover arbitrary declared 2D boxes."""
from pathlib import Path
import sys
import itertools
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_affine_disk_profiles import AffineDiskProfileOperator
from route2_disk_profiles import DiskProfileOperator
from solver.route2 import footprint,solve_route2,solve_route2_native,ordinary_jets
from solver.general_domains import ResidualDomain
from solver.general_adaptive import ResidualDeclaration
from solver.general_residual import ResidualBlock


def test_affine_box_corners_are_inside_reference_disk_and_direction_ranges():
    bounds=np.array([[-3.,1.],[2.,10.]])
    features=AffineDiskProfileOperator(bounds,12,centers=65)
    corners=np.array(list(itertools.product(*bounds)))
    normalized=(corners-features.midpoint)*features.scale
    np.testing.assert_allclose(abs(normalized),1/np.sqrt(2),atol=2e-16)
    assert np.max(np.linalg.norm(normalized,axis=1))<=1+2e-16
    assert np.max(abs(normalized@features.directions.T))<=1+2e-16
    assert features.metrics['geometry_map_bytes']==0
    assert 'not the restricted' in features.metrics['orthogonality_scope']


def test_affine_disk_reuses_existing_disk_map_normalization_and_physical_jets():
    features=AffineDiskProfileOperator([[-3.,1.],[2.,10.]],8,centers=129)
    reference=DiskProfileOperator(8,centers=129)
    rng=np.random.default_rng(820)
    points=np.array([-3.,2.])+rng.uniform(0,1,(29,2))*np.array([4.,8.])
    normalized=(points-features.midpoint)*features.scale
    coefficients=rng.normal(size=features.size)/(features.degrees+1)**3
    np.testing.assert_array_equal(features.to_directional(coefficients),reference.to_directional(coefficients))
    orders=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))
    actual=features.forward_jets(coefficients,points,orders)
    expected=reference.forward_jets(coefficients,normalized,orders)
    ideal=features.selected_ideal_columns(points,np.arange(features.size),orders)
    reference_ideal=reference.selected_ideal_columns(normalized,np.arange(features.size),orders)
    for order in orders:
        factor=np.prod(features.scale**np.array(order))
        np.testing.assert_allclose(actual[order],expected[order]*factor,atol=5e-13,rtol=3e-13)
        np.testing.assert_allclose(ideal[order],reference_ideal[order]*factor,atol=2e-11,rtol=3e-13)
    model=features.torch_model(coefficients)
    assert [type(layer) for layer in model]==[torch.nn.Linear,torch.nn.Tanh,torch.nn.Linear]
    ordinary=ordinary_jets(model,points,orders)
    for order in orders:
        np.testing.assert_allclose(ordinary[order].numpy().ravel(),actual[order],atol=3e-11,rtol=3e-11)


def bubble_declaration(bounds):
    domain=ResidualDomain(bounds)
    midpoint=np.asarray(bounds).mean(axis=1)
    halfwidth=np.diff(np.asarray(bounds),axis=1).ravel()/2
    def blocks(count,seed):
        points=domain.interior(count,seed)
        trace=domain.boundary(max(80,count//2),seed+1).points
        def equation(x,j,p):
            normalized=(x-torch.tensor(midpoint))/torch.tensor(halfwidth)
            source=2/halfwidth[0]**2*(1-normalized[:,1:2]**2)+2/halfwidth[1]**2*(1-normalized[:,:1]**2)
            return -j[(2,0)]-j[(0,2)]-source
        return [ResidualBlock('poisson',points,equation,((2,0),(0,2))),
                ResidualBlock('zero_boundary',trace,lambda x,j,p:j[(0,0)],((0,0),))]
    return ResidualDeclaration(domain,blocks)


def test_generic_native_driver_solves_rectangle_using_affine_disk_chart():
    declaration=bubble_declaration([[-3.,1.],[2.,10.]])
    solution=solve_route2_native(declaration,coordinates='affine_disk',degrees=(2,4,6),
        centers=129,tolerance=1e-10,max_seconds=20,check_points=37)
    assert solution.status=='sampled_residual_checks_passed',solution.history
    assert isinstance(solution.problem.features,AffineDiskProfileOperator)
    assert solution.metrics['coordinates']=='affine_disk'
    points=declaration.domain.interior(43,812)
    z=(points-np.array([-1.,6.]))/np.array([2.,4.])
    expected=(1-z[:,0]**2)*(1-z[:,1]**2)
    with torch.no_grad():predicted=solution.export()(torch.tensor(points)).numpy().ravel()
    np.testing.assert_allclose(predicted,expected,atol=2e-11,rtol=0.)
    assert max(row['maximum_scaled_rms'] for row in solution.history[-1]['ordinary_export_validation'])<1e-10
    changed=bubble_declaration([[-4.,1.],[2.,10.]])
    with pytest.raises(ValueError,match='affine coordinate chart'):
        solve_route2_native(changed,coordinates='affine_disk',degrees=(8,),initial_solution=solution)


def test_affine_disk_preflight_and_explicit_scope_guards():
    cost=footprint(2,32,257,coordinates='affine_disk',execution='streamed')
    assert cost==footprint(2,32,257,coordinates='disk',execution='streamed')
    assert cost['map_workspace_bytes']==cost['coordinate_transform_state_bytes']==0
    declaration=bubble_declaration([[-1.,1.],[-1.,1.]])
    with pytest.raises(ValueError,match='execution=streamed'):
        solve_route2(declaration,coordinates='affine_disk')
    with pytest.raises(ValueError,match='omit disk_radius'):
        solve_route2_native(declaration,coordinates='affine_disk',disk_radius=2.)
    with pytest.raises(ValueError,match='dimension=2'):
        footprint(3,4,129,coordinates='affine_disk',execution='streamed')
    with pytest.raises(ValueError,match='exactly two dimensions'):
        AffineDiskProfileOperator([[-1.,1.]]*3,4)


def test_existing_explicit_disk_already_supports_the_enclosing_square_circle():
    affine=AffineDiskProfileOperator([[-1.,1.]]*2,4,centers=65)
    existing=DiskProfileOperator(4,centers=65,midpoint=[0.,0.],scale=1/np.sqrt(2))
    np.testing.assert_array_equal(affine.scale,existing.scale)
    np.testing.assert_array_equal(affine.base.physical_directions,existing.base.physical_directions)
