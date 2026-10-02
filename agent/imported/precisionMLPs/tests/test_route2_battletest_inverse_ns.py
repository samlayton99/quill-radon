from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_battletest_inverse_ns import DOMAIN,declaration
from route2_physical_flow import boundary_velocity
from solver.route2_inverse import freeze_physics_parameters


def test_blind_gate_domain_and_boundary_are_the_declared_disk():
    probes=DOMAIN.interior(101,19)
    assert np.max(np.sum(probes**2,axis=1))<1
    blocks=declaration().make_blocks(79,72)
    assert np.max(np.sum(blocks[0].points**2,axis=1))<1
    wall=blocks[1]
    np.testing.assert_allclose(np.linalg.norm(wall.points,axis=1),1,atol=3e-16)
    values=torch.tensor(np.column_stack([boundary_velocity(wall.points,'disk_stirring'),np.zeros(len(wall.points))]))
    params=torch.full((len(wall.points),1),.05,dtype=torch.float64)
    np.testing.assert_allclose(wall.function(torch.tensor(wall.points),{(0,0):values},params),0,atol=1e-15)
    rows=np.array([1,5,17])
    np.testing.assert_array_equal(wall.batch_function(torch.tensor(wall.points[rows]),{(0,0):values[rows]},params[rows],rows),np.zeros((3,2)))


def test_blind_gate_has_zero_force_and_no_free_inner_viscosity():
    frozen=freeze_physics_parameters(declaration(),[.05])
    assert frozen.fields==3 and len(frozen.parameter_initial)==0
    physics=frozen.make_blocks(17,83)[0]
    jets={d:torch.zeros((17,3),dtype=torch.float64) for d in physics.derivatives}
    np.testing.assert_array_equal(physics.function(torch.tensor(physics.points),jets,torch.empty((17,0))),np.zeros((17,3)))
    jets[(2,0)][:,:2]=1
    expected=np.tile([-.05,-.05,0.],(17,1))
    np.testing.assert_array_equal(physics.function(torch.tensor(physics.points),jets,torch.empty((17,0))),expected)


def test_inverse_sensor_generation_quarantines_reference_from_physics(monkeypatch):
    import route2_battletest_inverse_ns as module
    import pytest
    class Reference(torch.nn.Module):
        def forward(self,x):return torch.column_stack([x[:,0]+2*x[:,1],x[:,0]-x[:,1],x[:,0]*0])
    monkeypatch.setattr(module,'load_measurement_reference',lambda:Reference())
    obs=module.make_observations(.001)
    assert obs.values.shape==(16,2) and obs.fields.tolist()==[0,1]
    monkeypatch.setattr(module,'load_measurement_reference',lambda:pytest.fail('Physics read the reference'))
    frozen=freeze_physics_parameters(module.declaration(),[.05])
    for block in frozen.make_blocks(5,73):
        x=torch.tensor(block.points);j={d:torch.zeros((len(x),3),dtype=torch.float64) for d in block.derivatives}
        assert torch.all(torch.isfinite(block.function(x,j,torch.empty((len(x),0)))))


def test_reference_and_field_audits_bound_activation_rows():
    from route2_battletest_inverse_ns import batched_values
    class Counted(torch.nn.Module):
        def __init__(self):super().__init__();self.rows=[]
        def forward(self,x):self.rows.append(len(x));return torch.column_stack([x[:,0],x[:,1],x.sum(1)])
    model=Counted();points=np.arange(514,dtype=float).reshape(257,2)
    values=batched_values(model,points,batch_size=32)
    assert max(model.rows)==32 and sum(model.rows)==257
    np.testing.assert_array_equal(values,np.column_stack([points,points.sum(1)]))
