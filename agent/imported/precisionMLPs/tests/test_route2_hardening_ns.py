"""Verify Navier--Stokes benchmark mathematics independently of the solver."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_hardening_ns import (make_declaration, manufactured_jets, orders, physical_terms,
                                 validation_field_torch, known_trigonometric_coefficients)
from route2_hardening_ns import load_native_resume


@pytest.mark.parametrize('family', ['quadratic', 'trigonometric', 'bubble'])
@pytest.mark.parametrize('transient', [False, True])
def test_manufactured_ns_forcing_matches_independent_torch_derivatives(family, transient):
    d = 3+int(transient)
    points = np.random.default_rng(17).uniform(-.8, .8, (11, d))
    if transient:
        points[:, 3] = np.linspace(.01, .23, len(points))
    symbolic = manufactured_jets(points, family, .7, transient)
    torch_jets = {}
    for derivative in symbolic:
        x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
        outputs = validation_field_torch(x, family, .7, transient)
        columns = []
        for field in range(4):
            value = outputs[:, field]
            for axis, count in enumerate(derivative):
                for _ in range(count):
                    if value.requires_grad:
                        value = torch.autograd.grad(value.sum(), x, create_graph=True)[0][:, axis]
                    else:
                        value = torch.zeros_like(value)
            columns.append(value.detach().numpy())
        torch_jets[derivative] = np.column_stack(columns)
        np.testing.assert_allclose(symbolic[derivative], torch_jets[derivative], atol=3e-14, rtol=2e-13)
    expected, divergence = physical_terms(symbolic, .2, d)
    actual, actual_divergence = physical_terms(torch_jets, .2, d)
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=2e-13)
    np.testing.assert_allclose(divergence, 0, atol=2e-15)
    np.testing.assert_allclose(actual_divergence, 0, atol=2e-15)


@pytest.mark.parametrize('transient', [False, True])
def test_all_exact_blocks_vanish_and_batches_select_original_rows(transient):
    spec = make_declaration('trigonometric', transient=transient)
    for block in spec.make_blocks(83, 901):
        d = 3+int(transient)
        jets = {k: torch.tensor(v) for k, v in manufactured_jets(block.points, transient=transient).items()}
        parameters = torch.empty((len(block.points), 0), dtype=torch.float64)
        raw = block.function(torch.tensor(block.points), jets, parameters)
        np.testing.assert_allclose(raw, 0, atol=2e-15)
        ids = np.arange(len(block.points))[::3]
        callback = block.batch_function
        if callback:
            batch = callback(torch.tensor(block.points[ids]), {k: v[ids] for k, v in jets.items()}, parameters[ids], ids)
            np.testing.assert_allclose(batch, raw[ids], atol=0, rtol=0)
    names = {block.name for block in spec.make_blocks(83, 901)}
    assert names == {'momentum', 'incompressibility', 'velocity_boundary', 'pressure_gauge'} | ({'velocity_initial'} if transient else set())
    assert spec.fields == 4
    assert not hasattr(spec, 'initial_coefficients')


def test_bubble_has_zero_velocity_on_every_face_but_nonzero_interior():
    spec = make_declaration('bubble')
    boundary = spec.domain.boundary(102, 771).points
    values = validation_field_torch(torch.tensor(boundary), 'bubble').numpy()
    np.testing.assert_array_equal(values[:, :3], 0)
    interior = spec.domain.interior(73, 321)
    assert np.linalg.norm(validation_field_torch(torch.tensor(interior), 'bubble').numpy()[:, :3]) > .1
    blocks = spec.make_blocks(73, 321)
    zero, first, second, _ = orders(3)
    zero_jets = {order: torch.zeros((73, 4), dtype=torch.float64) for order in (zero,)+first+second}
    # A zero wall trace does not make the zero interior a solution.
    residual = blocks[0].function(torch.tensor(blocks[0].points), zero_jets, torch.empty((73, 0)))
    assert torch.linalg.norm(residual) > .1


def test_trigonometric_ns_has_nonzero_convective_interaction():
    points = np.random.default_rng(13).uniform(-1, 1, (31, 3))
    jets = manufactured_jets(points)
    zero, first, _, _ = orders(3)
    advective = sum(jets[zero][:, axis:axis+1]*jets[first[axis]][:, :3] for axis in range(3))
    assert np.min(np.linalg.norm(advective, axis=0)) > .01


def test_polynomial_body_force_matches_hand_calculation():
    points = np.random.default_rng(217).uniform(-1, 1, (23, 3))
    jets = manufactured_jets(points, 'quadratic')
    momentum, _ = physical_terms(jets, .2, 3)
    expected = np.column_stack([.04*points[:, axis]*(points[:, (axis+1) % 3]**2+
                               points[:, (axis+2) % 3]**2)+
                               .15*(points[:, (axis+1) % 3]+points[:, (axis+2) % 3])+.03
                               for axis in range(3)])
    np.testing.assert_allclose(momentum, expected, rtol=3e-15, atol=3e-17)


@pytest.mark.parametrize('transient', [False, True])
def test_analytical_capacity_coefficients_are_correct_without_a_fit(transient):
    from solver.general_features import _compositions
    dimension = 3+int(transient)
    indices = np.array([a for degree in range(17) for a in _compositions(degree, dimension)])
    points = np.random.default_rng(711).uniform(-1, 1, (37, dimension))
    basis = np.ones((len(points), len(indices)))
    for axis in range(dimension):
        basis *= np.polynomial.legendre.legvander(points[:, axis], 16)[:, indices[:, axis]]
    actual = basis@known_trigonometric_coefficients(indices)
    if transient:
        points[:, 3] = (points[:, 3]+1)/8
    expected = validation_field_torch(torch.tensor(points), transient=transient).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-16, rtol=5e-15)


def test_native_restart_validates_problem_and_does_not_accept_capacity_marker(tmp_path):
    from solver.general_features import _compositions
    indices = np.array([a for degree in range(3) for a in _compositions(degree, 3)])
    config = dict(kind='native_ns_iterate', family='trigonometric', viscosity=.2,
                  frequency=.7, transient=False, centers=65, lam=.2)
    path = tmp_path/'native.npz'
    coeffs = np.arange(len(indices)*4).reshape(-1, 4)*1e-4
    np.savez(path, coefficients=coeffs, multiindices=indices, metadata=json.dumps(config))
    args = dict(family='trigonometric', viscosity=.2, frequency=.7, transient=False, batch_size=17)
    loaded = load_native_resume(path, make_declaration(), **args)
    np.testing.assert_array_equal(loaded.coefficients, coeffs)
    assert loaded.problem.execution == 'streamed'
    with pytest.raises(ValueError, match='changes viscosity'):
        load_native_resume(path, make_declaration(), **dict(args, viscosity=.1))
    config['kind'] = 'known_target_capacity'
    np.savez(path, coefficients=coeffs, multiindices=indices, metadata=json.dumps(config))
    with pytest.raises(ValueError, match='explicitly marked native'):
        load_native_resume(path, make_declaration(), **args)


@pytest.mark.parametrize('transient', [False, True])
def test_reflection_sampling_preserves_spatial_orbits_and_time(transient):
    spec = make_declaration(reflection_orbits=True, transient=transient)
    blocks = spec.make_blocks(97, 547)
    for block in blocks:
        if block.name == 'pressure_gauge':
            continue
        assert len(block.points) % 8 == 0
        grouped = block.points.reshape(-1, 8, 3+int(transient))
        np.testing.assert_allclose(grouped[:, :, :3].mean(axis=1), 0, atol=1e-16)
        np.testing.assert_allclose(np.abs(grouped[:, :, :3]), np.broadcast_to(abs(grouped[:, :1, :3]), grouped[:, :, :3].shape))
        if transient:
            np.testing.assert_array_equal(grouped[:, :, 3], np.broadcast_to(grouped[:, :1, 3], grouped[:, :, 3].shape))
            assert np.min(grouped[:, :, 3]) >= 0
            assert np.max(grouped[:, :, 3]) <= .25
        jets = {d: torch.tensor(v) for d, v in manufactured_jets(block.points, transient=transient).items()}
        residual = block.function(torch.tensor(block.points), jets, torch.empty((len(block.points), 0)))
        np.testing.assert_allclose(residual, 0, atol=2e-15)


@pytest.mark.parametrize('transient', [False, True])
def test_combined_physics_preserves_residual_objective_and_gradient(transient):
    separate = make_declaration(transient=transient).make_blocks(13, 8161)
    combined = make_declaration(transient=transient, combined_physics=True).make_blocks(13, 8161)
    for first_block, second_block in zip(separate[2:], combined[1:]):
        assert first_block.name == second_block.name
        np.testing.assert_array_equal(first_block.points, second_block.points)
    np.testing.assert_array_equal(separate[0].points, combined[0].points)
    # Arbitrary, imperfect jets make this a nontrivial objective/gradient test.
    random = np.random.default_rng(9123)
    jets = {order: torch.tensor(random.normal(size=(13, 4))*.1, requires_grad=True)
            for order in combined[0].derivatives}
    points, parameters = torch.tensor(combined[0].points), torch.empty((13, 0))
    momentum = separate[0].function(points, jets, parameters)
    divergence = separate[1].function(points, jets, parameters)
    joint = combined[0].function(points, jets, parameters)
    torch.testing.assert_close(joint[:, :3], momentum, rtol=0, atol=0)
    torch.testing.assert_close(joint[:, 3], divergence, rtol=0, atol=0)
    original_objective = (momentum.square().sum()+divergence.square().sum())/len(points)
    joint_objective = joint.square().sum()/len(points)
    torch.testing.assert_close(joint_objective, original_objective, rtol=2e-15, atol=1e-16)
    old_gradient = torch.autograd.grad(original_objective, tuple(jets.values()), retain_graph=True)
    new_gradient = torch.autograd.grad(joint_objective, tuple(jets.values()))
    for original, new in zip(old_gradient, new_gradient):
        torch.testing.assert_close(new, original, rtol=2e-15, atol=1e-16)
    indices = np.array([0, 3, 8, 12])
    batch = combined[0].batch_function(points[indices], {d: j[indices] for d, j in jets.items()},
                                       parameters[indices], indices)
    torch.testing.assert_close(batch, joint[indices], rtol=0, atol=0)


def test_native_reencoding_audit_never_uses_target_values_or_target_coefficients(tmp_path, monkeypatch):
    import route2_hardening_ns as benchmark
    from solver.general_features import _compositions
    indices = np.array([a for degree in range(3) for a in _compositions(degree, 3)])
    metadata = dict(kind='native_ns_iterate', family='trigonometric', viscosity=.2,
                    frequency=.7, transient=False, centers=33, lam=.2)
    path = tmp_path/'native.npz'
    np.savez(path, coefficients=np.arange(len(indices)*4).reshape(-1, 4)*1e-3,
             multiindices=indices, metadata=json.dumps(metadata))
    def forbidden(*args, **kwargs):
        raise AssertionError('Native geometry checks must not use an interior reference solution')
    monkeypatch.setattr(benchmark, 'validation_field_torch', forbidden)
    monkeypatch.setattr(benchmark, 'known_trigonometric_coefficients', forbidden)
    report = benchmark.native_reencoding_audit(path, ((33, .2), (65, .25)), transient=False, count=5)
    assert len(report['rows']) == 2
    for row in report['rows']:
        physics = row['blocks'][0]
        assert physics['name'] == 'momentum_and_incompressibility'
        assert len(physics['actual_rms_per_equation']) == 4
        assert np.all(np.isfinite(physics['actual_rms_per_equation']))
        assert max(physics['actual_rms_per_equation']) > .01
