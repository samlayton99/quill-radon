"""Independent geometry coverage and equation-declaration checks."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'experiments/expF19_radon_direct_pde'))
from solver.general_domains import ResidualDomain
from general_residual_study import cases, exact_jets, space_time_constraints


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_box_faces_cover_every_free_coordinate_and_have_outward_normals(dimension):
    bounds = np.column_stack([np.arange(dimension)-2.,
                              np.arange(dimension)+np.arange(1, dimension+1)])
    domain = ResidualDomain(bounds)
    samples = domain.boundary(256*2*dimension, seed=917)
    np.testing.assert_allclose(np.linalg.norm(samples.normals, axis=1), 1.)
    assert len(samples.points) == 256*2*dimension
    for axis in range(dimension):
        for side in range(2):
            expected_normal = np.zeros(dimension)
            expected_normal[axis] = 2*side-1
            selected = np.all(samples.normals == expected_normal, axis=1)
            assert selected.sum() == 256
            points = samples.points[selected]
            np.testing.assert_allclose(points[:, axis], bounds[axis, side])
            for free in range(dimension):
                if free == axis:
                    continue
                unit = (points[:, free]-bounds[free, 0])/np.diff(bounds[free])[0]
                # Detects the dangerous Sobol-bit/face assignment correlation,
                # where an entire half/quarter of a face can go unsampled.
                assert unit.min() < .02
                assert unit.max() > .98
                counts, _ = np.histogram(unit, bins=np.linspace(0, 1, 9))
                assert counts.min() >= 24


def test_space_time_boundaries_cover_full_time_on_each_side():
    domain = ResidualDomain([[-1., 2.], [0., .5]])
    exact = lambda x: x[:, 0]+x[:, 1]
    blocks = space_time_constraints(domain, exact, initial_derivative=True)(512, 736)
    for side in [-1., 2.]:
        selected = blocks[0].points[:, 0] == side
        t = blocks[0].points[selected, 1]
        assert len(t) >= 250
        assert t.min() < .01 and t.max() > .49
        counts, _ = np.histogram(t, bins=np.linspace(0, .5, 9))
        assert counts.min() >= 24
    np.testing.assert_equal(blocks[1].points[:, 1], 0.)
    assert blocks[1].points[:, 0].min() < -.95
    assert blocks[1].points[:, 0].max() > 1.95


def test_perforated_ellipse_samples_both_boundary_components_and_normal_orientation():
    domain = cases()['irregular_variable_diffusion'].domain
    for seed in [17, 829]:
        interior = domain.interior(1000, seed)
        outer = interior[:, 0]**2+(interior[:, 1]/.8)**2
        hole = (interior[:, 0]-.25)**2+(interior[:, 1]+.08)**2
        assert np.all(outer < 1)
        assert np.all(hole > .22**2)
        samples = domain.boundary(1000, seed)
        x = samples.points
        radius2 = (x[:, 0]-.25)**2+(x[:, 1]+.08)**2
        inner = np.abs(radius2-.22**2) < 1e-8
        outer = np.abs(x[:, 0]**2+(x[:, 1]/.8)**2-1) < 1e-8
        assert np.all(inner | outer)
        assert inner.sum() > 50 and outer.sum() > 200
        expected = np.column_stack([x[:, 0], x[:, 1]/.8**2])
        expected[inner] = np.column_stack([.25-x[inner, 0], -.08-x[inner, 1]])
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        np.testing.assert_allclose(samples.normals, expected, atol=1e-8)
        # Each component must include each angular octant; agreement at a few
        # points of the hole is not adequate boundary coverage.
        for mask, center, scale in [(inner, [.25, -.08], [1., 1.]),
                                    (outer, [0., 0.], [1., .8])]:
            z = (x[mask]-center)/scale
            angle = np.arctan2(z[:, 1], z[:, 0])
            counts, _ = np.histogram(angle, bins=np.linspace(-np.pi, np.pi, 9))
            assert counts.min() > 0


def test_exact_jets_independent_mixed_polynomial_and_zero_derivatives():
    x = np.array([[-.7, .2], [.3, -.8], [.6, .4]])
    f = lambda z: torch.stack([z[:, 0]**3*z[:, 1]**2,
                               2*z[:, 0]+3*z[:, 1]+4,
                               torch.ones_like(z[:, 0])], axis=1)
    jets = exact_jets(f, x, [(1, 1), (2, 2), (0, 4), (0, 0)])
    np.testing.assert_allclose(jets[(1, 1)][:, 0], 6*x[:, 0]**2*x[:, 1])
    np.testing.assert_allclose(jets[(2, 2)][:, 0], 12*x[:, 0])
    np.testing.assert_equal(jets[(0, 4)].numpy(), np.zeros((3, 3)))
    np.testing.assert_equal(jets[(1, 1)][:, 1:].numpy(), np.zeros((3, 2)))


@pytest.mark.parametrize('name', ['burgers_ivp', 'wave_ivp', 'steady_navier_stokes',
                                 'inverse_diffusivity'])
def test_unforced_equation_truth_satisfies_declared_operator(name):
    case = cases()[name]
    x = case.domain.interior(113, 723)
    jets = exact_jets(case.exact, x, case.derivatives)
    parameters = torch.tensor(np.broadcast_to(case.parameter_truth,
        (len(x), len(case.parameter_truth))).copy(), dtype=torch.float64)
    residual = case.operator(torch.tensor(x, dtype=torch.float64), jets, parameters)
    assert torch.max(torch.abs(residual)).item() < 2e-13


def test_ivps_supply_no_final_time_interior_constraints():
    for name in ['burgers_ivp', 'wave_ivp']:
        case = cases()[name]
        blocks = case.blocks(200, 372)
        for block in blocks[1:]:
            x = block.points
            on_initial = x[:, 1] == case.domain.bounds[1, 0]
            on_space = ((x[:, 0] == case.domain.bounds[0, 0]) |
                        (x[:, 0] == case.domain.bounds[0, 1]))
            assert np.all(on_initial | on_space)
