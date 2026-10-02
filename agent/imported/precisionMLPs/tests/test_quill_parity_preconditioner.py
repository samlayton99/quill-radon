"""Physics-coupled parity groups inferred from local residual derivatives."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import lsmr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_box_profiles import BoxProfileOperator
from route2_disk_profiles import DiskProfileOperator
from solver.general_residual import (ResidualBlock, ResidualProblem, linearize_residual,
    infer_field_parity_shifts, _RightPreconditioner, _coordinate_parities, solve_residual)


def stokes_sensitivities(dimension=3, with_time=False):
    inputs = dimension+int(with_time)
    zero = (0,)*inputs
    first = [tuple(int(j == axis) for j in range(inputs)) for axis in range(dimension)]
    second = [tuple(2*int(j == axis) for j in range(inputs)) for axis in range(dimension)]
    orders = (zero,)+tuple(first)+tuple(second)
    if with_time:
        orders += ((0,)*dimension+(1,),)
    points = np.random.default_rng(832).uniform(-1, 1, size=(7, inputs))
    block = ResidualBlock('coupled', points, lambda x, j, p: j[zero], orders)
    sensitivity = np.zeros((len(points), dimension+1, len(orders), dimension+1))
    for field in range(dimension):
        for axis in range(dimension):
            sensitivity[:, field, orders.index(second[axis]), field] = -.1
        sensitivity[:, field, orders.index(first[field]), dimension] = 1.
        sensitivity[:, dimension, orders.index(first[field]), field] = 1.
        if with_time:
            sensitivity[:, field, -1, field] = 1.
    return block, sensitivity


@pytest.mark.parametrize('dimension', [2, 3])
def test_constant_local_sensitivities_recover_velocity_pressure_parity(dimension):
    block, sensitivity = stokes_sensitivities(dimension)
    result = infer_field_parity_shifts(dimension+1, dimension, [(block, sensitivity)])
    shifts = np.asarray(result['shifts'])
    assert all(result['axis_mask'])
    for field in range(dimension):
        np.testing.assert_array_equal(shifts[field] ^ shifts[-1], np.eye(dimension, dtype=int)[field])


def test_time_parity_conflict_preserves_spatial_coupling_and_ignores_variable_terms():
    block, sensitivity = stokes_sensitivities(3, with_time=True)
    # A varying local convection coefficient must not supply a spurious
    # constant-coefficient parity constraint.
    sensitivity[:, 0, 0, 1] = block.points[:, 0]
    result = infer_field_parity_shifts(4, 4, [(block, sensitivity)])
    assert result['axis_mask'] == [True, True, True, False]
    assert result['inconsistent_axes'] == [3]
    assert result['varying_terms_ignored'] == 1
    shifts = np.asarray(result['shifts'])
    for field in range(3):
        np.testing.assert_array_equal((shifts[field] ^ shifts[-1])[:3], np.eye(3, dtype=int)[field])


def test_aggregation_inequality_and_singleton_do_not_infer_symmetry():
    block, sensitivity = stokes_sensitivities(2)
    aggregated = ResidualBlock('integral', block.points, block.function, block.derivatives,
                               aggregation=np.ones((1, len(block.points))))
    inequality = ResidualBlock('inequality', block.points, block.function, block.derivatives,
                               relation='le')
    singleton = ResidualBlock('single', block.points[:1], block.function, block.derivatives)
    result = infer_field_parity_shifts(3, 2, [(aggregated, sensitivity),
        (inequality, sensitivity), (singleton, sensitivity[:1])])
    assert result['constraint_edges'] == 0
    assert result['blocks_ignored'] == 3
    assert result['shifts'] == [[0, 0]]*3


def test_disk_mode_reflection_parities_are_explicit_geometry_properties():
    operator = DiskProfileOperator(5, centers=129)
    parity = _coordinate_parities(operator)
    points = np.random.default_rng(84).uniform(-.5, .5, size=(7, 2))
    reference = operator.evaluate(points)
    for axis in range(2):
        reflected = points.copy(); reflected[:, axis] *= -1.
        np.testing.assert_allclose(operator.evaluate(reflected), reference*(-1.)**parity[:, axis],
                                   atol=4e-11, rtol=4e-11)


def actual_ns_problem():
    features = BoxProfileOperator([[-1., 1.]]*2, 4, centers=129)
    nodes, _ = leggauss(8)
    x, y = np.meshgrid(nodes, nodes, indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel()))
    boundary = np.r_[np.c_[np.full(8, -1.), nodes], np.c_[np.full(8, 1.), nodes],
                     np.c_[nodes, np.full(8, -1.)], np.c_[nodes, np.full(8, 1.)]]
    zero, dx, dy, xx, yy = (0, 0), (1, 0), (0, 1), (2, 0), (0, 2)
    orders = (zero, dx, dy, xx, yy)
    def residual(x, jets, parameters):
        u, v = jets[zero][:, 0], jets[zero][:, 1]
        return torch.stack((
            -.1*(jets[xx][:, 0]+jets[yy][:, 0])+jets[dx][:, 2]+u*jets[dx][:, 0]+v*jets[dy][:, 0],
            -.1*(jets[xx][:, 1]+jets[yy][:, 1])+jets[dy][:, 2]+u*jets[dx][:, 1]+v*jets[dy][:, 1],
            jets[dx][:, 0]+jets[dy][:, 1]), dim=1)
    pde = ResidualBlock('ns', points, residual, orders)
    blocks = [pde, ResidualBlock('wall', boundary, lambda x, j, p: j[zero][:, :2], (zero,)),
              ResidualBlock('gauge', np.zeros((1, 2)), lambda x, j, p: j[zero][:, 2], (zero,))]
    problem = ResidualProblem(features, blocks, fields=3)
    return problem, pde


def test_actual_ns_linearization_field_parity_blocks_improve_conditioning_and_iterations():
    problem, pde = actual_ns_problem()
    linear = linearize_residual(problem, np.zeros((problem.features.size, 3)))
    # Metadata is built from local derivatives, independently of the matrix
    # assembled below ONLY for this 45-unknown conditioning diagnostic.
    jets = {order: torch.zeros((len(pde.points), 3), dtype=torch.float64, requires_grad=True)
            for order in pde.derivatives}
    raw = pde.function(torch.tensor(pde.points), jets, torch.empty((len(pde.points), 0)))
    channels = []
    for field in range(3):
        gradients = torch.autograd.grad(raw[:, field].sum(), list(jets.values()),
                                        allow_unused=True, retain_graph=True)
        channels.append([np.zeros((len(pde.points), 3)) if value is None else value.detach().numpy()
                         for value in gradients])
    sensitivity = np.transpose(np.asarray(channels), (2, 0, 1, 3))
    linear.operator.field_parity_inference = infer_field_parity_shifts(3, 2, [(pde, sensitivity)])
    count = problem.features.size*problem.fields
    identity = np.eye(count)
    matrix = np.column_stack([linear.operator @ column for column in identity])
    inverse_scale = 1/np.linalg.norm(matrix, axis=0)
    right_hand_side = matrix @ np.random.default_rng(158).normal(size=count)
    conditions, iterations = {}, {}
    for mode in ('diagonal', 'none', 'auto'):
        if mode == 'diagonal':
            right = np.diag(inverse_scale)
        else:
            preconditioner = _RightPreconditioner(problem, linear.operator, inverse_scale,
                                                   64, 1e-12, field_parity=mode)
            right = np.column_stack([preconditioner.apply(column) for column in identity])
            assert preconditioner.metrics['maximum_block_size'] <= count//2
            rng = np.random.default_rng(72)
            first, second = rng.normal(size=count), rng.normal(size=count)
            np.testing.assert_allclose(preconditioner.apply(first) @ second,
                first @ preconditioner.transpose(second), atol=3e-13, rtol=3e-13)
        conditioned = matrix @ right
        conditions[mode] = np.linalg.cond(conditioned)
        solution = lsmr(conditioned, right_hand_side, atol=1e-13, btol=1e-13, maxiter=500)
        iterations[mode] = solution[2]
        assert np.linalg.norm(conditioned@solution[0]-right_hand_side)/np.linalg.norm(right_hand_side) < 1e-11
    assert conditions['auto'] < conditions['none']/3
    assert conditions['auto'] < conditions['diagonal']/3
    assert iterations['auto'] < iterations['none']/3
    assert iterations['auto'] < iterations['diagonal']/3


def test_option_validation_preserves_none_default_and_explicit_shifts():
    problem, pde = actual_ns_problem()
    with pytest.raises(ValueError, match='preconditioner_field_parity'):
        solve_residual(problem, max_iterations=0, preconditioner_field_parity='unknown')
    with pytest.raises(ValueError, match='binary'):
        solve_residual(problem, max_iterations=0, preconditioner_field_parity=np.zeros((2, 2), int))
    linear = linearize_residual(problem, np.zeros((problem.features.size, 3)))
    shifts = np.array([[0, 0], [1, 1], [1, 0]])
    preconditioner = _RightPreconditioner(problem, linear.operator,
        np.ones(linear.operator.shape[1]), field_parity=shifts)
    assert preconditioner.metrics['field_parity']['mode'] == 'explicit'
