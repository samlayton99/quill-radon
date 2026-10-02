"""Native boundary-driven steady Navier--Stokes without manufactured truth.

The square cavity has lid u=(1-x^2)^2 at y=1 and stationary other walls.
The optional disk has smooth prescribed tangential stirring. Both use zero
body force, incompressibility, and p(0)=0. No interior velocity/pressure data
or external PDE solution enters initialization, equations, or continuation.

These are moderate-Reynolds-number steady boundary-value tests, not turbulent
flow simulations. Square corners can limit regularity despite the C1 lid;
the disk is a companion that removes the polygon-corner limitation.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import inspect
import json
from pathlib import Path
import resource
import sys
import time

import torch
import numpy as np

from route2_box_profiles import BoxProfileOperator
from route2_disk_profiles import DiskProfileOperator
from route2_streamed_study import disk_points, boundary_points
from solver.general_domains import ResidualDomain
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual
from solver.route2 import export_mlp, ordinary_jets

OUT = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/physical_flow'
ORDERS = ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2))
CASES = ('cavity', 'disk_stirring', 'disk_counterrotating')


def boundary_velocity(points, case='cavity', lid_speed=1.):
    """Prescribed boundary data, defined independently of any interior field."""
    points = np.asarray(points, float)
    x, y = points.T
    if case == 'cavity':
        top = np.isclose(y, 1., atol=1e-14, rtol=0.)
        return np.column_stack((lid_speed*top*(1-x*x)**2, np.zeros(len(points))))
    if case in ('disk_stirring', 'disk_counterrotating'):
        theta = np.arctan2(y, x)
        speed = lid_speed*((1+.25*np.cos(2*theta)+.15*np.sin(3*theta))
                           if case == 'disk_stirring' else
                           (.8*np.cos(3*theta)+.6*np.sin(5*theta)-.4*np.cos(7*theta)))
        return speed[:, None]*np.column_stack((-y, x))
    raise ValueError(f'Unknown physical flow case {case}')


def momentum_and_divergence(jets, viscosity):
    value, dx, dy, dxx, dyy = (jets[a] for a in ORDERS)
    momentum = (value[:, 0:1]*dx[:, :2]+value[:, 1:2]*dy[:, :2]
                -viscosity*(dxx[:, :2]+dyy[:, :2]))
    if isinstance(value, torch.Tensor):
        pressure_gradient = torch.stack((dx[:, 2], dy[:, 2]), dim=1)
        return torch.cat((momentum+pressure_gradient, (dx[:, 0]+dy[:, 1])[:, None]), dim=1)
    pressure_gradient = np.column_stack((dx[:, 2], dy[:, 2]))
    return np.column_stack((momentum+pressure_gradient, dx[:, 0]+dy[:, 1]))


def samples(case, interior, boundary_per_side, seed):
    if case == 'cavity':
        domain = ResidualDomain([[-1., 1.]]*2)
        points = domain.interior(interior, seed)
        # Lobatto points resolve the compatible but not infinitely smooth
        # corner transition; endpoint duplicates impose consistent zero data.
        t = np.cos(np.linspace(0., np.pi, boundary_per_side))
        boundary = np.concatenate((np.column_stack((t, np.ones_like(t))),
                                   np.column_stack((t, -np.ones_like(t))),
                                   np.column_stack((-np.ones_like(t), t)),
                                   np.column_stack((np.ones_like(t), t))))
    elif case in ('disk_stirring', 'disk_counterrotating'):
        points = disk_points(interior, seed)
        boundary = boundary_points(4*boundary_per_side)
    else:
        raise ValueError(f'Unknown physical flow case {case}')
    return points, boundary


def make_problem(case='cavity', degree=8, centers=257, lam=.2, viscosity=.1,
                 lid_speed=1., interior=None, boundary_per_side=65, batch_size=64,
                 coordinate_basis='auto'):
    if case not in CASES or viscosity <= 0 or not np.isfinite(viscosity):
        raise ValueError('Known case and finite positive viscosity required')
    if coordinate_basis not in ('auto', 'box', 'disk'):
        raise ValueError('coordinate_basis must be auto, box or disk')
    basis = ('box' if case == 'cavity' else 'disk') if coordinate_basis == 'auto' else coordinate_basis
    # A disk reference basis can cover a square without changing its physical
    # domain, equations or boundary. The enclosing radius keeps all arguments
    # inside the reference disk, avoiding extrapolated high-degree polynomials.
    features = (BoxProfileOperator([[-1., 1.]]*2, degree, centers=centers, lam=lam, block_size=batch_size)
                if basis == 'box' else DiskProfileOperator(degree, centers=centers, lam=lam,
                    scale=1/np.sqrt(2.) if case == 'cavity' else 1., block_size=batch_size))
    count = max(768, 6*features.size) if interior is None else int(interior)
    points, boundary = samples(case, count, boundary_per_side, 1741)
    trace = torch.tensor(boundary_velocity(boundary, case, lid_speed), dtype=torch.float64)
    blocks = [ResidualBlock('zero_force_momentum_and_divergence', points,
                lambda x, j, p: momentum_and_divergence(j, viscosity), ORDERS),
              ResidualBlock('prescribed_wall_velocity', boundary,
                lambda x, j, p: j[(0, 0)][:, :2]-trace, ((0, 0),), weight=5.,
                batch_function=lambda x, j, p, rows: j[(0, 0)][:, :2]-trace[rows]),
              ResidualBlock('pressure_gauge', np.zeros((1, 2)),
                lambda x, j, p: j[(0, 0)][:, 2], ((0, 0),))]
    return ResidualProblem(features, blocks, fields=3, execution='streamed', batch_size=batch_size)


def coordinate_keys(features):
    if hasattr(features, 'multiindices'):
        return tuple(tuple(int(x) for x in a) for a in features.multiindices)
    return tuple((int(n), int(k), int(s)) for n, k, s in
                 zip(features.degrees, features.harmonics, features.is_sine))


def transfer_coefficients(coefficients, old_keys, features):
    lookup = {key: i for i, key in enumerate(coordinate_keys(features))}
    if any(tuple(key) not in lookup for key in old_keys):
        raise ValueError('Continuation must retain every previous coordinate')
    result = np.zeros((features.size, 3))
    for key, row in zip(old_keys, coefficients):
        result[lookup[tuple(key)]] = row
    return result


def load_native(path, case, lid_speed):
    """Viscosity continuation is allowed, but external/reference states are not."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        if metadata.get('kind') != 'native_boundary_driven_ns':
            raise ValueError('Resume requires a native boundary-driven NS archive')
        if metadata.get('case') != case or metadata.get('lid_speed') != lid_speed:
            raise ValueError('Resume must preserve physical domain and boundary velocity')
        coefficients = np.asarray(saved['coefficients'], float)
        keys = tuple(map(tuple, np.asarray(saved['coordinate_keys'], int)))
        if coefficients.shape != (len(keys), 3) or not np.all(np.isfinite(coefficients)):
            raise ValueError('Native archive has invalid coefficients')
    return coefficients, keys, metadata


def audit(solution, case, viscosity, lid_speed=1., count=2048, previous=None,
          requested_tolerance=1e-14):
    """Independent locations and ordinary exported MLP differentiation."""
    model = export_mlp(solution.problem.features, solution.coefficients)
    points = (ResidualDomain([[-1., 1.]]*2).interior(count, 819193)
              if case == 'cavity' else disk_points(count, 819193))
    jets = ordinary_jets(model, points, ORDERS)
    values = jets[(0, 0)].numpy()
    physical = momentum_and_divergence(jets, viscosity).numpy()
    dx, dy = jets[(1, 0)].numpy(), jets[(0, 1)].numpy()
    advective = values[:, 0:1]*dx[:, :2]+values[:, 1:2]*dy[:, :2]
    viscous = -viscosity*(jets[(2, 0)].numpy()[:, :2]+jets[(0, 2)].numpy()[:, :2])
    pressure_gradient = np.column_stack((dx[:, 2], dy[:, 2]))
    t, weights = np.polynomial.legendre.leggauss(96)
    if case == 'cavity':
        boundary = np.concatenate((np.column_stack((t, -np.ones_like(t))),
            np.column_stack((t, np.ones_like(t))), np.column_stack((-np.ones_like(t), t)),
            np.column_stack((np.ones_like(t), t))))
        normals = np.repeat(((0., -1.), (0., 1.), (-1., 0.), (1., 0.)), len(t), axis=0)
        boundary_weights = np.tile(weights, 4)
        # Evaluate points very close to each corner, separately from the
        # uniformly sampled interior audit. Corner singularity cannot hide.
        distances = np.geomspace(1e-5, .15, 24)
        corner_points = np.concatenate([np.column_stack((sx*(1-distances), sy*(1-distances)))
                                         for sx in (-1, 1) for sy in (-1, 1)])
        corner_physics = momentum_and_divergence(ordinary_jets(model, corner_points, ORDERS), viscosity).numpy()
    else:
        boundary = boundary_points(389, .173)
        normals = boundary
        boundary_weights = np.full(len(boundary), 2*np.pi/len(boundary))
        corner_physics = None
    edge = ordinary_jets(model, boundary, ((0, 0),))[(0, 0)].numpy()
    boundary_error = edge[:, :2]-boundary_velocity(boundary, case, lid_speed)
    normal_velocity = np.sum(edge[:, :2]*normals, axis=1)
    section_positions = np.linspace(-.8, .8, 5)
    fluxes = []
    for position in section_positions:
        span = 1. if case == 'cavity' else np.sqrt(1-position*position)
        vertical = np.column_stack((np.full_like(t, position), span*t))
        horizontal = np.column_stack((span*t, np.full_like(t, position)))
        uv = ordinary_jets(model, np.concatenate((vertical, horizontal)), ((0, 0),))[(0, 0)].numpy()
        fluxes.extend((span*weights@uv[:len(t), 0], span*weights@uv[len(t):, 1]))
    reflected = points.copy(); reflected[:, 0] *= -1
    mirrored = ordinary_jets(model, reflected, ((0, 0),))[(0, 0)].numpy()
    parity_difference = values[:, :2]-mirrored[:, :2]*np.array([1., -1.])
    stable = solution.evaluate(points)
    # This compares two evaluations of the SAME fitted coordinates. It is
    # an encoding diagnostic, never an interior solution label or error-to-
    # truth claim. Only bounded point-by-coordinate panels are materialized.
    ideal_jets = {a: np.empty((len(points), 3)) for a in ORDERS}
    indices = np.arange(solution.problem.features.size)
    for first in range(0, len(points), 64):
        block = slice(first, first+64)
        columns = solution.problem.features.selected_ideal_columns(points[block], indices, ORDERS)
        for order in ORDERS:
            ideal_jets[order][block] = columns[order]@solution.coefficients
    encoding_rms = {str(order): float(np.sqrt(np.mean((jets[order].numpy()-ideal_jets[order])**2)))
                    for order in ORDERS}
    ideal_physical = momentum_and_divergence(
        {order: torch.from_numpy(value) for order, value in ideal_jets.items()}, viscosity).numpy()
    speed_scale = max(abs(lid_speed), np.finfo(float).tiny)
    out = dict(audit_points=count, momentum_rms=float(np.sqrt(np.mean(physical[:, :2]**2))),
        momentum_max_abs=float(np.max(abs(physical[:, :2]))),
        advective_rms=float(np.sqrt(np.mean(advective**2))),
        viscous_rms=float(np.sqrt(np.mean(viscous**2))),
        pressure_gradient_rms=float(np.sqrt(np.mean(pressure_gradient**2))),
        divergence_rms=float(np.sqrt(np.mean(physical[:, 2]**2))),
        divergence_max_abs=float(np.max(abs(physical[:, 2]))),
        wall_velocity_rms=float(np.sqrt(np.mean(boundary_error**2))),
        wall_velocity_max_abs=float(np.max(abs(boundary_error))),
        net_boundary_mass_flux=float(boundary_weights@normal_velocity),
        maximum_wall_normal_speed=float(np.max(abs(normal_velocity))),
        maximum_cross_section_flux=float(np.max(np.abs(fluxes))),
        cross_section_fluxes=[float(v) for v in fluxes],
        velocity_rms=float(np.sqrt(np.mean(values[:, :2]**2))),
        max_sampled_speed=float(np.max(np.linalg.norm(values[:, :2], axis=1))),
        pressure_rms=float(np.sqrt(np.mean(values[:, 2]**2))),
        pressure_gauge=float(model(torch.zeros((1, 2), dtype=torch.float64))[0, 2].detach()),
        export_stable_max_abs=float(np.max(abs(values-stable))),
        neural_vs_ideal_coordinate_jet_rms=encoding_rms,
        ideal_coordinate_momentum_rms=float(np.sqrt(np.mean(ideal_physical[:, :2]**2))),
        neural_vs_ideal_momentum_rms=float(np.sqrt(np.mean((physical[:, :2]-ideal_physical[:, :2])**2))),
        coordinate_diagnostic_note='Same learned coordinates evaluated by actual MLP versus ideal analytical coordinate basis; not truth error.',
        left_right_asymmetry_relative_to_lid=float(np.sqrt(np.mean(parity_difference**2))/speed_scale),
        symmetry_note='Descriptive only: the finite-Re driven flow need not be reflection-symmetric.',
        ordinary_export='Linear/Tanh/Linear, float64, ordinary Torch automatic differentiation',
        truth_error_available=False, requested_tolerance=requested_tolerance)
    if corner_physics is not None:
        out.update(near_corner_momentum_rms=float(np.sqrt(np.mean(corner_physics[:, :2]**2))),
                   near_corner_momentum_max_abs=float(np.max(abs(corner_physics[:, :2]))))
    out['richer_resolution_comparison'] = None
    if previous is not None:
        previous_model, previous_degree = previous
        previous_values = ordinary_jets(previous_model, points, ((0, 0),))[(0, 0)].numpy()
        degree = solution.problem.features.degree
        out['richer_resolution_comparison'] = dict(previous_degree=previous_degree, degree=degree,
            strictly_richer=degree > previous_degree,
            velocity_relative_l2=float(np.linalg.norm(values[:, :2]-previous_values[:, :2])/
                                       max(np.linalg.norm(values[:, :2]), np.finfo(float).tiny)),
            pressure_relative_l2=float(np.linalg.norm(values[:, 2]-previous_values[:, 2])/
                                       max(np.linalg.norm(values[:, 2]), np.finfo(float).tiny)))
    out['sampled_residual_reaches_requested_tolerance'] = bool(max(out['momentum_rms'],
        out['divergence_rms'], out['wall_velocity_rms'], abs(out['pressure_gauge'])) <= requested_tolerance)
    out['continuum_accuracy_certified'] = False
    return model, out


def run(case='cavity', degrees=(8, 12, 16), centers=257, lam=.2, viscosity=.1,
        lid_speed=1., seconds=120., krylov=400, max_iterations=12, interior=None,
        boundary_per_side=65, tolerance=1e-14, label='native', resume=None,
        linearization_basis='actual', audit_points=2048, inexact_newton=False,
        preconditioner_block_size=640, coordinate_basis='auto'):
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    previous = None
    previous_model = None
    saved = load_native(resume, case, lid_speed) if resume is not None else None
    basis = ('box' if case == 'cavity' else 'disk') if coordinate_basis == 'auto' else coordinate_basis
    if saved is not None:
        old_basis = saved[2].get('coordinate_basis', 'box' if case == 'cavity' else 'disk')
        if old_basis != basis:
            raise ValueError('Native continuation must preserve the coordinate basis and chart')
    if saved is not None and saved[2]['viscosity'] == viscosity:
        old = saved[2]
        old_features = make_problem(case, old['degree'], old['centers'], old['lam'],
                                   viscosity, lid_speed, 8, 5, coordinate_basis=basis).features
        if coordinate_keys(old_features) != saved[1] or old_features.size != len(saved[0]):
            raise ValueError('Native archive metadata does not match its coordinates')
        previous_model = (export_mlp(old_features, saved[0]), old['degree'])
    rows = []
    for degree in degrees:
        began = time.perf_counter()
        problem = make_problem(case, degree, centers, lam, viscosity, lid_speed,
                               interior, boundary_per_side, coordinate_basis=basis)
        initial = None
        if previous is not None:
            initial = transfer_coefficients(previous.coefficients, coordinate_keys(previous.problem.features), problem.features)
        elif saved is not None:
            initial = transfer_coefficients(saved[0], saved[1], problem.features)
        config = dict(case=case, degree=degree, centers=centers, lam=lam, viscosity=viscosity,
                      lid_speed=lid_speed, seconds=seconds, krylov=krylov, max_iterations=max_iterations,
                      interior=len(problem.blocks[0].points), boundary=len(problem.blocks[1].points),
                      linearization_basis=linearization_basis, inexact_newton=inexact_newton,
                      preconditioner_block_size=preconditioner_block_size,
                      tolerance=tolerance, coordinate_basis=basis)
        metadata = dict(kind='native_boundary_driven_ns', **config)
        stem = f'{label}_{case}_nu{viscosity:g}_p{degree}_n{centers}'
        def checkpoint(c, parameters, record):
            np.savez_compressed(OUT/(stem+'_latest.npz'), coefficients=c,
                coordinate_keys=np.asarray(coordinate_keys(problem.features)), metadata=json.dumps(metadata))
            print(json.dumps(dict(event='iteration', name=stem, **record)), flush=True)
        options = dict(initial_coefficients=initial, max_iterations=max_iterations, tolerance=tolerance,
            high_accuracy=True, max_seconds=seconds, lsmr_max_iterations=krylov,
            preconditioner='block', preconditioner_block_size=preconditioner_block_size, preconditioner_ridge=1e-12,
            preconditioner_field_parity='auto', preconditioner_basis='ideal', preconditioner_refresh=1,
            scaling_probes=4, linear_refinement_steps=0, validate_locality=False,
            iteration_callback=checkpoint, inexact_newton=inexact_newton)
        if linearization_basis != 'actual':
            if 'linearization_basis' not in inspect.signature(solve_residual).parameters:
                raise ValueError('This solver version does not expose linearization_basis yet')
            options['linearization_basis'] = linearization_basis
        setup_seconds = time.perf_counter()-began
        solution = solve_residual(problem, **options)
        solve_seconds = time.perf_counter()-began-setup_seconds
        model, checks = audit(solution, case, viscosity, lid_speed, audit_points,
                              previous_model, tolerance)
        boundary_speed_bound = abs(lid_speed)*dict(cavity=1., disk_stirring=1.4,
                                                 disk_counterrotating=1.8)[case]
        row = dict(config=config, status=solution.status, solver=solution.metrics,
            features=problem.features.metrics, history=solution.history,
            ordinary_audit=checks, setup_seconds=setup_seconds, solve_seconds=solve_seconds,
            total_seconds=time.perf_counter()-began,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
            reynolds=dict(length=2., velocity_scale_upper_bound=boundary_speed_bound,
                          value_upper_bound=2*boundary_speed_bound/viscosity,
                          convention='diameter/side length 2 and peak prescribed boundary speed'),
            source_policy=dict(interior_solution_labels=False, forcing='identically zero',
                exterior_solve=False, analytical_reference=False,
                initialization='zero readouts' if initial is None else 'previous native PDE iterate',
                resumed_archive=resume, resumed_viscosity=saved[2]['viscosity'] if saved is not None else None,
                boundary='u=(1-x^2)^2, v=0 on lid; zero other walls' if case == 'cavity'
                    else ('u_boundary=(1+.25cos(2theta)+.15sin(3theta))*(-sin(theta),cos(theta))'
                        if case == 'disk_stirring' else
                        'u_boundary=(.8cos(3theta)+.6sin(5theta)-.4cos(7theta))*(-sin(theta),cos(theta))'),
                pressure_gauge='p(0,0)=0'),
            limitations=['No exact interior solution: residual and refinement agreement are evidence, not a proof.',
                'Steady moderate-Re flow; no claim about turbulence or uniqueness at arbitrary Reynolds number.',
                'C1-compatible lid does not remove all polygon-corner singularities.' if case == 'cavity'
                    else 'Smooth circular boundary and analytic tangential trace avoid polygon corners.'])
        (OUT/(stem+'.json')).write_text(json.dumps(row, indent=2))
        np.savez_compressed(OUT/(stem+'.npz'), coefficients=solution.coefficients,
            coordinate_keys=np.asarray(coordinate_keys(problem.features)), metadata=json.dumps(metadata))
        torch.save(model.state_dict(), OUT/(stem+'_ordinary_mlp.pt'))
        print(json.dumps(dict(event='result', name=stem, status=solution.status, audit=checks)), flush=True)
        previous = solution
        previous_model = (model, degree)
        rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES, default='cavity')
    parser.add_argument('--degrees', type=int, nargs='+', default=[8, 12, 16])
    parser.add_argument('--centers', type=int, default=257)
    parser.add_argument('--lam', type=float, default=.2)
    parser.add_argument('--viscosity', type=float, default=.1)
    parser.add_argument('--lid-speed', type=float, default=1.)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--krylov', type=int, default=400)
    parser.add_argument('--max-iterations', type=int, default=12)
    parser.add_argument('--interior', type=int)
    parser.add_argument('--boundary-per-side', type=int, default=65)
    parser.add_argument('--tolerance', type=float, default=1e-14)
    parser.add_argument('--label', default='native')
    parser.add_argument('--resume')
    parser.add_argument('--linearization-basis', choices=['actual', 'ideal'], default='actual')
    parser.add_argument('--audit-points', type=int, default=2048)
    parser.add_argument('--inexact-newton', action='store_true')
    parser.add_argument('--preconditioner-block-size', type=int, default=640)
    parser.add_argument('--coordinate-basis', choices=['auto', 'box', 'disk'], default='auto')
    run(**vars(parser.parse_args()))
