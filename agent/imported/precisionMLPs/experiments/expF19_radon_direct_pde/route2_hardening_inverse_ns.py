"""Joint native disk Navier--Stokes/viscosity solve with five velocity anchors.

The manufactured body force is fixed before fitting. Only ten explicitly
declared interior velocity observations, zero wall velocity, a pressure
gauge and the PDE enter the fit. All readouts start at zero; no external PDE
solver, target-field fitting or constructed target coefficients initialize it.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from route2_disk_profiles import DiskProfileOperator
from route2_hardening_disk import ORDERS, exact, equations, source
from route2_streamed_study import disk_points, boundary_points
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual
from solver.route2 import export_mlp, ordinary_jets


OUT = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/inverse_ns'
OBSERVATION_POINTS = np.array([[0., 0.], [.3, .2], [-.45, .15], [.2, -.5], [-.3, -.4]])


def make_problem(degree=14, centers=257, lam=.2, nu_true=.3, nu_initial=.15,
                 interior=512, boundary=128, batch_size=128):
    features = DiskProfileOperator(degree, centers=centers, lam=lam, block_size=batch_size)
    points = disk_points(interior, 1741)
    # This array is prescribed problem data. It never depends on the unknown
    # parameter supplied to either callback below.
    force = torch.tensor(source(points, nu_true)[:, :2], dtype=torch.float64)

    def residual(x, jets, parameters):
        value = equations(jets, parameters[:, :1])
        return torch.cat((value[:, :2]-force, value[:, 2:3]), dim=1)

    def residual_batch(x, jets, parameters, rows):
        value = equations(jets, parameters[:, :1])
        return torch.cat((value[:, :2]-force[rows], value[:, 2:3]), dim=1)

    observations = exact(torch.tensor(OBSERVATION_POINTS, dtype=torch.float64))[:, :2].detach()

    def observed(x, jets, parameters):
        return jets[(0, 0)][:, :2]-observations

    def observed_batch(x, jets, parameters, rows):
        return jets[(0, 0)][:, :2]-observations[rows]

    blocks = [
        ResidualBlock('momentum_and_divergence', points, residual, ORDERS,
                      batch_function=residual_batch),
        ResidualBlock('no_slip', boundary_points(boundary),
                      lambda x, j, p: j[(0, 0)][:, :2], ((0, 0),), weight=5.),
        ResidualBlock('pressure_gauge', np.zeros((1, 2)),
                      lambda x, j, p: j[(0, 0)][:, 2], ((0, 0),)),
        ResidualBlock('velocity_observations', OBSERVATION_POINTS.copy(), observed,
                      ((0, 0),), weight=5., batch_function=observed_batch),
    ]
    problem = ResidualProblem(features, blocks, fields=3, parameter_initial=[nu_initial],
        parameter_bounds=([.05], [1.]), execution='streamed', batch_size=batch_size)
    return problem


def audit(solution, nu_true=.3, count=2048):
    model = export_mlp(solution.problem.features, solution.coefficients)
    points = disk_points(count, 812771)
    jets = ordinary_jets(model, points, ORDERS)
    truth = exact(torch.tensor(points, dtype=torch.float64)).detach().numpy()
    prediction = jets[(0, 0)].numpy()
    nu = float(solution.parameters[0])
    # The left-hand side uses fitted nu; the right-hand side always uses the
    # original prescribed forcing. Regenerating forcing at fitted nu would
    # make the inverse validation circular.
    residual = equations(jets, nu).numpy()
    forcing = source(points, nu_true)[:, :2]
    residual[:, :2] -= forcing
    walls = ordinary_jets(model, boundary_points(257, .317), ((0, 0),))[(0, 0)].numpy()
    observed = model(torch.tensor(OBSERVATION_POINTS, dtype=torch.float64)).detach().numpy()[:, :2]
    observation_truth = exact(torch.tensor(OBSERVATION_POINTS, dtype=torch.float64)).detach().numpy()[:, :2]
    stable = solution.evaluate(points)
    report = dict(
        viscosity=nu, viscosity_true=nu_true, viscosity_absolute_error=abs(nu-nu_true),
        viscosity_relative_error=abs(nu-nu_true)/abs(nu_true),
        velocity_relative_l2=float(np.linalg.norm(prediction[:, :2]-truth[:, :2])/np.linalg.norm(truth[:, :2])),
        pressure_relative_l2=float(np.linalg.norm(prediction[:, 2]-truth[:, 2])/np.linalg.norm(truth[:, 2])),
        momentum_rms=float(np.sqrt(np.mean(residual[:, :2]**2))),
        momentum_relative_rms=float(np.linalg.norm(residual[:, :2])/np.linalg.norm(forcing)),
        divergence_rms=float(np.sqrt(np.mean(residual[:, 2]**2))),
        no_slip_rms=float(np.sqrt(np.mean(walls[:, :2]**2))),
        pressure_gauge=float(model(torch.zeros((1, 2), dtype=torch.float64))[0, 2]),
        observation_rms=float(np.sqrt(np.mean((observed-observation_truth)**2))),
        observation_locations=len(OBSERVATION_POINTS), observation_scalar_values=observed.size,
        export_stable_max_abs=float(np.max(abs(prediction-stable))),
        heldout_points=count, forcing_uses_true_prescribed_parameter=True,
        ordinary_export='Linear/Tanh/Linear; float64; standard Torch AD',
        identifiability_scope='One noise-free manufactured example; no global identifiability or uncertainty certificate')
    return model, report


def run(label='inverse_p14_n257', degree=14, centers=257, lam=.2,
        nu_true=.3, nu_initial=.15, seconds=240., iterations=10, krylov=1000,
        tolerance=2e-14, block=128, batch_size=128, interior=512, boundary=128,
        preconditioner_basis='actual', preconditioner_refresh=0):
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    problem = make_problem(degree, centers, lam, nu_true, nu_initial, interior, boundary, batch_size)
    setup_seconds = time.perf_counter()-began
    progress = []

    def checkpoint(coefficients, parameters, record):
        entry = dict(record, viscosity=float(parameters[0]))
        progress.append(entry)
        (OUT/f'{label}_progress.json').write_text(json.dumps(progress, indent=2))
        np.savez_compressed(OUT/f'{label}_latest_native.npz', coefficients=coefficients, parameters=parameters)
        print(json.dumps(dict(event='iteration', **entry)), flush=True)

    solved = solve_residual(problem, max_iterations=iterations, tolerance=tolerance,
        high_accuracy=True, max_seconds=seconds, lsmr_max_iterations=krylov,
        preconditioner='block', preconditioner_block_size=block,
        preconditioner_field_parity='auto', preconditioner_ridge=1e-12,
        preconditioner_basis=preconditioner_basis,
        preconditioner_refresh=preconditioner_refresh,
        scaling_probes=4, linear_refinement_steps=0, validate_locality=False,
        iteration_callback=checkpoint)
    solve_seconds = time.perf_counter()-began-setup_seconds
    audit_started = time.perf_counter()
    model, checked = audit(solved, nu_true)
    config = dict(label=label, degree=degree, centers=centers, lam=lam,
        nu_true=nu_true, nu_initial=nu_initial, viscosity_bounds=[.05, 1.], seconds=seconds,
        iterations=iterations, krylov=krylov, tolerance=tolerance,
        preconditioner=f'{preconditioner_basis} block Gram', preconditioner_block_size=block,
        preconditioner_basis=preconditioner_basis,
        preconditioner_refresh=preconditioner_refresh,
        preconditioner_field_parity='auto', batch_size=batch_size, interior=interior, boundary=boundary)
    result = dict(config=config, source_policy=dict(initialization='zero readouts and nu_initial',
        exterior_solve=False, target_readout_fit=False, interior_solution_observation_locations=5,
        interior_solution_scalar_observations=10, observed_fields=['u', 'v'],
        observation_points=OBSERVATION_POINTS.tolist(), observation_noise=0.,
        other_interior_solution_labels=False, boundary_velocity='identically zero',
        gauge='p(0)=0', forcing='fixed manufactured source at nu_true; prescribed PDE input',
        reference='forcing, ten explicitly declared velocity observations, and independent validation'),
        features=problem.features.metrics, solver=solved.metrics, history=solved.history,
        ordinary_audit=checked, setup_seconds=setup_seconds, solve_seconds=solve_seconds,
        audit_seconds=time.perf_counter()-audit_started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        scope='Joint inverse scalar-parameter and three-field fit on a steady 2D smooth manufactured NS problem; not a general PDE or identifiability guarantee')
    (OUT/f'{label}.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT/f'{label}.npz', coefficients=solved.coefficients, parameters=solved.parameters)
    torch.save(model.state_dict(), OUT/f'{label}_ordinary_mlp.pt')
    print(json.dumps(dict(event='result', status=solved.status, **checked)), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='inverse_p14_n257')
    parser.add_argument('--degree', type=int, default=14)
    parser.add_argument('--centers', type=int, default=257)
    parser.add_argument('--lam', type=float, default=.2)
    parser.add_argument('--nu-true', type=float, default=.3)
    parser.add_argument('--nu-initial', type=float, default=.15)
    parser.add_argument('--seconds', type=float, default=240.)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--krylov', type=int, default=1000)
    parser.add_argument('--tolerance', type=float, default=2e-14)
    parser.add_argument('--block', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--interior', type=int, default=512)
    parser.add_argument('--boundary', type=int, default=128)
    parser.add_argument('--preconditioner-basis', choices=['actual', 'ideal'], default='actual')
    parser.add_argument('--preconditioner-refresh', type=int, default=0)
    run(**vars(parser.parse_args()))
