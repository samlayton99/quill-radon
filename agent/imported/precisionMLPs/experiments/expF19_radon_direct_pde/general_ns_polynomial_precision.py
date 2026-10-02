"""Manufactured polynomial 3D Navier--Stokes precision control in spacetime.

Velocity and pressure have total degree four. The forcing is explicitly
manufactured; only that forcing, velocity IC/BC, and a pressure gauge enter
the general residual solve. By default solves start from zero; optional
continuation only copies coefficients from a previous PDE residual solve.
This isolates float64 optimization/representation floors, not general NS
accuracy or the approximation difficulty of the separate ABC experiment.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quill-ns-polynomial-mpl')

import argparse
import gc
import hashlib
import json
from math import comb
from pathlib import Path
import resource
import sys
import time

import numpy as np
import sympy as sp
import torch

from solver.general_domains import ResidualDomain
from solver.general_features import ConstructedFeatures
from solver.general_residual import ResidualProblem, solve_residual, linearize_residual
from solver.general_symbolic import DifferentialResidual
from general_residual_study import exact_jets

OUT = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual'
DOMAIN = ResidualDomain([[-.5, .5]]*3+[[0., 1.]])
NU = .1
ZERO = (0, 0, 0, 0)
FIRST = tuple(tuple(int(i == j) for i in range(4)) for j in range(4))
SECOND = tuple(tuple(2*int(i == j) for i in range(4)) for j in range(3))
EPS = np.finfo(float).eps
TRACE_NUANCE = ('Below total degree6, values on all six cube faces determine each polynomial velocity component. '
               'A kinematic velocity perturbation vanishing on the six faces and t=0, before enforcing '
               'the PDE/divergence constraints, first appears at degree7: '
               't*(x²−1/4)*(y²−1/4)*(z²−1/4). Pressure is recovered through the PDE and gauge at every degree. '
               'A continued degree8 solve starts from an already resolved degree4 physical solution; '
               'it does not manufacture difficult unresolved content.')


def reference(points):
    """Data/manufacturing/audit only; no interior observations or initial fit."""
    x, y, z, t = points.unbind(1)
    a, b, c = 1+t+t*t, 1-.5*t+.25*t*t, .5+.2*t+.1*t*t
    return torch.stack((a*(y*y+y*z), b*(z*z+x*z), c*(x*x+x*y),
                        a*(x*y+y*z+x*z)), dim=1)


def declarations():
    coordinates = sp.symbols('x y z t', real=True)
    x, y, z, t = coordinates
    fields = tuple(sp.Function(name)(*coordinates) for name in ('u', 'v', 'w', 'p'))
    a, b, c = 1+t+t*t, 1-t/2+t*t/4, sp.Rational(1, 2)+t/5+t*t/10
    truth = (a*(y*y+y*z), b*(z*z+x*z), c*(x*x+x*y), a*(x*y+y*z+x*z))
    forcing = tuple(sp.diff(truth[i], t)+sum(truth[j]*sp.diff(truth[i], coordinates[j]) for j in range(3))+
                    sp.diff(truth[3], coordinates[i])-sp.Rational(1, 10)*sum(sp.diff(truth[i], coordinates[j], 2) for j in range(3))
                    for i in range(3))
    momentum = tuple(sp.diff(fields[i], t)+sum(fields[j]*sp.diff(fields[i], coordinates[j]) for j in range(3))+
                     sp.diff(fields[3], coordinates[i])-sp.Rational(1, 10)*sum(sp.diff(fields[i], coordinates[j], 2) for j in range(3))-
                     forcing[i] for i in range(3))
    divergence = sum(sp.diff(fields[i], coordinates[i]) for i in range(3))
    assert sp.simplify(sum(sp.diff(truth[i], coordinates[i]) for i in range(3))) == 0
    return dict(
        equation=DifferentialResidual(coordinates, fields, momentum+(divergence,)),
        velocity=DifferentialResidual(coordinates, fields, tuple(fields[i]-truth[i] for i in range(3))),
        gauge=DifferentialResidual(coordinates, fields, (fields[3],)),
        force=DifferentialResidual(coordinates, fields, forcing),
        formulas=dict(velocity=[str(v) for v in truth[:3]], pressure=str(truth[3]),
                      manufactured_forcing=[str(v) for v in forcing]))


def make_blocks(degree, declarations_, seed=1249):
    # A face or initial-time trace is a polynomial in d-1 coordinates.
    # Enforce at least twice that trace-space dimension independently per face.
    count = 4*comb(degree+4, 4)
    trace_count = 2*comb(degree+3, 3)
    interior = DOMAIN.interior(count, seed)
    spatial = []
    for axis in range(3):
        for side in range(2):
            points = DOMAIN.interior(trace_count, seed+104729*(1+2*axis+side))
            points[:, axis] = DOMAIN.bounds[axis, side]
            spatial.append(points)
    initial = DOMAIN.interior(trace_count, seed+87611)
    initial[:, 3] = 0.
    gauge = DOMAIN.interior(2*(degree+1), seed+99331)
    gauge[:, :3] = 0.
    return [declarations_['equation'].block('momentum_divergence', interior),
            declarations_['velocity'].block('six_spatial_faces', np.concatenate(spatial)),
            declarations_['velocity'].block('initial_velocity', initial),
            declarations_['gauge'].block('pressure_gauge', gauge)], dict(
                interior=count, per_spatial_face=trace_count, initial=trace_count,
                gauge=len(gauge), trace_space_dimension=comb(degree+3, 3))


def audit_reference(declarations_):
    points = DOMAIN.interior(131, 941171)
    derivatives = tuple(dict.fromkeys((ZERO,)+FIRST+SECOND))
    jets = exact_jets(reference, points, derivatives)
    tensor = torch.as_tensor(points)
    parameters = torch.empty((len(points), 0), dtype=torch.float64)
    force = declarations_['force'].function(tensor, jets, parameters)
    gradient = torch.stack([jets[d][:, :3] for d in FIRST[:3]], dim=2)
    convection = torch.einsum('qj,qij->qi', jets[ZERO][:, :3], gradient)
    laplacian = sum(jets[d][:, :3] for d in SECOND)
    pressure_gradient = torch.stack([jets[d][:, 3] for d in FIRST[:3]], dim=1)
    independent_residual = jets[FIRST[3]][:, :3]+convection+pressure_gradient-NU*laplacian-force
    compiled = declarations_['equation'].function(tensor, jets, parameters)
    divergence = sum(jets[FIRST[i]][:, i] for i in range(3))
    curl = torch.stack((gradient[:, 2, 1]-gradient[:, 1, 2],
                        gradient[:, 0, 2]-gradient[:, 2, 0],
                        gradient[:, 1, 0]-gradient[:, 0, 1]), dim=1)
    return dict(
        independent_torch_ad_momentum_rms=float(torch.sqrt(torch.mean(independent_residual**2))),
        independent_torch_ad_momentum_max=float(torch.max(abs(independent_residual))),
        compiled_symbolic_residual_max=float(torch.max(abs(compiled))),
        divergence_max=float(torch.max(abs(divergence))),
        convection_rms=float(torch.sqrt(torch.mean(convection**2))),
        viscosity_term_rms=float(torch.sqrt(torch.mean((NU*laplacian)**2))),
        pressure_gradient_rms=float(torch.sqrt(torch.mean(pressure_gradient**2))),
        vorticity_rms=float(torch.sqrt(torch.mean(curl**2))))


def validate(solution, declarations_, degree):
    points = DOMAIN.interior(1009, 414907)
    truth = reference(torch.as_tensor(points)).numpy()
    predicted = solution.evaluate(points)
    rows = {}
    for name, indices in [('velocity', slice(0, 3)), ('pressure', slice(3, 4))]:
        error = predicted[:, indices]-truth[:, indices]
        relative = float(np.linalg.norm(error)/np.linalg.norm(truth[:, indices]))
        rms = float(np.sqrt(np.mean(error**2)))
        rows[name] = dict(relative_l2=relative, absolute_rms=rms,
                          maximum_absolute_error=float(np.max(abs(error))),
                          relative_l2_in_machine_epsilons=relative/EPS,
                          absolute_rms_in_machine_epsilons=rms/EPS)
    fresh, _ = make_blocks(degree, declarations_, seed=734779)
    fresh_checks = []
    for block in fresh:
        jets = {d:torch.as_tensor(solution.evaluate(block.points, d)) for d in block.derivatives}
        residual = block.function(torch.as_tensor(block.points), jets,
                                  torch.empty((len(block.points), 0), dtype=torch.float64)).numpy()
        fresh_checks.append(dict(name=block.name, component_rms=np.sqrt(np.mean(residual**2, axis=0)).tolist(),
                                 maximum_absolute=float(np.max(abs(residual)))))
    rows['fresh_blocks'] = fresh_checks
    final_points = points.copy()
    final_points[:, 3] = 1.
    final_truth = reference(torch.as_tensor(final_points)).numpy()
    final_error = solution.evaluate(final_points)-final_truth
    rows['final_time'] = dict(velocity_relative_l2=float(np.linalg.norm(final_error[:, :3])/np.linalg.norm(final_truth[:, :3])),
                              pressure_relative_l2=float(np.linalg.norm(final_error[:, 3])/np.linalg.norm(final_truth[:, 3])))
    return rows


def conditioning(solution):
    """Post-solve audit only, bounded to the small degree-four control.

    This explicitly assembles a diagnostic Jacobian for singular values, never
    a solve, initializer, preconditioner, or data source for the solver.
    """
    began = time.perf_counter()
    linear = linearize_residual(solution.problem, solution.coefficients)
    n = linear.operator.shape[1]
    matrix = np.column_stack([linear.operator.matvec(np.eye(1, n, i).ravel()) for i in range(n)])
    singular = np.linalg.svd(matrix, compute_uv=False)
    scaled_singular = np.linalg.svd(matrix/np.linalg.norm(matrix, axis=0), compute_uv=False)
    return dict(unknowns=n, residual_entries=len(linear.residual),
                diagnostic_jacobian_bytes=matrix.nbytes,
                largest_singular_value=float(singular[0]), smallest_singular_value=float(singular[-1]),
                condition_number=float(singular[0]/singular[-1]),
                exact_column_scaled_condition_number=float(scaled_singular[0]/scaled_singular[-1]),
                inverse_smallest_singular_value=float(1/singular[-1]),
                seconds=time.perf_counter()-began,
                scope='Final sampled Jacobian local conditioning; not a global nonlinear or continuum stability bound. Dense matrix is diagnostic only.')


def run(degree, backend, args, declarations_):
    started = time.perf_counter()
    options = dict(max_derivative=2)
    if backend == 'quill':
        options.update(centers=args.centers, lam=args.lam, evaluation='anchored',
                       encoding_tolerance=args.encoding_tolerance)
    features = ConstructedFeatures(DOMAIN.bounds, degree, backend=backend, **options)
    blocks, samples = make_blocks(degree, declarations_)
    cache_bytes = sum(len(block.points)*len(block.derivatives)*features.size*8 for block in blocks)
    if cache_bytes > 95e6:
        raise MemoryError(f'Basis cache estimate {cache_bytes} exceeds the 95 MB working-array cap')
    problem = ResidualProblem(features, blocks, fields=4)
    initial, continuation = None, None
    if args.warmstart:
        source = Path(args.warmstart)
        with np.load(source) as previous:
            old_coefficients, old_indices = previous['coefficients'], previous['multiindices']
        if old_coefficients.shape != (len(old_indices), 4) or old_indices.shape[1] != 4:
            raise ValueError('Continuation requires four-field, four-coordinate coefficients and multiindices')
        lookup = {tuple(mode):index for index,mode in enumerate(features.multiindices)}
        initial = np.zeros((features.size, 4))
        for mode, values in zip(old_indices, old_coefficients):
            if tuple(mode) in lookup:
                initial[lookup[tuple(mode)]] = values
        continuation = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            source_features=len(old_indices),
                            scope='PDE-derived coefficient prolongation by matching multiindices; added coefficients zero. No analytic interior fit.')
    setup_seconds = time.perf_counter()-started
    solution = solve_residual(problem, initial_coefficients=initial,
                               tolerance=args.tolerance, high_accuracy=True,
                               preconditioner='block', max_iterations=25,
                               lsmr_max_iterations=1500, max_seconds=args.seconds,
                               max_residual_evaluations=500)
    solve_seconds = time.perf_counter()-started-setup_seconds
    validation = validate(solution, declarations_, degree)
    condition = conditioning(solution) if degree <= 4 else dict(scope='Dense conditioning diagnostic omitted above degree4 to keep working storage bounded')
    output = dict(degree=degree, backend=backend, viscosity=NU, bounds=DOMAIN.bounds.tolist(),
                  problem='Manufactured degree-four 3D incompressible Navier-Stokes with four spacetime inputs',
                  data_scope='Manufactured body force; velocity on six spatial faces and t=0; p(0,0,0,t)=0. No interior labels, final labels, reference initialization, or PDE-specific inverse.',
                  initialization='zero' if initial is None else 'previous_PDE_solution',
                  continuation=continuation,
                  status=solution.status, sample_counts=samples, features=features.metrics,
                  reference_audit=audit_reference(declarations_), formulas=declarations_['formulas'],
                  validation=validation, conditioning=condition,
                  solver=solution.metrics, history=solution.history,
                  setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                  estimated_basis_cache_bytes=cache_bytes,
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
                  memory_scope='Named basis arrays capped below95 MB; total process RSS includes Torch/SymPy/runtime libraries and exceeds100 MB before this solve.',
                  machine_epsilon=EPS,
                  engine_sha256=hashlib.sha256((Path(__file__).parent/'solver/general_residual.py').read_bytes()).hexdigest(),
                  limitation='Polynomial manufactured floor control; not the ABC test, turbulence, an arbitrary-domain result, or a universal NS accuracy guarantee',
                  trace_space_nuance=TRACE_NUANCE)
    OUT.mkdir(parents=True, exist_ok=True)
    label = '_'+args.label if args.label else ''
    stem=f'ns_polynomial_precision_{backend}{label}_p{degree}'
    (OUT/(stem+'.json')).write_text(json.dumps(output, indent=2)+'\n')
    np.savez_compressed(OUT/(stem+'.npz'), coefficients=solution.coefficients, multiindices=features.multiindices)
    print(json.dumps(dict(backend=backend, degree=degree, status=solution.status, seconds=solve_seconds,
                          training_residual=max(b['maximum_scaled_rms'] for b in solution.metrics['blocks']),
                          velocity=validation['velocity'], pressure=validation['pressure'],
                          conditioning=condition)), flush=True)
    return output


def summarize_and_plot():
    files = sorted(OUT.glob('ns_polynomial_precision_*_p*.json'))
    rows = []
    for path in files:
        row = json.loads(path.read_text())
        # Add scope annotations to earlier completed runs without relabeling
        # their numerical checks or timings as having been rerun.
        changed = 'continuation' not in row or row.get('trace_space_nuance') != TRACE_NUANCE
        row.setdefault('continuation', None)
        row['trace_space_nuance'] = TRACE_NUANCE
        if row['initialization'].startswith('All coefficients zero'):
            row['initialization'] = 'zero'
        if row['status'] == 'budget_exhausted' and 'budget' not in row['solver']:
            row['budget_policy_scope'] = ('This preserved run predates the engine fix that retains a useful '
                                         'already-computed Krylov direction after soft-time expiry. Its '
                                         'last direction may have been discarded; it was not rerun with the fix.')
            changed = True
        if changed:
            row['scope_metadata_added_after_run'] = True
            path.write_text(json.dumps(row, indent=2)+'\n')
        rows.append(row)
    summary = [{k:r.get(k) for k in ('degree', 'backend', 'status', 'initialization', 'continuation',
                                 'validation', 'conditioning', 'setup_seconds', 'solve_seconds',
                                 'estimated_basis_cache_bytes', 'trace_space_nuance',
                                 'budget_policy_scope')} for r in rows]
    (OUT/'ns_polynomial_precision_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.8), constrained_layout=True)
    for backend in ('polynomial', 'quill'):
        for warm in (False, True):
            selected = sorted([r for r in rows if r['backend'] == backend and
                               (r['initialization'] == 'previous_PDE_solution') == warm], key=lambda r:r['degree'])
            if not selected:
                continue
            label = backend+(' continuation' if warm else ' from zero')
            for ax, field in zip(axes, ('velocity', 'pressure')):
                ax.semilogy([r['degree'] for r in selected],
                            [r['validation'][field]['relative_l2']/EPS for r in selected],
                            ('s--' if warm else 'o-'), label=label)
                for r in selected:
                    if r['status'] != 'converged':
                        ax.annotate(r['status'].replace('_', ' '),
                                    (r['degree'], r['validation'][field]['relative_l2']/EPS),
                                    xytext=(-10, -20), textcoords='offset points', ha='right', fontsize=8)
    for ax, field in zip(axes, ('velocity', 'pressure')):
        ax.axhline(1., color='.6', linewidth=.8)
        ax.set(xlabel='Total polynomial degree', ylabel='Fresh relative L2 error / machine epsilon',
               title=field.capitalize()+' accuracy')
        ax.set_xticks(sorted({r['degree'] for r in rows}))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Manufactured 3D Navier–Stokes: general residual solver')
    fig.savefig(OUT/'ns_polynomial_precision.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degrees', type=int, nargs='+', default=[4, 6, 8])
    parser.add_argument('--backends', choices=['polynomial', 'quill'], nargs='+', default=['polynomial'])
    parser.add_argument('--tolerance', type=float, default=2e-15)
    parser.add_argument('--seconds', type=float, default=45.)
    parser.add_argument('--centers', type=int)
    parser.add_argument('--lam', type=float)
    parser.add_argument('--encoding-tolerance', type=float, default=1e-8)
    parser.add_argument('--warmstart', type=str)
    parser.add_argument('--label', default='')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.plot_only:
        summarize_and_plot()
        return
    declared = declarations()
    print(json.dumps(dict(event='reference_audit', audit=audit_reference(declared))), flush=True)
    outputs = []
    for backend in args.backends:
        for degree in args.degrees:
            outputs.append(run(degree, backend, args, declared))
            gc.collect()
    summarize_and_plot()


if __name__ == '__main__':
    main()
