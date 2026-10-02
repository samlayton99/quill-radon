"""Full public equation-only API proof on the manufactured NS floor control.

GeneralProblem constructs all PDE/face/initial samples. DifferentialResidual
discovers every derivative from SymPy expressions. Only prescribed pressure
gauge points are explicit problem data. No solver or feature tuning beyond a
requested accuracy, degree ladder, and resource budget is supplied.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quill-symbolic-ns-mpl')

import hashlib
import argparse
import json
from math import comb
from pathlib import Path
import resource
import sys
import time

import numpy as np
import sympy as sp
import torch

from solver import GeneralProblem, Condition
from solver.general_features import ConstructedFeatures
from solver.general_adaptive import check_residuals
from general_ns_polynomial_precision import (DOMAIN, EPS, TRACE_NUANCE, OUT,
                                               declarations, reference, audit_reference)
from general_residual_study import exact_jets

SOURCE_SHA256 = {name:hashlib.sha256((Path(__file__).parent/'solver'/name).read_bytes()).hexdigest()
                 for name in ('general_problem.py', 'general_adaptive.py', 'general_features.py',
                              'general_symbolic.py', 'general_residual.py')}


def problem_declaration():
    # Reuse exactly the same symbolic physics as the engine-level control.
    # The factory returns DifferentialResidual objects, not hand-coded jets.
    declared = declarations()
    gauge = np.zeros((17, 4))
    gauge[:, 3] = np.linspace(0., 1., len(gauge))
    conditions = [
        Condition('spatial_velocity', declared['velocity'], location='boundary',
                  selector=lambda points:np.any(np.isclose(abs(points[:, :3]), .5,
                                                           rtol=0., atol=1e-13), axis=1)),
        Condition('initial_velocity', declared['velocity'], location='slice', axis=3, value=0.),
        Condition('pressure_gauge', declared['gauge'], points=gauge),
    ]
    return GeneralProblem(DOMAIN, declared['equation'], conditions), declared


def divergence_free_bubble_audit():
    """Polynomial-space scope check only; never used in the forward solve."""
    sx, sy, sz, st = sp.symbols('x y z t')
    factors = [coordinate**2-sp.Rational(1, 4) for coordinate in (sx, sy, sz)]
    prefactor = st*sp.prod(factors)
    symbolic = (prefactor*factors[0]*sy, -prefactor*factors[1]*sx, sp.Integer(0))
    exact_divergence = sp.expand(sum(sp.diff(value, coordinate)
                                     for value,coordinate in zip(symbolic, (sx, sy, sz))))
    exact_traces_zero = all(sp.expand(value.subs(coordinate, side)) == 0
                           for value in symbolic for coordinate in (sx, sy, sz)
                           for side in (-sp.Rational(1, 2), sp.Rational(1, 2)))
    exact_initial_zero = all(value.subs(st, 0) == 0 for value in symbolic)
    total_degree = max(sp.Poly(value, sx, sy, sz, st).total_degree() for value in symbolic)
    assert exact_divergence == 0 and exact_traces_zero and exact_initial_zero and total_degree == 10
    def bubble(points):
        x, y, z, t = points.unbind(1)
        xx, yy, zz = x*x-.25, y*y-.25, z*z-.25
        factor = t*xx*yy*zz
        return factor[:, None]*torch.stack((xx*y, -yy*x, torch.zeros_like(x)), dim=1)
    points = DOMAIN.interior(127, 51237)
    first = ((1,0,0,0), (0,1,0,0), (0,0,1,0))
    jets = exact_jets(bubble, points, first)
    divergence = sum(jets[d][:, axis] for axis,d in enumerate(first))
    boundary_max = 0.
    for axis in range(3):
        for side in (-.5, .5):
            boundary = points.copy()
            boundary[:, axis] = side
            boundary_max = max(boundary_max, float(torch.max(abs(bubble(torch.as_tensor(boundary))))))
    initial = points.copy()
    initial[:, 3] = 0.
    return dict(
        minimum_total_degree=10,
        symbolic_total_degree=int(total_degree), symbolic_divergence_exactly_zero=exact_divergence == 0,
        symbolic_boundary_initial_exactly_zero=bool(exact_traces_zero and exact_initial_zero),
        bubble_formula='t B ((x²−1/4)y, −(y²−1/4)x, 0), B=(x²−1/4)(y²−1/4)(z²−1/4)',
        proof='Every trace/IC-zero polynomial perturbation is w=t B q. Setting div(w)=0 on x=±1/2 forces q1 divisible by x²−1/4, and similarly for q2/q3. Hence w_i=t B (x_i²−1/4) r_i. Below degree9 all r_i vanish. At degree9 they are constants and div(w)/(tB)=4(x r1+y r2+z r3), which forces them zero. At degree10 choose r=(y,−x,0), whose divergence cancels.',
        divergence_max=float(torch.max(abs(divergence))),
        boundary_max=boundary_max,
        initial_max=float(torch.max(abs(bubble(torch.as_tensor(initial))))),
        nonzero_bubble_rms=float(torch.sqrt(torch.mean(bubble(torch.as_tensor(points))**2))),
        scope='Exact polynomial-space statement. QUILL approximates this space by actual tanh products; this audit is not a forward initializer, constraint, or forcing term.')


def audit_saved(stem):
    """Fresh-seed checks after solving, without changing or fitting a field."""
    path = OUT/(stem+'.json')
    report = json.loads(path.read_text())
    with np.load(OUT/(stem+'.npz')) as data:
        final_coefficients, final_modes = data['coefficients'], data['multiindices']
    mapping = {tuple(mode):index for index,mode in enumerate(final_modes)}
    declared = declarations()
    equation = declared['equation']
    stages = []
    for index, entry in enumerate(report['history']):
        # A later zero-iteration prolongation preserves earlier coefficients.
        # If a later solve updated them, reconstruction would be invalid.
        if any(row['solver']['iterations'] != 0 for row in report['history'][index+1:]):
            continue
        settings = entry['solver']['feature_metrics']
        features = ConstructedFeatures(DOMAIN.bounds, entry['degree'], backend='quill',
            centers=settings['interior_centers_per_axis'], lam=settings['lam'],
            max_derivative=2, evaluation=settings['evaluation'],
            encoding_tolerance=settings['derivative_check_tolerance'])
        coefficients = final_coefficients[[mapping[tuple(mode)] for mode in features.multiindices]]
        rows = []
        for seed, final_time in [(441131, False), (819971, False), (936727, False), (717221, True)]:
            points = DOMAIN.interior(1031, seed)
            if final_time:
                points[:, 3] = 1.
            truth = reference(torch.as_tensor(points)).numpy()
            predicted = features.evaluate(points) @ coefficients
            error = predicted-truth
            row = dict(seed=seed, points=len(points), time_scope='t=1' if final_time else 'spacetime')
            for name, components in [('velocity', slice(0, 3)), ('pressure', slice(3, 4))]:
                relative = float(np.linalg.norm(error[:, components])/np.linalg.norm(truth[:, components]))
                row[name] = dict(relative_l2=relative, relative_l2_in_epsilons=relative/EPS,
                    absolute_rms=float(np.sqrt(np.mean(error[:, components]**2))),
                    maximum_absolute_error=float(np.max(abs(error[:, components]))))
            jets = {d:torch.as_tensor(features.evaluate(points, d) @ coefficients) for d in equation.derivatives}
            residual = equation.function(torch.as_tensor(points), jets,
                                          torch.empty((len(points), 0), dtype=torch.float64)).numpy()
            row['pde_component_rms'] = np.sqrt(np.mean(residual**2, axis=0)).tolist()
            row['pde_maximum_absolute'] = float(np.max(abs(residual)))
            rows.append(row)
        stages.append(dict(degree=entry['degree'], features=settings, checks=rows,
            reconstruction='Saved final coefficients restricted by multiindex; all subsequent stages performed zero coefficient updates'))
    output = dict(stages=stages,
                  scope='Post-solve verification on three new scrambled-Sobol spacetime sets and independent final-time points; no training or solution changes',
                  normalization='Velocity relative L2 combines all three components; pressure uses its own L2 norm. EPS multiples divide those relative errors by float64 epsilon. PDE component RMS uses physical residual scale1.',
                  constraint_scope='IC, BC, gauge, and incompressibility are sampled residual equations, not hard architectural constraints or exact lifting.',
                  divergence_free_trace_zero_bubble=divergence_free_bubble_audit())
    (OUT/(stem+'_additional_audit.json')).write_text(json.dumps(output, indent=2)+'\n')
    report['additional_audit_file'] = stem+'_additional_audit.json'
    final_check = stages[-1]['checks'][-1]
    report['validation']['independent_end_time_audit'] = final_check
    report['endpoint_accuracy_scope'] = (
        'The sampled spacetime status is unchanged by this post-solve audit. At independent t=1 points, '
        f"pressure relative L2 error is {final_check['pressure']['relative_l2']:.6g} "
        f"({final_check['pressure']['relative_l2_in_epsilons']:.3g} machine epsilons). "
        'Do not interpret the solver status as a uniform one-epsilon field-error certificate.')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(stages=[dict(degree=s['degree'], checks=s['checks']) for s in stages])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degrees', type=int, nargs='+', default=[4, 6])
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--tolerance', type=float, default=2e-15)
    parser.add_argument('--maximum-basis-cache-mb', type=float, default=95.)
    parser.add_argument('--label', default='')
    parser.add_argument('--audit-saved', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    stem = 'ns_symbolic_precision'+('_'+args.label if args.label else '')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.audit_saved:
        audit_saved(stem)
        return
    started = time.perf_counter()
    problem, declared = problem_declaration()
    declaration_seconds = time.perf_counter()-started
    reference_checks = audit_reference(declared)
    solve_options = dict(tolerance=args.tolerance, degrees=tuple(args.degrees), max_seconds=args.seconds,
                         maximum_basis_cache_mb=args.maximum_basis_cache_mb)
    progress = []
    def checkpoint(coefficients, parameters, record):
        degree = 0
        while comb(degree+4, 4) < len(coefficients):
            degree += 1
        entry = dict(degree=degree, **record)
        progress.append(entry)
        (OUT/(stem+'_progress.json')).write_text(json.dumps(progress, indent=2)+'\n')
        np.savez_compressed(OUT/(stem+'_checkpoint.npz'), coefficients=coefficients,
                            parameters=parameters, degree=degree, bounds=DOMAIN.bounds)
        print(json.dumps(dict(event='iterate', **entry)), flush=True)
    before = time.perf_counter()
    solution = problem.solve(**solve_options, solver_options={'iteration_callback': checkpoint})
    solve_seconds = time.perf_counter()-before

    # Analytic interior values are used only after the full public solve.
    points = DOMAIN.interior(1009, 177071)
    truth = reference(torch.as_tensor(points)).numpy()
    values = solution.evaluate(points)
    error = values-truth
    validation = {}
    for name, indices in [('velocity', slice(0, 3)), ('pressure', slice(3, 4))]:
        relative = float(np.linalg.norm(error[:, indices])/np.linalg.norm(truth[:, indices]))
        validation[name] = dict(relative_l2=relative, relative_l2_in_epsilons=relative/EPS,
                                absolute_rms=float(np.sqrt(np.mean(error[:, indices]**2))),
                                maximum_absolute_error=float(np.max(abs(error[:, indices]))))
    validation['second_independent_residual_check'] = check_residuals(solution.solution, problem.blocks(1237, 331981))
    fresh_gauge = np.zeros((73, 4))
    fresh_gauge[:, 3] = np.mod(np.arange(1, 74)*np.sqrt(2), 1.)
    gauge_values = solution.evaluate(fresh_gauge)[:, 3]
    validation['independent_pressure_gauge'] = dict(points=len(fresh_gauge),
        rms=float(np.sqrt(np.mean(gauge_values**2))), maximum_absolute=float(np.max(abs(gauge_values))),
        scope='Fresh times at the declared spatial origin; none are the17 prescribed gauge times')
    feature = solution.problem.features
    output = dict(
        problem='Manufactured degree-four three-dimensional incompressible Navier-Stokes in four spacetime inputs',
        public_interface='GeneralProblem + DifferentialResidual + Condition; GeneralProblem.solve',
        solver_options_supplied=solve_options,
        custom_optimizer_options_supplied=False, custom_feature_options_supplied=False,
        monitoring_callback_supplied=True,
        manual_derivative_lists_supplied=False, manual_pde_or_boundary_sampling=False,
        gauge_data='p(0,0,0,t)=0 at17 prescribed times; reused gauge locations are explicitly reported by validation',
        data_scope='Manufactured forcing, velocity IC/BC, pressure gauge. No analytic interior observations, oracle initialization, named-PDE inverse, or explicit time marching.',
        initial_state='Zero coefficients; higher-degree continuation generated only by the public solver',
        status=solution.status, metrics=solution.metrics, history=solution.history,
        selected_degree=feature.degree, selected_features=feature.metrics,
        selected_solver=solution.solution.metrics,
        declaration_seconds=declaration_seconds, full_public_solve_seconds=solve_seconds,
        reference_audit=reference_checks, validation=validation,
        equations=declared['equation'].metrics, formulas=declared['formulas'],
        trace_space_nuance=TRACE_NUANCE,
        divergence_free_trace_zero_bubble=divergence_free_bubble_audit(),
        starts_in_space_with_divergence_free_trace_zero_bubble=args.degrees[0] >= 10,
        limitation='Manufactured, low-degree smooth precision control; no inference that the separate ABC or arbitrary Navier-Stokes problem reaches this accuracy.',
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        source_sha256=SOURCE_SHA256)
    (OUT/(stem+'.json')).write_text(json.dumps(output, indent=2)+'\n')
    np.savez_compressed(OUT/(stem+'.npz'), coefficients=solution.coefficients,
                        multiindices=feature.multiindices, bounds=feature.bounds)
    plot(solution, output, stem)
    print(json.dumps(dict(status=solution.status, selected_degree=feature.degree,
                          seconds=solve_seconds, validation=validation,
                          feature_settings={k:feature.metrics.get(k) for k in
                                            ('interior_centers_per_axis', 'lam', 'evaluation',
                                             'derivative_check_tolerance', 'tanh_count')},
                          stages=[dict(degree=r['degree'], residual=r['maximum_validation_rms'],
                                       field_agreement=r['successive_field_difference'],
                                       solver_status=r['solver']['status']) for r in solution.history])), flush=True)


def plot(solution, output, stem='ns_symbolic_precision'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    axis = np.linspace(-.5, .5, 51)
    x, y = np.meshgrid(axis, axis, indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .2), np.full(x.size, .8)))
    truth = reference(torch.as_tensor(points)).numpy()
    predicted = solution.evaluate(points)
    arrays = [np.linalg.norm(predicted[:, :3], axis=1),
              np.linalg.norm(predicted[:, :3]-truth[:, :3], axis=1)/EPS,
              abs(predicted[:, 3]-truth[:, 3])/EPS]
    titles = ['Predicted speed', 'Absolute velocity-vector error / ε', 'Absolute pressure error / ε']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, values, title in zip(axes, arrays, titles):
        mesh = ax.pcolormesh(x, y, values.reshape(x.shape), shading='auto')
        ax.set(xlabel='x', ylabel='y', title=title, aspect='equal')
        fig.colorbar(mesh, ax=ax)
    fig.suptitle('Public symbolic API: manufactured 3D NS, z=0.2, t=0.8; '+output['status'])
    fig.savefig(OUT/(stem+'.png'), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
