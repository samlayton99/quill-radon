"""Bounded longer-time native QUILL NS study; comparisons are verification only."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/native-ns-extended-mpl')
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.stats import qmc
from solver.ns_extended import PeriodicNS, solve_ns_refined, taylor_green

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup'


def relative(x, y):
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(y), 1e-30))


def manufactured_initial(points):
    x, y, z = points.T
    return np.column_stack((np.sin(z), np.sin(x), np.sin(y)))


def manufactured_force(viscosity):
    def forcing(t, points):
        x, y, z = points.T
        amplitude, derivative = 1+.2*np.sin(2*t), .4*np.cos(2*t)
        nonlinear = np.column_stack((np.sin(y)*np.cos(z), np.sin(z)*np.cos(x), np.sin(x)*np.cos(y)))
        return (derivative+viscosity*amplitude)*manufactured_initial(points)+amplitude**2*nonlinear
    return forcing


def run(out=OUT, end=1., dt=.005):
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    points = 2*np.pi*(qmc.Sobol(3, scramble=True, seed=7289).random_base2(8)-.5)
    rows, references, stored = [], [], []
    for viscosity in (.05, .01):
        batch = []
        for cutoff in (3, 5, 7):
            model = PeriodicNS(cutoff, 3*cutoff+3, viscosity)
            result = model.solve(final_time=end, dt=dt)
            inference_start = time.perf_counter()
            values = result.evaluate(points)
            inference_seconds = time.perf_counter()-inference_start
            record = dict(viscosity=viscosity, cutoff=cutoff, grid=model.grid,
                          build=model.build, native=result.metrics,
                          native_offgrid_inference_seconds=inference_seconds,
                          history=result.history)
            # Actual network derivatives checked independently by finite differences.
            x = points[:12]
            analytic = result.evaluate(x, True)
            h = 2e-5
            fdgrad = np.stack([(result.evaluate(x+np.eye(3)[j]*h)-result.evaluate(x-np.eye(3)[j]*h))/(2*h) for j in range(3)], axis=2)
            fdlap = sum((result.evaluate(x+np.eye(3)[j]*h)-2*analytic['value']+result.evaluate(x-np.eye(3)[j]*h))/h**2 for j in range(3))
            record['gradient_finite_difference_relative'] = relative(fdgrad, analytic['gradient'])
            record['laplacian_finite_difference_relative'] = relative(fdlap, analytic['laplacian'])
            record['generated_vertical_velocity_rms'] = float(np.sqrt(np.mean(values[:, 2]**2)))
            record['initial_energy'] = result.history[0]['energy']
            record['final_energy'] = result.metrics['diagnostics']['energy']
            record['maximum_recorded_energy_increase'] = float(np.max(np.diff([s['energy'] for s in result.history])))
            print(json.dumps(dict(native_completed={k:v for k,v in record.items() if k != 'history'})), flush=True)
            batch.append((record, model, result, values))

        # Fresh references are produced only after this viscosity's native runs.
        refs = []
        for cutoff in (9, 11, 13):
            reference = PeriodicNS(cutoff, 3*cutoff+3, viscosity, architecture='fft')
            r = reference.solve(final_time=end, dt=dt/2, verify_time=(cutoff == 11), residual_tolerance=None)
            rv = r.evaluate(points)
            refs.append((cutoff, r, rv))
            print(json.dumps(dict(reference_completed=dict(viscosity=viscosity, cutoff=cutoff, metrics=r.metrics))), flush=True)
        reference_record = dict(viscosity=viscosity,
                                K9_vs_K11=relative(refs[0][2], refs[1][2]),
                                K11_vs_K13=relative(refs[1][2], refs[2][2]),
                                rows=[dict(cutoff=k, metrics=r.metrics, build=r.model.build) for k,r,_ in refs])
        references.append(reference_record)
        for record, model, result, values in batch:
            classical = PeriodicNS(model.cutoff, model.grid, viscosity, architecture='fft')
            c = classical.solve(final_time=end, dt=dt, verify_time=False)
            inf_start = time.perf_counter()
            cv = c.evaluate(points)
            inf_seconds = time.perf_counter()-inf_start
            record.update(classical=c.metrics, classical_build=classical.build,
                          classical_offgrid_inference_seconds=inf_seconds,
                          native_vs_same_cutoff_relative=relative(values, cv),
                          native_vs_K13_relative=relative(values, refs[-1][2]),
                          classical_vs_K13_relative=relative(cv, refs[-1][2]),
                          reference_K11_vs_K13=reference_record['K11_vs_K13'],
                          forward_cost_ratio=(model.build['seconds']+result.metrics['evolution_seconds'])/(classical.build['seconds']+c.metrics['evolution_seconds']))
            rows.append(record)
            np.savez_compressed(out/f'ns_K{model.cutoff}_nu{viscosity:g}.npz',
                                coefficients=result.coefficients, points=points,
                                native_velocity=values, classical_velocity=cv,
                                reference_velocity=refs[-1][2],
                                encoder_centers=model.encoder.centers,
                                encoder_weights=model.encoder.weights,
                                encoder_bias=model.encoder.bias,
                                encoder_gamma=model.encoder.gamma)
            stored.append((record, result))
            print(json.dumps(dict(comparison=dict(viscosity=viscosity, cutoff=model.cutoff,
                  total_error=record['native_vs_K13_relative'],
                  native_classical_gap=record['native_vs_same_cutoff_relative'],
                  forward_cost_ratio=record['forward_cost_ratio']))), flush=True)

    control = PeriodicNS(3, 12, .05)
    forced = control.solve(manufactured_initial, manufactured_force(.05), final_time=.5, dt=dt)
    exact = (1+.2*np.sin(1.))*manufactured_initial(points)
    forced.metrics['known_exact_relative_error'] = relative(forced.evaluate(points), exact)
    # Deliberately violate a numerical stability bound; return a failure status.
    stability = PeriodicNS(7, 24, .05).solve(final_time=1., dt=.5, verify_time=False)
    # A valid force outside the retained space: underresolution must be visible.
    def unresolved_force(t, points):
        return np.column_stack((np.sin(9*points[:, 1]), np.zeros((len(points), 2))))
    unresolved = PeriodicNS(3, 30, .05).solve(lambda p: np.zeros_like(p), unresolved_force,
                                           final_time=.1, dt=.005, verify_time=False)
    controls = dict(manufactured_forced=forced.metrics,
                    excessive_dt=stability.metrics,
                    unresolved_forcing=unresolved.metrics)
    metadata = dict(equation='3D periodic incompressible Navier-Stokes',
                    initial='standard interacting Taylor-Green vortex', final_time=end, dt=dt,
                    method='Actual shared QUILL 1D bank values/derivatives, tensor products, real-data FFT/Leray closure, RK4',
                    architecture_change='Product gates and shared banks replace prior flat ridge construction; this is a neural reparameterization of a spectral method',
                    no_oracle='All native evolution uses prescribed IC, viscosity and optional forcing only. Classical trajectories are generated afterward for verification.',
                    timing_scope='One CPU thread, setup+forward measured separately from verification/diagnostics. Fourier baseline uses FFT synthesis; both use equivalent real tensor state.',
                    limitations='Fixed mode cutoff, prescribed periodic cube, smooth positive-viscosity flows. Sampled residual/time refinement do not prove continuum accuracy, regularity, or blowup.',
                    rows=rows, references=references, controls=controls,
                    wall_seconds_before_plots=time.perf_counter()-began,
                    peak_process_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'ns_extended_metrics.json').write_text(json.dumps(metadata, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for nu in (.05, .01):
        r = [r for r in rows if r['viscosity'] == nu]
        k = [x['cutoff'] for x in r]
        axes[0].semilogy(k, [x['native_vs_K13_relative'] for x in r], 'o-', label=f'nu={nu:g}')
        axes[1].semilogy(k, [x['native']['diagnostics']['strong_residual_relative'] for x in r], 'o-', label=f'nu={nu:g}')
        axes[2].plot(k, [x['forward_cost_ratio'] for x in r], 'o-', label=f'nu={nu:g}')
    axes[0].set(title='Velocity error vs K=13 reference', xlabel='Cutoff K', ylabel='Relative error')
    axes[1].set(title='Independent physical residual', xlabel='Cutoff K', ylabel='Residual / force scale')
    axes[1].axhline(1e-3, color='gray', ls='--', lw=1)
    axes[2].set(title='Setup + forward cost ratio', xlabel='Cutoff K', ylabel='QUILL / FFT seconds')
    axes[2].axhline(1, color='gray', ls='--', lw=1)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(f'Constructed product QUILL: interacting 3D NS to T={end:g}')
    fig.savefig(out/'ns_extended_convergence.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for record, result in stored:
        nu, k = record['viscosity'], record['cutoff']
        axes[0].plot([h['time'] for h in result.history], [h['energy'] for h in result.history], label=f'nu={nu:g}, K={k}')
        axes[1].semilogy(k, record['native']['time_refinement_relative'], 'o', color='C0' if nu == .05 else 'C1')
    axes[0].set(title='Energy decay', xlabel='Time', ylabel='Mean kinetic energy')
    axes[0].legend(fontsize=7)
    axes[1].set(title='Native dt versus dt/2 difference', xlabel='Cutoff K', ylabel='Relative velocity difference')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out/'ns_extended_energy.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(completed_seconds=time.perf_counter()-began, controls=controls)), flush=True)
    return metadata


def run_refinement(out=OUT, end=1., dt=.005):
    """Append higher native resolutions without repeating the reference study."""
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for viscosity in (.05, .01):
        result = solve_ns_refined(viscosity=viscosity, final_time=end, dt=dt,
                                  cutoffs=(7, 9, 11, 13, 15), max_seconds=120.)
        solution = result.solution
        validation = np.load(out/f'ns_K7_nu{viscosity:g}.npz')
        points, reference = validation['points'], validation['reference_velocity']
        values = solution.evaluate(points)
        classical = PeriodicNS(solution.model.cutoff, solution.model.grid, viscosity, architecture='fft')
        baseline = classical.solve(final_time=end, dt=dt, verify_time=False)
        row = dict(viscosity=viscosity, status=result.status, adaptive=result.metrics,
                   attempts=result.attempts, final_cutoff=solution.model.cutoff,
                   final_history=solution.history,
                   native_vs_K13_relative=relative(values, reference),
                   native_vs_same_cutoff_relative=relative(values, baseline.evaluate(points)),
                   classical_build=classical.build, classical=baseline.metrics,
                   forward_cost_ratio=(solution.model.build['seconds']+solution.metrics['evolution_seconds'])/(classical.build['seconds']+baseline.metrics['evolution_seconds']))
        if solution.model.cutoff >= 13:
            finer = PeriodicNS(15, 48, viscosity, architecture='fft')
            fine = finer.solve(final_time=end, dt=dt/2, verify_time=False, residual_tolerance=None)
            fine_values = fine.evaluate(points)
            row['native_vs_K15_relative'] = relative(values, fine_values)
            row['reference_K13_vs_K15_relative'] = relative(reference, fine_values)
            row['fresh_K15_reference'] = fine.metrics
        records.append(row)
        np.savez_compressed(out/f'ns_adaptive_nu{viscosity:g}.npz', coefficients=solution.coefficients,
                            points=points, native_velocity=values, reference_velocity=reference,
                            encoder_centers=solution.model.encoder.centers,
                            encoder_weights=solution.model.encoder.weights,
                            encoder_bias=solution.model.encoder.bias,
                            encoder_gamma=solution.model.encoder.gamma)
        print(json.dumps(dict(adaptive_completed=row)), flush=True)
    metadata = dict(rows=records,
                    reference_note='Uses stored validation-only K13 trajectories from the fresh reference phase; no reference is passed to native solve or refinement decisions',
                    peak_process_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'ns_refinement_metrics.json').write_text(json.dumps(metadata, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for r in records:
        k = [a['cutoff'] for a in r['attempts']]
        axes[0].semilogy(k, [a['metrics']['diagnostics']['strong_residual_relative'] for a in r['attempts']], 'o-', label=f"nu={r['viscosity']:g}")
        axes[1].semilogy(k[1:], [a['field_refinement_relative'] for a in r['attempts'][1:]], 'o-', label=f"nu={r['viscosity']:g}")
    axes[0].axhline(1e-3, color='gray', ls='--', label='Tolerance')
    axes[1].axhline(1e-4, color='gray', ls='--', label='Tolerance')
    axes[0].set(title='Physical residual controls refinement', xlabel='Cutoff K', ylabel='Relative strong residual')
    axes[1].set(title='Successive field check', xlabel='Cutoff K', ylabel='Relative velocity change')
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'ns_adaptive_convergence.png', dpi=180)
    plt.close(fig)
    return metadata


def run_product_control(out=OUT, end=1., dt=.005):
    """Cost ablation: exact trig banks with identical tensor arithmetic."""
    metadata = json.loads((out/'ns_refinement_metrics.json').read_text())
    for row in metadata['rows']:
        k, nu = row['final_cutoff'], row['viscosity']
        saved = np.load(out/f'ns_adaptive_nu{nu:g}.npz')
        classical = PeriodicNS(k, 3*k+3, nu, architecture='trig_product')
        result = classical.solve(final_time=end, dt=dt, verify_time=False)
        start = time.perf_counter()
        values = result.evaluate(saved['points'])
        inference = time.perf_counter()-start
        native = PeriodicNS(k, 3*k+3, nu)
        start = time.perf_counter()
        actual = native.evaluate(saved['coefficients'], saved['points'])
        native_inference = time.perf_counter()-start
        row['matched_trig_product'] = dict(build=classical.build, metrics=result.metrics,
                                           offgrid_inference_seconds=inference,
                                           native_offgrid_inference_seconds=native_inference,
                                           native_vs_trig_relative=relative(actual, values),
                                           forward_cost_ratio=(row['attempts'][-1]['build']['seconds']+row['attempts'][-1]['metrics']['evolution_seconds'])/(classical.build['seconds']+result.metrics['evolution_seconds']))
        print(json.dumps(dict(matched_product=dict(viscosity=nu, cutoff=k, result=row['matched_trig_product']))), flush=True)
    metadata['cost_interpretation'] = 'FFT and exact-trig tensor comparisons distinguish arithmetic layout from QUILL encoding; a faster tensor layout is not intrinsically a neural speedup.'
    (out/'ns_refinement_metrics.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def run_checks(out=OUT, end=1., dt=.005):
    """Reproducible public-API/IC/derivative checks; no saved solution inputs."""
    from native_ns_pilot import Coordinates
    out.mkdir(parents=True, exist_ok=True)
    model = PeriodicNS(3, 12, .05)
    b = model.analyze(taylor_green(model.points))
    old = Coordinates(3, 12, .05)
    old_b = old.analyze(model.fields(b, derivatives=False))
    rhs_gap = float(np.max(abs(old.synthesize_spectral(old.spectral_rhs(old_b))-
                               model.fields(model.rhs(0, b), derivatives=False))))
    assert rhs_gap < 1e-12
    rng = np.random.default_rng(831)
    random_b = model.analyze(rng.normal(size=(12**3, 3)))
    exact = PeriodicNS(3, 12, .05, architecture='fft')
    gaps = [float(np.max(abs(a-c))) for a,c in zip(model.fields(random_b), exact.fields(random_b))]
    assert max(gaps) < 1e-11
    def alias_initial(p):
        return np.column_stack((np.sin(9*p[:, 1]), np.zeros((len(p), 2))))
    alias = model.solve(alias_initial, final_time=.01, dt=.005)
    assert alias.status == 'initial_underresolved' and alias.metrics['steps'] == 0
    def divergent_initial(p):
        return np.column_stack((np.sin(p[:, 0]), np.zeros((len(p), 2))))
    divergent = model.solve(divergent_initial, final_time=.01, dt=.005)
    assert divergent.status == 'initial_underresolved'
    assert divergent.metrics['initial_condition']['retained_leray_projection_change_relative'] > .99
    def fifth_mode(p):
        return np.column_stack((np.sin(5*p[:, 1]), np.zeros((len(p), 2))))
    recovery = solve_ns_refined(fifth_mode, viscosity=.05, final_time=.02, dt=.005, cutoffs=(3, 5, 7))
    assert recovery.status == 'converged_sampled'
    assert [a['metrics']['status'] for a in recovery.attempts] == ['initial_underresolved', 'converged_sampled', 'converged_sampled']
    def constant_initial(p):
        return np.column_stack((np.zeros(len(p)), np.ones(len(p)), np.zeros(len(p))))
    def transient_force(t, p):
        return np.column_stack((np.zeros((len(p), 2)), t*(1-t)*np.sin(30*p[:, 0])))
    transient_model = PeriodicNS(3, 12, .001)
    transient = transient_model.solve(constant_initial, transient_force, final_time=1., dt=.05)
    assert transient.status == 'underresolved'
    assert transient.metrics['max_sampled_strong_residual_relative'] > .9
    transient_refinement = solve_ns_refined(constant_initial, transient_force, viscosity=.001,
                                            final_time=1., dt=.05, cutoffs=(3, 5), n_centers=129)
    assert transient_refinement.status == 'resolution_exhausted'
    def commensurate_force(t, p):
        return np.column_stack((np.zeros((len(p), 2)), np.sin(10*np.pi*t)**2*np.sin(30*p[:, 0])))
    jitter = transient_model.solve(constant_initial, commensurate_force, final_time=1., dt=.05)
    assert jitter.status == 'underresolved'
    assert jitter.metrics['forcing_representation']['max_relative_error'] > .9
    def gradient_force(t, p):
        return np.cos(p)*(1+t)
    pressure_only = model.solve(constant_initial, gradient_force, final_time=.05, dt=.005)
    assert pressure_only.status == 'converged_sampled'
    invalid_rejected = 0
    for kwargs in (dict(cutoff=0), dict(grid=9), dict(viscosity=0), dict(architecture='ridge')):
        try:
            PeriodicNS(**kwargs)
        except ValueError:
            invalid_rejected += 1
        else:
            raise AssertionError(kwargs)
    try:
        model.solve(forcing=lambda t,p: np.zeros(len(p)), final_time=.01)
    except ValueError:
        invalid_rejected += 1
    else:
        raise AssertionError('Invalid forcing shape was accepted')
    metadata = dict(status='passed', old_independent_coordinate_rhs_max_difference=rhs_gap,
                    random_field_gradient_laplacian_max_differences=gaps,
                    aliased_initial=alias.metrics, divergent_initial=divergent.metrics,
                    recovery_attempts=recovery.attempts, invalid_options_rejected=invalid_rejected)
    metadata.update(transient_forcing=transient.metrics,
                    transient_refinement_status=transient_refinement.status,
                    temporal_alias_forcing=jitter.metrics,
                    pressure_gradient_forcing=pressure_only.metrics)
    (out/'ns_checks.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(checks='passed', rhs_gap=rhs_gap, random_gaps=gaps)), flush=True)
    return metadata


def plot_slices(out=OUT, end=1., dt=.005):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metadata = json.loads((out/'ns_refinement_metrics.json').read_text())
    axis = np.linspace(-np.pi, np.pi, 65)
    xx, yy = np.meshgrid(axis, axis, indexing='xy')
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, np.pi/4)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, row in zip(axes, metadata['rows']):
        nu, k = row['viscosity'], row['final_cutoff']
        data = np.load(out/f'ns_adaptive_nu{nu:g}.npz')
        model = PeriodicNS(k, 3*k+3, nu)
        values = np.concatenate([model.evaluate(data['coefficients'], points[i:i+256]) for i in range(0, len(points), 256)])
        speed = np.linalg.norm(values, axis=1).reshape(xx.shape)
        im = ax.pcolormesh(xx, yy, speed, shading='auto', cmap='viridis')
        uv = values.reshape(xx.shape+(3,))
        ax.quiver(xx[::6,::6], yy[::6,::6], uv[::6,::6,0], uv[::6,::6,1], color='white', alpha=.8, scale=12)
        fig.colorbar(im, ax=ax, label='Speed')
        ax.set(title=f'nu={nu:g}, K={k}', xlabel='x', ylabel='y', aspect='equal')
    fig.suptitle(f'Actual constructed neural velocity, T={end:g}, z=pi/4')
    fig.savefig(out/'ns_flow_slices.png', dpi=180)
    plt.close(fig)


def run_guard_recheck(out=OUT, end=1., dt=.005):
    """Recheck selected complete trajectories with the final guard semantics."""
    prior = json.loads((out/'ns_refinement_metrics.json').read_text())
    rows = []
    for old in prior['rows']:
        nu, k = old['viscosity'], old['final_cutoff']
        saved = np.load(out/f'ns_adaptive_nu{nu:g}.npz')
        model = PeriodicNS(k, 3*k+3, nu)
        result = model.solve(final_time=end, dt=dt, verify_time=False)
        coefficient_gap = relative(result.coefficients, saved['coefficients'])
        previous_dt = old['attempts'][-1]['metrics']['time_refinement_relative']
        assert result.status == 'completed_unverified'
        assert coefficient_gap < 1e-12 and previous_dt < 1e-6
        row = dict(viscosity=nu, cutoff=k,
                   status='converged_sampled_with_reused_time_check',
                   metrics=result.metrics, history=result.history,
                   coefficients_relative_to_prior=coefficient_gap,
                   reused_time_evidence=dict(source='ns_refinement_metrics.json',
                                             dt=dt, half_dt=dt/2,
                                             relative_difference=previous_dt,
                                             justification='New complete integration reproduces the prior final coefficients; only diagnostics changed. The actual dt/2 comparison remains recorded separately.'))
        rows.append(row)
        print(json.dumps(dict(guard_recheck=dict(viscosity=nu, cutoff=k,
              max_residual=result.metrics['max_sampled_strong_residual_relative'],
              coefficient_gap=coefficient_gap, status=row['status']))), flush=True)
    metadata = dict(rows=rows,
                    timing_note='Separate reruns with intermediate and jittered-force guards; original matched timing measurements are preserved in the earlier metrics files.',
                    peak_process_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'ns_guarded_validation.json').write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--end', type=float, default=1.)
    parser.add_argument('--dt', type=float, default=.005)
    parser.add_argument('--refine-only', action='store_true')
    parser.add_argument('--product-control-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--plot-slices-only', action='store_true')
    parser.add_argument('--guard-recheck-only', action='store_true')
    args = parser.parse_args()
    action = (run_checks if args.check_only else run_guard_recheck if args.guard_recheck_only else plot_slices if args.plot_slices_only else
              run_product_control if args.product_control_only else run_refinement if args.refine_only else run)
    action(args.out, args.end, args.dt)
