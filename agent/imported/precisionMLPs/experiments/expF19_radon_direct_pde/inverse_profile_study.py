"""Unknown-profile inversion, information loss, and mismatch negative controls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-inverse-profile-mpl')
from pathlib import Path
import json
import time
import hashlib
import numpy as np
from solver.inverse_profile import ProfileForward, ProfileObservations, fit_profile, fit_profile_refined
from native_tanh import modes, features
from native_burgers import eval_fourier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup'


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def independent_measurements(profile, times, x, nu=.15, points=192, dt=.0001):
    """Independent dealiased FFT solver ONLY generates synthetic observations."""
    grid = -np.pi+2*np.pi*np.arange(points)/points
    initial = modes(grid, (len(profile)-1)//2)@profile
    wave = np.fft.fftfreq(points, 1/points)
    integer = wave.astype(int)
    state = np.fft.fft(initial)/points
    keep = abs(wave) < points/2
    state *= keep

    def rhs(c):
        padded = np.zeros(2*points, complex)
        padded[integer % (2*points)] = c
        values = np.fft.ifft(padded*2*points).real
        square = np.fft.fft(values**2)/(2*points)
        return (-.5j*wave*square[integer % (2*points)]-nu*wave**2*c)*keep

    answers, now = [], 0.
    for target in times:
        count = int(np.ceil((target-now)/dt))
        if count:
            step = (target-now)/count
            for _ in range(count):
                a = rhs(state); b = rhs(state+step*a/2)
                c = rhs(state+step*b/2); d = rhs(state+step*c)
                state += step*(a+2*b+2*c+d)/6
        answers.append(eval_fourier(state, x))
        now = target
    return np.asarray(answers)


def verification(model, truth, x, times):
    direction = np.random.default_rng(372).normal(size=len(truth))
    direction /= np.linalg.norm(direction)
    jac = model.predict(truth, x, times, True)[:, :, 1:]
    eps = 1e-5
    difference = (model.predict(truth+eps*direction, x, times)[:, :, 0] -
                  model.predict(truth-eps*direction, x, times)[:, :, 0])/(2*eps)
    tangent_error = relative(difference, jac@direction)
    assert tangent_error < 2e-7
    direct = model.readouts(truth, times[-1])
    literal = np.tanh(direct['gamma']*(x[:, None]-direct['centers']))@direct['readout']+direct['bias']
    constructed = model.predict(truth, x, times[-1:])[-1, :, 0]
    readout_error = relative(literal, constructed)
    assert readout_error < 1e-11
    from unittest.mock import patch
    from contextlib import ExitStack
    def forbidden(*args, **kwargs):
        raise AssertionError('The forward evolution must not fit/readout-invert')
    with ExitStack() as guards:
        for name in ('lstsq', 'pinv', 'inv', 'solve', 'svd'):
            guards.enter_context(patch.object(np.linalg, name, forbidden))
        model.integrate(truth, np.array([.02]), True)
    return dict(directional_tangent_relative_error=tangent_error, explicit_readout_relative_error=readout_error,
                forward_and_tangents_passed_no_lstsq_pinv_inv_solve_svd_guard=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    truth = np.r_[.15, [.10, .20, 0., .06, -.04, .025, 0., .015],
                  [.85, 0., .11, 0., .05, 0., -.025, .01]]
    rng = np.random.default_rng(143)
    sensors = -np.pi+2*np.pi*(np.arange(12)+.4+rng.uniform(-.22, .22, 12))/12
    layouts = {'early': np.array([.03, .08, .15, .25]),
               'late': np.array([.8, 1., 1.2, 1.5])}
    x = np.linspace(-np.pi, np.pi, 1024, endpoint=False)+.127*2*np.pi/1024
    check_times = np.array([0., .25, .8, 1.6])
    references = {name: independent_measurements(truth, times, sensors) for name, times in layouts.items()}
    heldout = independent_measurements(truth, check_times, x)
    refined = independent_measurements(truth, check_times, x, points=256, dt=.00005)
    reference_refinement = relative(heldout, refined)
    model = ProfileForward(evolution_modes=32, dt=.003)
    validation = verification(model, truth, sensors, layouts['early'])
    rows, curves = [], {}
    noise_fraction = .005
    for layout, times in layouts.items():
        clean = references[layout]
        rms = float(np.sqrt(np.mean(clean**2)))
        for label, sigma, noise_seed in [('noiseless', 1e-8, None), ('noisy', noise_fraction*rms, 73)]:
            measured = clean if noise_seed is None else clean+np.random.default_rng(noise_seed).normal(size=clean.shape)*sigma
            obs = ProfileObservations(sensors, times, measured, sigma)
            case_model, estimate, row = fit_profile_refined(model, obs, regularize=noise_seed is not None)
            fields = case_model.predict(estimate, x, check_times)[:, :, 0]
            unreg = case_model.predict(np.asarray(row['unregularized_coefficients']), x, np.array([0.]))[0, :, 0]
            row.update(layout=layout, noise_label=label, noise_sigma=sigma,
                       profile_relative_l2=relative(fields[0], heldout[0]),
                       heldout_fields_relative_l2=relative(fields[1:], heldout[1:]),
                       future_t1_6_relative_l2=relative(fields[-1], heldout[-1]),
                       unregularized_profile_relative_l2=relative(unreg, heldout[0]),
                       neurons=case_model.geometry['neurons'], setup_seconds=case_model.setup_seconds)
            rows.append(row)
            curves[f'{layout}_{label}'] = fields
            curves[f'{layout}_{label}_unregularized'] = unreg
            print(json.dumps({key: row[key] for key in ('layout', 'noise_label', 'status', 'alpha', 'profile_relative_l2', 'future_t1_6_relative_l2', 'fit_seconds')}), flush=True)
    # Match the exact feature calculation, preserving equation, quadrature, dt,
    # observations, optimizer, initial profile, and regularization selection.
    exact = ProfileForward(evolution_modes=32, dt=.003, backend='trigonometric')
    early_sigma = noise_fraction*np.sqrt(np.mean(references['early']**2))
    observed = references['early']+np.random.default_rng(73).normal(size=references['early'].shape)*early_sigma
    exact_estimate, cost_control = fit_profile(exact, ProfileObservations(sensors, layouts['early'], observed, early_sigma))
    cost_control.update(setup_seconds=exact.setup_seconds,
                        profile_relative_l2=relative(exact.predict(exact_estimate, x, np.array([0.]))[0, :, 0], heldout[0]),
                        quill_coefficient_difference=relative(exact_estimate, np.asarray(rows[1]['coefficients'])))
    # Prediction-risk regularization can optimize the measured future while
    # failing to recover the past. It is a separate predeclared data-only rule,
    # not tuned against the heldout initial profile.
    late_sigma = noise_fraction*np.sqrt(np.mean(references['late']**2))
    late_observed = references['late']+np.random.default_rng(73).normal(size=references['late'].shape)*late_sigma
    risk_estimate, risk_selection = fit_profile(model, ProfileObservations(sensors, layouts['late'], late_observed, late_sigma), regularization_rule='sure')
    risk_fields = model.predict(risk_estimate, x, check_times)[:, :, 0]
    risk_selection.update(profile_relative_l2=relative(risk_fields[0], heldout[0]),
                          future_t1_6_relative_l2=relative(risk_fields[-1], heldout[-1]))
    curves['late_prediction_risk_regularized'] = risk_fields
    repeats = []
    for seed in (74, 75):
        for layout, times in layouts.items():
            sigma = noise_fraction*np.sqrt(np.mean(references[layout]**2))
            measured = references[layout]+np.random.default_rng(seed).normal(size=references[layout].shape)*sigma
            estimate, report = fit_profile(model, ProfileObservations(sensors, times, measured, sigma))
            fields = model.predict(estimate, x, check_times)[:, :, 0]
            repeats.append(dict(layout=layout, noise_seed=seed, status=report['status'], alpha=report['alpha'],
                                profile_relative_l2=relative(fields[0], heldout[0]),
                                future_t1_6_relative_l2=relative(fields[-1], heldout[-1]),
                                unregularized_optimizer_success=report['unregularized_optimizer_success'],
                                selected_optimizer_success=report['selected_optimizer_success'],
                                fit_seconds=report['fit_seconds']))
            print(json.dumps(repeats[-1]), flush=True)
    # Sensor bias grows with time; conservation of spatial mean makes it
    # incompatible with changing a single periodic initial profile.
    wrong = references['early']+.12*layouts['early'][:, None]
    wrong_estimate, mismatch = fit_profile(model, ProfileObservations(sensors, layouts['early'], wrong, .001*np.sqrt(np.mean(references['early']**2))))
    # Forward discretization and timestep refinements are independent of data-fit
    # model selection: this detects inverse crimes or inadequate forward fidelity.
    fine = ProfileForward(evolution_modes=48, dt=.0015)
    forward = model.predict(truth, sensors, layouts['late'])[:, :, 0]
    fine_forward = fine.predict(truth, sensors, layouts['late'])[:, :, 0]
    # The unregularized candidate is not automatically trustworthy merely
    # because it fits sensors. Check it with a separate, finer forward method.
    raw_profile = np.asarray(rows[3]['unregularized_coefficients'])
    raw_coarse = model.predict(raw_profile, sensors, layouts['late'])[:, :, 0]
    raw_independent = independent_measurements(raw_profile, layouts['late'], sensors, points=384, dt=.00005)
    validation.update(reference_refinement_relative_l2=reference_refinement,
                      quill_forward_vs_independent_reference=relative(forward, references['late']),
                      combined_forward_refinement_relative_l2=relative(forward, fine_forward),
                      unregularized_late_forward_rms_difference=float(np.sqrt(np.mean((raw_coarse-raw_independent)**2))),
                      unregularized_late_forward_difference_fraction_of_noise=float(np.sqrt(np.mean((raw_coarse-raw_independent)**2))/late_sigma),
                      unregularized_caution='Large inferred profile is less resolved than regularized result; its raw fit is not a validated inverse solution.')
    # Heat control makes loss of information explicit: mode k attenuates by
    # exp(-nu*k^2*t), independently of solver architecture.
    heat = {name: np.exp(-model.nu*np.arange(1, 9)**2*times[0]).tolist() for name, times in layouts.items()}
    result = dict(problem='u_t+u*u_x=.15 u_xx on [-pi,pi]; unknown 17-coordinate initial profile',
                  rows=rows, exact_feature_control=cost_control, mismatch=mismatch,
                  noisy_repeats=repeats, alternative_prediction_risk_rule=risk_selection,
                  validation=validation, sensor_positions=sensors.tolist(),
                  sensor_times={name: times.tolist() for name, times in layouts.items()},
                  truth_profile_coefficients=truth.tolist(), heat_first_observation_attenuation=heat,
                  wall_seconds=time.perf_counter()-started,
                  independent_data='192-point dealiased FFT RK4 generates measurements only; 256-point/half-step validation',
                  limitation='Finite 17-dimensional profile family, known viscosity, Gaussian iid noise, periodic 1D. Regularization is a smoothness assumption, not recovery of annihilated information.',
                  source_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), Path(__file__).parent/'solver/inverse_profile.py']})
    (OUT/'inverse_profile_metrics.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT/'inverse_profile_fields.npz', x=x, times=check_times, target=heldout, **curves)
    plot(result, x, heldout, curves)
    print(json.dumps(dict(validation=validation, mismatch_status=mismatch['status'], wall_seconds=result['wall_seconds']), indent=2))


def plot(result, x, heldout, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.7))
    for r, layout in enumerate(('early', 'late')):
        row = next(item for item in result['rows'] if item['layout'] == layout and item['noise_label'] == 'noisy')
        ax = axes[r, 0]
        ax.plot(x, heldout[0], color='black', label='Actual initial profile', lw=2)
        ax.plot(x, curves[f'{layout}_noisy_unregularized'], color='.65', label='No regularization', lw=1)
        ax.plot(x, curves[f'{layout}_noisy'][0], color='#be4b32', label='Noise-selected regularization', lw=1.8)
        ax.set(title=f'{layout.title()} sensors: recovered initial profile', xlabel='x', ylabel='u(x,0)', ylim=(-1.15, 1.15))
        ax.text(.02, .03, f'Profile error: {row["profile_relative_l2"]:.2%}', transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        if layout == 'late':
            ax.text(.02, .96, 'Rejected unregularized fit:\nforward refinement check fails (curve clipped)',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
        axes[r, 1].plot(x, heldout[-1], color='black', lw=2, label='Withheld solution')
        axes[r, 1].plot(x, curves[f'{layout}_noisy'][-1], color='#be4b32', lw=1.8, label='Prediction')
        axes[r, 1].set(title='Unobserved future, t = 1.6', xlabel='x', ylabel='u(x,1.6)')
        axes[r, 1].text(.02, .03, f'Field error: {row["future_t1_6_relative_l2"]:.2%}', transform=axes[r, 1].transAxes, fontsize=9,
                        bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        for label, marker in [('noiseless', 'o'), ('noisy', 's')]:
            item = next(v for v in result['rows'] if v['layout'] == layout and v['noise_label'] == label)
            # Convert weighted sensitivity back to absolute data sensitivity so
            # noise levels do not obscure the temporal information comparison.
            sensitivity = np.asarray(item['weighted_jacobian_singular_values'])*item['noise_sigma']
            if label == 'noisy':
                axes[r, 2].semilogy(np.arange(1, 18), sensitivity, marker+'-', color='#247ba0')
        axes[r, 2].axhline(row['noise_sigma'], color='#be4b32', ls='--', label='Known noise standard deviation')
        axes[r, 2].set(title='Observation sensitivity to 17 profile directions', xlabel='Singular-value index', ylabel='Unweighted singular value')
        axes[r, 2].text(.02, .04, f'Noisy fit: {row["status"].replace("_", " ")}', transform=axes[r, 2].transAxes, fontsize=8,
                        bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8); axes[0, 2].legend(fontsize=8)
    fig.suptitle('Recovering an unknown initial function from 48 sparse Burgers observations\n17 physical profile coefficients; actual QUILL dynamics; 0.5% measurement noise', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT/'inverse_profile_results.png', dpi=180)
    plt.close(fig)


def focused_checks():
    """Fast API checks plus the inverse-crime counterexample; no broad sweep."""
    OUT.mkdir(parents=True, exist_ok=True)
    model = ProfileForward(profile_modes=2, evolution_modes=4, equation='heat', dt=.002)
    profile = np.array([.1, .2, .3, .4, .5])
    x, times = np.linspace(-2.5, 2.5, 6), np.array([.1, .2])
    y = model.predict(profile, x, times)[:, :, 0]
    recovered, report = fit_profile(model, ProfileObservations(x.tolist(), times.tolist(), y.tolist(), .0001), regularize=False)
    assert np.linalg.norm(recovered-profile) < 1e-8
    assert report['status'] == 'data_consistent'
    bad = 0
    for call in [lambda: ProfileObservations(x, times, y, 0),
                 lambda: ProfileObservations(x, times, y, float('nan')),
                 lambda: ProfileObservations(x, times[::-1], y, .1),
                 lambda: ProfileObservations(x, times, y[:, :-1], .1),
                 lambda: ProfileForward(profile_modes=8, evolution_modes=4),
                 lambda: model.integrate(profile[:-1], times),
                 lambda: model.predict(profile, np.array([4.]), times),
                 lambda: fit_profile(model, ProfileObservations(x, times, y, .1), regularization_rule='oracle')]:
        try:
            call()
        except ValueError:
            bad += 1
    assert bad == 8
    result = dict(heat_profile_recovery_norm=float(np.linalg.norm(recovered-profile)),
                  invalid_input_checks=bad, forward_refinement=report['forward_refinement'])
    coarse = ProfileForward(profile_modes=4, evolution_modes=4, equation='burgers', dt=.002)
    profile = np.array([0., 0., 0., 0., 0., 1., 0., 0., .3])
    x, times = np.linspace(-2.9, 2.9, 12), np.array([.1, .3])
    y = coarse.predict(profile, x, times)[:, :, 0]
    _, report = fit_profile(coarse, ProfileObservations(x, times, y, 1e-5), regularize=False)
    assert report['status'] == 'forward_underresolved'
    result['inverse_crime_negative_control'] = dict(status=report['status'],
        fitted_data_discrepancy=report['discrepancy'], refinement=report['forward_refinement'])
    metrics_path = OUT/'inverse_profile_metrics.json'
    if metrics_path.exists():
        recorded = json.loads(metrics_path.read_text())
        result['inferred_profile_checks_from_recorded_study'] = [
            dict(layout=row['layout'], noise_label=row['noise_label'],
                 status=row['status'], refinements=row['refinement_attempts'],
                 final_forward_check=row['forward_refinement']) for row in recorded['rows']]
    result['reproduce'] = '.venv/bin/python experiments/expF19_radon_direct_pde/inverse_profile_study.py --checks-only'
    (OUT/'inverse_profile_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import sys
    focused_checks() if '--checks-only' in sys.argv else main()
