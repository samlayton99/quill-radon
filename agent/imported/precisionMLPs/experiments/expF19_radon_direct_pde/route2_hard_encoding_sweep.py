"""Quarantined hard-Helmholtz encoding and analytical capacity audits.

No PDE solver, least-squares fit, optimizer, or target initialization is used.
The primary sweep holds saved native coefficients fixed. A separately named
capacity audit uses explicit Fourier/Bessel coefficients of the known target;
those coefficients must never be supplied to a native PDE solve.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import numpy as np
from scipy.special import jv, eval_chebyu

from route2_disk_recurrence import RecurrenceAffineDiskProfileOperator
from solver.route2 import ordinary_jets


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/helmholtz_n4_s0.npz'
DEFAULT_OUTPUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/hard_encoding'
ORDERS = ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2))


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b), 1e-300))


def ideal_jets(features, coefficients, points, derivatives):
    result = {d: np.empty(len(points)) for d in derivatives}
    for start in range(0, len(points), 64):
        batch = points[start:start+64]
        prepared = features.prepare_ideal_columns(batch, derivatives)
        local = {d: np.zeros(len(batch)) for d in derivatives}
        for col in range(0, features.size, 64):
            indices = np.arange(col, min(col+64, features.size))
            panel = features.selected_ideal_columns(batch, indices, derivatives, prepared=prepared)
            for d in derivatives:
                local[d] += panel[d] @ coefficients[indices]
        for d in derivatives:
            result[d][start:start+len(batch)] = local[d]
    return result


def source_metadata(source):
    source = Path(source)
    metadata = json.loads(source.with_suffix('.json').read_text())
    arrays = np.load(source)
    if (metadata.get('name') != 'helmholtz_n4' or metadata.get('reference_used_in_fit') is not False
            or metadata['metrics'].get('reference_solution_used') is not False):
        raise ValueError('Expected the saved native Helmholtz n4 checkpoint with no reference initialization')
    coefficients = arrays['coefficients'][:, 0].copy()
    degree = metadata['degree']
    if len(coefficients) != (degree+1)*(degree+2)//2:
        raise ValueError('Saved coefficient count does not match declared disk degree')
    return coefficients, arrays, metadata


def halo_growth(degree, centers):
    radius = int(np.ceil(np.sqrt(centers)))
    endpoint = 1 + 2*radius/(centers-1)
    alpha = np.arccosh(endpoint)
    exact = np.sinh((degree+1)*alpha)/np.sinh(alpha)
    asymptotic_alpha = 2*centers**(-.25)
    return dict(halo_per_side=radius, outer_center=endpoint,
                acosh_outer_center=float(alpha), leading_acosh=float(asymptotic_alpha),
                chebyshev_u_at_outer_center=float(exact),
                independently_evaluated_u=float(eval_chebyu(degree, endpoint)),
                exponential_regime_log_u=float((degree+1)*alpha-np.log(2*np.sinh(alpha))),
                leading_exponent=float(2*(degree+1)*centers**(-.25)),
                limitation='Exponential approximation requires (degree+1)*acosh(endpoint) >> 1; for fixed degree and centers→infinity, U_degree→degree+1.')


def pde(jets, points, frequency):
    forcing = frequency**2*np.sin(frequency*points[:, 0])*np.sin(frequency*points[:, 1])
    return -jets[(2, 0)]-jets[(0, 2)]-frequency**2*jets[(0, 0)]-forcing


def audit_encoding(features, coefficients, value_points, derivative_points, ideal_values, ideal_derivatives, frequency,
                   *, reference_values=None):
    began = time.perf_counter()
    model = features.torch_model(coefficients)
    ordinary_values = ordinary_jets(model, value_points, ((0, 0),))[(0, 0)].numpy().ravel()
    ordinary = {d: value.numpy().ravel() for d, value in ordinary_jets(model, derivative_points, ORDERS).items()}
    stable = features.forward_jets(coefficients, derivative_points, ORDERS)
    stable_values = features.forward_jets(coefficients, value_points, ((0, 0),))[(0, 0)]
    ordinary_residual = pde(ordinary, derivative_points, frequency)
    stable_residual = pde(stable, derivative_points, frequency)
    ideal_residual = pde(ideal_derivatives, derivative_points, frequency)
    arrays = features.compile(coefficients)
    _, directional = features.base.unpack(features.to_directional(coefficients))
    encoding = features.base.encoding
    baseline = directional @ encoding.baseline_weights[:, 1:].T
    weights = arrays['output_weights'].reshape(features.direction_count, -1)
    correction = weights-baseline
    radius = features.halo_per_side
    halo_mask = np.zeros(weights.shape[1], bool)
    halo_mask[:radius] = True
    halo_mask[-radius:] = True
    result = dict(degree=features.degree, centers=features.interior_centers,
                lam=encoding.lam, gamma=encoding.gamma, neurons=features.tanh_count,
                readout_l1=float(abs(weights).sum()), readout_max=float(abs(weights).max()),
                output_bias=float(arrays['output_bias'][0]),
                baseline_readout_l1=float(abs(baseline).sum()),
                boundary_correction_readout_l1=float(abs(correction).sum()),
                halo_readout_l1=float(abs(weights[:, halo_mask]).sum()),
                interior_readout_l1=float(abs(weights[:, ~halo_mask]).sum()),
                encoding_bank_highest_degree_l1=float(abs(features.base.weights[:, -1]).sum()),
                actual_directional_profiles_l1=float(abs(directional).sum()),
                ordinary_vs_ideal_value_relative_l2=relative(ordinary_values, ideal_values),
                stable_vs_ideal_value_relative_l2=relative(stable_values, ideal_values),
                ordinary_vs_stable_value_relative_l2=relative(ordinary_values, stable_values),
                ordinary_vs_ideal_jet_rms={str(d):rms(ordinary[d]-ideal_derivatives[d]) for d in ORDERS},
                stable_vs_ideal_jet_rms={str(d):rms(stable[d]-ideal_derivatives[d]) for d in ORDERS},
                ordinary_vs_stable_jet_rms={str(d):rms(ordinary[d]-stable[d]) for d in ORDERS},
                ordinary_pde_rms=rms(ordinary_residual), ordinary_pde_max=float(abs(ordinary_residual).max()),
                stable_pde_rms=rms(stable_residual), ideal_pde_rms=rms(ideal_residual),
                ordinary_minus_ideal_pde_rms=rms(ordinary_residual-ideal_residual),
                stable_minus_ideal_pde_rms=rms(stable_residual-ideal_residual),
                ordinary_minus_stable_pde_rms=rms(ordinary_residual-stable_residual),
                halo_profile_growth=halo_growth(features.degree, features.interior_centers),
                seconds=time.perf_counter()-began)
    if reference_values is not None:
        result.update(ordinary_target_relative_l2=relative(ordinary_values, reference_values),
                      stable_target_relative_l2=relative(stable_values, reference_values))
    return result


def fixed_native_sweep(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coefficients, arrays, metadata = source_metadata(source)
    digest = hashlib.sha256(coefficients.tobytes()).hexdigest()
    points, derivative_points = arrays['points'], arrays['raw_pde_points']
    frequency = 2*np.pi*metadata['case']['n']
    features = RecurrenceAffineDiskProfileOperator([[0, 1], [0, 1]], metadata['degree'], centers=257, lam=.2)
    ideal_values = ideal_jets(features, coefficients, points, ((0, 0),))[(0, 0)]
    ideal_derivatives = ideal_jets(features, coefficients, derivative_points, ORDERS)
    # Verification-only target values; not used in geometry choice or readouts.
    truth = np.sin(frequency*points[:, 0])*np.sin(frequency*points[:, 1])
    report = dict(source_archive=str(Path(source).resolve()), source_archive_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  coefficient_sha256=digest, degree=metadata['degree'], coefficient_l2=float(np.linalg.norm(coefficients)),
                  coefficient_l1=float(abs(coefficients).sum()), value_points=len(points), derivative_points=len(derivative_points),
                  source_policy='Same saved native coefficients in every geometry; no optimizer, least squares, external solver, target fit, or oracle initialization.',
                  reference_scope='Known analytic target is used only once to quantify pre-existing ideal-coordinate approximation error; never to select geometry or modify coefficients.',
                  ideal_coordinate_target_relative_l2=relative(ideal_values, truth), ideal_coordinate_pde_rms=rms(pde(ideal_derivatives, derivative_points, frequency)),
                  saved_ordinary_target_relative_l2=relative(arrays['prediction'].ravel(), truth),
                  rows=[])
    for centers in (257, 513, 1025):
        for lam in (.16, .20, .24):
            features = RecurrenceAffineDiskProfileOperator([[0, 1], [0, 1]], metadata['degree'], centers=centers, lam=lam)
            row = audit_encoding(features, coefficients, points, derivative_points, ideal_values, ideal_derivatives, frequency)
            assert hashlib.sha256(coefficients.tobytes()).hexdigest() == digest
            report['rows'].append(row)
            (output/'fixed_native_helmholtz.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(row), flush=True)
    return report


def analytical_target_coefficients(features, frequency):
    """Normalized disk Fourier/Bessel expansion of sin(kx)sin(ky).

    This is a KNOWN-TARGET CAPACITY construction, never native PDE input.
    For x=.5+z_x/sqrt(2), y=.5+z_y/sqrt(2), integral target modes are
    .5*[cos(k*w_minus·z)-cos(k+k*w_plus·z)], w angles -pi/4,+pi/4.
    Under normalized disk area a plane-wave mode coefficient is
    2*normalization*i**degree*J_(degree+1)(k)/k * angular(harmonic).
    """
    if not np.allclose(features.midpoint, .5) or not np.allclose(features.scale, np.sqrt(2)):
        raise ValueError('This explicit target formula requires the declared [0,1]^2 affine disk chart')
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError('frequency must be finite and positive')
    n, m = features.degrees, features.harmonics
    minus = np.where(features.is_sine, np.sin(-m*np.pi/4), np.cos(-m*np.pi/4))
    plus = np.where(features.is_sine, np.sin(m*np.pi/4), np.cos(m*np.pi/4))
    radial = features.normalizations*(1j**n)*jv(n+1, frequency)/frequency
    return np.real(radial*(minus-np.exp(1j*frequency)*plus))


def capacity_audit(output=DEFAULT_OUTPUT, degrees=(32, 48, 64)):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(633273)
    points = rng.uniform(0, 1, (1024, 2))
    derivative_points = rng.uniform(0, 1, (257, 2))
    frequency = 8*np.pi
    truth = np.sin(frequency*points[:, 0])*np.sin(frequency*points[:, 1])
    report = dict(source_policy='Known-target analytical Fourier/Bessel construction for CAPACITY ONLY. No native initialization or PDE success is claimed.',
                  formula='c_lm=normalization*i^l*J_(l+1)(k)/k * [angular_m(-pi/4)-exp(ik)*angular_m(pi/4)], real part.',
                  centers=513, lam=.2, value_points=len(points), derivative_points=len(derivative_points), rows=[])
    for degree in degrees:
        features = RecurrenceAffineDiskProfileOperator([[0, 1], [0, 1]], degree, centers=513, lam=.2)
        coefficients = analytical_target_coefficients(features, frequency)
        ideal_values = ideal_jets(features, coefficients, points, ((0, 0),))[(0, 0)]
        ideal_derivatives = ideal_jets(features, coefficients, derivative_points, ORDERS)
        row = audit_encoding(features, coefficients, points, derivative_points, ideal_values, ideal_derivatives, frequency,
                             reference_values=truth)
        row.update(coefficient_l2=float(np.linalg.norm(coefficients)), coefficient_l1=float(abs(coefficients).sum()),
                   ideal_coordinate_target_relative_l2=relative(ideal_values, truth),
                   native_solve=False, analytical_known_target=True)
        report['rows'].append(row)
        (output/'analytic_capacity_helmholtz.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    return report


def figure(output=DEFAULT_OUTPUT):
    """Plot completed diagnostics only; no fit or additional experiment."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    native = json.loads((output/'fixed_native_helmholtz.json').read_text())
    capacity = json.loads((output/'analytic_capacity_helmholtz.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    colors = { .16:'#a94442', .20:'#337ab7', .24:'#3b8a5a'}
    for lam, color in colors.items():
        rows = [r for r in native['rows'] if r['lam'] == lam]
        n = [r['centers'] for r in rows]
        axes[0].loglog(n, [r['ordinary_vs_ideal_value_relative_l2'] for r in rows], '-o', color=color, label=f'λ={lam:.2f}')
        axes[1].loglog(n, [r['readout_l1'] for r in rows], '-o', color=color, label=f'λ={lam:.2f}')
    p = [r['degree'] for r in capacity['rows']]
    for key, label, color, marker in [
        ('ideal_coordinate_target_relative_l2', 'Ideal coordinates', '#444444', 'o'),
        ('stable_target_relative_l2', 'Stable neural evaluation', '#3b8a5a', 's'),
        ('ordinary_target_relative_l2', 'Ordinary tanh MLP', '#c97824', '^')]:
        axes[2].semilogy(p, [r[key] for r in capacity['rows']], '-'+marker, color=color, label=label)
    axes[0].set(title='Same native p32 coefficients', xlabel='Interior centers per direction',
                ylabel='Ordinary vs ideal coordinate value error')
    axes[1].set(title='Halo cancellation becomes smaller', xlabel='Interior centers per direction',
                ylabel='Sum of absolute readout weights')
    axes[2].set(title='Separate known-target capacity audit', xlabel='Total coordinate degree p',
                ylabel='Relative target error; N=513, λ=0.20')
    for ax in axes:
        ax.grid(alpha=.22)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.13), frameon=False, fontsize=9)
    axes[0].set_xticks([257, 513, 1025], ['257', '513', '1025'])
    axes[1].set_xticks([257, 513, 1025], ['257', '513', '1025'])
    from matplotlib.ticker import NullFormatter
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xticks(p)
    fig.suptitle('Helmholtz: fitting error and ordinary-MLP encoding error are separate limits', y=.99, fontsize=14)
    fig.text(.5, .018,
        f"Left/middle: fixed native coefficients, no refit; their ideal target error remains {100*native['ideal_coordinate_target_relative_l2']:.3f}%. "
        'Right: explicit Bessel coefficients, capacity only, never used to initialize a PDE solve.',
        ha='center', fontsize=9)
    fig.subplots_adjust(top=.71, bottom=.17, left=.065, right=.98, wspace=.32)
    fig.savefig(output/'encoding_limits.png', dpi=170)
    fig.savefig(output/'encoding_limits.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--capacity-only', action='store_true')
    parser.add_argument('--figure-only', action='store_true')
    options = parser.parse_args()
    if options.figure_only:
        figure(options.output)
    elif options.capacity_only:
        capacity_audit(options.output)
    else:
        fixed_native_sweep(options.source, options.output)
