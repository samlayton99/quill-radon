"""Append exactly one K=5 native NS refinement, preserving K=2/3 artifacts."""
from native_ns_pilot import Coordinates, NeuralCoordinates, OUT, initial, integrate, relative
import json
import resource
import time
import numpy as np


def run(out=OUT):
    began = time.perf_counter()
    metadata = json.loads((out/'ns_native_metrics.json').read_text())
    assert [r['cutoff'] for r in metadata['rows']] == [2, 3], 'Append once to existing coarse results'
    # Reuse only validation coordinates, never any saved solution as native input.
    points = np.load(out/'ns_native_K3.npz')['evaluation_points']
    model = NeuralCoordinates(5, 18, metadata['viscosity'], n_interior=257, lam=.2)
    b0 = model.analyze(initial(model.points))
    b, timing = integrate(model.rhs, b0, metadata['final_time'], .002)
    u, grad, lap = model.fields(b)
    direct, dgrad, dlap = model.direct(b, model.points[::97], True)
    record = dict(cutoff=5, grid=18, build=model.build, native=timing,
                  initial_relative_error=relative(model.direct(b0, points), initial(points)),
                  direct_vs_cache_relative=relative(direct, u[::97]),
                  direct_gradient_vs_cache_relative=relative(dgrad, grad[::97]),
                  direct_laplacian_vs_cache_relative=relative(dlap, lap[::97]),
                  divergence_max=float(np.max(abs(np.trace(grad, axis1=1, axis2=2)))),
                  initial_energy=float(.5*np.mean(np.sum((model.value@b0)**2, axis=1))),
                  final_energy=float(.5*np.mean(np.sum(u*u, axis=1))),
                  newly_generated_mode_norm=float(np.linalg.norm((b-b0)[np.linalg.norm(b0, axis=1)<1e-13])),
                  velocity_change_relative=relative(u, model.value@b0),
                  generated_vertical_velocity_rms=float(np.sqrt(np.mean(u[:, 2]**2))))
    print(json.dumps(dict(native_completed=record)), flush=True)

    # Only now create fresh conventional solutions for verification.
    reference = Coordinates(9, 30, model.viscosity)
    bref, reference_time = integrate(reference.spectral_rhs, reference.analyze(initial(reference.points)),
                                    metadata['final_time'], .001)
    ordinary, ordinary_time = integrate(model.spectral_rhs, b0, metadata['final_time'], .002)
    actual = model.direct(b, points)
    conventional = model.heldout_spectral(ordinary, points)
    ref_values = reference.heldout_spectral(bref, points)
    record.update(same_cutoff_spectral=ordinary_time,
                  fresh_cutoff9_reference=reference_time,
                  native_vs_same_cutoff_spectral_relative=relative(actual, conventional),
                  native_vs_cutoff9_relative=relative(actual, ref_values),
                  spectral_same_cutoff_vs_cutoff9_relative=relative(conventional, ref_values),
                  native_vs_dt_half_relative=None,
                  time_refinement_note='Not repeated for K=5; K=2/3 time refinement remains recorded',
                  native_evolution_cost_ratio=timing['seconds']/ordinary_time['seconds'],
                  construction_plus_evolution_seconds=model.build['seconds']+timing['seconds'])
    np.savez_compressed(out/'ns_native_K5.npz', coefficients=b, initial_coefficients=b0, k=model.k,
                        evaluation_points=points, native_velocity=actual,
                        same_cutoff_spectral_velocity=conventional, reference_velocity=ref_values)
    network = model.export(b)
    np.savez_compressed(out/'ns_native_network_K5.npz', **network)
    exported = np.broadcast_to(network['bias'], (len(points), 3)).copy()
    for v, c, g, w in zip(network['directions'], network['centers'], network['gamma'], network['weights']):
        exported += np.tanh(g*((points@v)[:, None]-c))@w
    record['exported_readout_relative_difference'] = relative(exported, actual)
    record['peak_process_rss_bytes_macos'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record['append_wall_seconds'] = time.perf_counter()-began

    assert record['native_vs_cutoff9_relative'] < metadata['rows'][-1]['native_vs_cutoff9_relative']/100
    assert record['native_vs_same_cutoff_spectral_relative'] < 1e-11
    assert record['divergence_max'] < 1e-11
    assert record['exported_readout_relative_difference'] < 1e-12
    assert record['direct_gradient_vs_cache_relative'] < 1e-12
    assert record['direct_laplacian_vs_cache_relative'] < 1e-12
    assert record['generated_vertical_velocity_rms'] > .001
    metadata['rows'].append(record)
    metadata['reference_protocol'] = 'K2/K3 references were created after the coarse native batch; K5 was appended later and its fresh K9 comparison was created after K5 evolution. No comparison solution enters any native RHS.'
    (out/'ns_native_metrics.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(refinement_complete=record)), flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = metadata['rows']
    k = [r['cutoff'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    axes[0].semilogy(k, [r['native_vs_cutoff9_relative'] for r in rows], 'o-', label='Total vs K=9')
    axes[0].semilogy(k, [r['native_vs_same_cutoff_spectral_relative'] for r in rows], 's-', label='Neural vs same K')
    axes[0].semilogy(k[:2], [r['native_vs_dt_half_relative'] for r in rows[:2]], '^-', label='dt vs dt/2 (K=2,3)')
    axes[0].set(title='Separate numerical errors', xlabel='Fourier cutoff K', ylabel='Relative velocity error')
    axes[0].legend(fontsize=8)
    axes[1].bar(np.array(k)-.18, [r['native']['seconds'] for r in rows], .36, label='Native evolution')
    axes[1].bar(np.array(k)+.18, [r['same_cutoff_spectral']['seconds'] for r in rows], .36, label='Spectral evolution')
    axes[1].set(title='Evolution cost (one CPU thread)', xlabel='Fourier cutoff K', ylabel='Seconds')
    axes[1].legend(fontsize=8)
    axes[2].bar(k, [r['build']['seconds'] for r in rows])
    axes[2].set(title='One-time QUILL/caching cost', xlabel='Fourier cutoff K', ylabel='Seconds')
    for ax in axes:
        ax.set_xticks(k)
        ax.grid(alpha=.2)
    fig.suptitle('Native QUILL: interacting 3D Taylor–Green flow, T=.05')
    fig.savefig(out/'ns_native_comparison.png', dpi=180)
    plt.close(fig)
    return record


if __name__ == '__main__':
    run()
