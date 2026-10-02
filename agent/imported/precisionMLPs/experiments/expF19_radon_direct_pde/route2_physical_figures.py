"""Rebuild physical-flow figures and catalogue from saved native runs only."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/precisionmpls-matplotlib')
import csv
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from route2_physical_flow import OUT, ORDERS, load_native, make_problem, momentum_and_divergence
from solver.route2 import export_mlp, ordinary_jets


def record(stem):
    return json.loads((OUT/(stem+'.json')).read_text())


def compact():
    """Plot one independently solved physical field and its resolution checks."""
    stem = 'lowviscosity_refined_disk_stirring_nu0.03_p48_n257'
    r = record(stem)
    c, _, metadata = load_native(OUT/(stem+'.npz'), 'disk_stirring', 1.)
    features = make_problem('disk_stirring', metadata['degree'], metadata['centers'],
        metadata['lam'], metadata['viscosity'], interior=8, boundary_per_side=5).features
    model = export_mlp(features, c)
    axis = np.linspace(-1., 1., 91)
    xx, yy = np.meshgrid(axis, axis)
    mask = xx*xx+yy*yy <= 1.
    points = np.column_stack((xx[mask], yy[mask]))
    jets = ordinary_jets(model, points, ORDERS)
    uv = jets[(0, 0)].numpy()[:, :2]
    residual = momentum_and_divergence(jets, metadata['viscosity']).numpy()[:, :2]
    arrays = [np.full(xx.shape, np.nan) for _ in range(4)]
    arrays[0][mask] = np.linalg.norm(uv, axis=1)
    arrays[1][mask] = np.log10(np.maximum(np.linalg.norm(residual, axis=1), 1e-17))
    arrays[2][mask], arrays[3][mask] = uv.T
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    fig.suptitle('Boundary-driven Navier–Stokes: zero body force, no interior solution labels', fontsize=16)
    im = axes[0].pcolormesh(xx, yy, arrays[0], shading='auto', cmap='viridis')
    axes[0].streamplot(axis, axis, arrays[2], arrays[3], color='white', density=1.1, linewidth=.65)
    fig.colorbar(im, ax=axes[0], label='Speed')
    axes[0].set_title(r'Stirred disk: $\nu=0.03$'+'\n14,259 tanh neurons')
    im = axes[1].pcolormesh(xx, yy, arrays[1], shading='auto', cmap='magma', vmin=-16, vmax=-12)
    fig.colorbar(im, ax=axes[1], label=r'$\log_{10}$ momentum residual magnitude')
    axes[1].set_title('Actual network + ordinary automatic derivatives\nSeparate 4,096-point RMS: '+f"{r['ordinary_audit']['momentum_rms']:.2e}")
    for ax in axes[:2]:
        ax.set(xlabel='x', ylabel='y', aspect='equal', xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
    prefixes = {
        .1: ['tight_ideal_disk_stirring_nu0.1_p8_n257',
             'tight_ideal_disk_stirring_nu0.1_p12_n257',
             'tight_ideal_disk_stirring_nu0.1_p16_n257',
             'refined_ideal_disk_stirring_nu0.1_p20_n257',
             'extended_ideal_disk_stirring_nu0.1_p24_n257',
             'largeblock_ideal_disk_stirring_nu0.1_p32_n257',
             'highdegree_ideal_disk_stirring_nu0.1_p40_n257',
             'resolution_check_disk_stirring_nu0.1_p48_n513'],
        .03: ['lowviscosity_ideal_disk_stirring_nu0.03_p40_n257', stem,
              'lowviscosity_resolution_check_disk_stirring_nu0.03_p56_n513']}
    for viscosity, names in prefixes.items():
        entries = [record(name) for name in names]
        count = [v['features']['neurons'] if 'neurons' in v['features'] else
                 (v['config']['degree']+1)*(v['config']['centers']+2*int(np.ceil(np.sqrt(v['config']['centers']))))
                 for v in entries]
        err = [v['ordinary_audit']['momentum_rms'] for v in entries]
        line, = axes[2].loglog(count, err, '-o', color='#1769b0' if viscosity==.1 else '#bd3f38',
                              label=fr'$\nu={viscosity:g}$')
        axes[2].plot(count[-1], err[-1], 'o', mfc='white', mec=line.get_color(), mew=1.6)
        for width, error, v in zip(count, err, entries):
            axes[2].annotate(f"p{v['config']['degree']}", (width,error), xytext=(3,5), textcoords='offset points',
                             color=line.get_color(), fontsize=8)
    axes[2].axhline(1e-14, ls='--', color='.5', label=r'Requested RMS $10^{-14}$')
    axes[2].set(xlabel='Tanh neurons', ylabel='Held-out momentum RMS',
                title='Native resolution sequence\nOrdinary MLP + automatic derivatives', xlim=(2400, 38000))
    axes[2].xaxis.set_major_locator(FixedLocator([3000, 6000, 10000, 20000, 30000]))
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x, _:f'{x/1000:g}k'))
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].grid(alpha=.2, which='both')
    axes[2].legend(loc='lower center', bbox_to_anchor=(.5,1.16), ncol=3, fontsize=9, frameon=False)
    fig.supxlabel('Hollow markers: richer encoding/constraint checks, no extra optimization. No exact interior solution supplied.', fontsize=10)
    fig.savefig(OUT/'disk_compact.png', dpi=160)
    fig.savefig(OUT/'disk_compact.pdf')
    plt.close(fig)


def completed():
    rows = []
    for path in sorted(OUT.glob('*.json')):
        value = json.loads(path.read_text())
        if isinstance(value, dict) and 'ordinary_audit' in value and 'source_policy' in value and 'config' in value:
            rows.append((path.stem, value))
    return rows


def catalogue():
    rows = []
    for name, value in completed():
        cfg, audit = value['config'], value['ordinary_audit']
        comparison = audit.get('richer_resolution_comparison') or {}
        rows.append(dict(run=name, case=cfg['case'], viscosity=cfg['viscosity'],
            coordinate_basis=cfg.get('coordinate_basis', 'box' if cfg['case']=='cavity' else 'disk'),
            degree=cfg['degree'], interior_centers=cfg['centers'],
            neurons=(cfg['degree']+1)*(cfg['centers']+2*int(np.ceil(np.sqrt(cfg['centers'])))),
            unknowns=3*(cfg['degree']+1)*(cfg['degree']+2)//2,
            interior_points=cfg['interior'], status=value['status'],
            nonlinear_steps=value['solver'].get('iterations'),
            krylov_steps=value['solver'].get('lsmr_iterations'),
            solve_seconds=value['solve_seconds'], process_peak_mib=value['process_peak_rss_bytes']/1024**2,
            momentum_rms=audit['momentum_rms'], momentum_max=audit['momentum_max_abs'],
            divergence_rms=audit['divergence_rms'], wall_rms=audit['wall_velocity_rms'],
            mass_flux=audit['net_boundary_mass_flux'],
            previous_degree=comparison.get('previous_degree'), strictly_richer=comparison.get('strictly_richer'),
            velocity_refinement_relative=comparison.get('velocity_relative_l2'),
            pressure_refinement_relative=comparison.get('pressure_relative_l2'),
            momentum_audit_points=audit['audit_points'],
            source_archive=value['source_policy']['resumed_archive'], body_force='zero', interior_solution_labels=False))
    with (OUT/'physical_benchmark_summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    torch.set_num_threads(1)
    catalogue()
    compact()
