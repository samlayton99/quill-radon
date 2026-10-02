"""Plot native physical-flow archives; no reference solution or solve occurs."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quill-physical-flow-mpl')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from route2_physical_flow import make_problem, ORDERS, momentum_and_divergence
from solver.route2 import export_mlp, ordinary_jets


def plot_compact(archives, output):
    """Show the first archive's flow plus independently audited convergence.

    Remaining archives supply the convergence points, not solution labels.
    Zero-iteration re-encoding checks use hollow markers and explicit text.
    """
    archive = Path(archives[0])
    with np.load(archive, allow_pickle=False) as saved:
        config = json.loads(str(saved['metadata']))
        coefficients = np.asarray(saved['coefficients'])
    problem = make_problem(config['case'], config['degree'], config['centers'], config['lam'],
                           config['viscosity'], config['lid_speed'], 8, 5)
    model = export_mlp(problem.features, coefficients)
    record = json.loads(archive.with_suffix('.json').read_text())
    t = np.linspace(-1., 1., 81)
    xx, yy = np.meshgrid(t, t)
    valid = np.ones(xx.shape, bool) if config['case'] == 'cavity' else xx*xx+yy*yy <= 1.
    points = np.column_stack((xx[valid], yy[valid]))
    jets = ordinary_jets(model, points, ORDERS)
    values = jets[(0, 0)].numpy()
    residual = momentum_and_divergence(jets, config['viscosity']).numpy()
    u, v, speed, error = [np.full(xx.shape, np.nan) for _ in range(4)]
    for grid, quantity in zip((u, v, speed, error), (values[:, 0], values[:, 1],
                              np.linalg.norm(values[:, :2], axis=1),
                              np.log10(np.maximum(np.linalg.norm(residual[:, :2], axis=1), 1e-17)))):
        grid[valid] = quantity

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    fig.subplots_adjust(left=.045, right=.99, bottom=.17, top=.75, wspace=.38)
    image = axes[0].pcolormesh(xx, yy, speed, shading='nearest', cmap='viridis')
    axes[0].streamplot(t, t, np.ma.masked_invalid(u), np.ma.masked_invalid(v),
                       color='white', density=1.2, linewidth=.6, arrowsize=.7)
    fig.colorbar(image, ax=axes[0], fraction=.045, pad=.035, label='speed')
    image = axes[1].pcolormesh(xx, yy, error, shading='nearest', cmap='magma',
                             vmin=-16, vmax=max(-13, np.ceil(np.nanmax(error))))
    fig.colorbar(image, ax=axes[1], fraction=.045, pad=.035, label='log10 residual magnitude')
    for axis in axes[:2]:
        axis.set(xlabel='x', ylabel='y', aspect='equal')
    axes[0].set_title(f"Stirred disk: speed + streamlines\nν = {config['viscosity']:g}; "
                      f"{problem.features.tanh_count:,} tanh neurons", fontsize=11)
    axes[1].set_title('Actual network momentum residual\n'
                      f"held-out RMS {record['ordinary_audit']['momentum_rms']:.2e}", fontsize=11)

    groups = {}
    for path in dict.fromkeys(map(str, archives)):
        row = json.loads(Path(path).with_suffix('.json').read_text())
        if row['config']['case'] != config['case']:
            continue
        groups.setdefault(row['config']['viscosity'], []).append(row)
    colors = {0.1: '#1764b4', 0.03: '#bd3a31'}
    handles = []
    for viscosity, rows in sorted(groups.items(), reverse=True):
        rows.sort(key=lambda row: row['features']['neurons'])
        neurons = [row['features']['neurons'] for row in rows]
        errors = [row['ordinary_audit']['momentum_rms'] for row in rows]
        color = colors.get(viscosity)
        line, = axes[2].loglog(neurons, errors, '-o', color=color, linewidth=1.6,
                               markersize=4, label=f'ν = {viscosity:g}')
        handles.append(line)
        for row, x, y in zip(rows, neurons, errors):
            if row['solver']['iterations'] == 0:
                axes[2].plot(x, y, 'o', color=line.get_color(), markerfacecolor='white',
                             markersize=6, markeredgewidth=1.5)
            axes[2].annotate(f"p{row['config']['degree']}", (x, y), xytext=(3, 5),
                             textcoords='offset points', fontsize=7, color=line.get_color())
    threshold = axes[2].axhline(1e-14, color='.45', linestyle='--', linewidth=1,
                               label='requested residual 10⁻¹⁴')
    handles.append(threshold)
    axes[2].set(xlabel='Tanh neurons', ylabel='Held-out momentum RMS',
                title='Native degree refinement\nordinary MLP + automatic derivatives')
    axes[2].set_xlim(2400, max(row['features']['neurons'] for rows in groups.values() for row in rows)*1.22)
    axes[2].xaxis.set_major_locator(FixedLocator([3000, 6000, 10000, 20000, 40000]))
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value/1000:g}k'))
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].grid(alpha=.18, which='both')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .885),
               ncol=len(handles), frameon=False, fontsize=10)
    fig.suptitle('Boundary-driven Navier–Stokes: zero body force, no interior solution labels',
                 y=.99, fontsize=14)
    fig.text(.5, .03, 'Residual is measured on unseen points; no exact solution is supplied. '
             'Hollow markers: denser re-encoding checks, without another optimization step.',
             ha='center', fontsize=9)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output.resolve())


def plot(archives, output, omit_pressure=False):
    columns = 2 if omit_pressure else 3
    fig, axes = plt.subplots(len(archives), columns, figsize=(10.5 if omit_pressure else 14, 4.2*len(archives)), squeeze=False,
                              layout='constrained')
    t = np.linspace(-1., 1., 65)
    xx, yy = np.meshgrid(t, t)
    for row, archive in enumerate(archives):
        with np.load(archive, allow_pickle=False) as saved:
            config = json.loads(str(saved['metadata']))
            coefficients = np.asarray(saved['coefficients'])
        problem = make_problem(config['case'], config['degree'], config['centers'], config['lam'],
                               config['viscosity'], config['lid_speed'], 8, 5)
        model = export_mlp(problem.features, coefficients)
        record_path = Path(archive).with_suffix('.json')
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        valid = np.ones(xx.shape, bool) if config['case'] == 'cavity' else xx*xx+yy*yy <= 1.
        points = np.column_stack((xx[valid], yy[valid]))
        jets = ordinary_jets(model, points, ORDERS)
        values = jets[(0, 0)].numpy()
        residual = momentum_and_divergence(jets, config['viscosity']).numpy()
        grids = [np.full(xx.shape, np.nan) for _ in range(5)]
        for grid, quantity in zip(grids, (values[:, 0], values[:, 1], values[:, 2],
                                          np.linalg.norm(values[:, :2], axis=1),
                                          np.log10(np.maximum(np.linalg.norm(residual[:, :2], axis=1), 1e-16)))):
            grid[valid] = quantity
        u, v, p, speed, error = grids
        image = axes[row, 0].pcolormesh(xx, yy, speed, shading='nearest', cmap='viridis')
        axes[row, 0].streamplot(t, t, np.ma.masked_invalid(u), np.ma.masked_invalid(v),
                               color='white', density=1.3, linewidth=.65, arrowsize=.7)
        fig.colorbar(image, ax=axes[row, 0], shrink=.75, fraction=.045, pad=.025)
        if not omit_pressure:
            image = axes[row, 1].pcolormesh(xx, yy, p, shading='nearest', cmap='coolwarm')
            fig.colorbar(image, ax=axes[row, 1], shrink=.75, fraction=.045, pad=.025)
        residual_axis = axes[row, -1]
        image = residual_axis.pcolormesh(xx, yy, error, shading='nearest', cmap='magma',
                                         vmin=-15.5 if omit_pressure else None,
                                         vmax=-1. if omit_pressure else None)
        fig.colorbar(image, ax=residual_axis, shrink=.75, fraction=.045, pad=.025)
        label = 'Lid-driven square' if config['case'] == 'cavity' else 'Tangentially stirred disk'
        titles = (f"{label}: speed + streamlines\nviscosity {config['viscosity']:g}, degree {config['degree']}, "
                  f"{problem.features.tanh_count:,} neurons", 'Pressure, with p(0)=0',
                  'log10 |momentum residual|\n'+
                  (f"held-out RMS {record['ordinary_audit']['momentum_rms']:.2e}" if record is not None
                   else 'ordinary exported MLP + automatic derivatives'))
        if omit_pressure:
            titles = (titles[0], titles[2])
        for axis, title in zip(axes[row], titles):
            axis.set(xlabel='x', ylabel='y', aspect='equal')
            axis.set_title(title, fontsize=11)
    fig.suptitle('Boundary-driven Navier–Stokes: prescribed walls, zero body force, no interior solution labels\n'
                 'Ordinary tanh networks; equation residuals evaluated independently', fontsize=14)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archives', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--omit-pressure', action='store_true')
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    if args.compact:
        plot_compact(args.archives, args.output)
    else:
        plot(args.archives, args.output, args.omit_pressure)
