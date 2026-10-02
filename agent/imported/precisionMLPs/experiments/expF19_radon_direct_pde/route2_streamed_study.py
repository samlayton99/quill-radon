"""Native streamed Radon-MLP forward/inverse diffusion checks and memory audit.

All solves start from zero field coefficients. Manufactured formulas provide
only prescribed source, Dirichlet data, and sparse inverse observations. Actual
tanh values and derivatives are used in every solve and an independently
exported ordinary Torch MLP is evaluated on separate audit points.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quill-streamed-mpl')

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc

import numpy as np
from scipy.stats import qmc
import torch

from route2_directional_operator import DirectionalProfileOperator
from solver.general_residual import (ResidualBlock, ResidualProblem, _RightPreconditioner,
                                     linearize_residual, solve_residual)

OUT = Path(__file__).resolve().parents[2] / 'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/streamed_solver'
ORDERS = ((0, 0), (2, 0), (0, 2))
TRUE_DIFFUSIVITY = 1.7
REACTION = .2


def disk_points(count, seed):
    raw = qmc.Sobol(2, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count]
    radius, angle = np.sqrt(raw[:, 0]), 2*np.pi*raw[:, 1]
    return np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))


def boundary_points(count, offset=0.):
    angle = 2*np.pi*(np.arange(count)+offset)/count
    return np.column_stack((np.cos(angle), np.sin(angle)))


def exact(points, family):
    """Pointwise analytic validation formula; works for NumPy or Torch inputs."""
    x, y = points[:, 0], points[:, 1]
    if family == 'quadratic':
        value = .2+.3*x+.1*y+.2*x*x+.1*x*y-.05*y*y
        laplacian = 0*x+.3
    elif family == 'smooth':
        exponential = torch.exp(.4*x) if isinstance(points, torch.Tensor) else np.exp(.4*x)
        cosine = torch.cos(1.2*y) if isinstance(points, torch.Tensor) else np.cos(1.2*y)
        term = .25*exponential*cosine
        value = .1+term+.05*x*y
        laplacian = (.4**2-1.2**2)*term
    else:
        raise ValueError(family)
    return value, laplacian


def forcing(points, family):
    value, laplacian = exact(points, family)
    nonlinear = torch.tanh(value) if isinstance(points, torch.Tensor) else np.tanh(value)
    return -TRUE_DIFFUSIVITY*laplacian+value+REACTION*nonlinear


def make_problem(degree=2, centers=65, lam=.2, interior=64, boundary=32,
                 family='quadratic', inverse=False, execution='streamed', batch_size=64,
                 directions=None, anchors=8, coordinates='direct'):
    count = degree+1 if directions is None else int(directions)
    angles = np.pi*np.arange(count)/count
    if coordinates == 'direct':
        features = DirectionalProfileOperator(np.column_stack((np.cos(angles), np.sin(angles))),
                                             degree=degree, centers=centers, lam=lam,
                                             block_size=batch_size)
    elif coordinates == 'disk':
        from route2_disk_profiles import DiskProfileOperator
        features = DiskProfileOperator(degree=degree, centers=centers, lam=lam,
                                       direction_count=count, block_size=batch_size)
    else:
        raise ValueError(coordinates)

    def pde(x, j, parameters):
        nu = parameters[:, 0] if inverse else TRUE_DIFFUSIVITY
        value = j[(0, 0)][:, 0]
        return (-nu*(j[(2, 0)][:, 0]+j[(0, 2)][:, 0])+value
                +REACTION*torch.tanh(value)-forcing(x, family))

    def value_constraint(x, j, parameters):
        return j[(0, 0)][:, 0]-exact(x, family)[0]

    blocks = [ResidualBlock('pde', disk_points(interior, 116), pde, ORDERS),
              ResidualBlock('dirichlet', boundary_points(boundary), value_constraint,
                            ((0, 0),), weight=1.)]
    if inverse:
        blocks.append(ResidualBlock('observations', disk_points(anchors, 831), value_constraint,
                                    ((0, 0),), weight=1.))
    problem = ResidualProblem(features, blocks, parameter_initial=[.9] if inverse else [],
                              parameter_bounds=([.05], [5.]) if inverse else None,
                              execution=execution, batch_size=batch_size)
    return problem


def ordinary_audit(features, coefficients, parameter, family, inverse, count=1024):
    model = features.torch_model(coefficients[:, 0])
    points = disk_points(count, 10423)
    values, residuals = [], []
    nu = float(parameter[0]) if inverse else TRUE_DIFFUSIVITY
    for start in range(0, count, 64):
        x = torch.tensor(points[start:start+64], dtype=torch.float64, requires_grad=True)
        u = model(x)[:, 0]
        gradient = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        uxx = torch.autograd.grad(gradient[:, 0].sum(), x, retain_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(gradient[:, 1].sum(), x)[0][:, 1]
        residual = -nu*(uxx+uyy)+u+REACTION*torch.tanh(u)-forcing(x, family)
        values.append(u.detach().numpy())
        residuals.append(residual.detach().numpy())
    predicted, residual = np.concatenate(values), np.concatenate(residuals)
    target = exact(points, family)[0]
    boundary = boundary_points(257, offset=.381)
    with torch.no_grad():
        boundary_values = np.concatenate([model(torch.tensor(boundary[start:start+64], dtype=torch.float64))[:, 0].numpy()
                                          for start in range(0, len(boundary), 64)])
        boundary_error = boundary_values-exact(boundary, family)[0]
    result = dict(relative_l2=float(np.linalg.norm(predicted-target)/np.linalg.norm(target)),
                  field_max_abs=float(np.max(abs(predicted-target))),
                  pde_rms=float(np.sqrt(np.mean(residual**2))),
                  pde_max_abs=float(np.max(abs(residual))),
                  boundary_rms=float(np.sqrt(np.mean(boundary_error**2))),
                  boundary_max_abs=float(np.max(abs(boundary_error))),
                  ordinary_export='torch.nn.Sequential(Linear,Tanh,Linear); float64; ordinary autograd',
                  audit_interior_points=count, audit_boundary_points=len(boundary))
    readout = model[2].weight.detach().numpy().ravel()
    result.update(readout_l1=float(np.sum(abs(readout))), readout_l2=float(np.linalg.norm(readout)),
                  readout_max_abs=float(np.max(abs(readout))),
                  output_bias=float(model[2].bias.detach().numpy()[0]))
    if inverse:
        result.update(diffusivity=nu, diffusivity_abs_error=abs(nu-TRUE_DIFFUSIVITY),
                      diffusivity_relative_error=abs(nu-TRUE_DIFFUSIVITY)/TRUE_DIFFUSIVITY)
    return result


def run_case(name, degree=2, centers=65, lam=.2, interior=64, boundary=32,
             family='quadratic', inverse=False, execution='streamed', batch_size=64,
             tolerance=1e-13, max_seconds=60., max_iterations=20,
             linear_refinement_steps=1, preconditioner='block', directions=None,
             inner_iterations=None, preconditioner_block_size=16, coordinates='direct'):
    started = time.perf_counter()
    problem = make_problem(degree, centers, lam, interior, boundary, family, inverse,
                           execution, batch_size, directions, coordinates=coordinates)
    setup_seconds = time.perf_counter()-started
    # No reference solution, interpolation coefficients, or PDE warmstart.
    solution = solve_residual(problem, max_iterations=max_iterations, tolerance=tolerance,
                              max_seconds=max_seconds, high_accuracy=True,
                              lsmr_max_iterations=inner_iterations or min(4*(problem.features.size+int(inverse)), 400),
                              linear_refinement_steps=linear_refinement_steps,
                              preconditioner=preconditioner, preconditioner_block_size=preconditioner_block_size,
                              preconditioner_ridge=1e-12, scaling_probes=4,
                              validate_locality=False)
    audit = ordinary_audit(problem.features, solution.coefficients, solution.parameters,
                           family, inverse)
    sampled_blocks = solution.metrics['blocks']
    audit['final_sampled_tolerance_met'] = bool(sampled_blocks and all(
        block['maximum_scaled_rms'] <= tolerance for block in sampled_blocks))
    audit['final_sampled_maximum_scaled_rms'] = max(
        (block['maximum_scaled_rms'] for block in sampled_blocks), default=None)
    config = dict(name=name, degree=degree, centers=centers, lam=lam, interior=interior,
                  boundary=boundary, family=family, inverse=inverse, execution=execution,
                  batch_size=batch_size, tolerance=tolerance, max_seconds=max_seconds,
                  directions=problem.features.direction_count,
                  coordinates=coordinates,
                  inner_iterations=inner_iterations, preconditioner_block_size=preconditioner_block_size,
                  preconditioner=preconditioner,
                  tanh_neurons=problem.features.tanh_count,
                  unknowns=problem.features.size+int(inverse),
                  initial_field='all coefficients exactly zero', initial_diffusivity=.9 if inverse else None)
    result = dict(config=config, construction_seconds=setup_seconds, solver=solution.metrics,
                  audit=audit, history=solution.history,
                  scope='Manufactured unit-disk nonlinear reaction-diffusion with ordinary Dirichlet boundary; no exterior solver or oracle field initialization')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{name}.json').write_text(json.dumps(result, indent=2))
    np.savez(OUT/f'{name}.npz', coefficients=solution.coefficients, parameters=solution.parameters,
             **problem.features.compile(solution.coefficients[:, 0]))
    print(json.dumps(dict(name=name, status=solution.status, seconds=solution.metrics['seconds'],
                          iterations=solution.metrics['iterations'], lsmr=solution.metrics['lsmr_iterations'],
                          cache=solution.metrics['basis_cache_bytes'], **audit)), flush=True)
    return result


def current_rss_bytes():
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (psutil.Error, OSError):
        return None


def memory_child(execution, count, directions, degree=8, centers=65, batch_size=64):
    """Isolated-process *incremental* traced allocations and total process RSS.

    Import/model-bank memory precedes baseline. NumPy allocations are traced by
    this Python build. RSS includes runtime/imports and must not be interpreted
    as feature storage; both measurements are retained explicitly.
    """
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    baseline_current_rss = current_rss_bytes()
    tracemalloc.start()
    initial = tracemalloc.get_traced_memory()[0]
    began = time.perf_counter()
    problem = make_problem(degree=degree, centers=centers, interior=count, boundary=32,
                           execution=execution, batch_size=batch_size, directions=directions)
    feature_constructor_seconds = time.perf_counter()-began
    after_constructor, constructor_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    linearization_began = time.perf_counter()
    linearization = linearize_residual(problem, np.zeros((problem.features.size, 1)),
                                      validate_locality=False)
    construction_seconds = time.perf_counter()-linearization_began
    vector = np.linspace(-.1, .1, problem.features.size)
    product = linearization.operator.matvec(vector)
    transpose = linearization.operator.rmatvec(product)
    preconditioner = _RightPreconditioner(problem, linearization.operator,
                                          np.ones(problem.features.size), 16, 1e-12)
    preconditioned = preconditioner.apply(vector)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current_rss = current_rss_bytes()
    # On macOS ru_maxrss is bytes; on Linux it is KiB.
    rss_factor = 1 if sys.platform == 'darwin' else 1024
    return dict(execution=execution, physics_points=count, directions=directions,
                degree=degree, centers=centers, unknowns=problem.features.size,
                batch_size=batch_size, feature_constructor_seconds=feature_constructor_seconds,
                linearization_seconds=construction_seconds,
                total_seconds=time.perf_counter()-began,
                traced_constructor_retained_bytes=after_constructor-initial,
                traced_constructor_peak_bytes=constructor_peak-initial,
                traced_retained_bytes=retained-initial,
                traced_peak_bytes=max(peak, constructor_peak)-initial,
                traced_postconstructor_peak_increment_bytes=peak-after_constructor,
                process_peak_rss_bytes=rss*rss_factor,
                baseline_process_peak_rss_bytes=baseline_rss*rss_factor,
                process_peak_rss_increment_bytes=(rss-baseline_rss)*rss_factor,
                baseline_current_rss_bytes=baseline_current_rss,
                final_current_rss_bytes=current_rss,
                current_rss_increment_bytes=(current_rss-baseline_current_rss
                                             if current_rss is not None and baseline_current_rss is not None else None),
                product_norm=float(np.linalg.norm(product)), transpose_norm=float(np.linalg.norm(transpose)),
                preconditioned_norm=float(np.linalg.norm(preconditioned)),
                preconditioner=preconditioner.metrics, metrics=linearization.metrics)


def memory_study():
    rows = []
    configs = sorted(set([(q, 32) for q in (512, 2048, 8192)]+[(4096, m) for m in (8, 16, 32)]))
    for count, directions in configs:
        for execution in ('cached', 'streamed'):
            completed = subprocess.run([sys.executable, __file__, '--memory-child', execution,
                                        '--interior', str(count), '--directions', str(directions)],
                                       check=True, capture_output=True, text=True,
                                       # Permit posix_spawn rather than invoking OpenMP's
                                       # unsafe fork handlers after Torch is imported.
                                       close_fds=False, env=dict(os.environ), timeout=120)
            row = json.loads(completed.stdout)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('execution', 'physics_points', 'directions',
                                                    'traced_peak_bytes', 'total_seconds')}), flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT/'memory.json').write_text(json.dumps(rows, indent=2))
    return rows


def summarize():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    cases = [json.loads(path.read_text()) for path in sorted(OUT.glob('*.json'))
             if not path.name.startswith('memory') and path.name != 'summary.json']
    memory = json.loads((OUT/'memory.json').read_text()) if (OUT/'memory.json').exists() else []
    summary = dict(cases=cases, memory=memory,
                   interpretation='Native zero-field-start MLP solves; memory audit includes constructor, residual linearization, Jv/JTv, and bounded-block preconditioner, not export or a complete solve trajectory',
                   complexity_scope='Streaming eliminates dense Q-by-coefficient storage. Whole-solve runtime still includes Krylov work and preconditioner construction. Selected-column block preconditioning can revisit every direction for every coefficient block, with conservative O(Q*B*P) setup work for disk coordinates; diagonal probe scaling avoids that setup at potentially greater Krylov cost. Reported preconditioner_setup_seconds separates measured setup from total solve time.',
                   allocation_scope='tracemalloc measures tracked Python/NumPy allocations; process RSS includes Torch/native runtime baseline. Post-constructor peak is incremental above retained geometry. Peak RSS minus previous peak is a high-water difference, not a separate allocation measurement.')
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    direct_smooth = [r for r in cases if r['config']['family'] == 'smooth'
                     and r['config']['execution'] == 'streamed'
                     and r['config'].get('coordinates', 'direct') == 'direct']
    # Keep capped pilot runs in JSON, but plot the best completed accuracy
    # at each resolution, including the longer p16 run explicitly requested.
    best_by_degree = {}
    for row in direct_smooth:
        p = row['config']['degree']
        if p not in best_by_degree or row['audit']['relative_l2'] < best_by_degree[p]['audit']['relative_l2']:
            best_by_degree[p] = row
    smooth = [best_by_degree[p] for p in sorted(best_by_degree)]
    if smooth:
        last_seconds = smooth[-1]['solver']['seconds']
        axes[0].semilogy([r['config']['degree'] for r in smooth],
                         [r['audit']['relative_l2'] for r in smooth], 'o-',
                         label=f'Field relative L2 (p16: {last_seconds:.0f}s)')
        axes[0].semilogy([r['config']['degree'] for r in smooth],
                         [r['audit']['pde_rms'] for r in smooth], 's-', label='PDE residual RMS')
    capped = next((r for r in cases if r['config']['name'] == 'smooth_p16_n257_streamed'), None)
    if capped:
        axes[0].semilogy([16], [capped['audit']['relative_l2']], 'D', color='tab:orange',
                         markersize=7, label='p16: earlier capped solve')
        axes[0].annotate(f"Capped run: {capped['solver']['seconds']:.0f}s", (16, capped['audit']['relative_l2']),
                         xytext=(-95, 10), textcoords='offset points', fontsize=8)
    disk = next((r for r in cases if r['config']['name'] == 'smooth_p16_n257_disk'), None)
    if disk:
        axes[0].semilogy([16], [disk['audit']['relative_l2']], '*', color='tab:green',
                         markersize=12, label=f"p16: disk coordinates ({disk['solver']['seconds']:.0f}s)")
        axes[0].semilogy([16], [disk['audit']['pde_rms']], 'P', color='tab:green', markersize=8,
                         label='Disk coordinates: PDE residual')
    axes[0].set(xlabel='Directional profile order', ylabel='Independent exported-MLP error',
                title='Nonpolynomial target; zero field start\nBest tested budget at each order')
    axes[0].legend()
    for ax, varying, fixed, fixed_value, title in [
            (axes[1], 'physics_points', 'directions', 32, 'Fixed 32 directions, order 8'),
            (axes[2], 'directions', 'physics_points', 4096, 'Fixed 4096 points, order 8')]:
        for execution in ('cached', 'streamed'):
            rows = sorted([r for r in memory if r['execution'] == execution and r[fixed] == fixed_value],
                          key=lambda r:r[varying])
            if rows:
                ax.loglog([r[varying] for r in rows], [r['traced_postconstructor_peak_increment_bytes']/2**20 for r in rows],
                          'o-', label=execution)
        ax.set(xlabel='Physics points' if varying == 'physics_points' else 'Directions',
               ylabel='Peak traced solver allocation (MiB)', title=title)
        if memory:
            tick_values = sorted(set(r[varying] for r in memory if r[fixed] == fixed_value))
            ax.set_xticks(tick_values, [f'{v:,}' for v in tick_values])
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend()
    for ax in axes:
        ax.grid(alpha=.25, which='both')
    diagonal = next((r for r in cases if r['config']['name'] == 'smooth_p16_n257_disk_diagonal'), None)
    if diagonal:
        fig.suptitle(f"Diagonal-only control: field error {diagonal['audit']['relative_l2']:.1e}, "
                     f"PDE RMS {diagonal['audit']['pde_rms']:.1e} in {diagonal['solver']['seconds']:.0f}s; "
                     'no block-preconditioner construction', fontsize=11)
    fig.savefig(OUT/'streamed_summary.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='tiny_streamed')
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--centers', type=int, default=65)
    parser.add_argument('--lam', type=float, default=.2)
    parser.add_argument('--interior', type=int, default=64)
    parser.add_argument('--boundary', type=int, default=32)
    parser.add_argument('--directions', type=int)
    parser.add_argument('--family', choices=['quadratic', 'smooth'], default='quadratic')
    parser.add_argument('--execution', choices=['cached', 'streamed'], default='streamed')
    parser.add_argument('--coordinates', choices=['direct', 'disk'], default='direct')
    parser.add_argument('--inverse', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=60.)
    parser.add_argument('--max-iterations', type=int, default=20)
    parser.add_argument('--tolerance', type=float, default=1e-13)
    parser.add_argument('--refinement', type=int, default=1)
    parser.add_argument('--inner-iterations', type=int)
    parser.add_argument('--preconditioner-block-size', type=int, default=16)
    parser.add_argument('--preconditioner', choices=['diagonal', 'block', 'auto'], default='block')
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--memory-child', choices=['cached', 'streamed'])
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.memory_child:
        print(json.dumps(memory_child(args.memory_child, args.interior, args.directions or 32)))
    elif args.memory:
        memory_study()
    else:
        run_case(args.name, degree=args.degree, centers=args.centers, lam=args.lam,
                 interior=args.interior, boundary=args.boundary, family=args.family,
                 inverse=args.inverse, execution=args.execution, max_seconds=args.max_seconds,
                 max_iterations=args.max_iterations, tolerance=args.tolerance,
                 linear_refinement_steps=args.refinement, directions=args.directions,
                 inner_iterations=args.inner_iterations,
                 preconditioner_block_size=args.preconditioner_block_size,
                 coordinates=args.coordinates, preconditioner=args.preconditioner)


if __name__ == '__main__':
    main()
