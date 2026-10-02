"""Frozen, bounded five-input native semilinear elliptic scaling probe.

This is a smooth manufactured scaling control, not a real-world 5D claim.
All five coordinates interact through a full-rank dense quadratic form.
Only prescribed forcing and Dirichlet values enter the native solver.
No active subspace, sparse direction rule, exact readout, or external PDE
solution initializes any stage. Every attempted resolution is retained.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import hashlib
import itertools
import json
from math import comb
from pathlib import Path
import resource
import sys
import time
import traceback
import torch
import numpy as np

from solver.general_domains import ResidualDomain
from solver.general_adaptive import ResidualDeclaration
from solver.general_residual import ResidualBlock
from solver.route2 import solve_route2_native, ordinary_jets


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/five_dimensional'
DIMENSION = 5
_axis = np.arange(1, DIMENSION+1)
B = np.diag(np.array([.018, .022, .026, .030, .034])) + .005*np.cos(_axis[:, None]*_axis[None, :])
VALUE = (0,)*DIMENSION
SECONDS = tuple(tuple(2 if j == i else 0 for j in range(DIMENSION)) for i in range(DIMENSION))
ORDERS = (VALUE,) + SECONDS
PROTOCOL = dict(version=1, equation='-Delta u + u^3 = forcing', dimension=DIMENSION,
    domain=[[-1., 1.]]*DIMENSION, exact_validation_family='exp(x^T B x)', B=B.tolist(),
    degrees=[4, 6, 8], centers=129, lam=.25, angular_rule='tensor',
    tolerance=1e-14, oversampling=6, batch_size=64, maximum_working_array_mb=2048,
    maximum_neurons=2000000, max_seconds=900, max_iterations=8,
    solver_options=dict(lsmr_max_iterations=150),
    check_points=129, export_check_points=65,
    source_policy='Prescribed forcing and boundary values only; zero first-stage readout, native continuation only; no interior labels or known-function readout construction.',
    parameter_policy='No learned PDE parameters and no discovered active axes or symmetry reduction.',
    resource_scope='Single CPU thread; 2GiB named-array estimate, not process RSS cap; 900s soft solve budget. Concurrent research jobs may affect wall times.',
    interpretation='Easy smooth full-five-input scaling control, not a difficult real-world five-dimensional PDE or general dimensional limit.',
    heldout=dict(values=257, derivatives=97, corners=32, boundary=257, seed=573927))


def exact_numpy(points):
    points = np.asarray(points, float)
    quadratic = np.einsum('qi,ij,qj->q', points, B, points)
    value = np.exp(quadratic)
    gradient_log = 2*np.einsum('ij,qj->qi', B, points)
    second = (gradient_log**2+2*np.diag(B))*value[:, None]
    return value, second


def forcing_torch(points):
    matrix = torch.tensor(B, dtype=points.dtype, device=points.device)
    transformed = points @ matrix
    value = torch.exp((transformed*points).sum(dim=1, keepdim=True))
    laplace = (2*torch.trace(matrix)+4*(transformed**2).sum(dim=1, keepdim=True))*value
    return -laplace+value**3


def boundary_torch(points):
    matrix = torch.tensor(B, dtype=points.dtype, device=points.device)
    return torch.exp(torch.sum((points @ matrix)*points, dim=1, keepdim=True))


def make_declaration():
    domain = ResidualDomain(PROTOCOL['domain'])
    def blocks(count, seed):
        interior = domain.interior(count, seed)
        boundary = domain.boundary(max(256, count//2), seed+7919).points
        def physics(x, jets, parameters):
            return -sum(jets[d] for d in SECONDS)+jets[VALUE]**3-forcing_torch(x)
        return [ResidualBlock('semilinear_elliptic', interior, physics, ORDERS),
                ResidualBlock('Dirichlet', boundary, lambda x, j, p: j[VALUE]-boundary_torch(x), (VALUE,))]
    return ResidualDeclaration(domain, blocks, 1)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def audit(result, output):
    return audit_native_state(result.solution.problem.features, result.solution.coefficients, output)


def audit_native_state(features, coefficients, output):
    """Independent audit of a completed native state; no fitting or updates."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    domain = features.bounds
    sampler = ResidualDomain(domain)
    seed = PROTOCOL['heldout']['seed']
    values = sampler.interior(PROTOCOL['heldout']['values'], seed)
    derivatives = sampler.interior(PROTOCOL['heldout']['derivatives'], seed+1)
    corners = np.asarray(list(itertools.product((-1., 1.), repeat=DIMENSION)))
    boundary = sampler.boundary(PROTOCOL['heldout']['boundary'], seed+2).points
    model = features.torch_model(np.asarray(coefficients).reshape(features.size))
    prediction = ordinary_jets(model, values, (VALUE,))[VALUE].numpy().ravel()
    jets = {d: v.numpy().ravel() for d, v in ordinary_jets(model, derivatives, ORDERS).items()}
    target, _ = exact_numpy(values)
    force = forcing_torch(torch.tensor(derivatives, dtype=torch.float64)).numpy().ravel()
    residual = -sum(jets[d] for d in SECONDS)+jets[VALUE]**3-force
    edge_prediction = ordinary_jets(model, boundary, (VALUE,))[VALUE].numpy().ravel()
    corner_prediction = ordinary_jets(model, corners, (VALUE,))[VALUE].numpy().ravel()
    edge_truth, _ = exact_numpy(boundary)
    corner_truth, _ = exact_numpy(corners)
    stable = features.forward_jets(np.asarray(coefficients).reshape(features.size), values, (VALUE,))[VALUE].ravel()
    report = dict(field_relative_l2=float(np.linalg.norm(prediction-target)/np.linalg.norm(target)),
                  field_max_absolute=float(abs(prediction-target).max()),
                  pde_rms=float(np.sqrt(np.mean(residual**2))), pde_max=float(abs(residual).max()),
                  boundary_rms=float(np.sqrt(np.mean((edge_prediction-edge_truth)**2))),
                  boundary_max=float(abs(edge_prediction-edge_truth).max()),
                  corner_rms=float(np.sqrt(np.mean((corner_prediction-corner_truth)**2))),
                  corner_max=float(abs(corner_prediction-corner_truth).max()),
                  ordinary_minus_stable_rms=float(np.sqrt(np.mean((prediction-stable)**2))),
                  readout_l1=float(model[2].weight.detach().abs().sum()),
                  model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                  architecture=[type(layer).__name__ for layer in model],
                  interior_truth_used_only_after_solver_returned=True)
    torch.save(dict(state_dict=model.state_dict(), dimension=DIMENSION, width=model[0].out_features,
                    architecture='Linear/Tanh/Linear'), output/'ordinary_mlp.pt')
    np.savez_compressed(output/'final_native.npz', coefficients=coefficients,
                        values=values, prediction=prediction, target=target,
                        derivative_points=derivatives, residual=residual)
    return report


def run(output=OUTPUT, refine_on_ideal_stall=False):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL)
    protocol['solver_options'] = dict(PROTOCOL['solver_options'])
    if refine_on_ideal_stall:
        protocol['version'] = 2
        protocol['solver_options']['refine_on_ideal_stall'] = True
        protocol['control_scope'] = 'Same cold-start five-dimensional protocol except explicit unproven ideal-stall refinement exit; not an accuracy or stationarity certificate.'
    frozen = json.dumps(protocol, indent=2, sort_keys=True)+'\n'
    protocol_file = output/'protocol.json'
    if protocol_file.exists() and protocol_file.read_text() != frozen:
        raise ValueError('Existing frozen protocol differs; preserve it and use a new output directory')
    protocol_file.write_text(frozen)
    source_names = ['route2_native_five_dimensional.py', 'route2_box_profiles.py', 'route2_box_tensor.py',
                    'route2_directional_operator.py', 'quill_boundary.py', 'solver/route2.py',
                    'solver/general_residual.py', 'solver/streamed_residual.py']
    source_hashes = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in source_names}
    (output/'startup_source_hashes.json').write_text(json.dumps(source_hashes, indent=2)+'\n')
    trace = []
    def checkpoint(coefficients, parameters, record):
        degree = next(p for p in PROTOCOL['degrees'] if comb(p+DIMENSION, DIMENSION) == len(coefficients))
        row = dict(record, degree=degree, saved_initialization='zero then native previous-stage coefficients only')
        trace.append(row)
        np.savez_compressed(output/f'p{degree}_latest_native.npz', coefficients=coefficients, parameters=parameters)
        (output/f'p{degree}_latest_native.json').write_text(json.dumps(row, indent=2, default=_json)+'\n')
        (output/'progress.json').write_text(json.dumps(trace, indent=2, default=_json)+'\n')
        print(json.dumps(dict(event='native_iteration', **row), default=_json), flush=True)
    options = {key: protocol[key] for key in ('degrees', 'centers', 'lam', 'angular_rule',
        'tolerance', 'oversampling', 'batch_size', 'maximum_working_array_mb',
        'maximum_neurons', 'max_seconds', 'max_iterations', 'check_points', 'export_check_points')}
    options['solver_options'] = dict(protocol['solver_options'], iteration_callback=checkpoint)
    began = time.perf_counter()
    report = dict(protocol_sha256=hashlib.sha256(frozen.encode()).hexdigest(), B_eigenvalues=np.linalg.eigvalsh(B).tolist(),
                  B_rank=int(np.linalg.matrix_rank(B)), B_off_diagonal_nonzero_count=int(np.count_nonzero(B-np.diag(np.diag(B)))),
                  reference_used_to_fit=False, startup_source_hashes=source_hashes)
    try:
        result = solve_route2_native(make_declaration(), **options)
        report.update(status=result.status, solver_status=result.solution.status,
                      degree=result.solution.problem.features.degree,
                      neurons=result.solution.problem.features.tanh_count,
                      directions=result.solution.problem.features.direction_count,
                      coordinates=result.solution.problem.features.size,
                      solve_seconds=time.perf_counter()-began, history=result.history, metrics=result.metrics)
        # Save completed solve results before the independent ordinary audit.
        (output/'results.json').write_text(json.dumps(report, indent=2, default=_json)+'\n')
        audit_started = time.perf_counter()
        report['ordinary_audit'] = audit(result, output)
        report['audit_seconds'] = time.perf_counter()-audit_started
    except Exception as error:
        report.update(status='exception', error=repr(error), traceback=traceback.format_exc())
    report.update(total_seconds=time.perf_counter()-began,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024))
    (output/'results.json').write_text(json.dumps(report, indent=2, default=_json)+'\n')
    print(json.dumps(dict(event='result', **{key:value for key,value in report.items() if key not in ('history','metrics')}), default=_json), flush=True)
    return report


def plot_saved_control(output):
    """Plot saved audits only; does not refit or choose a geometry."""
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-ns-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    result = json.loads((output/'results.json').read_text())
    provenance = json.loads((output/'comparison_provenance.json').read_text())
    if not (provenance['p4_coefficients_bitwise_identical']
            and provenance['p8_initial_lower_modes_bitwise_equal_to_baseline_last_accepted_p6']
            and provenance['p8_new_modes_exactly_zero']):
        raise ValueError('The saved lower-degree audits are not verified control states')
    baseline = ROOT/provenance['baseline']
    rows = [json.loads((baseline/name).read_text())
            for name in ('p4_completed_audit.json', 'p6_checkpoint_audit.json')]
    rows.append(dict(result['ordinary_audit'], degree=result['degree'], neurons=result['neurons']))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    positions = np.arange(len(rows))
    for key, label, color in (
            ('field_relative_l2', 'Relative field error', '#2874a6'),
            ('pde_rms', 'Raw PDE residual RMS', '#b03a2e'),
            ('corner_max', 'Largest corner error', '#9a7d0a')):
        axes[0].semilogy(positions, [row[key] for row in rows], 'o-', label=label, color=color)
    axes[0].set(xticks=positions, xticklabels=[f"p={row['degree']}\n{row['neurons']:,} neurons" for row in rows],
                ylabel='Error (each measure labeled)', title='More resolution improves accuracy')
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.10), ncol=1, frameon=False, fontsize=9)
    stage_seconds = [row['solver_work']['seconds'] for row in result['history']]
    bars = axes[1].bar(positions, stage_seconds, color=['#85c1e9', '#5dade2', '#2874a6'])
    for bar, stage in zip(bars, result['history']):
        seconds = stage['solver_work']['seconds']
        krylov = stage['solver_work']['lsmr_iterations']
        axes[1].text(bar.get_x()+bar.get_width()/2, seconds+12,
                     f'{seconds:.0f} s\n{krylov} Krylov iterations', ha='center', fontsize=9)
    axes[1].set(xticks=positions, xticklabels=[f"p={row['degree']}" for row in rows],
                ylabel='Native stage solve time (seconds)', ylim=(0, max(stage_seconds)*1.2),
                title='One million neurons: time becomes costly')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle('Five interacting inputs: cold native semilinear PDE solve', y=.98, fontsize=14)
    fig.text(.5, .04,
             'Smooth manufactured control; forcing + boundary values only. 917 s solve; 476 MB peak RSS.\n'
             'Independent ordinary-MLP audit: 257 interior values, 97 PDE points, 257 walls, all 32 corners.\n'
             'Soft time budget reached after p8 correction; accuracy goal not reached. No continuum certificate.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.72, wspace=.27)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'five_dimensional_accuracy_cost.{suffix}', dpi=180)
    plt.close(fig)
    summary = dict(audited_stages=rows, stage_solver_seconds=stage_seconds,
                   stages=[dict(degree=row['degree'], status=row['solver_status'],
                                work=row['solver_work']) for row in result['history']],
                   solve_seconds=result['solve_seconds'], audit_seconds=result['audit_seconds'],
                   peak_rss_bytes=result['peak_rss_bytes'],
                   baseline_disposition='Manual resource interruption at 929.05 seconds inside incomplete p6 actual-J pass.',
                   control_disposition='Normal soft time-budget return after accepted p8 ideal-J correction.',
                   performance_scope='Counters establish no p8 actual-J fallback and only 17 inner iterations; separate per-kernel timings were not instrumented.',
                   conclusion='Memory-bounded five-input native scaling demonstrated; precision floor and difficult five-dimensional PDE performance not established.')
    (output/'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--refine-on-ideal-stall', action='store_true')
    parser.add_argument('--figure-only', action='store_true')
    args = parser.parse_args()
    if args.figure_only:
        plot_saved_control(args.output)
    else:
        run(args.output, args.refine_on_ideal_stall)
