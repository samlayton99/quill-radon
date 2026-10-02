"""Small operator-only timing study; no PDE solve or solver-speed claim."""
from __future__ import annotations

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quill-matplotlib')

import argparse
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
from route2_directional_operator import DirectionalProfileOperator
from solver.ridge_pinn_features import RidgePINNFeatures


def timed(call, repetitions):
    call()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter()-start)
    return dict(median_seconds=float(np.median(samples)), minimum_seconds=min(samples),
                repetitions=repetitions)


def peak_allocated(call):
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def persistent_array_bytes(obj):
    """Count owned persistent ndarray payloads once, including encoding metadata."""
    seen = set()
    def visit(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(visit(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(v) for v in value)
        if hasattr(value, '__dict__'):
            return visit(vars(value))
        return 0
    return visit(obj)


def run(output, queries=2048, repetitions=5):
    rng = np.random.default_rng(919)
    radii = np.sqrt(rng.random(queries))*.98
    angles = 2*np.pi*rng.random(queries)
    x = radii[:, None]*np.column_stack([np.cos(angles), np.sin(angles)])
    derivatives = [(0, 0), (1, 0), (2, 0)]
    cotangents = {a: rng.normal(size=queries) for a in derivatives}
    rows = []
    for m in [8, 16, 32, 64]:
        angles = np.pi*np.arange(m)/m
        directions = np.column_stack([np.cos(angles), np.sin(angles)])
        start = time.perf_counter()
        op = DirectionalProfileOperator(directions, 8, centers=129)
        setup_seconds = time.perf_counter()-start
        theta = rng.normal(size=op.size)/m
        forward = lambda: op.forward_jets(theta, x, derivatives)
        adjoint = lambda: op.adjoint_jets(cotangents, x)
        rows.append(dict(directions=m, profile_degree=8, interior_centers=129,
                         halo_per_side=op.halo_per_side, centers_including_halo=op.centers_per_direction,
                         neurons=op.tanh_count, direct_unknowns=op.size, points=queries,
                         setup_seconds=setup_seconds,
                         forward=timed(forward, repetitions), adjoint=timed(adjoint, repetitions),
                         persistent_ndarray_bytes=persistent_array_bytes(op),
                         forward_peak_traced_allocation_bytes=peak_allocated(forward),
                         adjoint_peak_traced_allocation_bytes=peak_allocated(adjoint),
                         single_streamed_panel_bytes=op.block_size*op.centers_per_direction*8))
    # Matched geometry, different coefficient spaces.  This disk basis uses
    # its existing degree-block angular formulas, not BallRidgeFeatures or
    # analytic_profile_map.  It isolates dense Q-by-P storage/evaluation.
    start = time.perf_counter()
    dense = RidgePINNFeatures(8, directions=9, centers=129)
    dense_setup = time.perf_counter()-start
    op = DirectionalProfileOperator(dense.directions, 8, centers=129)
    a = rng.normal(size=dense.size)*.01
    b = np.zeros((9, 8))
    start_index = 1
    for n in range(1, 9):
        b[:, n-1] = dense.angular_maps[n]@a[start_index:start_index+n+1]
        start_index += n+1
    theta = op.pack(a[0]/np.sqrt(np.pi), b)
    start = time.perf_counter()
    cache = {order: dense.evaluate(x, order) for order in derivatives}
    cache_setup_seconds = time.perf_counter()-start
    dense_forward = lambda: {order: matrix@a for order, matrix in cache.items()}
    dense_adjoint = lambda: sum(cache[order].T@y for order, y in cotangents.items())
    direct_forward = lambda: op.forward_jets(theta, x, derivatives)
    direct_adjoint = lambda: op.adjoint_jets(cotangents, x)
    actual = direct_forward()
    reference = dense_forward()
    grad_direct = direct_adjoint()
    beta_grad, b_grad = op.unpack(grad_direct)
    converted_grad = np.zeros(dense.size)
    converted_grad[0] = beta_grad/np.sqrt(np.pi)
    start_index = 1
    for n in range(1, 9):
        converted_grad[start_index:start_index+n+1] = dense.angular_maps[n].T@b_grad[:, n-1]
        start_index += n+1
    comparison = dict(directions=9, degree=8, neurons=op.tanh_count,
        direct_unknowns=op.size, dense_coordinate_unknowns=dense.size,
        dense_feature_setup_seconds=dense_setup,
        dense_three_jet_cache_setup_seconds=cache_setup_seconds,
        dense_three_jet_cache_bytes=sum(matrix.nbytes for matrix in cache.values()),
        direct_persistent_ndarray_bytes=persistent_array_bytes(op),
        direct_peak_traced_allocation_bytes=peak_allocated(direct_forward),
        dense_cached_forward=timed(dense_forward, repetitions),
        dense_cached_adjoint=timed(dense_adjoint, repetitions),
        direct_forward=timed(direct_forward, repetitions),
        direct_adjoint=timed(direct_adjoint, repetitions),
        same_function_maximum_absolute_jet_discrepancy={str(order):float(np.max(np.abs(actual[order]-reference[order])))
                                                      for order in derivatives},
        pulled_back_adjoint_maximum_absolute_discrepancy=float(np.max(np.abs(converted_grad-dense_adjoint()))),
        interpretation='Same actual tanh geometry and corresponding coefficients; unequal parameter spaces. '
                       'The dense cache is reused and is faster per matvec; streaming trades computation for memory. '
                       'There is no nonlinear solve or matched solver performance comparison.')
    result = dict(scope='operator correctness, cost and storage only; no PDE solve',
        seed=919, precision='float64', threads=1, derivatives=derivatives,
        asymptotic=dict(L='N+2*ceil(sqrt(N)); N counts interior centers',
            direct_parameter_count='1+M*p', neurons='M*L',
            one_jet_forward_or_adjoint='O(M*L*p + Q*M*(L+d))',
            streamed_storage_beyond_inputs_outputs='O(L*p + M*d + block_size*L)',
            outputs_for_J_jets='O(J*Q)', dense_coordinate_cache='O(J*Q*P)',
            removed_full_coordinate_map='O(M*(p+1)*P)',
            caveat='Constant profiles share one bias; higher-degree angular redundancies remain. '
                   'No conditioning, convergence or dimension-free approximation guarantee is established.'),
        memory_note='tracemalloc measures Python/NumPy-tracked peak allocation, not whole-process RSS; '
                    'persistent_ndarray_bytes counts stored array payloads; inputs/outputs reported separately by formulas.',
        direction_scaling=rows, dense_cache_comparison=comparison)
    output.mkdir(parents=True, exist_ok=True)
    (output/'operator_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    m = [row['directions'] for row in rows]
    for key, label in [('forward', 'Forward: value, dx, dxx'), ('adjoint', 'Adjoint: same three jets')]:
        axes[0].plot(m, [row[key]['median_seconds'] for row in rows], 'o-', label=label)
    axes[0].set(xlabel='Directions M', ylabel='Seconds per operator call',
                title=f'Actual tanh evaluation; Q={queries}, N=129, p=8')
    axes[0].legend()
    axes[1].plot(m, [row['forward_peak_traced_allocation_bytes']/2**20 for row in rows],
                 'o-', label='Forward transient allocation')
    axes[1].plot(m, [row['persistent_ndarray_bytes']/2**20 for row in rows],
                 'o-', label='Persistent operator arrays')
    axes[1].set(xlabel='Directions M', ylabel='MiB', title='No point × feature cache')
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.25)
    fig.savefig(output/'operator_scaling.png', dpi=170)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/
                        'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/directional_operator')
    parser.add_argument('--queries', type=int, default=2048)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args()
    run(args.output, args.queries, args.repetitions)
