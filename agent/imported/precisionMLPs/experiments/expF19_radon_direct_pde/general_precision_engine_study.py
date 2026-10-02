"""Reproduce generic precision and bounded-column preconditioner checks.

The conditioned algebraic systems isolate optimization error from feature
approximation. Their known coefficients only define/verify these manufactured
test problems; both solves start from zero. The optional NS4D setup profile
measures construction at zero state and does not solve or verify that PDE.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quill-precision-engine-mpl')

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator
import torch

from solver.general_residual import (ResidualBlock, ResidualProblem,
                                     solve_residual, linearize_residual,
                                     _RightPreconditioner)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual'


def conditioned_systems():
    rows = []
    for exponent in (5, 7, 9):
        rng = np.random.default_rng(104)
        left = np.linalg.qr(rng.normal(size=(60, 14)))[0]
        right = np.linalg.qr(rng.normal(size=(14, 14)))[0]
        matrix = (left*np.geomspace(1., 10.**-exponent, 14)) @ right.T
        exact = rng.normal(size=14)
        target = torch.tensor(matrix @ exact)

        class MatrixFeatures:
            dimension, size = 1, 14
            metrics = dict(kind='manufactured_matrix', prescribed_condition_number=10.**exponent)

            def evaluate(self, points, derivative=None):
                return matrix[np.asarray(points[:, 0], int)]

        block = ResidualBlock('linear', np.arange(60.)[:, None],
                              lambda x,j,p:j[(0,)][:, 0]-target, ((0,),))
        problem = ResidualProblem(MatrixFeatures(), [block])
        for high_accuracy in (False, True):
            solution = solve_residual(problem, tolerance=1e-15, high_accuracy=high_accuracy,
                                       max_iterations=15, lsmr_max_iterations=500)
            error = solution.coefficients[:, 0]-exact
            rows.append(dict(prescribed_condition_number=10.**exponent,
                             measured_condition_number=float(np.linalg.cond(matrix)),
                             high_accuracy=high_accuracy, status=solution.status,
                             residual_rms=solution.metrics['blocks'][0]['rms'],
                             coefficient_absolute_l2=float(np.linalg.norm(error)),
                             coefficient_relative_l2=float(np.linalg.norm(error)/np.linalg.norm(exact)),
                             solver=solution.metrics, history=solution.history))
    return rows


def setup_profile(degree, compare_old):
    from general_ns4d_study import DOMAIN, make_blocks
    from solver.general_features import ConstructedFeatures
    began = time.perf_counter()
    features = ConstructedFeatures(DOMAIN.bounds, degree, backend='polynomial', max_derivative=2)
    problem = ResidualProblem(features, make_blocks(6*features.size, 113+1009*degree), fields=4)
    linear = linearize_residual(problem, np.zeros((features.size, 4)))
    linearize_seconds = time.perf_counter()-began
    scale = np.ones(4*features.size)
    fast = _RightPreconditioner(problem, linear.operator, scale)
    result = dict(degree=degree, unknowns=len(scale), linearize_seconds=linearize_seconds,
                  new=fast.metrics, basis_cache_bytes=linear.metrics['basis_cache_bytes'],
                  product_counts=dict(linear.metrics['product_counts']),
                  state='zero coefficients', scaling='identity, same for both builders',
                  scope='Preconditioner construction only; no NS solution or accuracy claim')
    if compare_old:
        wrapped = LinearOperator(linear.operator.shape, matvec=linear.operator.matvec,
                                 rmatvec=linear.operator.rmatvec, dtype=float)
        old = _RightPreconditioner(problem, wrapped, scale)
        result.update(old=old.metrics,
                      maximum_factor_entry_difference=max(float(np.max(abs(a[1]-b[1])))
                                                          for a,b in zip(fast.factors, old.factors)),
                      matched_setup_speedup=old.metrics['setup_seconds']/fast.metrics['setup_seconds'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-degrees', type=int, nargs='*', default=[6, 10])
    args = parser.parse_args()
    torch.set_num_threads(1)
    began = time.perf_counter()
    rows = conditioned_systems()
    profiles = []
    for degree in args.profile_degrees:
        profiles.append(setup_profile(degree, degree <= 6))
        gc.collect()
    result = dict(
        engine_sha256=hashlib.sha256((Path(__file__).parent/'solver/general_residual.py').read_bytes()).hexdigest(),
        precision_cases=rows, preconditioner_profiles=profiles,
        seconds=time.perf_counter()-began,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        scope='Generic algebraic accuracy and local preconditioner setup tests; not a machine-epsilon PDE solution certificate',
        interpretation='Near-epsilon residuals coexist with amplified coefficient errors in ill-conditioned spaces. High accuracy never relaxes the requested tolerance.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'precision_engine.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for high in (False, True):
        selected = [r for r in rows if r['high_accuracy'] == high]
        axes[0].loglog([r['prescribed_condition_number'] for r in selected],
                       [r['residual_rms'] for r in selected], '-o',
                       label='high accuracy' if high else 'defaults')
        axes[1].loglog([r['prescribed_condition_number'] for r in selected],
                       [r['coefficient_relative_l2'] for r in selected], '-o',
                       label='high accuracy' if high else 'defaults')
    axes[0].set(ylabel='Residual RMS', title='Declared residual accuracy')
    axes[1].set(ylabel='Relative coefficient error', title='Conditioning still limits solution accuracy')
    for ax in axes:
        ax.set(xlabel='Matrix condition number')
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(OUT/'precision_engine.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(
        precision=[{k:r[k] for k in ('prescribed_condition_number', 'high_accuracy', 'status',
                                    'residual_rms', 'coefficient_relative_l2')} for r in rows],
        profiles=profiles, seconds=result['seconds'], peak_rss_bytes=result['process_peak_rss_bytes'])), flush=True)


if __name__ == '__main__':
    main()
