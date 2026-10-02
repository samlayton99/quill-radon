"""Matched generic diagonal versus bounded-block preconditioner study."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from general_residual_study import cases, evaluate_case, clean, OUT
from solver.general_features import ConstructedFeatures
from solver.general_residual import ResidualProblem, solve_residual


def run(case, degree=12, backend='quill', iterations=35, methods=('diagonal', 'block')):
    torch.set_num_threads(1)
    began = time.perf_counter()
    order = max(map(sum, case.derivatives))
    features = ConstructedFeatures(case.domain.bounds, degree, backend=backend, max_derivative=order)
    points = max(256, 5*features.size)
    problem = ResidualProblem(features, case.blocks(points, 113), fields=case.fields,
                              parameter_initial=case.parameter_initial,
                              parameter_bounds=case.parameter_bounds)
    construction_seconds = time.perf_counter()-began
    rows = []
    for method in methods:
        start = time.perf_counter()
        solution = solve_residual(problem, max_iterations=iterations, tolerance=1e-9,
                                  preconditioner=method, preconditioner_block_size=64,
                                  max_seconds=120.)
        seconds = time.perf_counter()-start
        validation = evaluate_case(case, solution)
        row = dict(case=case.name, degree=degree, backend=backend, preconditioner=method,
                   shared_feature_problem_setup_seconds=construction_seconds,
                   solve_seconds=seconds, validation=validation,
                   features=features.metrics, solver=solution.metrics, history=solution.history,
                   peak_process_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rows.append(row)
        OUT.mkdir(parents=True, exist_ok=True)
        stem=f'precondition_{case.name}_{backend}_p{degree}_{method}'
        (OUT/(stem+'.json')).write_text(json.dumps(clean(row), indent=2))
        np.savez_compressed(OUT/(stem+'.npz'), coefficients=solution.coefficients,
                            parameters=solution.parameters)
        print(json.dumps(dict(case=case.name, preconditioner=method, seconds=seconds,
                              validation=validation, status=solution.status,
                              iterations=solution.metrics['iterations'],
                              lsmr_iterations=solution.metrics['lsmr_iterations'],
                              preconditioner_setup_seconds=solution.metrics['preconditioner_setup_seconds'],
                              products=solution.metrics['product_counts'])), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', default=['biharmonic_plate', 'three_dimensional_ellipsoid'])
    parser.add_argument('--degree', type=int, default=12)
    parser.add_argument('--backend', choices=('quill', 'polynomial'), default='quill')
    parser.add_argument('--iterations', type=int, default=35)
    parser.add_argument('--methods', nargs='+', choices=('diagonal', 'block', 'auto'), default=['diagonal', 'block'])
    args = parser.parse_args()
    definitions = cases()
    for name in args.cases:
        run(definitions[name], args.degree, args.backend, args.iterations, args.methods)
