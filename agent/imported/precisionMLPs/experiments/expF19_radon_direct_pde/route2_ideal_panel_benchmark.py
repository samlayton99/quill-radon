"""Bounded Cartesian disk correction panels against the angular identity.

This is a derivative-panel timing/consistency check, not a PDE solve or a
prediction of full-solver acceleration. Both methods are target independent.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import json
from pathlib import Path
import time
import tracemalloc
import torch
import numpy as np
from route2_disk_profiles import DiskProfileOperator


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(519)
    angle = rng.uniform(0, 2*np.pi, 64)
    radius = np.sqrt(rng.uniform(0, 1, 64))
    points = np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))
    orders = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    rows = []
    for degree in (12, 24, 40):
        operator = DiskProfileOperator(degree, centers=129)
        indices = np.arange(operator.size-64, operator.size)
        oracle = operator._selected_ideal_columns_ridge(points, indices, orders)
        actual = operator.selected_ideal_columns(points, indices, orders)
        timings = {}
        for name, evaluate in [('angular', operator._selected_ideal_columns_ridge),
                               ('cartesian', operator.selected_ideal_columns)]:
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                evaluate(points, indices, orders)
                samples.append(time.perf_counter()-start)
            timings[name] = float(np.median(samples))
        tracemalloc.start()
        operator.selected_ideal_columns(points, indices, orders)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(dict(degree=degree, points=len(points), columns=len(indices),
            derivative_orders=orders, seconds=timings,
            speedup=timings['angular']/timings['cartesian'], traced_peak_bytes=peak,
            relative_product_differences={str(d): float(np.linalg.norm(actual[d]-oracle[d])/
                max(np.linalg.norm(oracle[d]), 1.)) for d in orders}))
    out = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening'
    out.mkdir(parents=True, exist_ok=True)
    (out/'disk_ideal_panel_benchmark.json').write_text(json.dumps(dict(
        scope='Bounded ideal derivative panels only, not full PDE runtime', rows=rows), indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
