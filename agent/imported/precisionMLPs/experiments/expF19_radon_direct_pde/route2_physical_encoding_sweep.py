"""Change only the encoding of a saved native physical-flow solution.

No coefficients are fitted, no exact interior solution is used, and no run
here is claimed to be a new PDE solve. The diagnostic tests bandwidth and
center count against the PDE itself and the solved analytical coordinates.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import json
from pathlib import Path
import time
import torch
import numpy as np
from route2_disk_profiles import DiskProfileOperator
from route2_physical_flow import (load_native, coordinate_keys, boundary_velocity,
                                 momentum_and_divergence, ORDERS)
from route2_streamed_study import disk_points, boundary_points
from solver.route2 import export_mlp, ordinary_jets


def run(path):
    torch.set_num_threads(1)
    coefficients, keys, metadata = load_native(path, 'disk_stirring', 1.)
    degree = metadata['degree']
    viscosity = metadata['viscosity']
    points = disk_points(257, 238913)
    edge = boundary_points(137, .091)
    reference = DiskProfileOperator(degree, centers=metadata['centers'], lam=metadata['lam'])
    if coordinate_keys(reference) != keys:
        raise ValueError('Native coordinate metadata mismatch')
    ideal = {order: np.zeros((len(points), 3)) for order in ORDERS}
    for start in range(0, len(points), 64):
        rows = slice(start, min(start+64, len(points)))
        for col in range(0, reference.size, 64):
            indices = np.arange(col, min(col+64, reference.size))
            panels = reference.selected_ideal_columns(points[rows], indices, ORDERS)
            for order in ORDERS:
                ideal[order][rows] += panels[order] @ coefficients[indices]
    records = []
    for centers in (257, 513):
        for lam in (.12, .16, .20, .24, .28):
            started = time.perf_counter()
            features = DiskProfileOperator(degree, centers=centers, lam=lam)
            model = export_mlp(features, coefficients)
            jets = {d: value.numpy() for d, value in ordinary_jets(model, points, ORDERS).items()}
            residual = momentum_and_divergence(jets, viscosity)
            wall = ordinary_jets(model, edge, ((0, 0),))[(0, 0)].numpy()[:, :2]
            row = dict(degree=degree, centers=centers, lam=lam, neurons=features.tanh_count,
                gamma=lam/(2/(centers-1)), momentum_rms=float(np.sqrt(np.mean(residual[:, :2]**2))),
                divergence_rms=float(np.sqrt(np.mean(residual[:, 2]**2))),
                wall_rms=float(np.sqrt(np.mean((wall-boundary_velocity(edge, 'disk_stirring'))**2))),
                coordinate_encoding_rms={str(d):float(np.sqrt(np.mean((jets[d]-ideal[d])**2))) for d in ORDERS},
                seconds=time.perf_counter()-started)
            records.append(row)
            print(json.dumps(row), flush=True)
    output = Path(path).with_name(Path(path).stem+'_encoding_sweep.json')
    output.write_text(json.dumps(dict(source_archive=str(Path(path).resolve()),
        source_policy='Saved native PDE coefficients only; no target values, no new least-squares/PDE fit',
        interpretation='Encoding/geometry diagnostic; same coefficients in reference coordinates. Not ten independently solved PDEs.',
        viscosity=viscosity, audit_points=len(points), rows=records), indent=2)+'\n')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    run(**vars(parser.parse_args()))
