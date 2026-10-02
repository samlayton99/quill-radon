"""Independent larger audits of already solved native NS networks.

Never fits or updates coefficients. Includes interior, boundary, initial-time
and corner checks in ordinary float64 Torch, without custom derivative rules.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import itertools
import json
from pathlib import Path
import resource
import sys
import time
import torch
import numpy as np
from route2_hardening_ns import (make_declaration, load_native_resume,
    ordinary_audit, orders, manufactured_jets, physical_terms, validation_field_torch)
from solver.route2 import Route2Solution, ordinary_jets


def audit(path, count=2048, derivative_count=1024):
    torch.set_num_threads(1)
    path = Path(path)
    source_record = json.loads(path.with_suffix('.json').read_text())
    cfg = source_record['config']
    spec = {key: cfg[key] for key in ('family', 'viscosity', 'frequency', 'transient')}
    declaration = make_declaration(**spec)
    started = time.perf_counter()
    native = load_native_resume(path, declaration, batch_size=64, **spec)
    solution = Route2Solution(native, status='saved_native_audit_only')
    result = ordinary_audit(solution, count=count, derivative_count=derivative_count, **spec)
    model = solution.export()
    dimension = declaration.domain.dimension
    zero, first, second, time_order = orders(dimension)
    derivatives = (zero,)+first+second+((time_order,) if cfg['transient'] else ())
    corners = np.array(list(itertools.product(*declaration.domain.bounds)))
    values = ordinary_jets(model, corners, derivatives)
    truth = validation_field_torch(torch.tensor(corners), cfg['family'], cfg['frequency'], cfg['transient']).numpy()
    source, _ = physical_terms(manufactured_jets(corners, cfg['family'], cfg['frequency'], cfg['transient']), cfg['viscosity'], dimension)
    momentum, div = physical_terms({d: v.numpy() for d, v in values.items()}, cfg['viscosity'], dimension)
    result['all_corners'] = dict(points=len(corners),
        maximum_field_error=float(np.max(abs(values[zero].numpy()-truth))),
        momentum_rms=float(np.sqrt(np.mean((momentum-source)**2))),
        momentum_max_abs=float(np.max(abs(momentum-source))),
        divergence_rms=float(np.sqrt(np.mean(div**2))))
    constraints = {}
    for block in declaration.make_blocks(2048, 978377):
        if block.name in ('momentum', 'incompressibility'):
            continue
        jets = ordinary_jets(model, block.points, block.derivatives)
        with torch.no_grad():
            r = block.function(torch.tensor(block.points), jets, torch.empty((len(block.points), 0))).numpy()
        constraints[block.name] = dict(points=len(block.points), rms=float(np.sqrt(np.mean(r*r))), maximum_absolute=float(np.max(abs(r))))
    result['larger_constraint_audit'] = constraints
    result.update(source_archive=str(path.resolve()), seconds=time.perf_counter()-started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        policy='No solve, no coefficient update; saved native network only. Finite sampled checks, not a continuum certificate.')
    output = path.with_name(path.stem+'_extended_audit.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--count', type=int, default=2048)
    parser.add_argument('--derivative-count', type=int, default=1024)
    audit(**vars(parser.parse_args()))
