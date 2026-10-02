"""Predeclared harder physical flows, no interior reference or body forcing."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '1')
import argparse
import json
from pathlib import Path
import time
import torch
import numpy as np
import route2_physical_flow as physical

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/flows'
SOURCE = physical.OUT/'lowviscosity_refined_disk_stirring_nu0.03_p48_n257.npz'
PROTOCOL = dict(version=1,
    purpose='Unchanged native solver on harder zero-force wall-driven Navier-Stokes; no known interior solution.',
    source_policy=dict(external_pde_solution=False, interior_solution_labels=False, target_initialized_readouts=False,
                       body_force='zero', continuation='Only preceding native PDE coefficients'),
    cases=[dict(name='counterrotation_nu0.1', case='disk_counterrotating', viscosity=.1, initialization='zero'),
           dict(name='counterrotation_nu0.03', case='disk_counterrotating', viscosity=.03, initialization='previous native counterrotation'),
           dict(name='stirring_nu0.01', case='disk_stirring', viscosity=.01, initialization=str(SOURCE)),
           dict(name='stirring_nu0.003', case='disk_stirring', viscosity=.003, initialization='previous native stirring')],
    degrees_cold=[8,16,24,32,48,64], degrees_warm=[48,64], centers=513, lam=.2,
    max_seconds_per_degree=120., inner_iteration_cap=400, outer_iteration_cap=12,
    tolerance=1e-14, coordinate_basis='disk', preconditioner_block_size=640,
    linearization_basis='ideal', inexact_newton=True, audit_points=4096,
    stopping='Keep all scheduled degrees and outcomes, including budget/iteration exits. No exact error available.',
    validation='Unseen ordinary Torch AD residuals, wall data, mass flux, richer-field agreement; no continuum certificate.',
    scope='Steady solution branch; no claim about its dynamical stability or turbulence.',
    resource_scope='Bounded operator/preconditioner panels and measured RSS; time limits soft around inner solves.')


def compatible_native_start(path,first_degree):
    """Use the preceding run's saved matching stage, never truncate silently."""
    path=Path(path)
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
    if metadata['degree']<=first_degree:return str(path)
    matching=path.with_name(path.name.replace(f'_p{metadata["degree"]}_',f'_p{first_degree}_'))
    if matching==path or not matching.exists():
        raise ValueError('A saved lower native stage is required; continuation will not discard coordinates')
    with np.load(matching,allow_pickle=False) as archive:
        prior=json.loads(str(archive['metadata']))
    for key in ('kind','case','viscosity','lid_speed','coordinate_basis'):
        if prior.get(key)!=metadata.get(key):raise ValueError('Matching native stage changes the problem or chart')
    if prior['degree']!=first_degree:raise ValueError('Matching archive has an unexpected degree')
    return str(matching)


def run(selected=None):
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != PROTOCOL:
            raise ValueError('Frozen protocol changed: create a separately named campaign')
    else:
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2)+'\n')
    physical.OUT=OUT
    previous={}
    summary=[]
    for spec in PROTOCOL['cases']:
        if selected is not None and spec['name'] != selected:
            continue
        resume=None
        if spec['initialization'].startswith('previous'):
            resume=previous.get(spec['case'])
            if resume is None:
                candidates=sorted(OUT.glob(f"battle_v1_{spec['case']}_nu{'0.1' if spec['case']=='disk_counterrotating' else '0.01'}_p64_n513.npz"))
                if not candidates:
                    raise ValueError('Required preceding native state is not available')
                resume=str(candidates[-1])
        elif spec['case']=='disk_stirring':
            resume=str(SOURCE)
        if resume is not None:
            resume=compatible_native_start(resume,PROTOCOL['degrees_warm'][0])
        start=time.perf_counter()
        rows=physical.run(case=spec['case'], degrees=tuple(PROTOCOL['degrees_cold'] if resume is None else PROTOCOL['degrees_warm']),
            centers=PROTOCOL['centers'],lam=PROTOCOL['lam'],viscosity=spec['viscosity'],
            seconds=PROTOCOL['max_seconds_per_degree'],krylov=PROTOCOL['inner_iteration_cap'],
            max_iterations=PROTOCOL['outer_iteration_cap'],tolerance=PROTOCOL['tolerance'],
            label='battle_v1',resume=resume,linearization_basis='ideal',audit_points=PROTOCOL['audit_points'],
            inexact_newton=True,preconditioner_block_size=PROTOCOL['preconditioner_block_size'],coordinate_basis='disk')
        last=rows[-1]['config']
        archive=OUT/f"battle_v1_{spec['case']}_nu{spec['viscosity']:g}_p{last['degree']}_n{last['centers']}.npz"
        previous[spec['case']]=str(archive)
        summary.append(dict(case=spec['name'],seconds=time.perf_counter()-start,initial_archive=resume,
            final_archive=str(archive),final_status=rows[-1]['status'],audit=rows[-1]['ordinary_audit']))
        (OUT/(spec['name']+'_summary.json')).write_text(json.dumps(summary[-1],indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selected',choices=[row['name'] for row in PROTOCOL['cases']])
    run(**vars(parser.parse_args()))
