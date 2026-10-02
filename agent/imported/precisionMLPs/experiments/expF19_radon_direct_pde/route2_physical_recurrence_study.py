"""Generic memory-for-time control on four harder native physical flows.

The original counterrotation campaign completed its p64 correction before a
runner bug attempted to continue its p64 archive into a smaller p48 space.
This separate protocol retains that outcome. It always continues at p64,
uses bounded recurrence products, and spends a declared 1GiB allowance on
preconditioner work. There are no interior reference fields or body forces.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import hashlib
import json
from pathlib import Path
import torch
import route2_physical_flow as physical
from route2_disk_recurrence import RecurrenceDiskProfileOperator
from solver.route2 import _memory_bounded_preconditioner

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde'
OUT=BASE/'route2_battletest/flows_recurrence'
OLD=BASE/'route2_hardening/physical_flow'
PROTOCOL=dict(version=1,degree=64,centers=513,lam=.2,seconds=600.,krylov=150,
    max_iterations=12,tolerance=1e-14,requested_block_size=2048,
    preconditioner_workspace_allowance_bytes=1024**3,
    cases=[dict(case='disk_counterrotating',viscosity=.1,
                resume=str(BASE/'route2_battletest/flows/battle_v1_disk_counterrotating_nu0.1_p48_n513.npz')),
           dict(case='disk_counterrotating',viscosity=.03,resume='previous native case'),
           dict(case='disk_stirring',viscosity=.01,
                resume=str(OLD/'lowviscosity_refined_disk_stirring_nu0.03_p48_n257.npz')),
           dict(case='disk_stirring',viscosity=.003,resume='previous native case')],
    source_policy='Only prescribed wall motion and zero-force PDE; native coefficient continuation; no interior truth, sensors, external trajectories or exact-solution coefficients.',
    changes='Separate continuation study: larger memory-bounded parity preconditioner, opt-in bounded Jacobi recurrence, uniform p64/150K/600s settings. No isolated end-to-end speed claim.',
    arithmetic='Actual ordinary float64 Linear/Tanh/Linear residual, acceptance and audit unchanged.',
    budgets='1GiB for estimated preconditioner/correction work only, not process RSS cap; soft time budget; retain all exits and overshoots.')


def run():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    frozen=json.dumps(PROTOCOL,indent=2,sort_keys=True)+'\n'
    protocol=OUT/'protocol.json'
    if protocol.exists() and protocol.read_text()!=frozen:raise ValueError('Frozen protocol changed')
    protocol.write_text(frozen)
    physical.DiskProfileOperator=RecurrenceDiskProfileOperator
    physical.OUT=OUT
    sources=['route2_physical_flow.py','route2_disk_recurrence.py','solver/general_residual.py',
             'solver/streamed_residual.py','solver/route2.py','route2_physical_recurrence_study.py']
    snapshots={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in sources}
    (OUT/'startup_source_hashes.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    previous=None;summary=[]
    for spec in PROTOCOL['cases']:
        resume=previous if spec['resume']=='previous native case' else spec['resume']
        if resume is None:raise ValueError('No previous native state')
        problem=physical.make_problem(spec['case'],PROTOCOL['degree'],PROTOCOL['centers'],PROTOCOL['lam'],spec['viscosity'])
        f=problem.features
        cost=dict(dimension=2,degree=f.degree,readout_coordinates=f.size,directions=f.direction_count)
        options,workspace=_memory_bounded_preconditioner(cost,problem.blocks,3,0,64,
            PROTOCOL['centers'],'disk',dict(preconditioner='block',preconditioner_basis='ideal',
            linearization_basis='ideal',preconditioner_block_size=PROTOCOL['requested_block_size']),
            PROTOCOL['preconditioner_workspace_allowance_bytes'])
        row=dict(case=spec['case'],viscosity=spec['viscosity'],resume=resume,
                 source_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest(),
                 workspace=workspace,selection=workspace['block_budget_selection'])
        stem=f'recurrence_{spec["case"]}_nu{spec["viscosity"]:g}_p64_n513'
        (OUT/(stem+'_preflight.json')).write_text(json.dumps(row,indent=2)+'\n')
        del problem,f
        completed=physical.run(case=spec['case'],degrees=(64,),centers=513,lam=.2,
            viscosity=spec['viscosity'],seconds=600.,krylov=150,max_iterations=12,tolerance=1e-14,
            label='recurrence',resume=resume,linearization_basis='ideal',audit_points=4096,
            inexact_newton=True,preconditioner_block_size=options['preconditioner_block_size'],coordinate_basis='disk')
        previous=str(OUT/(stem+'.npz'));result=completed[-1]
        row.update(status=result['status'],audit=result['ordinary_audit'],
                   solve_seconds=result['solve_seconds'],source='native previous physical solve')
        summary.append(row);(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':run()
