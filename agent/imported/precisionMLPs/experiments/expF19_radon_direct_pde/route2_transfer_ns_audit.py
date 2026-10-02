"""Additional saved-state NS checks; never discovers or modifies coefficients."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,itertools,json,time,traceback
import torch
import numpy as np
from route2_transfer_ns import OUT,CASES,physical_terms,orders,manufactured_jets
from route2_gap_diagnosis import log,stamp,dump,emit
from route2_box_profiles import BoxProfileOperator
from solver.route2 import ordinary_jets


def run(run_id):
    folder=OUT/run_id;dest=folder/'additional_audit.json'
    if dest.exists():raise ValueError('Preserve existing additional audit')
    config=json.loads((folder/'protocol.json').read_text());case=config['case']
    started=time.perf_counter()
    log(f'\n### {run_id}_saved_audit — START {stamp()}\n\nRead-only saved ordinary-model corner/near-edge PDE checks and ideal-coordinate comparison. No refit, new initialization, or target data supplied to a solver. Separate audit file, original metrics unchanged.')
    try:
        archive=torch.load(folder/'ordinary_mlp.pt',map_location='cpu',weights_only=False)
        d=archive['dimension'];h=archive['width']
        model=torch.nn.Sequential(torch.nn.Linear(d,h,dtype=torch.float64),torch.nn.Tanh(),torch.nn.Linear(h,4,dtype=torch.float64))
        model.load_state_dict(archive['state_dict']);model.eval()
        zero,first,second,dt=orders(d);ders=(zero,)+first+second+((dt,) if dt else ())
        corners=np.array(list(itertools.product((-1.,1.),repeat=3)))
        near=list(corners*.999)
        for axes in itertools.combinations(range(3),2):
            free=next(k for k in range(3) if k not in axes)
            for signs in itertools.product((-.999,.999),repeat=2):
                for value in np.linspace(-.9,.9,9):
                    pt=np.zeros(3);pt[list(axes)]=signs;pt[free]=value;near.append(pt)
        groups={'corners':corners,'near_edges':np.array(near)}
        if d==4:
            groups={name:np.concatenate([np.column_stack((pts,np.full(len(pts),t))) for t in (0.,.125,.25)]) for name,pts in groups.items()}
        report={'run_id':run_id,'checks':{},'scope':'Postfit saved-state audits only; no continuum certificate or known cavity field.'}
        for name,pts in groups.items():
            jets={key:value.numpy() for key,value in ordinary_jets(model,pts,ders).items()}
            m,div=physical_terms(jets,case['viscosity'],d)
            if case['family']!='cavity':
                expected=manufactured_jets(pts,**{k:case[k] for k in ('family','frequency','transient')})
                force,_=physical_terms(expected,case['viscosity'],d);m-=force
            report['checks'][name]=dict(points=len(pts),momentum_rms=float(np.sqrt(np.mean(m*m))),
                momentum_max=float(np.max(abs(m))),divergence_rms=float(np.sqrt(np.mean(div*div))),
                divergence_max=float(np.max(abs(div))))
        cfile=np.load(folder/'coefficients.npz');p=int(cfile['degree']);coeff=cfile['coefficients']
        features=BoxProfileOperator(cfile['bounds'],p,int(cfile['centers']),float(cfile['lam']),block_size=64)
        saved=np.load(folder/'audit_arrays.npz');points=saved['value_points']
        ideal=np.concatenate([features.selected_ideal_columns(points[i:i+64],np.arange(features.size),(zero,))[zero]@coeff for i in range(0,len(points),64)])
        ordinary=saved['values'];report['ordinary_vs_ideal_max']=float(np.max(abs(ordinary-ideal)))
        if 'target' in saved:
            target=saved['target'];report.update(ideal_velocity_relative_l2=float(np.linalg.norm(ideal[:,:3]-target[:,:3])/np.linalg.norm(target[:,:3])),
                ideal_pressure_relative_l2=float(np.linalg.norm(ideal[:,3]-target[:,3])/np.linalg.norm(target[:,3])))
        report['seconds']=time.perf_counter()-started;dump(dest,report)
        log(f"\n**{run_id}_saved_audit FINISH {stamp()}.** Near-edge momentum RMS/max={report['checks']['near_edges']['momentum_rms']:.6g}/{report['checks']['near_edges']['momentum_max']:.6g}; ordinary-vs-ideal field max={report['ordinary_vs_ideal_max']:.6g}; {report['seconds']:.1f}s. Original fit unchanged.")
        emit('additional_audit',**report)
    except BaseException as error:
        dump(folder/'additional_audit_failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        log(f'\n**{run_id}_saved_audit FINISH {stamp()} — failed.** {error!r}; original fit unchanged.');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id',choices=CASES);run(parser.parse_args().run_id)
