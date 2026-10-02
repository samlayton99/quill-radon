"""Read-only encoding audit of saved harder-2D fits; never refits a coefficient."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, time
from pathlib import Path
import torch
torch.set_num_threads(1)
import numpy as np
from route2_gap_diagnosis import ROOT, log, stamp, dump, emit
from route2_transfer_hard2d import OUT, RUNS
from route2_box_profiles import BoxProfileOperator
from route2_battletest_forward import CASES, make_declaration, equation

def run(run_id):
    folder=OUT/run_id;dest=folder/'encoding_audit.json'
    if dest.exists():raise FileExistsError(dest)
    audit_id=run_id+'_encoding_audit'
    log(f'\n### {audit_id} — START {stamp()}\n\nSaved-state audit only: same coefficients in ideal box coordinates versus saved ordinary MLP values. No fitting, changed geometry or access to references during coefficient discovery. Artifacts: `{dest.relative_to(ROOT)}`.')
    began=time.perf_counter()
    try:
        coeffs=np.load(folder/'coefficients.npz');degree=int(coeffs['degree']);c=coeffs['coefficients'][:,0]
        data=np.load(folder/'field_audit.npz');raw=np.load(folder/'ordinary_audit_before_reference.npz')
        case=CASES[RUNS[run_id]];decl=make_declaration(case)
        features=BoxProfileOperator(decl.domain.bounds,degree,257,.2,block_size=64)
        selector=np.arange(features.size)
        def ideal_values(points,orders):
            out={order:[] for order in orders}
            for begin in range(0,len(points),64):
                panel=features.selected_ideal_columns(points[begin:begin+64],selector,orders)
                for order in orders:out[order].append((panel[order]@c)[:,None])
            return {order:np.concatenate(values) for order,values in out.items()}
        ideal=ideal_values(data['points'],((0,0),))[(0,0)]
        callback,orders=equation(case);x=raw['raw_pde_points']
        jets={key:torch.tensor(value,dtype=torch.float64) for key,value in ideal_values(x,orders).items()}
        pde=callback(torch.tensor(x,dtype=torch.float64),jets,torch.empty(0)).detach().numpy()
        truth=data['reference'];pred=data['prediction'];den=np.linalg.norm(truth)
        row=dict(run_id=run_id,degree=degree,coordinates=features.size,
            ideal_relative_l2=float(np.linalg.norm(ideal-truth)/den),
            ordinary_relative_l2=float(np.linalg.norm(pred-truth)/den),
            ordinary_vs_ideal_relative_l2=float(np.linalg.norm(pred-ideal)/den),
            ideal_raw_pde_rms=float(np.sqrt(np.mean(pde**2))),
            ordinary_raw_pde_rms=float(np.sqrt(np.mean(raw['raw_pde']**2))),
            seconds=time.perf_counter()-began,
            coefficients_sha256=hashlib.sha256((folder/'coefficients.npz').read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            interpretation='Saved coefficient state only. An inaccurate ideal field does not by itself distinguish insufficient coordinate capacity from incomplete solving. No refit is performed.')
        dump(dest,row);np.savez_compressed(folder/'encoding_audit_values.npz',ideal=ideal,ideal_raw_pde=pde)
        log(f"\n**{audit_id} FINISH {stamp()} — completed.** Ideal field relL2 {row['ideal_relative_l2']:.6g}; ordinary field {row['ordinary_relative_l2']:.6g}; ordinary-to-ideal gap {row['ordinary_vs_ideal_relative_l2']:.6g}; ideal raw PDE {row['ideal_raw_pde_rms']:.6g} versus ordinary {row['ordinary_raw_pde_rms']:.6g}. Same saved coefficients; no solve. {row['seconds']:.1f}s.")
        emit('hard2d_encoding_audit',**row)
    except BaseException as exc:
        dump(folder/'encoding_audit_failure.json',dict(error=repr(exc),seconds=time.perf_counter()-began))
        log(f'\n**{audit_id} FINISH {stamp()} — failed.** {exc!r}. No fit altered.')
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True,choices=RUNS);args=p.parse_args();run(args.run_id)
