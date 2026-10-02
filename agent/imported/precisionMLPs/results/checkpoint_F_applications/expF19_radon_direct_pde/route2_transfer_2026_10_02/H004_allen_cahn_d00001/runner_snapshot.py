"""Frozen transfer test of the improved native solve on harder 2D equations.

Only PDE/BC/IC data enter the solve. References are opened after coefficients
and the ordinary MLP have been saved. Every invocation is append-logged.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from route2_gap_diagnosis import ROOT, log, stamp, dump, emit
from route2_battletest_forward import (CASES, make_declaration, equation,
    evaluate_raw, analytical_reference)
from solver.residual_scaling import normalize_equations
from solver.route2 import solve_route2_native

OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02'
RUNS={'H001_helmholtz_n2':'helmholtz_n2', 'H002_helmholtz_n4':'helmholtz_n4',
      'H003_allen_cahn_d001':'allen_cahn_d001', 'H004_allen_cahn_d00001':'allen_cahn_d00001'}
OPTIONS=dict(coordinates='box',centers=257,lam=.2,degrees=(4,8,12,16,24,32,36,40),
    tolerance=1e-13,oversampling=6,check_points=257,export_check_points=257,
    maximum_working_array_mb=1024,maximum_neurons=2000000,max_seconds=480.,
    max_iterations=8,batch_size=64)

def reference_after_save(case, points, folder):
    if not (folder/'ordinary_mlp.pt').exists() or not (folder/'coefficients.npz').exists():
        raise RuntimeError('Save fit before opening reference')
    if case['family']!='allen_cahn':
        return analytical_reference(case,points),dict(kind='analytic evaluation-only',reference_used_in_fit=False)
    name=next(name for name,value in CASES.items() if value==case)
    root=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2'
    path=root/(name+'_evaluation_reference.npz');meta_path=path.with_suffix('.json')
    data=np.load(path);meta=json.loads(meta_path.read_text())
    n=len(points)
    if not np.array_equal(data['queries'][:n],points):
        raise RuntimeError('Archived reference query coordinates differ')
    if meta['coarse']['grid']!=512 or meta['fine']['grid']!=1024:
        raise RuntimeError('Unexpected archived reference grids')
    for value in (meta['coarse'],meta['fine']):
        if value['rtol']!=2e-11 or value['atol']!=2e-12:
            raise RuntimeError('Unexpected archived reference tolerances')
    truth=data['truth'][:n];coarse=data['coarse'][:n]
    gap=float(np.linalg.norm(truth-coarse)/np.linalg.norm(truth))
    info=dict(kind='archived independent Fourier/DOP853 evaluator, opened after saved fit',
        path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata=meta,queries_exactly_match=True,reference_resolution_difference_relative_l2=gap,
        reference_used_in_fit=False,
        caveat='Reference disagreement is empirical uncertainty, not a certified bound. No machine-floor claim from this reference.')
    return truth,info

def run(run_id):
    case_name=RUNS[run_id];case=CASES[case_name]
    folder=OUT/run_id;folder.mkdir(parents=True,exist_ok=False)
    damping=1e-30 if case['family']=='helmholtz' else 1e-6
    options=dict(OPTIONS,solver_options=dict(inexact_newton=False,refine_on_ideal_stall=True,
        lsmr_max_iterations=4000,damping=damping))
    files=[Path(__file__),ROOT/'experiments/expF19_radon_direct_pde/route2_battletest_forward.py']
    files.extend(ROOT/'experiments/expF19_radon_direct_pde'/p for p in
        ['route2_box_profiles.py','route2_directional_operator.py','solver/route2.py',
         'solver/general_residual.py','solver/streamed_residual.py','solver/residual_scaling.py'])
    config=dict(run_id=run_id,case_name=case_name,case=case,options=options,
        seed=0,normalization='fixed zero-jet PDE scaling, all prescribed conditions unchanged',
        initialization='cold zero, then only native coefficient continuation between resolutions',
        source_policy='PDE forcing, BC and IC only. No interior reference values, no external PDE solve or dense fit feeds the method.',
        audit=dict(field_count=2048,field_seed=884713,pde_count=257,pde_seed=663277,
                   constraints_count=257,constraints_seed=987513),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        compatibility_caveat='Allen-Cahn published IC x^2 cos(pi x) has derivatives +2 and -2 at the periodic endpoints, so periodic derivative continuity fails at t=0.' if case['family']=='allen_cahn' else None)
    dump(folder/'protocol.json',config)
    (folder/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    log(f"\n### {run_id} — START {stamp()}\n\nHarder 2D transfer `{case_name}`. Frozen before outcomes: box coordinates, N257, lambda .2, degree ladder {OPTIONS['degrees']}, exact inner tolerance, ideal/block-parity native correction, 8 outer / 4000 inner iterations per stage, 480s soft total budget, 1GiB named arrays, 2 million neurons. Generic PDE normalization. Damping {damping:g} from linear/nonlinear class policy. Cold zero; no interior truth or external solve enters fit. Artifacts: `{folder.relative_to(ROOT)}`.")
    started=time.perf_counter();torch.set_num_threads(1)
    def progress(coeff,params,record):
        with (folder/'iterations.jsonl').open('a') as f:
            f.write(json.dumps(dict(coordinates=len(coeff),**record),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v))+'\n')
        np.savez_compressed(folder/'latest.npz',coefficients=coeff,parameters=params)
        emit('hard2d_iteration',run_id=run_id,coordinates=len(coeff),iteration=record.get('iteration'),
             objective=record.get('objective'),rms=record.get('maximum_scaled_block_rms'))
    options['solver_options']['iteration_callback']=progress
    try:
        raw=make_declaration(case);decl=normalize_equations(raw,['PDE'])
        result=solve_route2_native(decl,**options)
        solve_seconds=time.perf_counter()-started
        feature=result.problem.features;model=result.export()
        assert [type(layer).__name__ for layer in model]==['Linear','Tanh','Linear']
        np.savez_compressed(folder/'coefficients.npz',coefficients=result.coefficients,degree=feature.degree)
        torch.save(dict(state_dict=model.state_dict(),width=model[0].out_features,dimension=2,
                   architecture='Linear/Tanh/Linear'),folder/'ordinary_mlp.pt')
        row=dict(run_id=run_id,case_name=case_name,case=case,status=result.status,
            solver_status=result.solution.status,history=result.history,metrics=result.metrics,
            inner_history=result.solution.history,solve_seconds=solve_seconds,
            degree=feature.degree,coordinates=feature.size,neurons=feature.tanh_count,
            reference_used_in_fit=False,ordinary_readout_l1=float(model[2].weight.detach().abs().sum()))
        dump(folder/'fit_saved_before_reference.json',row)
        pts=raw.domain.interior(2048,884713);check=raw.domain.interior(257,663277)
        prediction,_=evaluate_raw(model,pts)
        callback,ders=equation(case);_,residual=evaluate_raw(model,check,callback,ders)
        constraints=[]
        for block in raw.make_blocks(257,987513):
            _,value=evaluate_raw(model,block.points,block.function,block.derivatives)
            if block.aggregation is not None:value=block.aggregation@value
            constraints.append(dict(name=block.name,points=len(block.points),
                raw_rms=float(np.sqrt(np.mean(value**2))),raw_max=float(abs(value).max())))
        np.savez_compressed(folder/'ordinary_audit_before_reference.npz',points=pts,prediction=prediction,
                            raw_pde_points=check,raw_pde=residual)
        truth,refmeta=reference_after_save(case,pts,folder)
        dump(folder/'reference_audit.json',refmeta)
        np.savez_compressed(folder/'field_audit.npz',points=pts,prediction=prediction,reference=truth)
        row.update(ordinary_relative_l2=float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth)),
            ordinary_max_absolute=float(abs(prediction-truth).max()),
            ordinary_pde_rms=float(np.sqrt(np.mean(residual**2))),ordinary_pde_max=float(abs(residual).max()),
            ordinary_heldout_constraints=constraints,reference=refmeta,
            seconds=time.perf_counter()-started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            maximum_named_array_estimate_bytes=max((h.get('estimated_named_arrays_bytes',0) for h in result.history),default=0))
        dump(folder/'result.json',row)
        log(f"\n**{run_id} FINISH {stamp()} — {row['status']}.** Returned degree {feature.degree}, {feature.tanh_count} neurons / {feature.size} coordinates. Ordinary field relative L2 {row['ordinary_relative_l2']:.6g}; raw PDE RMS {row['ordinary_pde_rms']:.6g}; {row['solve_seconds']:.1f}s solve / {row['seconds']:.1f}s including audit; {row['process_peak_rss_bytes']/1024**2:.1f}MiB peak process RSS. Reference only after frozen fit; Allen-Cahn reference uncertainty and endpoint compatibility caveat retained. All stage exits preserved.")
        emit('hard2d_result',**{k:row[k] for k in ['run_id','status','degree','neurons','ordinary_relative_l2','ordinary_pde_rms','seconds','process_peak_rss_bytes']})
    except BaseException as exc:
        failure=dict(run_id=run_id,status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                     error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
        dump(folder/'failure.json',failure)
        log(f"\n**{run_id} FINISH {stamp()} — {failure['status']}.** {exc!r}. Failure and any completed state preserved.")
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',choices=RUNS);p.add_argument('--all',action='store_true');args=p.parse_args()
    if args.all:
        for run_id in RUNS:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-id',run_id],check=False)
    elif args.run_id:run(args.run_id)
    else:p.error('Choose --run-id or --all')
