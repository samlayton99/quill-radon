"""Bounded-stage native comparison after fixed-space access diagnostics.

No dense reference weights/matrices enter this run. No production solver code
changes. Same policy for both diffusion contrasts, cold zero followed by native
continuation. Every invocation appends the requested scratchpad.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,resource,time,traceback
from pathlib import Path
import numpy as np
from route2_gap_diagnosis import ROOT,OUT,log,stamp,dump,audit,emit
from route2_battletest_forward import CASES,make_declaration
from solver.residual_scaling import normalize_equations
from solver.route2 import solve_route2_native

def run(args):
    run_dir=OUT/args.run_id;run_dir.mkdir(parents=True,exist_ok=False)
    degrees=tuple(int(v) for v in args.degrees.split(','))
    if not degrees or any(v<1 for v in degrees) or list(degrees)!=sorted(set(degrees)):
        raise ValueError('Degrees must be positive and strictly increasing')
    options=dict(degrees=degrees,centers=257,lam=.2,coordinates=args.coordinates,
        tolerance=1e-13,oversampling=6,check_points=257,export_check_points=257,
        maximum_working_array_mb=1024,max_seconds=args.seconds,max_iterations=args.outer,batch_size=64,
        solver_options=dict(lsmr_max_iterations=args.inner,refine_on_ideal_stall=True,
            inexact_newton=not args.exact_inner,damping=args.damping))
    config=dict(run_id=args.run_id,case=args.case,options=options,normalization='fixed zero-jet PDE scaling',
        source_policy='PDE/BC only, cold zero then native continuation. No dense diagnostic coefficients or interior labels.',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),ROOT/'experiments/expF19_radon_direct_pde/solver/route2.py',
        ROOT/'experiments/expF19_radon_direct_pde/solver/general_residual.py',
        ROOT/'experiments/expF19_radon_direct_pde/solver/streamed_residual.py']})
    dump(run_dir/'protocol.json',config);(run_dir/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    log(f"\n### {args.run_id} — START {stamp()}\n\nCold native {args.case}; coordinates={args.coordinates}; normalized PDE, stage cap {args.outer} outer / {args.inner} Krylov, {args.seconds}s soft total budget, degrees={degrees}, exact_inner={args.exact_inner}, damping={args.damping}, ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `{run_dir.relative_to(ROOT)}`.")
    def progress(c,p,record):
        with (run_dir/'iterations.jsonl').open('a') as f:f.write(json.dumps(dict(coordinates=len(c),**record),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v))+'\n')
        np.savez_compressed(run_dir/'latest.npz',coefficients=c,parameters=p)
        emit('native_iteration',run_id=args.run_id,coordinates=len(c),iteration=record.get('iteration'),
             objective=record.get('objective'),rms=record.get('maximum_scaled_block_rms'))
    options['solver_options']['iteration_callback']=progress
    started=time.perf_counter()
    try:
        case=CASES[args.case];decl=normalize_equations(make_declaration(case),['PDE'])
        sol=solve_route2_native(decl,**options)
        c=sol.coefficients[:,0];features=sol.problem.features
        np.savez_compressed(run_dir/'coefficients.npz',coefficients=c,degree=features.degree)
        result=dict(run_id=args.run_id,status=sol.status,solver_status=sol.solution.status,
            history=sol.history,metrics=sol.metrics,degree=features.degree,neurons=features.tanh_count,
            coordinates=features.size,solve_seconds=time.perf_counter()-started,
            inner_history=sol.solution.history)
        result.update(audit(features,c,case,'actual_native'));result.update(seconds=time.perf_counter()-started,
             process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(run_dir/'result.json',result)
        log(f"\n**{args.run_id} FINISH {stamp()} — {sol.status}.** Returned p{features.degree}; ordinary field relL2 {result['ordinary_relative_l2']:.6g}; raw PDE RMS {result['ordinary_pde_rms']:.6g}; readout L1 {result['readout_l1']:.4g}; {result['seconds']:.1f}s, {result['process_peak_rss_bytes']/1024**2:.1f}MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.")
        emit('native_result',**{k:result[k] for k in ['run_id','status','degree','ordinary_relative_l2','ordinary_pde_rms','readout_l1','seconds','process_peak_rss_bytes']})
    except BaseException as e:
        result=dict(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
        dump(run_dir/'failure.json',result);log(f"\n**{args.run_id} FINISH {stamp()} — {result['status']}.** {e!r}; checkpoint/failure preserved.");raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--case',choices=['diffusion_c10','diffusion_c1000'],required=True)
    p.add_argument('--seconds',type=float,default=180.);p.add_argument('--outer',type=int,default=6);p.add_argument('--inner',type=int,default=80)
    p.add_argument('--coordinates',choices=['affine_disk','box'],default='affine_disk')
    p.add_argument('--degrees',default='4,8,12,16,24,32,40')
    p.add_argument('--exact-inner',action='store_true')
    p.add_argument('--damping',type=float,default=1e-6)
    run(p.parse_args())
