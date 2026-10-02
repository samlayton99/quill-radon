"""Fixed-space PDE access/encoding diagnosis; dense matrices are controls only.

Fits prescribed PDE and BC callbacks, never interior target values. Freeze
the same features/locations as the existing stage. Log EVERY invocation before
work and every result/failure. Do not use these dense controls as the proposed
memory-scalable solver. Native production modules are not changed.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse, fcntl, hashlib, json, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import lstsq
from route2_affine_disk_profiles import AffineDiskProfileOperator
from route2_battletest_forward import CASES,make_declaration,manufactured_jet,equation
from solver.route2 import export_mlp,ordinary_jets
from solver.general_residual import ResidualProblem,_make_engine,linearize_residual

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis'
LOG=ROOT/'docs/radon_results_library/current_findings/scratchpad.md'

def safe(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def dump(path,x):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=safe)+'\n');tmp.replace(path)

def log(text):
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(text+'\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)

def stamp():return time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())

def emit(event,**kw):print(json.dumps(dict(event=event,**kw),default=safe),flush=True)

def assemble(features,blocks,basis):
    """Scalar linear equality equations only, checked via callback derivatives.

    Reproduce the production block norm: weight/row_count, scale, PDE row
    normalization. Full dense diagnostic columns are retained intentionally;
    feature temporaries use 64-point tiles. No normal equations are formed.
    """
    selector=features.selected_columns if basis=='actual' else features.selected_ideal_columns
    As=[];bs=[];scales=[];meta=[];cursor=0
    for block in blocks:
        if block.aggregation is not None or block.relation!='eq':raise ValueError('Pointwise scalar linear control required')
        n=len(block.points);A=np.zeros((n,features.size));rhs=np.empty(n);scale=np.empty(n)
        factor=float(np.sqrt(block.weight/n)/block.scale)
        for start in range(0,n,64):
            stop=min(n,start+64);pts=block.points[start:stop];x=torch.tensor(pts,dtype=torch.float64)
            jets={d:torch.zeros((len(x),1),dtype=x.dtype,requires_grad=True) for d in block.derivatives}
            raw=block.function(x,jets,torch.empty((len(x),0),dtype=x.dtype))
            assert tuple(raw.shape)==(len(x),1)
            grad=torch.autograd.grad(raw.sum(),list(jets.values()),allow_unused=True)
            cols=selector(pts,np.arange(features.size),block.derivatives)
            sq=np.zeros(len(x))
            for d,g in zip(block.derivatives,grad):
                if g is None:continue
                values=g.detach().numpy().ravel();A[start:stop]+=cols[d]*values[:,None];sq+=values**2
            rhs[start:stop]=-raw.detach().numpy().ravel()
            scale[start:stop]=np.maximum(np.sqrt(sq),np.finfo(float).tiny) if block.name=='PDE' else 1.
        As.append(A*factor);bs.append(rhs*factor);scales.append(scale)
        meta.append(dict(name=block.name,start=cursor,stop=cursor+n,factor=factor));cursor+=n
    return np.concatenate(As),np.concatenate(bs),np.concatenate(scales),meta

def audit(features,c,case,fit_basis):
    """Only after solve: independent values, raw PDE, boundary, ordinary AD."""
    rng=np.random.default_rng(884713);points=rng.random((2048,2));dpoints=np.random.default_rng(663277).random((257,2))
    callback,ders=equation(case)
    truth=manufactured_jet(torch.tensor(points,dtype=torch.float64))[0].numpy().ravel()
    ideal=features.selected_ideal_columns(points,np.arange(features.size),((0,0),))[(0,0)]@c
    actual=features.forward(c,points)
    model=export_mlp(features,c[:,None]);pred=ordinary_jets(model,points,((0,0),))[(0,0)].numpy().ravel()
    jets=ordinary_jets(model,dpoints,ders)
    residual=callback(torch.tensor(dpoints),jets,torch.empty(0)).detach().numpy().ravel()
    ij=features.selected_ideal_columns(dpoints,np.arange(features.size),ders)
    ideal_jets={d:torch.tensor((v@c)[:,None]) for d,v in ij.items()}
    ideal_res=callback(torch.tensor(dpoints),ideal_jets,torch.empty(0)).numpy().ravel()
    bnd=make_declaration(case).domain.boundary(512,98231).points
    bc_truth=manufactured_jet(torch.tensor(bnd,dtype=torch.float64))[0].numpy().ravel()
    bc=ordinary_jets(model,bnd,((0,0),))[(0,0)].numpy().ravel()-bc_truth
    den=np.linalg.norm(truth)
    return dict(ordinary_relative_l2=np.linalg.norm(pred-truth)/den,ideal_relative_l2=np.linalg.norm(ideal-truth)/den,
        stable_neural_relative_l2=np.linalg.norm(actual-truth)/den,
        ordinary_vs_ideal_relative_l2=np.linalg.norm(pred-ideal)/den,
        ordinary_pde_rms=np.sqrt(np.mean(residual**2)),ordinary_pde_max=np.max(np.abs(residual)),
        ideal_pde_rms=np.sqrt(np.mean(ideal_res**2)),boundary_rms=np.sqrt(np.mean(bc**2)),
        readout_l1=float(model[2].weight.abs().sum()),coordinate_norm=np.linalg.norm(c),
        fit_basis=fit_basis,field_points=2048,pde_points=257,reference_used_in_fit=False)

def run(args):
    OUT.mkdir(parents=True,exist_ok=True);run_dir=OUT/args.run_id
    run_dir.mkdir(exist_ok=False)
    config=vars(args).copy();config.update(started=stamp(),architecture='Linear/Tanh/Linear',
        purpose='Dense small diagnostic only, not scalable production solve',
        truth_policy='Prescribed PDE forcing and BC only; interior truth used after solving',
        dtype='float64',threads=1,source_hashes={})
    for p in [Path(__file__),ROOT/'experiments/expF19_radon_direct_pde/route2_battletest_forward.py',
              ROOT/'experiments/expF19_radon_direct_pde/route2_disk_profiles.py',
              ROOT/'experiments/expF19_radon_direct_pde/route2_directional_operator.py']:
        config['source_hashes'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(run_dir/'protocol.json',config)
    (run_dir/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    log(f"\n### {args.run_id} — START {config['started']}\n\nFixed-degree {args.case}, p={args.degree}, N={args.centers}, lambda={args.lam}, {args.basis} columns, {args.weighting} weighting, SVD cutoff={args.rcond}. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `{run_dir.relative_to(ROOT)}`.")
    started=time.perf_counter()
    try:
        case=CASES[args.case];decl=make_declaration(case,seed=0)
        features=AffineDiskProfileOperator(decl.domain.bounds,args.degree,args.centers,args.lam,block_size=64)
        count=max(256,6*features.size);blocks=decl.make_blocks(count,113+args.degree*1009)
        emit('assembly_started',run_id=args.run_id,columns=features.size,rows=sum(len(b.points) for b in blocks))
        t=time.perf_counter();A,b,scale,meta=assemble(features,blocks,args.basis);assembly_seconds=time.perf_counter()-t
        if args.weighting=='normalized':A/=scale[:,None];b/=scale
        # Unit column scaling is invertible and alters cutoff behavior explicitly.
        column_scale=np.linalg.norm(A,axis=0) if args.column_scale else np.ones(features.size)
        column_scale=np.maximum(column_scale,np.finfo(float).tiny);A/=column_scale
        emit('assembly_finished',run_id=args.run_id,seconds=assembly_seconds,bytes=A.nbytes)
        t=time.perf_counter();z,_,rank,s=lstsq(A,b,cond=args.rcond,lapack_driver='gelsd');c=z/column_scale
        solve_seconds=time.perf_counter()-t
        algebra=A@z-b;normal=A.T@algebra
        emit('fit_finished',run_id=args.run_id,rank=rank,residual_norm=np.linalg.norm(algebra))
        problem=ResidualProblem(features,blocks,1,execution='streamed',batch_size=64)
        engine=_make_engine(problem,False)
        row_scale=scale if args.weighting=='normalized' else np.ones(len(b))
        native_zero=engine.evaluate(np.zeros(features.size))[0]/row_scale
        assembly_rhs_discrepancy=float(np.linalg.norm(native_zero+b)/max(np.linalg.norm(b),1e-300))
        # Independent native-operator check on deterministic non-solution vectors.
        linearization=linearize_residual(problem,np.zeros((features.size,1)))
        operator=linearization.operator if args.basis=='actual' else linearization.operator.ideal_operator
        probe=np.random.default_rng(46729).normal(size=features.size)/np.sqrt(features.size)
        predicted=A@(probe*column_scale);native_product=operator.matvec(probe)/row_scale
        assembly_operator_discrepancy=float(np.linalg.norm(predicted-native_product)/max(np.linalg.norm(native_product),1e-300))
        if assembly_rhs_discrepancy>1e-13 or assembly_operator_discrepancy>1e-10:
            raise RuntimeError('Diagnostic assembly differs from native objective/operator')
        refinements=[]
        for iteration in range(args.refinements):
            residual=engine.evaluate(c)[0]/row_scale
            step_scaled,_,step_rank,_=lstsq(A,-residual,cond=args.rcond,lapack_driver='gelsd')
            trial=c+step_scaled/column_scale
            new_residual=engine.evaluate(trial)[0]/row_scale
            accepted=np.linalg.norm(new_residual)<np.linalg.norm(residual)
            refinements.append(dict(iteration=iteration,before_norm=np.linalg.norm(residual),
                after_norm=np.linalg.norm(new_residual),step_norm=np.linalg.norm(step_scaled/column_scale),
                rank=int(step_rank),accepted=bool(accepted)))
            emit('refinement',run_id=args.run_id,**refinements[-1])
            if accepted:c=trial
            else:break
        z=c*column_scale;algebra=A@z-b;normal=A.T@algebra
        result=dict(run_id=args.run_id,status='complete',case=args.case,degree=args.degree,neurons=features.tanh_count,
            coordinates=features.size,rows=len(b),basis=args.basis,weighting=args.weighting,
            rank=int(rank),singular_max=s[0],singular_min=s[-1],condition=s[0]/s[-1],
            algebraic_residual_norm=np.linalg.norm(algebra),normal_residual_norm=np.linalg.norm(normal),
            dense_matrix_bytes=A.nbytes,assembly_seconds=assembly_seconds,solve_seconds=solve_seconds,
            column_scaled=args.column_scale,rcond=args.rcond,blocks=meta,
            assembly_rhs_discrepancy=assembly_rhs_discrepancy,
            assembly_operator_discrepancy=assembly_operator_discrepancy,
            refinements=refinements,
            final_native_objective_residual_norm=np.linalg.norm(engine.evaluate(c)[0]/row_scale))
        np.savez_compressed(run_dir/'coefficients.npz',coefficients=c,singular_values=s,column_scale=column_scale)
        result.update(audit(features,c,case,args.basis));result.update(seconds=time.perf_counter()-started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(run_dir/'result.json',result)
        log(f"\n**FINISH {stamp()} — complete.** Ordinary field relL2 {result['ordinary_relative_l2']:.6g}; ideal {result['ideal_relative_l2']:.6g}; raw ordinary PDE RMS {result['ordinary_pde_rms']:.6g}; boundary RMS {result['boundary_rms']:.6g}. Rank {rank}/{features.size}; condition {result['condition']:.3g}; readout L1 {result['readout_l1']:.3g}. {result['seconds']:.1f}s; process peak {result['process_peak_rss_bytes']/1024**2:.1f} MiB. Interpretation pending paired controls; no automatic floor claim.")
        emit('result',**result)
    except BaseException as e:
        info=dict(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
        dump(run_dir/'failure.json',info);log(f"\n**FINISH {stamp()} — {info['status']}.** {e!r}. Saved failure record; no numerical success claimed.")
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--case',choices=['diffusion_c10','diffusion_c1000'],default='diffusion_c1000')
    p.add_argument('--degree',type=int,default=24);p.add_argument('--centers',type=int,default=257);p.add_argument('--lam',type=float,default=.2)
    p.add_argument('--basis',choices=['actual','ideal'],default='actual');p.add_argument('--weighting',choices=['raw','normalized'],default='raw')
    p.add_argument('--rcond',type=float,default=1e-14);p.add_argument('--column-scale',action='store_true')
    p.add_argument('--refinements',type=int,default=0)
    run(p.parse_args())
