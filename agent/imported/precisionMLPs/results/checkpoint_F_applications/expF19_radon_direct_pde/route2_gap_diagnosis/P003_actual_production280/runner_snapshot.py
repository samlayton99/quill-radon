"""Small dense diagnostic of bounded-block QR versus regularized Gram factors.

PDE forcing/boundary only during solve. Every invocation is logged. The dense
matrix and its global singular values are diagnostic controls, not a proposed
scalable solver. No production modules are modified.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, resource, time, traceback
from pathlib import Path
import numpy as np
from scipy.linalg import qr, svdvals, solve_triangular
from scipy.sparse.linalg import LinearOperator, lsmr
from route2_gap_diagnosis import ROOT, OUT, log, stamp, dump, assemble, audit, emit
from route2_affine_disk_profiles import AffineDiskProfileOperator
from route2_battletest_forward import CASES, make_declaration


def factors(matrix, groups, kind):
    result=[]; stats=[]
    for ids in groups:
        panel=matrix[:,ids]
        singular=svdvals(panel)
        stats.append(dict(size=len(ids),singular_min=singular[-1],singular_max=singular[0],
                          condition=singular[0]/singular[-1]))
        if kind=='diagonal':
            r=np.eye(len(ids))
        elif kind=='gram_ridge':
            gram=panel.T@panel; gram=(gram+gram.T)/2
            ridge=1e-10*max(1.,float(np.max(np.diag(gram))))
            r=np.linalg.cholesky(gram+ridge*np.eye(len(ids))).T
        elif kind=='qr':
            if singular[-1] <= 1e-13*singular[0]:
                raise ValueError('Block numerically rank deficient under declared QR guard')
            r=qr(panel,mode='economic',check_finite=False)[1]
        else:raise ValueError(kind)
        result.append((ids,r))
    return result,stats


def run(args):
    directory=OUT/args.run_id;directory.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    config=dict(vars(args),started=stamp(),threads=1,dtype='float64',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_used_in_fit=False,scope='Dense diagnostic only, not production scaling',
        basis=args.basis,weighting='fixed zero-jet PDE normalization',
        grouping='production default disk contiguous coefficient-order blocks',
        lsmr_tolerance=1e-14,gram_ridge=1e-10,qr_relative_rank_guard=1e-13)
    dump(directory/'protocol.json',config)
    (directory/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    log(f"\n### {args.run_id} — START {stamp()}\n\nFixed p32 diffusion_c1000, N257, lambda0.2, {args.basis} columns, normalized PDE and unit column scaling. Same matrix and contiguous {args.block_size}-coordinate blocks: diagonal, regularized Gram, and guarded QR. LSMR cap {args.iterations}; no target values in solve. Dense diagnostic only. Artifacts: `{directory.relative_to(ROOT)}`.")
    try:
        case=CASES['diffusion_c1000'];decl=make_declaration(case,seed=0)
        features=AffineDiskProfileOperator(decl.domain.bounds,32,257,.2,block_size=64)
        blocks=decl.make_blocks(max(256,6*features.size),113+32*1009)
        A,b,row_scale,meta=assemble(features,blocks,args.basis)
        A/=row_scale[:,None];b/=row_scale
        column_scale=np.linalg.norm(A,axis=0);A/=column_scale
        groups=[np.arange(start,min(start+args.block_size,A.shape[1])) for start in range(0,A.shape[1],args.block_size)]
        base_singular=svdvals(A)
        results=[]
        for kind in ['diagonal','gram_ridge','qr']:
            begin=time.perf_counter();fs,stats=factors(A,groups,kind)
            def apply(z,transpose=False):
                out=np.empty_like(z)
                for ids,r in fs:
                    out[ids]=solve_triangular(r.T if transpose else r,z[ids],lower=transpose,check_finite=False)
                return out
            # Singular values diagnose cross-block geometry; they do not enter fitting.
            B=np.empty_like(A)
            for ids,r in fs:B[:,ids]=solve_triangular(r.T,A[:,ids].T,lower=True,check_finite=False).T
            singular=svdvals(B);del B
            setup_seconds=time.perf_counter()-begin
            counts=dict(matvec=0,rmatvec=0)
            def mv(z):
                counts['matvec']+=1
                return A@apply(z)
            def rmv(z):
                counts['rmatvec']+=1
                return apply(A.T@z,True)
            op=LinearOperator(A.shape,matvec=mv,rmatvec=rmv,dtype=float)
            begin=time.perf_counter()
            solution=lsmr(op,b,atol=1e-14,btol=1e-14,conlim=1e16,maxiter=args.iterations)
            seconds=time.perf_counter()-begin
            c=apply(solution[0])/column_scale
            residual=A@(c*column_scale)-b
            item=dict(kind=kind,lsmr_istop=int(solution[1]),lsmr_iterations=int(solution[2]),
                estimated_residual_norm=solution[3],algebraic_residual_norm=np.linalg.norm(residual),
                normal_residual_norm=np.linalg.norm(A.T@residual),setup_seconds=setup_seconds,
                solve_seconds=seconds,work_counts=counts,block_spectra=stats,
                preconditioned_condition=singular[0]/singular[-1],
                preconditioned_singular_min=singular[-1],preconditioned_singular_max=singular[0],
                factor_bytes=sum(r.nbytes for _,r in fs))
            item.update(audit(features,c,case,args.basis))
            results.append(item)
            dump(directory/f'{kind}.json',item)
            np.savez_compressed(directory/f'{kind}_coefficients.npz',coefficients=c,singular_values=singular)
            log(f"\n{args.run_id} / {kind}: {solution[2]} LSMR iterations, algebraic residual {item['algebraic_residual_norm']:.6g}, ordinary field relL2 {item['ordinary_relative_l2']:.6g}, raw PDE RMS {item['ordinary_pde_rms']:.6g}, transformed condition {item['preconditioned_condition']:.4g}. Setup {setup_seconds:.2f}s, solve {seconds:.2f}s. Finite iteration diagnostic; low objective alone is not a field certificate.")
            emit('variant',run_id=args.run_id,**{k:item[k] for k in ['kind','lsmr_iterations','algebraic_residual_norm','ordinary_relative_l2','ordinary_pde_rms','preconditioned_condition','solve_seconds']})
        result=dict(config=config,status='complete',variants=results,
            unpreconditioned_column_scaled_condition=base_singular[0]/base_singular[-1],
            shape=A.shape,matrix_bytes=A.nbytes,
            seconds=time.perf_counter()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(directory/'result.json',result)
        log(f"\n**{args.run_id} FINISH {stamp()} — complete.** Three preconditioner variants retained; total {result['seconds']:.2f}s, process peak {result['process_peak_rss_bytes']/1024**2:.1f}MiB. Dense matrix diagnostics deliberately exceed scalable-storage policy; factor storage alone is recorded separately.")
    except BaseException as e:
        result=dict(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
        dump(directory/'failure.json',result)
        log(f"\n**{args.run_id} FINISH {stamp()} — {result['status']}.** {e!r}; failure retained.")
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--basis',choices=['actual','ideal'],default='actual')
    p.add_argument('--block-size',type=int,default=64);p.add_argument('--iterations',type=int,default=800)
    run(p.parse_args())
