"""PDE-only coefficient-driven sparse refinement of the generic NS declaration.

The basis support comes exclusively from a previous PDE solve, never from the
analytic reference. All coordinate directions are expanded equally from that
support. Reference values are consulted only AFTER each solve for the audit.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quill-ns4d-sparse-mpl')
import argparse,json,time,resource,sys
from pathlib import Path
import numpy as np
from solver.general_sparse import (selected_feature_bank,coefficient_support,
                                  extend_support,embed_coefficients)
from solver.general_residual import ResidualProblem,ResidualSolution,solve_residual
from solver.general_adaptive import check_residuals
from general_ns4d_study import DOMAIN,OUT,validate_solution,DERIVATIVES
from general_ns4d_precision import make_precision_blocks,temporal_errors
from general_residual_study import clean


class _CheckpointStop(Exception):
    pass


def run(args):
    data=np.load(args.warmstart)
    old=selected_feature_bank(DOMAIN.bounds,data['multiindices'],backend='polynomial')
    coefficients=data['coefficients'];source=args.warmstart
    source_scope='PDE-solved coefficients only; no reference initialization or support selection'
    rows=[]
    for stage,budget in enumerate(args.discard_budgets):
        seed_support,selection=coefficient_support(old,coefficients,budget,
                         derivatives=DERIVATIVES if args.derivative_aware else None)
        indices=extend_support(seed_support,layers=args.growth_layers,mode=args.growth_mode,
                               max_degree=args.maximum_degree)
        feature_options=dict(max_derivative=2)
        if args.backend=='quill':
            feature_options.update(centers=args.centers,lam=args.lam,
                                   evaluation='anchored',encoding_tolerance=1e-13)
        features=selected_feature_bank(DOMAIN.bounds,indices,backend=args.backend,**feature_options)
        count=max(256,int(args.oversampling*features.size))
        blocks=make_precision_blocks(count,113+1009*features.degree,features.degree,args.trace_oversampling)
        estimate=sum(len(b.points)*len(b.derivatives)*features.size*8 for b in blocks)
        if estimate>args.maximum_cache_gb*1e9:
            raise MemoryError(f'Basis jets require{estimate/1e9:.2f}GB, overdeclaredcap')
        problem=ResidualProblem(features,blocks,fields=4)
        initial=embed_coefficients(old,coefficients,features)
        stem=f'ns4d_sparse_{args.label}_{args.backend}_stage{stage}_p{features.degree}'
        checkpoint_state={};checkpoint_history=[]
        def checkpoint(c,p,record):
            checkpoint_state.update(coefficients=c.copy(),parameters=p.copy())
            checkpoint_history.append(dict(record))
            temp=OUT/(stem+'_live.tmp.npz')
            np.savez_compressed(temp,coefficients=c,parameters=p,multiindices=features.multiindices,bounds=features.bounds)
            temp.replace(OUT/(stem+'_live.npz'))
            (OUT/(stem+'_live.json')).write_text(json.dumps(clean(record),indent=2)+'\n')
            print(json.dumps(dict(event='iterate',stage=stage,**record)),flush=True)
            if args.stop_file and Path(args.stop_file).exists():
                raise _CheckpointStop('Requested stop at an accepted saved iterate')
        options=dict(tolerance=args.tolerance,max_iterations=args.iterations,
                     max_seconds=args.seconds,high_accuracy=True,
                     preconditioner='block',preconditioner_block_size=args.block_size,
                     lsmr_max_iterations=args.krylov_iterations,
                     linear_refinement_steps=args.linear_refinements,
                     max_residual_evaluations=3000,iteration_callback=checkpoint)
        print(json.dumps(dict(event='launch',degree=features.degree,features=features.size,
            full_features=features.metrics['full_space_coefficient_count'],cache_gb=estimate/1e9,
            source=source,selection=selection,block_size=args.block_size)),flush=True)
        start=time.perf_counter()
        try:
            solution=solve_residual(problem,initial_coefficients=initial,**options)
        except (KeyboardInterrupt,_CheckpointStop) as interruption:
            if not checkpoint_state:raise
            status=('stopped_at_checkpoint' if isinstance(interruption,_CheckpointStop)
                    else 'interrupted_at_checkpoint')
            solution=ResidualSolution(problem,checkpoint_state['coefficients'],
                checkpoint_state['parameters'],dict(status=status,converged=False),checkpoint_history)
            solution.metrics.update(blocks=check_residuals(solution,blocks),
                iterations=checkpoint_history[-1]['iteration'],lsmr_iterations=None,
                interruption_scope='Latest saved accepted iterate; incomplete linear work discarded')
            for check in solution.metrics['blocks']:
                check['reuses_training_locations']=True
        seconds=time.perf_counter()-start
        path=OUT/(stem+'.npz')
        np.savez_compressed(path,coefficients=solution.coefficients,parameters=solution.parameters,
                            multiindices=features.multiindices,bounds=features.bounds)
        validation=validate_solution(solution)
        row=dict(stem=stem,backend=args.backend,degree=features.degree,
                 unknown_coefficients=features.size*4,features=features.metrics,
                 warmstart=source,warmstart_scope=source_scope,selection=selection,
                 discard_budget=budget,growth_layers=args.growth_layers,
                 growth_mode=args.growth_mode,maximum_degree=args.maximum_degree,
                 trace_oversampling=args.trace_oversampling,estimated_basis_cache_bytes=estimate,
                 process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                 options={k:v for k,v in options.items() if k!='iteration_callback'},
                 solve_seconds=seconds,status=solution.status,solver=solution.metrics,
                 history=solution.history,validation=validation,error_by_time=temporal_errors(solution))
        (OUT/(stem+'.json')).write_text(json.dumps(clean(row),indent=2)+'\n')
        print(json.dumps(dict(event='finished',stem=stem,status=solution.status,seconds=seconds,
                              validation=validation)),flush=True)
        rows.append(row);old=features;coefficients=solution.coefficients;source=str(path)
        (OUT/f'ns4d_sparse_{args.label}_summary.json').write_text(json.dumps(clean([
            {k:r[k] for k in ('stem','degree','unknown_coefficients','solve_seconds','status','validation')}
            for r in rows]),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmstart',required=True)
    parser.add_argument('--discard-budgets',type=float,nargs='+',default=[1e-9])
    parser.add_argument('--growth-layers',type=int,default=4)
    parser.add_argument('--growth-mode',choices=['total','axis'],default='total')
    parser.add_argument('--maximum-degree',type=int)
    parser.add_argument('--backend',choices=['polynomial','quill'],default='polynomial')
    parser.add_argument('--label',default='discovery')
    parser.add_argument('--centers',type=int,default=513);parser.add_argument('--lam',type=float,default=.2)
    parser.add_argument('--oversampling',type=float,default=6)
    parser.add_argument('--trace-oversampling',type=float,default=2)
    parser.add_argument('--seconds',type=float,default=1800)
    parser.add_argument('--maximum-cache-gb',type=float,default=2.5)
    parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--krylov-iterations',type=int,default=2500)
    parser.add_argument('--linear-refinements',type=int,default=1)
    parser.add_argument('--block-size',type=int,default=512)
    parser.add_argument('--tolerance',type=float,default=2e-15)
    parser.add_argument('--derivative-aware',action='store_true')
    parser.add_argument('--stop-file',help='Stop at the next accepted saved iterate when this file exists')
    run(parser.parse_args())
