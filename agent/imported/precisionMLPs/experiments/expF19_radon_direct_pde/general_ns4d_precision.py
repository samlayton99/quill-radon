"""Resolution/optimization precision study for the declared 4D NS problem.

Only the previously solved coefficients provide warm starts. No analytic
interior or final field is fitted or used to initialize coefficients. Backend
comparisons and precision controls use the same general residual engine.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quill-ns4d-precision-mpl')
from pathlib import Path
import argparse,gc,hashlib,inspect,json,math,resource,sys,time
import numpy as np
import torch
from solver.general_features import ConstructedFeatures
from solver.general_sparse import selected_feature_bank
from solver.general_residual import ResidualBlock,ResidualProblem,ResidualSolution,solve_residual
from solver.general_adaptive import check_residuals
from general_ns4d_study import DOMAIN,OUT,ZERO,make_blocks,reference,validate_reference,validate_solution
from general_residual_study import clean

ENGINE_PATH=Path(__file__).parent/'solver/general_residual.py'
ENGINE_SHA256=hashlib.sha256(ENGINE_PATH.read_bytes()).hexdigest()


class _CheckpointStop(Exception):
    pass


def load_start(path):
    data=np.load(path)
    return data['coefficients'],data['multiindices']


def map_start(coefficients,indices,features):
    result=np.zeros((features.size,4));lookup={tuple(k):i for i,k in enumerate(features.multiindices)}
    for c,index in zip(coefficients,indices):
        if tuple(index) in lookup:result[lookup[tuple(index)]]=c
    return result


def temporal_errors(solution):
    space=DOMAIN.interior(769,136897);rows=[]
    for t in np.linspace(0,1,21):
        points=space.copy();points[:,3]=t
        target=reference(torch.as_tensor(points)).numpy();predicted=solution.evaluate(points)
        error=predicted-target
        rows.append(dict(time=float(t),
            velocity_relative_l2=float(np.linalg.norm(error[:,:3])/np.linalg.norm(target[:,:3])),
            pressure_relative_l2=float(np.linalg.norm(error[:,3])/np.linalg.norm(target[:,3])),
            velocity_absolute_rms=float(np.sqrt(np.mean(error[:,:3]**2))),
            pressure_absolute_rms=float(np.sqrt(np.mean(error[:,3]**2))),
            velocity_linf=float(np.max(abs(error[:,:3]))),pressure_linf=float(np.max(abs(error[:,3])))))
    return rows


def validate_saved_checkpoint(path,backend='polynomial',centers=513,lam=.2):
    """Validate a saved full or sparse PDE iterate using its actual support.

    Reference fields are evaluated only after the saved coefficients and
    geometry have been reconstructed. No solve or support selection occurs.
    """
    began=time.perf_counter();data=np.load(path)
    if not np.array_equal(data['bounds'],DOMAIN.bounds):
        raise ValueError('Checkpoint bounds do not match this declared problem')
    options=dict(max_derivative=2)
    if backend=='quill':
        options.update(centers=centers,lam=lam,evaluation='anchored',encoding_tolerance=1e-12)
    features=selected_feature_bank(data['bounds'],data['multiindices'],backend=backend,**options)
    coefficients=map_start(data['coefficients'],data['multiindices'],features)
    parameters=data['parameters'] if 'parameters' in data else np.empty(0)
    problem=ResidualProblem(features,make_blocks(32,136199),fields=4)
    solution=ResidualSolution(problem,coefficients,parameters,{'status':'checkpoint_validation_only'})
    result=dict(source=str(path),backend=backend,features=features.metrics,
        scope='Validation of saved PDE coefficients on their saved support; no optimization or reference-derived coefficients',
        validation=validate_solution(solution),error_by_time=temporal_errors(solution))
    result['validation_seconds']=time.perf_counter()-began
    return result


def make_precision_blocks(count,seed,degree,trace_oversampling=0.):
    """Keep the declared equations, with dimension-aware trace sampling.

    A degree-p total-degree basis restricted to a 3D face has C(p+3,3)
    coefficients. This is a sample-count policy, not an NS-specific solve.
    Zero retains the earlier study's sqrt(interior-count) sampling baseline.
    """
    blocks=make_blocks(count,seed)
    if trace_oversampling<=0:return blocks
    trace_size=math.comb(degree+3,3)
    per_face=max(math.ceil(len(blocks[1].points)/6),
                 math.ceil(trace_oversampling*trace_size))
    boundary=DOMAIN.boundary(8*per_face,seed+13)
    spatial=boundary.points[abs(boundary.normals[:,3])<.5]
    target=reference(torch.as_tensor(spatial))[:,:3].detach()
    blocks[1]=ResidualBlock('spatial_velocity',spatial,
        lambda x,j,p:j[ZERO][:,:3]-target,(ZERO,))
    initial=DOMAIN.interior(max(len(blocks[2].points),
        math.ceil(trace_oversampling*trace_size)),seed+104729)
    initial[:,3]=0.
    initial_target=reference(torch.as_tensor(initial))[:,:3].detach()
    blocks[2]=ResidualBlock('initial_velocity',initial,
        lambda x,j,p:j[ZERO][:,:3]-initial_target,(ZERO,))
    gauge=DOMAIN.interior(max(len(blocks[3].points),
        math.ceil(trace_oversampling*(degree+1))),seed+29989)
    gauge[:,:3]=0.
    blocks[3]=ResidualBlock('pressure_gauge',gauge,
        lambda x,j,p:j[ZERO][:,3],(ZERO,))
    return blocks


def run(args,degree,backend,warm,warm_indices,source):
    began=time.perf_counter()
    kwargs=dict(max_derivative=2)
    if backend=='quill':
        if args.evaluation:
            if 'evaluation' not in inspect.signature(ConstructedFeatures).parameters:
                raise ValueError('Requested evaluation mode is not yet supported by the feature API')
            kwargs['evaluation']=args.evaluation
        if args.centers:kwargs['centers']=args.centers
        if args.lam:kwargs['lam']=args.lam
        if args.encoding_tolerance:
            if 'encoding_tolerance' not in inspect.signature(ConstructedFeatures).parameters:
                raise ValueError('Requested encoding_tolerance is not yet supported by the feature API')
            kwargs['encoding_tolerance']=args.encoding_tolerance
    features=ConstructedFeatures(DOMAIN.bounds,degree,backend=backend,**kwargs)
    n=max(256,int(args.oversampling*features.size))
    blocks=make_precision_blocks(n,113+1009*degree,degree,args.trace_oversampling)
    estimate=sum(len(b.points)*len(b.derivatives)*features.size*8 for b in blocks)
    if estimate>args.maximum_cache_gb*1e9:
        raise MemoryError(f'Estimated derivative cache {estimate/1e9:.3f} GB exceeds declared cap {args.maximum_cache_gb} GB')
    problem=ResidualProblem(features,blocks,fields=4)
    initial=map_start(warm,warm_indices,features)
    label=args.label+'_' if args.label else ''
    stem=f'ns4d_precision_{label}{backend}_p{degree}'
    checkpoint_state={};checkpoint_history=[]
    def checkpoint(coefficients,parameters,record):
        checkpoint_state.update(coefficients=coefficients.copy(),parameters=parameters.copy())
        checkpoint_history.append(dict(record))
        temporary=OUT/(stem+'_live.tmp.npz')
        np.savez_compressed(temporary,coefficients=coefficients,parameters=parameters,
                            multiindices=features.multiindices,bounds=features.bounds)
        temporary.replace(OUT/(stem+'_live.npz'))
        (OUT/(stem+'_live.json')).write_text(json.dumps(clean(record),indent=2)+'\n')
        print(json.dumps(dict(event='iterate',backend=backend,degree=degree,**record)),flush=True)
        if args.stop_file and Path(args.stop_file).exists():
            raise _CheckpointStop('Stop file requested termination at a saved accepted iterate')
    options=dict(max_iterations=args.iterations,tolerance=args.tolerance,
        damping=1e-12,lsmr_tolerance=1e-12,lsmr_max_iterations=args.krylov_iterations,
        gradient_tolerance=1e-15,step_tolerance=1e-16,
        max_seconds=args.seconds,max_residual_evaluations=3000,
        preconditioner=args.preconditioner,
        preconditioner_block_size=args.preconditioner_block_size)
    if args.high_accuracy:
        if 'high_accuracy' not in inspect.signature(solve_residual).parameters:
            raise ValueError('High-accuracy engine API not yet available')
        options['high_accuracy']=True
    if args.linear_refinement_steps is not None:
        options['linear_refinement_steps']=args.linear_refinement_steps
    if 'iteration_callback' in inspect.signature(solve_residual).parameters:
        options['iteration_callback']=checkpoint
    recorded_options={k:('atomic solved-coefficient checkpoint' if k=='iteration_callback' else v)
                      for k,v in options.items()}
    print(json.dumps(dict(event='launch',backend=backend,degree=degree,features=features.size,
                          unknowns=4*features.size,estimated_cache_gb=estimate/1e9,
                          engine_sha256=ENGINE_SHA256,warmstart=source,options=recorded_options)),flush=True)
    setup=time.perf_counter()-began
    try:
        solution=solve_residual(problem,initial_coefficients=initial,**options)
    except (KeyboardInterrupt,_CheckpointStop) as interruption:
        if not checkpoint_state:raise
        status='stopped_at_checkpoint' if isinstance(interruption,_CheckpointStop) else 'interrupted_at_checkpoint'
        solution=ResidualSolution(problem,checkpoint_state['coefficients'],
            checkpoint_state['parameters'],dict(status=status,converged=False),checkpoint_history)
        recovered_checks=check_residuals(solution,blocks)
        for check in recovered_checks:check['reuses_training_locations']=True
        solution.metrics.update(blocks=recovered_checks,
            iterations=checkpoint_history[-1]['iteration'],lsmr_iterations=None,
            interruption_scope='Latest atomically saved accepted iterate; in-flight linear work discarded, iteration totals unavailable')
    solve_seconds=time.perf_counter()-began-setup
    # Save transferable coefficients immediately, before all reference checks.
    path=OUT/(stem+'.npz')
    np.savez_compressed(path,coefficients=solution.coefficients,multiindices=features.multiindices,
                        bounds=features.bounds,parameters=solution.parameters)
    validation_start=time.perf_counter()
    validation=validate_solution(solution)
    by_time=temporal_errors(solution)
    modal_degrees=np.sum(features.multiindices,axis=1)
    coefficient_norms={str(p):np.linalg.norm(solution.coefficients[modal_degrees==p],axis=0).tolist()
                       for p in range(degree+1)}
    validation_seconds=time.perf_counter()-validation_start
    output=dict(backend=backend,degree=degree,stem=stem,warmstart=source,
        warmstart_scope='Previously solved coefficients mapped by polynomial multiindex; no oracle initialization',
        engine_sha256=ENGINE_SHA256,options=recorded_options,features=features.metrics,
        unknown_coefficients=features.size*4,interior_points=n,
        trace_oversampling=args.trace_oversampling,
        block_point_counts={b.name:len(b.points) for b in blocks},
        estimated_basis_cache_bytes=estimate,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        memory_scope='Peak RSS is cumulative for this runner process, cache bytes are named derivative arrays',
        setup_seconds=setup,solve_seconds=solve_seconds,validation_seconds=validation_seconds,
        status=solution.status,solver=solution.metrics,history=solution.history,
        validation=validation,error_by_time=by_time,coefficient_norm_by_total_degree=coefficient_norms,
        reference_checks=validate_reference())
    (OUT/(stem+'.json')).write_text(json.dumps(clean(output),indent=2)+'\n')
    plot(output)
    print(json.dumps(dict(event='finished',backend=backend,degree=degree,status=solution.status,
        seconds=solve_seconds,velocity=validation['velocity_relative_l2'],
        pressure=validation['pressure_relative_l2'],final_velocity=validation['final_velocity_relative_l2'],
        final_pressure=validation['final_pressure_relative_l2'],solver_residual=max(b['maximum_scaled_rms'] for b in solution.metrics['blocks']))),flush=True)
    return solution.coefficients.copy(),features.multiindices.copy(),str(path),output


def plot(row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for key,label in [('velocity_relative_l2','velocity'),('pressure_relative_l2','pressure')]:
        axes[0].semilogy([a['time'] for a in row['error_by_time']],
            [a[key] for a in row['error_by_time']],'-o',ms=3,label=label)
    axes[0].set(xlabel='Physical time',ylabel='Held-out relative L2 error',title='Accuracy through the whole trajectory')
    axes[0].legend()
    history=row['history']
    axes[1].semilogy([a['elapsed_seconds'] for a in history],
        [a['maximum_scaled_block_rms'] for a in history],'-o')
    axes[1].set(xlabel='Solver elapsed seconds',ylabel='Maximum declared block RMS',title='Optimization progress')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"{row['backend']}, degree {row['degree']}: {row['status']}")
    fig.savefig(OUT/(row['stem']+'.png'),dpi=180);plt.close(fig)


def plot_scaling():
    """Summarize measured runs without fitting or choosing coefficients."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    paths=list(OUT.glob('ns4d_precision_*_p*.json'))+list(OUT.glob('ns4d_sparse_*_stage*_p*.json'))
    for path in sorted(paths):
        row=json.loads(path.read_text())
        if 'error_by_time' in row and 'backend' in row and 'degree' in row:rows.append(row)
    if not rows:return
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for row in rows:
        geometry='sparse' if row['features'].get('selected_multiindices',False) else 'full'
        label=f"{row['backend']} {geometry} degree{row['degree']}: {row['status']}"
        for ax,key in zip(axes,['velocity_relative_l2','pressure_relative_l2']):
            ax.semilogy([v['time'] for v in row['error_by_time']],
                [v[key] for v in row['error_by_time']],label=label)
    for ax,title in zip(axes,['Velocity accuracy over physical time','Pressure accuracy over physical time']):
        ax.set(xlabel='Physical time',ylabel='Relative L2 error',title=title)
        ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'ns4d_precision_over_time.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for row in rows:
        geometry='sparse' if row['features'].get('selected_multiindices',False) else 'full'
        label=f"{row['backend']} {geometry} p{row['degree']}: {row['status']}"
        axes[0].scatter(row['degree'],row['validation']['velocity_relative_l2'],label=label)
        axes[0].scatter(row['degree'],row['validation']['pressure_relative_l2'],marker='x')
        axes[1].scatter(row['solve_seconds'],row['validation']['pressure_relative_l2'],label=label)
        axes[2].scatter(row['unknown_coefficients']/4,row['estimated_basis_cache_bytes']/1e9,label=label)
    axes[0].set(yscale='log',xlabel='Total polynomial degree',ylabel='Relative L2 error',title='Circles: velocity; crosses: pressure')
    axes[1].set(xscale='log',yscale='log',xlabel='Solver seconds',ylabel='Pressure relative L2 error',title='Measured cost and accuracy')
    axes[2].set(xscale='log',yscale='log',xlabel='Scalar feature count P',ylabel='Basis cache GB',title='Cache grows approximately as P²')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'ns4d_precision_scaling.png',dpi=180);plt.close(fig)
    capacity_path=OUT/'ns4d_reference_approximation_floor.json'
    if capacity_path.exists():
        capacity=json.loads(capacity_path.read_text())['rows']
        selected_capacity=[]
        for path in OUT.glob('ns4d_precision_*checkpoint_validation*.json'):
            audit=json.loads(path.read_text()).get('capacity_audit')
            if audit is not None:selected_capacity.append(audit)
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for ax,key,title in zip(axes,['velocity_relative_l2','pressure_relative_l2'],
                                ['Velocity','Pressure']):
            for backend,style in [('polynomial','--'),('quill',':')]:
                reference_rows=[r for r in capacity if r['backend']==backend]
                ax.semilogy([r['degree'] for r in reference_rows],
                    [r[key] for r in reference_rows],style,
                    label=f'{backend}: full-space reference coefficients')
            for audit in selected_capacity:
                ax.semilogy(audit['degree'],audit[key],'D',mfc='none',ms=7,
                    label=f"Selected-support capacity, {audit['features']} features")
            for row in rows:
                geometry='sparse' if row['features'].get('selected_multiindices',False) else 'full'
                ax.semilogy(row['degree'],row['validation'][key],'o',
                    label=f"PDE: {row['backend']} {geometry} p{row['degree']} ({row['status']})")
            ax.set(xlabel='Total degree',ylabel='Held-out relative L2 error',title=title)
            ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Representational capacity versus coefficients actually found from the PDE')
        fig.savefig(OUT/'ns4d_precision_capacity_gap.png',dpi=180);plt.close(fig)
    plot_main_precision()


def plot_main_precision():
    """Keep one useful result per refinement stage; omit aborted controls."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['ns4d_precision_pilot_polynomial_p10.json',
           'ns4d_precision_trace_polynomial_p12.json',
           'ns4d_sparse_axis14_polynomial_stage0_p14.json',
           'ns4d_sparse_mixed14_quill_stage0_p14.json']
    polished='ns4d_sparse_polish14_quill_stage0_p14.json'
    if (OUT/polished).exists():
        candidate=json.loads((OUT/polished).read_text())
        if candidate.get('history') and candidate['history'][-1]['iteration']>0:
            names[-1]=polished
    rows=[json.loads((OUT/name).read_text()) for name in names if (OUT/name).exists()]
    if not rows:return
    capacity=json.loads((OUT/'ns4d_reference_approximation_floor.json').read_text())['rows']
    capacity={r['degree']:r for r in capacity if r['backend']=='polynomial'}
    labels=[f"Degree {r['degree']}\n{r['unknown_coefficients']//4:,} features\n{r['backend'].upper() if r['backend']=='quill' else 'Polynomial'}"
            for r in rows]
    x=np.arange(len(rows))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,key,title in zip(axes,['velocity_relative_l2','pressure_relative_l2'],['Velocity','Pressure']):
        actual=[r['validation'][key] for r in rows]
        ax.semilogy(x,actual,'-',color='#4e6688',alpha=.65,lw=1.3)
        for backend,marker,color,label in [('polynomial','o','#2863ad','PDE solve: polynomial features'),
                                         ('quill','*','#b94342','PDE solve: actual QUILL features')]:
            mask=np.array([r['backend']==backend for r in rows])
            if mask.any():ax.semilogy(x[mask],np.asarray(actual)[mask],marker,color=color,
                                     ms=10 if backend=='quill' else 6,ls='none',label=label)
        ax.semilogy(x,[capacity[r['degree']][key] for r in rows],'--',color='#777777',
                    lw=1.5,label='Full-degree reference expansion: capacity only')
        ax.set(xticks=x,xticklabels=labels,ylabel='Space-time relative L2 error',title=title)
        ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper right')
        ax.tick_params(axis='x',labelsize=9)
    fig.suptitle('Accuracy obtained from the PDE, initial values, and boundary values',y=.985,fontsize=14)
    fig.get_layout_engine().set(rect=(0,.06,1,.86))
    fig.text(.5,.018,'Dashed lines use known reference coefficients, never used to initialize the solver. '
             'The full degree-14 reference uses 3,060 features.',ha='center',fontsize=8)
    fig.savefig(OUT/'ns4d_precision_main.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,key,title in zip(axes,['velocity_relative_l2','pressure_relative_l2'],['Velocity','Pressure']):
        for r in rows:
            label=f"{r['backend']}, degree {r['degree']}, {r['unknown_coefficients']//4:,} features"
            ax.semilogy([a['time'] for a in r['error_by_time']],
                        [a[key] for a in r['error_by_time']],label=label,
                        lw=2.2 if r['backend']=='quill' else 1.2)
        ax.set(xlabel='Physical time',ylabel='Relative L2 error',title=title)
        ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Held-out accuracy throughout the solved trajectory')
    fig.savefig(OUT/'ns4d_precision_main_over_time.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--degrees',type=int,nargs='+',default=[10,12])
    p.add_argument('--backends',nargs='+',choices=['polynomial','quill'],default=['polynomial'])
    p.add_argument('--warmstart',default=str(OUT/'ns4d_coefficients.npz'))
    p.add_argument('--seconds',type=float,default=900)
    p.add_argument('--tolerance',type=float,default=1e-12)
    p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--krylov-iterations',type=int,default=5000)
    p.add_argument('--preconditioner',choices=['auto','diagonal','block'],default='auto')
    p.add_argument('--preconditioner-block-size',type=int,default=64)
    p.add_argument('--linear-refinement-steps',type=int)
    p.add_argument('--stop-file',help='When this file appears, stop at the next saved accepted iterate and validate it')
    p.add_argument('--maximum-cache-gb',type=float,default=2.5)
    p.add_argument('--oversampling',type=float,default=6)
    p.add_argument('--trace-oversampling',type=float,default=0,
        help='Minimum samples per spatial face / initial trace, divided by C(p+3,3); 0 preserves old counts')
    p.add_argument('--centers',type=int);p.add_argument('--lam',type=float)
    p.add_argument('--encoding-tolerance',type=float)
    p.add_argument('--evaluation',choices=['standard','anchored'])
    p.add_argument('--high-accuracy',action='store_true');p.add_argument('--label',default='')
    args=p.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    validate_reference()
    warm,indices=load_start(args.warmstart);source=args.warmstart;rows=[]
    for degree in args.degrees:
        for backend in args.backends:
            warm,indices,source,row=run(args,degree,backend,warm,indices,source)
            rows.append({k:row[k] for k in ['stem','degree','backend','status','solve_seconds','validation','estimated_basis_cache_bytes','process_peak_rss_bytes']})
            del row;gc.collect()
    label=args.label+'_' if args.label else ''
    (OUT/f'ns4d_precision_{label}summary.json').write_text(json.dumps(clean(rows),indent=2)+'\n')
    plot_scaling()


if __name__=='__main__':main()
