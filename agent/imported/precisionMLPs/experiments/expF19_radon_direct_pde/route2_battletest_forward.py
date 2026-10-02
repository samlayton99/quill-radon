"""Frozen-protocol native Radon forward stress tests; references are evaluation-only.

No interior solution labels, reference coefficients, externally evolved state,
or discovered active axes enter the ResidualDeclaration. The shared native
preset is unchanged between equations. Failed/budget-limited runs are retained.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-route2-battletest-mpl')
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import marshal
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
import torch
from scipy import sparse
from scipy.integrate import solve_ivp
from solver.general_domains import ResidualDomain
from solver.general_adaptive import ResidualDeclaration
from solver.general_residual import ResidualBlock
from solver.route2 import solve_route2_native

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward'
SOURCES={
 'benchmark_context':'https://arxiv.org/abs/2306.08827',
 'helmholtz':'https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.dirichlet.html',
 'allen_cahn':'https://github.com/maziarraissi/PINNs/blob/master/docs/index.md',
 'diffusion':'Explicitly manufactured nonseparable verification control; not claimed to reproduce a PINNacle instance.'}
CASES={
 'helmholtz_n2':dict(family='helmholtz',difficulty='published',n=2),
 'helmholtz_n4':dict(family='helmholtz',difficulty='harder frequency',n=4),
 'diffusion_c10':dict(family='diffusion',difficulty='easier contrast',contrast=10.),
 'diffusion_c1000':dict(family='diffusion',difficulty='harder contrast',contrast=1000.),
 'allen_cahn_d001':dict(family='allen_cahn',difficulty='easier diffusion',diffusion=.01),
 'allen_cahn_d00001':dict(family='allen_cahn',difficulty='published stiff case',diffusion=.0001)}
OPTIONS=dict(coordinates='affine_disk',centers=257,lam=.2,tolerance=1e-8,
             oversampling=6,check_points=257,export_check_points=257,
             maximum_working_array_mb=2048,max_seconds=60.,batch_size=64)
PROTOCOL=dict(version=1,description='Native ordinary-MLP stress test; fixed settings before runs',
 cases=CASES,seeds=[0,1],solver_options=OPTIONS,
 preset='solve_route2_native defaults, no per-equation overrides',
 degrees=[4,8,12,16,24,32,40,48],
 constraints='All blocks weight=scale=1. PDE and prescribed BC/IC only; no interior observations.',
 initialization='zero first resolution; only native previous-resolution coefficients thereafter',
 reference_quarantine='No reference callbacks accepted by declarations. Numerical Allen-Cahn reference is generated only in --evaluate after fit artifacts exist.',
 reference_controls=dict(allen_cahn='Fourier collocation + DOP853, evaluation only',
                         grids=[512,1024],rtol=2e-11,atol=2e-12,
                         validation='same held-out points on both resolutions; disagreement reported, no claimed reference floor'),
 heldout=dict(field_count=2048,field_seed=884713,pde_count=257,pde_seed=663277,
              plotting_grid=81,plot_derivatives=False),
 resource='Single CPU thread per child. 2GiB named-array estimate, not hard process RSS cap; 60s soft solve budget per case.',
 source_urls=SOURCES,
 caveats=['Published Allen-Cahn IC has a periodic derivative compatibility defect at t=0.',
          'Helmholtz exact targets are separable benchmark functions but the solver does not exploit their factors or active directions.',
          'Manufactured diffusion labels enter boundary data and explicit forcing only.',
          'No function-space capacity fit is performed in this campaign; residual stagnation alone does not prove capacity limitation.'])
BASE_OUT=OUT
BASE_OPTIONS=copy.deepcopy(OPTIONS)
BASE_PROTOCOL=copy.deepcopy(PROTOCOL)


def configure(campaign):
    """Choose a whole-campaign protocol, never a target-specific solver setting."""
    global OUT,OPTIONS,PROTOCOL
    OUT=BASE_OUT;OPTIONS=copy.deepcopy(BASE_OPTIONS);PROTOCOL=copy.deepcopy(BASE_PROTOCOL)
    if campaign=='accuracy_v2':
        OUT=BASE_OUT/'accuracy_v2'
        OPTIONS.update(max_seconds=600.,max_iterations=60)
        PROTOCOL.update(version=2,seeds=[0],solver_options=OPTIONS,
                        description='Cold uniform accuracy extension of V1; all six cases, seed0, no case-specific changes',
                        preset='solve_route2_native defaults with max_iterations=60 uniformly across every case',
                        resource='Single CPU thread per child. 2GiB named-array estimate, not hard process RSS cap; 600s soft solve budget per case.',
                        comparison='Only outer-iteration cap20->60 and time60->600s change; same degree ladder, centers, lambda, constraints, sampling, tolerances; every case restarts at zero, no V1 weights reused.')
    elif campaign=='normalized_diffusion_precision_v3':
        OUT=BASE_OUT/'normalized_diffusion_precision_v3'
        OPTIONS.update(max_seconds=600.,max_iterations=60,tolerance=1e-13)
        PROTOCOL.update(version=3,seeds=[0],solver_options=OPTIONS,
                        cases={key:copy.deepcopy(CASES[key]) for key in ['diffusion_c10','diffusion_c1000']},
                        description='Paired diffusion precision extension with the same generic zero-jet differential-operator scaling',
                        preset='solve_route2_native defaults with max_iterations=60 uniformly across both contrasts',
                        normalization=dict(method='solver.residual_scaling.normalize_equations',blocks=['PDE'],
                                           expansion='zero jets before fitting',length_scales='declared domain side lengths',
                                           field_scales='one for every field',fit_truth_or_reference_used=False,
                                           source_sha256=hashlib.sha256((ROOT/'experiments/expF19_radon_direct_pde/solver/residual_scaling.py').read_bytes()).hexdigest()),
                        constraints='PDE receives the fixed generic sensitivity scaling; Dirichlet data unchanged. No interior observations. Independent audits always use the original unscaled PDE.',
                        resource='Single CPU thread per child, at most two concurrent paired fits. 2GiB named-array estimate, not hard process RSS cap; 600s soft solve budget per case.',
                        comparison='Both contrasts cold-start at zero using identical settings. N257, lambda.2, default degree ladder, all sampling unchanged; normalized stopping tolerance1e-13. Raw physical residual and field error are separately reported.')
    elif campaign!='screening_v1':raise ValueError('Unknown campaign')


def safe(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def freeze_protocol(out=None):
    if out is None:out=OUT
    out.mkdir(parents=True,exist_ok=True)
    payload=dict(PROTOCOL)
    payload['core_sha256']=hashlib.sha256((ROOT/'experiments/expF19_radon_direct_pde/solver/route2.py').read_bytes()).hexdigest()
    text=json.dumps(payload,sort_keys=True,indent=2)+'\n'
    path=out/'protocol.json'
    if path.exists():
        previous=json.loads(path.read_text())
        comparable=dict(previous);comparable.pop('core_sha256',None)
        if comparable!=PROTOCOL:raise RuntimeError('Frozen protocol differs; preserve it rather than silently changing settings')
        return hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def manufactured_jet(x):
    a,b=x[:,0:1],x[:,1:2]
    q=np.pi*a*b;r=2*np.pi*a+.4*np.pi*b
    e=torch.exp(.3*torch.sin(q));s=torch.sin(q);c=torch.cos(q)
    gx=.3*np.pi*b*c;gy=.3*np.pi*a*c
    value=e+.2*torch.sin(r)
    dx=e*gx+.4*np.pi*torch.cos(r)
    dy=e*gy+.08*np.pi*torch.cos(r)
    dxx=e*(gx*gx-.3*(np.pi*b)**2*s)-.8*np.pi**2*torch.sin(r)
    dyy=e*(gy*gy-.3*(np.pi*a)**2*s)-.032*np.pi**2*torch.sin(r)
    return value,dx,dy,dxx,dyy


def diffusion_coefficients(x,contrast):
    a,b=x[:,0:1],x[:,1:2];k=np.log(contrast)
    coef=torch.exp(.5*k*torch.sin(2*np.pi*a)*torch.sin(2*np.pi*b))
    ax=coef*k*np.pi*torch.cos(2*np.pi*a)*torch.sin(2*np.pi*b)
    ay=coef*k*np.pi*torch.sin(2*np.pi*a)*torch.cos(2*np.pi*b)
    return coef,ax,ay


def equation(case):
    family=case['family']
    if family=='helmholtz':
        k=2*np.pi*case['n']
        def residual(x,j,p):
            forcing=k*k*torch.sin(k*x[:,0:1])*torch.sin(k*x[:,1:2])
            return -j[(2,0)]-j[(0,2)]-k*k*j[(0,0)]-forcing
        return residual,((0,0),(2,0),(0,2))
    if family=='diffusion':
        def residual(x,j,p):
            a,ax,ay=diffusion_coefficients(x,case['contrast'])
            _,ux,uy,uxx,uyy=manufactured_jet(x)
            forcing=-a*(uxx+uyy)-ax*ux-ay*uy
            return -a*(j[(2,0)]+j[(0,2)])-ax*j[(1,0)]-ay*j[(0,1)]-forcing
        return residual,((1,0),(0,1),(2,0),(0,2))
    def residual(x,j,p):
        u=j[(0,0)]
        return j[(0,1)]-case['diffusion']*j[(2,0)]+5*(u**3-u)
    return residual,((0,0),(0,1),(2,0))


def make_declaration(case,seed=0):
    bounds=[[-1,1],[0,1]] if case['family']=='allen_cahn' else [[0,1],[0,1]]
    domain=ResidualDomain(bounds);callback,ders=equation(case)
    def blocks(count,local_seed):
        draw_seed=int(local_seed)+1000003*seed
        interior=domain.interior(count,draw_seed)
        result=[ResidualBlock('PDE',interior,callback,ders)]
        n=max(80,8*int(np.sqrt(count)))
        if case['family']=='allen_cahn':
            initial=domain.interior(n,draw_seed+31);initial[:,1]=0.
            def ic(x,j,p):return j[(0,0)]-x[:,0:1]**2*torch.cos(np.pi*x[:,0:1])
            result.append(ResidualBlock('initial',initial,ic,((0,0),)))
            times=domain.interior(n,draw_seed+79)[:,1]
            pair=np.r_[np.column_stack([np.full(n,-1.),times]),np.column_stack([np.full(n,1.),times])]
            aggregation=sparse.hstack([sparse.eye(n),-sparse.eye(n)],format='csr')
            result.extend([ResidualBlock('periodic_value',pair,lambda x,j,p:j[(0,0)],((0,0),),aggregation=aggregation),
                           ResidualBlock('periodic_dx',pair,lambda x,j,p:j[(1,0)],((1,0),),aggregation=aggregation)])
        else:
            boundary=domain.boundary(4*n,draw_seed+61).points
            bc=(lambda x,j,p:j[(0,0)]) if case['family']=='helmholtz' else (lambda x,j,p:j[(0,0)]-manufactured_jet(x)[0])
            result.append(ResidualBlock('Dirichlet',boundary,bc,((0,0),)))
        return result
    return ResidualDeclaration(domain,blocks,1)


def ordinary_jets(model,points,ders):
    x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    u=model(x)
    first=torch.autograd.grad(u.sum(),x,create_graph=True)[0] if any(sum(a)>0 for a in ders) else None
    jets={(0,0):u}
    for order in ders:
        if sum(order)==1:jets[order]=first[:,0:1] if order==(1,0) else first[:,1:2]
        if sum(order)==2:
            axis=0 if order[0] else 1
            jets[order]=torch.autograd.grad(first[:,axis].sum(),x,create_graph=True)[0][:,axis:axis+1]
    return x,jets


def evaluate_raw(model,points,callback=None,ders=((0,0),)):
    values=[];residuals=[]
    for chunk in np.array_split(points,max(1,int(np.ceil(len(points)/64)))):
        x,j=ordinary_jets(model,chunk,ders)
        values.append(j[(0,0)].detach().numpy())
        if callback is not None:residuals.append(callback(x,j,torch.empty(0)).detach().numpy())
    return np.concatenate(values),np.concatenate(residuals) if residuals else None


def source_snapshot():
    """Record current disk files separately from already-loaded function code."""
    files=['experiments/expF19_radon_direct_pde/route2_battletest_forward.py',
           'experiments/expF19_radon_direct_pde/solver/route2.py',
           'experiments/expF19_radon_direct_pde/solver/general_residual.py',
           'experiments/expF19_radon_direct_pde/solver/streamed_residual.py']
    if PROTOCOL.get('normalization'):
        files.append('experiments/expF19_radon_direct_pde/solver/residual_scaling.py')
    functions={'make_declaration':make_declaration,'equation':equation,
               'solve_route2_native':solve_route2_native}
    for module_name,function_name in [('solver.route2','solve_route2'),
                                     ('solver.general_residual','solve_residual'),
                                     ('solver.residual_scaling','normalize_equations')]:
        module=sys.modules.get(module_name)
        if module is not None and hasattr(module,function_name):
            functions[module_name+'.'+function_name]=getattr(module,function_name)
    return dict(disk_files_sha256={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in files},
                loaded_function_code_sha256={name:hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
                                             for name,function in functions.items()},
                scope='Disk hashes are measured at snapshot time. Code-object hashes identify the named loaded functions, not a complete transitive dependency snapshot.')


def fit(name,seed):
    """Atomically claim a case when independent single-thread workers share a queue."""
    if seed not in PROTOCOL['seeds']:raise ValueError('Seed is outside this frozen campaign')
    if name not in PROTOCOL['cases']:raise ValueError('Case is outside this frozen campaign')
    freeze_protocol();stem=f'{name}_s{seed}'
    saved=OUT/(stem+'.json')
    if saved.exists():return json.loads(saved.read_text())
    claim=OUT/(stem+'.running.json')
    try:handle=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        row=dict(name=name,seed=seed,status='already_running_elsewhere')
        print(json.dumps(row),flush=True)
        return row
    with os.fdopen(handle,'w') as stream:
        json.dump(dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                       campaign=OUT.name,name=name,seed=seed),stream)
    try:return _fit_case(name,seed)
    finally:claim.unlink(missing_ok=True)


def _fit_case(name,seed):
    if seed not in PROTOCOL['seeds']:raise ValueError('Seed is outside this frozen campaign')
    fingerprint=freeze_protocol();case=CASES[name];stem=f'{name}_s{seed}'
    dest=OUT/(stem+'.json')
    if dest.exists():return json.loads(dest.read_text())
    started=time.perf_counter()
    if PROTOCOL.get('normalization'):
        from solver.residual_scaling import normalize_equations
    startup=source_snapshot()
    row=dict(name=name,seed=seed,case=case,protocol_sha256=fingerprint,reference_used_in_fit=False,
             source_snapshot_before_fit=startup)
    try:
        declaration=make_declaration(case,seed)
        solving_declaration=declaration
        if PROTOCOL.get('normalization'):
            solving_declaration=normalize_equations(declaration,['PDE'])
            row['normalization']=PROTOCOL['normalization']
            row['raw_audit_scope']='Original unscaled equations evaluated using ordinary Torch AD; normalized stopping is not a raw-accuracy certificate.'
        result=solve_route2_native(solving_declaration,**OPTIONS)
        model=result.export();assert [type(layer).__name__ for layer in model]==['Linear','Tanh','Linear']
        points=declaration.domain.interior(PROTOCOL['heldout']['field_count'],PROTOCOL['heldout']['field_seed'])
        check=declaration.domain.interior(PROTOCOL['heldout']['pde_count'],PROTOCOL['heldout']['pde_seed'])
        y,_=evaluate_raw(model,points)
        callback,ders=equation(case);_,raw=evaluate_raw(model,check,callback,ders)
        axis0=np.linspace(*declaration.domain.bounds[0],81);axis1=np.linspace(*declaration.domain.bounds[1],81)
        a,b=np.meshgrid(axis0,axis1,indexing='ij');grid=np.column_stack([a.ravel(),b.ravel()])
        plot_y,_=evaluate_raw(model,grid)
        torch.save(dict(state_dict=model.state_dict(),width=model[0].out_features,dimension=2,architecture='Linear/Tanh/Linear'),OUT/(stem+'.pt'))
        np.savez_compressed(OUT/(stem+'.npz'),points=points,prediction=y,raw_pde_points=check,raw_pde=raw,grid=grid,grid_prediction=plot_y,coefficients=result.solution.coefficients)
        row.update(status=result.status,solver_status=result.solution.status,
                   neurons=model[0].out_features,degree=result.solution.problem.features.degree,
                   coordinates=result.solution.problem.features.size,
                   metrics=result.metrics,history=result.history,
                   raw_pde_rms=float(np.sqrt(np.mean(raw**2))),raw_pde_max=float(np.max(abs(raw))),
                   ordinary_readout_l1=float(model[2].weight.detach().abs().sum()),
                   heldout_reference_evaluated=False,
                   source_files_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/expF19_radon_direct_pde/solver/route2.py','experiments/expF19_radon_direct_pde/solver/general_residual.py','experiments/expF19_radon_direct_pde/solver/streamed_residual.py']})
    except Exception as exc:
        row.update(status='exception',error=repr(exc),traceback=traceback.format_exc())
    row['all_fit_export_seconds']=time.perf_counter()-started
    row['source_snapshot_at_completion']=source_snapshot()
    row['legacy_source_files_sha256_scope']='Disk hashes at fit completion; do not interpret this legacy field as startup or loaded-source provenance.'
    row['disk_source_files_changed_during_fit']=[path for path,value in startup['disk_files_sha256'].items()
        if row['source_snapshot_at_completion']['disk_files_sha256'].get(path)!=value]
    row['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    dest.write_text(json.dumps(row,indent=2,default=safe)+'\n')
    print(json.dumps({k:row.get(k) for k in ['name','seed','status','degree','neurons','raw_pde_rms','all_fit_export_seconds','error']}),flush=True)
    return row


def analytical_reference(case,points):
    x=torch.tensor(points,dtype=torch.float64)
    if case['family']=='helmholtz':
        k=2*np.pi*case['n'];return (torch.sin(k*x[:,0:1])*torch.sin(k*x[:,1:2])).numpy()
    if case['family']=='diffusion':return manufactured_jet(x)[0].numpy()
    raise ValueError('Allen-Cahn has no analytic solution oracle here')


def allen_reference(diffusion,queries,n):
    """Quarantined conventional solution: ONLY evaluation after neural fitting."""
    grid=-1+2*np.arange(n)/n;frequency=np.pi*np.fft.fftfreq(n,d=1/n)
    initial=grid**2*np.cos(np.pi*grid)
    def rhs(t,u):return diffusion*np.fft.ifft(-frequency**2*np.fft.fft(u)).real+5*(u-u**3)
    began=time.perf_counter()
    sol=solve_ivp(rhs,(0,1),initial,method='DOP853',rtol=2e-11,atol=2e-12,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    output=[]
    for chunk in np.array_split(queries,max(1,int(np.ceil(len(queries)/128)))):
        nodes=sol.sol(chunk[:,1]);coeff=np.fft.fft(nodes,axis=0)/n
        phases=np.exp(1j*frequency[:,None]*(chunk[:,0][None,:]+1))
        output.append(np.sum(coeff*phases,axis=0).real[:,None])
    return np.concatenate(output),dict(grid=n,seconds=time.perf_counter()-began,nfev=sol.nfev,accepted_steps=len(sol.t),rtol=2e-11,atol=2e-12,scope='Evaluation-only Fourier collocation/DOP853 reference, never supplied to native model')


def evaluate_saved():
    # Two independent fit queues may finish close together. Serialize only
    # post-fit cache/report writes; this lock never changes fitting behavior.
    with (OUT/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _evaluate_saved()


def _evaluate_saved():
    """The only execution path that constructs an Allen-Cahn reference."""
    for name,case in PROTOCOL['cases'].items():
        present=[OUT/f'{name}_s{seed}.npz' for seed in PROTOCOL['seeds'] if (OUT/f'{name}_s{seed}.npz').exists()]
        if not present:continue
        first=np.load(present[0]);queries=np.r_[first['points'],first['grid']]
        cache=OUT/(name+'_evaluation_reference.npz');refmeta={}
        if case['family']=='allen_cahn':
            if cache.exists():
                previous=np.load(cache);truth=previous['truth'];coarse=previous['coarse'];refmeta=json.loads((OUT/(name+'_evaluation_reference.json')).read_text())
            else:
                coarse,coarse_meta=allen_reference(case['diffusion'],queries,512)
                truth,fine_meta=allen_reference(case['diffusion'],queries,1024)
                refmeta=dict(coarse=coarse_meta,fine=fine_meta,fit_artifacts_existed_before_reference=True)
                np.savez_compressed(cache,truth=truth,coarse=coarse,queries=queries)
                (OUT/(name+'_evaluation_reference.json')).write_text(json.dumps(refmeta,indent=2))
        else:truth=analytical_reference(case,queries);coarse=truth
        n=len(first['points']);reference_difference=float(np.linalg.norm(truth[:n]-coarse[:n])/np.linalg.norm(truth[:n]))
        for file in present:
            arrays=np.load(file);row=json.loads(file.with_suffix('.json').read_text())
            pred=arrays['prediction'];real=truth[:n]
            saved=torch.load(file.with_suffix('.pt'),map_location='cpu',weights_only=True)
            model=torch.nn.Sequential(torch.nn.Linear(saved['dimension'],saved['width']),
                                      torch.nn.Tanh(),torch.nn.Linear(saved['width'],1)).double()
            model.load_state_dict(saved['state_dict'])
            # Conditions receive the same ordinary-network audit as the PDE.
            # These fresh points and any numerical reference remain evaluation-only.
            checks=[]
            for block in make_declaration(case,row['seed']).make_blocks(257,987513):
                _,residual=evaluate_raw(model,block.points,block.function,block.derivatives)
                if block.aggregation is not None:residual=block.aggregation@residual
                checks.append(dict(name=block.name,points=len(block.points),
                                   raw_rms=float(np.sqrt(np.mean(residual**2))),
                                   raw_max=float(abs(residual).max())))
            row.update(heldout_reference_evaluated=True,heldout_relative_l2=float(np.linalg.norm(pred-real)/np.linalg.norm(real)),
                       heldout_max_absolute=float(abs(pred-real).max()),
                       ordinary_heldout_constraints=checks,
                       reference_resolution_difference_relative_l2=reference_difference,
                       reference_scope='analytic manufactured verification' if case['family']!='allen_cahn' else 'independent numerical evaluator; difference is empirical reference uncertainty, not a proof',
                       reference_not_used_to_fit=True)
            file.with_suffix('.json').write_text(json.dumps(row,indent=2,default=safe)+'\n')
        np.savez_compressed(OUT/(name+'_plot_reference.npz'),grid=first['grid'],truth=truth[n:])
    return summarize()


def summarize():
    rows=[]
    for name in PROTOCOL['cases']:
        for seed in PROTOCOL['seeds']:
            path=OUT/f'{name}_s{seed}.json'
            if path.exists():rows.append(json.loads(path.read_text()))
    summary=dict(protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),cases=rows,
                 scope='Fixed-preset finite-budget stress test, no oracle fitting; all failure statuses retained')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=safe)+'\n')
    if not rows:return summary
    columns=['name','seed','status','solver_status','degree','neurons','coordinates',
             'heldout_relative_l2','heldout_max_absolute','raw_pde_rms','raw_pde_max',
             'all_fit_export_seconds','process_peak_rss_bytes','maximum_named_array_estimate_bytes',
             'reference_resolution_difference_relative_l2','ordinary_readout_l1','error']
    with (OUT/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for row in rows:
            compact={key:row.get(key,'') for key in columns}
            compact['maximum_named_array_estimate_bytes']=max((h.get('estimated_named_arrays_bytes',0) for h in row.get('history',[])),default=0)
            writer.writerow(compact)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
    labels=list(PROTOCOL['cases'])
    for index,name in enumerate(labels):
        for row in [r for r in rows if r['name']==name]:
            x=index+(.1 if row['seed'] else -.1)
            for ax,key in zip(axes,['heldout_relative_l2','raw_pde_rms','all_fit_export_seconds']):
                if key in row:ax.semilogy(x,max(row[key],1e-17),'o' if row['seed']==0 else 's',color='tab:blue' if row['status']=='sampled_residual_checks_passed' else 'tab:orange')
    for ax,title in zip(axes,['Held-out solution relative L2','Ordinary Torch AD PDE residual RMS','Fit + export + audit wall seconds']):
        ax.set_title(title);ax.set_xticks(range(len(labels)),[x.replace('_','\n') for x in labels],fontsize=8);ax.grid(alpha=.25)
    if not PROTOCOL.get('normalization'):
        axes[0].axhline(1e-8,color='gray',ls='--',lw=1);axes[1].axhline(1e-8,color='gray',ls='--',lw=1)
    seed_legend='circle = seed0, square = seed1' if len(PROTOCOL['seeds'])>1 else 'seed0 cold starts'
    campaign_label='Shared generic operator normalization, paired precision control' if PROTOCOL.get('normalization') else 'One native preset across three hard PDE families'
    fig.suptitle(campaign_label+'; '+seed_legend+'\nOrange retains non-passing/budget-limited runs; references used only after fitting',fontsize=11)
    fig.savefig(OUT/'forward_summary.png',dpi=170);plt.close(fig)
    plotted=list(PROTOCOL['cases']) if len(PROTOCOL['cases'])<=3 else ['helmholtz_n4','diffusion_c1000','allen_cahn_d00001']
    fig,axes=plt.subplots(len(plotted),3,figsize=(12,3.5*len(plotted)),constrained_layout=True,squeeze=False)
    for line,name in enumerate(plotted):
        path=OUT/(name+'_s0.npz');ref=OUT/(name+'_plot_reference.npz')
        if not path.exists() or not ref.exists():continue
        data=np.load(path);truth=np.load(ref)['truth'];grid=data['grid'];values=data['grid_prediction']
        for col,(value,title) in enumerate([(truth,'Reference'),(values,'Native ordinary MLP'),(values-truth,'Error')]):
            im=axes[line,col].pcolormesh(grid[:,0].reshape(81,81),grid[:,1].reshape(81,81),value.reshape(81,81),shading='auto',cmap='coolwarm')
            axes[line,col].set(title=name+'\n'+title,xlabel='x',ylabel='t' if CASES[name]['family']=='allen_cahn' else 'y');fig.colorbar(im,ax=axes[line,col],shrink=.8)
    fig.savefig(OUT/'hard_fields.png',dpi=150);plt.close(fig)
    if OUT.name=='accuracy_v2':
        comparisons=[]
        for row in rows:
            original=BASE_OUT/f"{row['name']}_s0.json"
            if not original.exists():continue
            baseline=json.loads(original.read_text())
            comparison=dict(name=row['name'],v1_status=baseline['status'],v2_status=row['status'])
            for key in ['degree','neurons','heldout_relative_l2','raw_pde_rms','all_fit_export_seconds']:
                comparison['v1_'+key]=baseline.get(key)
                comparison['v2_'+key]=row.get(key)
            comparisons.append(comparison)
        (OUT/'budget_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for ax,key,title in zip(axes,['heldout_relative_l2','raw_pde_rms'],
                               ['Ordinary-MLP heldout field relative L2','Ordinary Torch-AD PDE residual RMS']):
            for i,row in enumerate(comparisons):
                if row['v1_'+key] is None or row['v2_'+key] is None:continue
                ax.semilogy([i-.12,i+.12],[row['v1_'+key],row['v2_'+key]],color='.65',lw=1)
                ax.semilogy(i-.12,row['v1_'+key],'o',color='tab:orange',label='60s / 20 outer iterations' if i==0 else None)
                ax.semilogy(i+.12,row['v2_'+key],'s',color='tab:blue',label='600s / 60 outer iterations' if i==0 else None)
            ax.set_xticks(range(len(comparisons)),[r['name'].replace('_','\n') for r in comparisons],fontsize=8)
            ax.set_title(title);ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle('Uniform cold-start accuracy extension; seed0 for every case\nGeometry, data, sampling and tolerances unchanged; larger solve budgets only',fontsize=11)
        fig.savefig(OUT/'budget_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps([dict(name=r['name'],seed=r['seed'],status=r['status'],error=r.get('heldout_relative_l2'),pde=r.get('raw_pde_rms'),reference_gap=r.get('reference_resolution_difference_relative_l2')) for r in rows],indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',choices=['screening_v1','accuracy_v2','normalized_diffusion_precision_v3'],default='screening_v1');parser.add_argument('--freeze',action='store_true');parser.add_argument('--case',choices=list(CASES));parser.add_argument('--seed',type=int,choices=[0,1],default=0);parser.add_argument('--evaluate',action='store_true');parser.add_argument('--summarize',action='store_true');args=parser.parse_args();torch.set_num_threads(1);configure(args.campaign)
    if args.freeze:print(freeze_protocol())
    if args.case:fit(args.case,args.seed)
    if args.evaluate:evaluate_saved()
    elif args.summarize:summarize()
