"""Cold-start nonlinear PDE discovery in an actual flat QUILL ridge MLP.

Run from repository root:
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
 experiments/expF19_radon_direct_pde/ridge_pinn_study.py

The forcing and boundary data define a manufactured, uniquely solvable
semilinear elliptic problem. Interior reference values are validation only.
All solves start at a=0 and evaluate actual neural jets for backend=quill.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from scipy.stats import qmc

from solver.ridge_pinn_features import RidgePINNFeatures
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/ridge_pinn'


def disk_points(n,seed):
    q=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
    return np.sqrt(q[:,0,None])*np.column_stack([np.cos(2*np.pi*q[:,1]),np.sin(2*np.pi*q[:,1])])


def exact(x):
    return np.exp(.3*x[:,0:1])*np.cos(1.3*x[:,1:2])+.1*x[:,0:1]*x[:,1:2]


def forcing(x):
    # Declared forcing for -Laplacian(u)+u^3; not a solution-value loss.
    return 1.6*np.exp(.3*x[:,0:1])*np.cos(1.3*x[:,1:2])+exact(x)**3


def blocks(count,seed):
    x=disk_points(count,seed)
    a=2*np.pi*(np.arange(max(80,int(np.sqrt(count))*8))+.173)/max(80,int(np.sqrt(count))*8)
    b=np.column_stack([np.cos(a),np.sin(a)])
    source=torch.tensor(forcing(x),dtype=torch.float64)
    boundary=torch.tensor(exact(b),dtype=torch.float64)
    return [ResidualBlock('PDE',x,lambda x,j,p:-j[(2,0)]-j[(0,2)]+j[(0,0)]**3-source,((0,0),(2,0),(0,2))),
            ResidualBlock('boundary',b,lambda x,j,p:j[(0,0)]-boundary,((0,0),))]


def torch_export_audit(features,coefficients,points,source=None,save_path=None):
    """Evaluate an ordinary two-Linear/Tanh module, including Torch AD jets."""
    net=features.compile(coefficients)
    layer1=torch.nn.Linear(2,len(net['first_bias']),dtype=torch.float64)
    layer2=torch.nn.Linear(len(net['first_bias']),1,dtype=torch.float64)
    with torch.no_grad():
        layer1.weight.copy_(torch.tensor(net['first_weights']))
        layer1.bias.copy_(torch.tensor(net['first_bias']))
        layer2.weight.copy_(torch.tensor(net['output_weights'].T))
        layer2.bias.copy_(torch.tensor(net['output_bias']))
    model=torch.nn.Sequential(layer1,torch.nn.Tanh(),layer2)
    if save_path is not None:
        torch.save(model.state_dict(),save_path)
    x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    y=model(x)
    d=torch.autograd.grad(y.sum(),x,create_graph=True)[0]
    dxx=torch.autograd.grad(d[:,0].sum(),x,retain_graph=True)[0][:,0:1]
    dyy=torch.autograd.grad(d[:,1].sum(),x)[0][:,1:2]
    y0=y.detach().numpy()
    checks={'value_maximum_difference':float(np.max(np.abs(y0-features.evaluate(points)@coefficients))),
        'dxx_maximum_difference':float(np.max(np.abs(dxx.detach().numpy()-features.evaluate(points,(2,0))@coefficients))),
        'dyy_maximum_difference':float(np.max(np.abs(dyy.detach().numpy()-features.evaluate(points,(0,2))@coefficients)))}
    if source is not None:
        r=(-dxx-dyy+y**3).detach().numpy()-source(points)
        checks['ordinary_torch_pde_rms']=float(np.sqrt(np.mean(r*r)))
        checks['ordinary_torch_value_relative_l2']=float(np.linalg.norm(y0-exact(points))/np.linalg.norm(exact(points)))
    return checks


def run(degree=8,centers=257,directions=None,backend='quill',seconds=60.,label='main'):
    began=time.perf_counter()
    f=RidgePINNFeatures(degree,centers=centers,directions=directions,backend=backend,
                         encoding_tolerance=1e3,allow_angular_underresolution=True)
    problem=ResidualProblem(f,blocks(max(512,8*f.size),813))
    history=[]
    def callback(a,p,record):
        history.append(record)
    s=solve_residual(problem,max_iterations=14,tolerance=1e-12,max_seconds=seconds,
        high_accuracy=True,linear_refinement_steps=0,damping=1e-10,
        lsmr_max_iterations=600,preconditioner='auto',preconditioner_block_size=64,
        iteration_callback=callback)
    solved=time.perf_counter()
    held=disk_points(2048,18973)
    y=s.evaluate(held);truth=exact(held)
    residual=-s.evaluate(held,(2,0))-s.evaluate(held,(0,2))+y**3-forcing(held)
    a=2*np.pi*(np.arange(401)+.5913)/401
    b=np.column_stack([np.cos(a),np.sin(a)])
    boundary=s.evaluate(b)-exact(b)
    errors=dict(relative_l2=float(np.linalg.norm(y-truth)/np.linalg.norm(truth)),
        maximum_absolute=float(np.max(np.abs(y-truth))),
        fresh_pde_rms=float(np.sqrt(np.mean(residual**2))),
        fresh_pde_maximum=float(np.max(np.abs(residual))),
        fresh_boundary_rms=float(np.sqrt(np.mean(boundary**2))))
    audit={}
    if backend=='quill':
        for derivative in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]:
            stable=s.evaluate(held,derivative)
            flat=f.evaluate_flat(held,s.coefficients,derivative)
            audit[str(derivative)]=dict(maximum_difference=float(np.max(np.abs(flat-stable))),
                                       rms_difference=float(np.sqrt(np.mean((flat-stable)**2))))
            if derivative==(0,0):
                errors['ordinary_flat_mlp_relative_l2']=float(np.linalg.norm(flat-truth)/np.linalg.norm(truth))
        compiled=f.compile(s.coefficients)
        audit['readout_l1']=float(np.sum(np.abs(compiled['output_weights'])))
        audit['readout_max']=float(np.max(np.abs(compiled['output_weights'])))
        audit['output_bias']=compiled['output_bias'].tolist()
        audit['ordinary_layers']=[f'Linear(2,{f.metrics["tanh_count"]})','Tanh',f'Linear({f.metrics["tanh_count"]},1)']
        OUT.mkdir(parents=True,exist_ok=True)
        stem=f'ridge_pinn_{label}_{backend}_p{degree}_m{f.direction_count}_n{centers}'
        audit['torch_autograd']=torch_export_audit(f,s.coefficients,held[:257],forcing,OUT/(stem+'_torch_state.pt'))
        audit['torch_autograd_boundary']=torch_export_audit(f,s.coefficients,b[:257],forcing)
    row=dict(label=label,features=f.metrics,errors=errors,solver=s.metrics,history=s.history,
        architecture_audit=audit,construction_and_solve_seconds=solved-began,
        all_setup_solve_evaluation_seconds=time.perf_counter()-began,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        reference_formula='exp(0.3*x)*cos(1.3*y)+0.1*x*y',
        PDE='-(u_xx+u_yy)+u^3=1.6*exp(0.3*x)*cos(1.3*y)+(exp(0.3*x)*cos(1.3*y)+0.1*x*y)^3',
        metric_normalization={'relative_l2':'Euclidean error norm / reference norm on 2048 independent uniform-area disk Sobol points',
                              'fresh_pde_rms':'absolute sqrt(mean(residual^2)); physical coordinate derivatives, no source normalization',
                              'fresh_boundary_rms':'absolute RMS on 401 independently shifted boundary angles',
                              'ordinary_torch':'257 fresh interior points and 257 fresh boundary points; standard torch.nn.Linear/Tanh/Linear and automatic differentiation'},
        data_policy=dict(initial_coefficients='exactly zero; no warmstart',
                         observed_data='PDE source and boundary values only',
                         interior_solution_labels=False,
                         reference_use='manufactured forcing, boundary, heldout validation',
                         no_solution_computed_before_neural_solve=True),
        limitation='Smooth manufactured 2D elliptic problem; global numerical Gauss-Newton solve, not a closed-form PDE solution.')
    OUT.mkdir(parents=True,exist_ok=True)
    stem=f'ridge_pinn_{label}_{backend}_p{degree}_m{f.direction_count}_n{centers}'
    (OUT/(stem+'.json')).write_text(json.dumps(row,indent=2))
    arrays=dict(coefficients=s.coefficients,points=held,target=truth,prediction=y,residual=residual)
    if backend=='quill':arrays.update(f.compile(s.coefficients))
    np.savez_compressed(OUT/(stem+'.npz'),**arrays)
    print(json.dumps(dict(stem=stem,status=s.status,**errors,
                         seconds=row['all_setup_solve_evaluation_seconds'])),flush=True)
    return row


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(p.read_text()) for p in OUT.glob('ridge_pinn_*_p*_m*_n*.json')]
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for backend,color in [('polynomial','#5271a5'),('quill','#b54937')]:
        selected=sorted([r for r in rows if r.get('label')=='resolution' and r['features']['backend']==backend],key=lambda r:r['features']['degree'])
        if selected:
            ax[0].semilogy([r['features']['degree'] for r in selected],[r['errors']['relative_l2'] for r in selected],'-o',color=color,label=backend)
    ax[0].set(xlabel='Maximum profile degree p',ylabel='Heldout relative L2 error',title='Same PDE solver, zero initialization')
    ax[0].legend()
    for label,title,index in [('centers','Center resolution at fixed directions',1),('directions','Direction resolution at fixed centers',2)]:
        selected=sorted([r for r in rows if r.get('label')==label],key=lambda r:r['features']['interior_centers_per_direction'] if label=='centers' else r['features']['directions'])
        key='interior_centers_per_direction' if label=='centers' else 'directions'
        if selected:
            x=[r['features'][key] for r in selected]
            ax[index].semilogy(x,[r['errors']['relative_l2'] for r in selected],'-o',label='function error')
            ax[index].semilogy(x,[r['errors']['fresh_pde_rms'] for r in selected],'-s',label='fresh PDE residual')
        ax[index].set(xlabel='Interior centers N' if label=='centers' else 'Directions M',title=title)
        ax[index].legend()
    for a in ax:a.grid(alpha=.2)
    fig.suptitle('A flat tanh ridge MLP solves −Δu + u³ = s on the disk')
    fig.tight_layout();fig.savefig(OUT/'ridge_pinn_resolution.png',dpi=180);plt.close(fig)
    result=OUT/'ridge_pinn_resolution_quill_p18_m19_n257.npz'
    if result.exists():
        data=np.load(result)
        points=data['points'];target=data['target'][:,0];pred=data['prediction'][:,0]
        fig,ax=plt.subplots(1,3,figsize=(13,4))
        lo=min(target.min(),pred.min());hi=max(target.max(),pred.max())
        for index,values,title in [(0,target,'Target (validation only)'),(1,pred,'Actual tanh network')]:
            art=ax[index].tripcolor(points[:,0],points[:,1],values,shading='gouraud',vmin=lo,vmax=hi,cmap='viridis')
            fig.colorbar(art,ax=ax[index]);ax[index].set_title(title)
        error=np.log10(np.maximum(np.abs(pred-target),1e-17))
        art=ax[2].tripcolor(points[:,0],points[:,1],error,shading='flat',cmap='magma',vmin=-17,vmax=-14)
        fig.colorbar(art,ax=ax[2],label='log10 absolute error');ax[2].set_title('Float64 error; no error certificate')
        for a in ax:a.set(aspect='equal',xlabel='x',ylabel='y')
        fig.suptitle('Nonlinear PDE from zero: 5529 tanh neurons, 190 solved coordinates')
        fig.tight_layout();fig.savefig(OUT/'ridge_pinn_solution.png',dpi=180);plt.close(fig)
    burgers=[r for r in rows if r.get('case')=='unforced Burgers IVP']
    if burgers:
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        for r in burgers:
            p=r['features']['degree'];strong=r['solver']['settings']['lsmr_max_iterations']>900
            label=f'p={p} {r["features"]["backend"]}'+(' more Krylov steps' if strong else '')
            ax[0].semilogy([h['iteration'] for h in r['history']],[h['maximum_scaled_block_rms'] for h in r['history']],label=label)
        ax[0].set(xlabel='Gauss–Newton iteration',ylabel='Training maximum block RMS',title='Burgers: conditioning remains consequential')
        ax[0].legend(fontsize=8)
        baseline=sorted([r for r in burgers if r['features']['backend']=='quill' and r['solver']['settings']['lsmr_max_iterations']==900],key=lambda r:r['features']['degree'])
        ax[1].semilogy([r['features']['degree'] for r in baseline],[r['errors']['relative_l2'] for r in baseline],'-o',label='standard solve budget')
        for r in burgers:
            if r['solver']['settings']['lsmr_max_iterations']>900:
                ax[1].semilogy([r['features']['degree']],[r['errors']['relative_l2']],'*',markersize=13,label='more Krylov steps; still iteration-limited')
        ax[1].set(xlabel='Maximum profile degree',ylabel='Heldout space-time relative L2',title='Unforced nonlinear IVP, initial/boundary data only')
        ax[1].legend(fontsize=8)
        for a in ax:a.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUT/'ridge_pinn_burgers.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if not args.plot_only:
        for p in ([4,8] if args.quick else [4,8,12,16,18]):
            for backend in ['polynomial','quill']:
                run(p,257,backend=backend,label='resolution',seconds=45.)
        if not args.quick:
            for n in [33,65,129,257]:run(12,n,label='centers',seconds=45.)
            for m in [2,4,8,13]:run(12,257,m,label='directions',seconds=45.)
    plot()
