"""Nonlinear Dirichlet pilot: preconditioned residuals in actual QUILL ridges.

Solve -Delta u + beta*u**3 = 3*sin(x)*sin(y)+.3*sin(3*x)*sin(2*y)
on (0,pi)^2 with zero boundary values. No solution labels, neural training,
or readout least squares. Sine coefficients are explicit real quadrature;
inverse of a reference shifted Laplacian is scalar division per mode.
Actual tanh values/Laplacians enter every nonlinear residual. This is a
classical spectral preconditioning strategy implemented in QUILL coordinates.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-native-quill-mpl')
from pathlib import Path
import json
import time
import numpy as np
from quill_boundary import encode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/capabilities'

def force(xy):
    x,y=xy.T
    return 3*np.sin(x)*np.sin(y)+.3*np.sin(3*x)*np.sin(2*y)

def quadrature(k):
    # q interior nodes on each axis; >4K resolves projected cubic products.
    q=4*k+1
    x=np.pi*np.arange(1,q+1)/(q+1)
    xx,yy=np.meshgrid(x,x,indexing='ij')
    xy=np.column_stack([xx.ravel(),yy.ravel()])
    mm,nn=np.meshgrid(np.arange(1,k+1),np.arange(1,k+1),indexing='ij')
    mn=np.column_stack([mm.ravel(),nn.ravel()])
    basis=np.sin(xy[:,0,None]*mn[:,0])*np.sin(xy[:,1,None]*mn[:,1])
    analysis=4*basis.T/(q+1)**2
    return xy,mn,basis,analysis

def encoded_modes(k,n,lam):
    freqs=np.arange(2,2*k+1)
    enc=encode(lambda z:np.cos(np.asarray(z)[...,None]*freqs),n-1,lam,
               interval=(-np.pi,np.pi),halo=int(np.ceil(np.sqrt(n))))
    return enc

def ridge_tables(xy,mn,enc,derivatives=True):
    """sin(mx)sin(ny)=.5[cos(mx-ny)-cos(mx+ny)], with each cosine QUILL encoded.

    s=(mx +/- ny)/(m+n) lies inside [-pi,pi]. The actual normal direction
    has length sqrt(m*m+n*n)/(m+n); the chain factor is included in Delta.
    """
    out=np.empty((len(xy),len(mn)))
    lap=np.empty_like(out) if derivatives else None
    for j,(m,n) in enumerate(mn):
        total=m+n
        vv=[];dd=[]
        for sign in [-1,1]:
            arg=(m*xy[:,0]+sign*n*xy[:,1])/total
            z=enc.gamma*(arg[:,None]-enc.centers)
            t=np.tanh(z)
            w=enc.weights[:,total-2]
            vv.append(t@w+enc.bias[total-2])
            if derivatives:
                r=np.exp(-2*np.abs(z));sech=4*r/(1+r)**2
                dd.append((-2*enc.gamma**2*t*sech)@w)
        out[:,j]=.5*(vv[0]-vv[1])
        if derivatives:
            lap[:,j]=.5*(m*m+n*n)/(total*total)*(dd[0]-dd[1])
    return out,lap

def construct(k,n,lam):
    start=time.perf_counter()
    xy,mn,basis,a=quadrature(k)
    enc=encoded_modes(k,n,lam)
    p,lap=ridge_tables(xy,mn,enc)
    return dict(k=k,N=n,lam=lam,xy=xy,mn=mn,A=a,P=p,Lap=lap,enc=enc,
                eigen=(mn*mn).sum(axis=1),setup_seconds=time.perf_counter()-start,
                relative_mode_value_error=float(np.linalg.norm(p-basis)/np.linalg.norm(basis)),
                relative_mode_lap_error=float(np.linalg.norm(lap+basis*(mn*mn).sum(axis=1))/
                                             np.linalg.norm(basis*(mn*mn).sum(axis=1))))

def solve(model,beta=5.,shift=5.,tol=2e-12,max_iter=1500):
    """b <- b - (diag(m²+n²)+shift)^(-1) A residual(S E b).

    The denominator is a preconditioner only: Lap uses actual neural derivatives.
    There is no assumed analytic solution and no matrix inverse/linear solve.
    """
    a,p,lap,eigen=(model[key] for key in ['A','P','Lap','eigen'])
    forcing=force(model['xy'])
    scale=np.linalg.norm(a@forcing)
    b=np.zeros(p.shape[1]);history=[]
    start=time.perf_counter();success=False;updates=0
    for iteration in range(max_iter):
        u=p@b
        if not np.all(np.isfinite(u)) or np.max(abs(u))>1e6:
            break
        residual=a@(-lap@b+beta*u**3-forcing)
        error=float(np.linalg.norm(residual)/scale)
        history.append(error)
        if error<tol:
            success=True
            break
        b-=residual/(eigen+shift)
        updates+=1
    return b,dict(beta=beta,shift=shift,converged=success,iterations=iteration+1,
                  coefficient_updates=updates,residual_evaluations=len(history),
                  seconds=time.perf_counter()-start,projected_relative_residual=history[-1],
                  residual_history=history)

def classical(k,beta=5.,shift=5.):
    xy,mn,p,a=quadrature(k)
    model=dict(xy=xy,mn=mn,P=p,A=a,eigen=(mn*mn).sum(axis=1),Lap=-p*(mn*mn).sum(axis=1))
    b,stats=solve(model,beta,shift,tol=3e-14)
    if not stats['converged']:
        raise RuntimeError('Independent spectral reference did not converge')
    return b,mn,stats

def classical_values(xy,b,mn):
    return (np.sin(xy[:,0,None]*mn[:,0])*np.sin(xy[:,1,None]*mn[:,1]))@b

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    # Select bandwidth from value and second-derivative encoding accuracy,
    # with no solution information. Keep sweep output, not just best value.
    xcheck=np.linspace(-np.pi,np.pi,2003)
    sweep=[]
    for lam in [.15,.20,.25,.30]:
        enc=encoded_modes(12,385,lam)
        freq=np.arange(2,25)
        target=np.cos(xcheck[:,None]*freq)
        v=enc.evaluate(xcheck);d=enc.evaluate(xcheck,2)
        row=dict(lam=lam,value_error=float(np.linalg.norm(v-target)/np.linalg.norm(target)),
                 d2_error=float(np.linalg.norm(d+target*freq**2)/np.linalg.norm(target*freq**2)))
        sweep.append(row)
    lam=min(sweep,key=lambda r:r['d2_error'])['lam']
    models=[];states=[];rows=[]
    for k in [4,8,12]:
        model=construct(k,32*k+1,lam)
        b,stats=solve(model)
        rows.append(dict(K=k,N=model['N'],lambda_selected=lam,
                         halo_per_side=model['enc'].halo_per_side,
                         unmerged_ridge_blocks=2*k*k,
                         unmerged_neurons=2*k*k*len(model['enc'].centers),
                         cached_matrix_bytes=model['P'].nbytes+model['Lap'].nbytes+model['A'].nbytes,
                         setup_seconds=model['setup_seconds'],
                         mode_value_error=model['relative_mode_value_error'],
                         mode_lap_error=model['relative_mode_lap_error'],**stats))
        models.append(model);states.append(b)
        print(json.dumps({key:val for key,val in rows[-1].items() if key!='residual_history'}),flush=True)
    # Failure control: the same nonlinear equation, no shift in the preconditioner.
    bad,negative=solve(models[-1],shift=0,max_iter=100)
    # Only after all production solves: independent higher-resolution references.
    ref,mnref,refstats=classical(28)
    ref2,mnref2,refstats2=classical(24)
    rng=np.random.default_rng(819)
    held=rng.uniform(0,np.pi,(1500,2))
    truth=classical_values(held,ref,mnref)
    refinement=np.linalg.norm(truth-classical_values(held,ref2,mnref2))/np.linalg.norm(truth)
    edge=np.linspace(0,np.pi,121)
    boundary=np.vstack([np.column_stack([np.zeros_like(edge),edge]),
                        np.column_stack([np.full_like(edge,np.pi),edge]),
                        np.column_stack([edge,np.zeros_like(edge)]),
                        np.column_stack([edge,np.full_like(edge,np.pi)])])
    for model,b,row in zip(models,states,rows):
        p,lap=ridge_tables(held,model['mn'],model['enc'])
        pred=p@b
        residual=-lap@b+5*pred**3-force(held)
        bp,_=ridge_tables(boundary,model['mn'],model['enc'],False)
        matched,matchedmn,matchedstats=classical(model['k'])
        row.update(relative_l2=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)),
                   offgrid_relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(force(held))),
                   boundary_max_abs=float(np.max(abs(bp@b))),
                   same_resolution_classical_gap=float(np.linalg.norm(pred-classical_values(held,matched,matchedmn))/np.linalg.norm(truth)),
                   same_resolution_classical_iteration_seconds=matchedstats['seconds'])
    result=dict(equation='-Delta u + 5 u^3 = 3 sin(x)sin(y)+.3 sin(3x)sin(2y)',
                domain='[0,pi]^2',boundary='u=0 (sine basis, up to measured QUILL encoding error)',
                method='preconditioned native neural residual with scalar shifted-Laplacian inverse',
                no_solution_labels=True,no_neural_readout_solve=True,lambda_sweep=sweep,
                rows=rows,negative_control=negative,reference_K=28,reference_refinement_gap=float(refinement),
                note='Fixed-point iterative numerical PDE solver, not a closed-form construction; sine/QUILL bridge and simple square boundary are essential here.')
    (OUT/'elliptic_metrics.json').write_text(json.dumps(result,indent=2))
    model,b=models[-1],states[-1]
    x=np.linspace(0,np.pi,61);xx,yy=np.meshgrid(x,x,indexing='ij')
    grid=np.column_stack([xx.ravel(),yy.ravel()])
    p,_=ridge_tables(grid,model['mn'],model['enc'],False)
    field=(p@b).reshape(xx.shape)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    im=axs[0].pcolormesh(xx,yy,field,shading='auto',cmap='viridis')
    fig.colorbar(im,ax=axs[0],label='u(x,y)')
    axs[0].set(title='Nonlinear steady solution from forcing + BC',xlabel='x',ylabel='y')
    for row in rows:
        axs[1].semilogy(row['residual_history'],label=f"K={row['K']}, shift=5")
    axs[1].semilogy(negative['residual_history'],'--',color='#bb4444',label='K=12, shift=0 (fails)')
    axs[1].set(title='Choice of PDE preconditioner matters',xlabel='Iteration',ylabel='Projected relative PDE residual',ylim=(1e-14,1e4))
    axs[1].legend(fontsize=8);axs[1].grid(alpha=.2)
    axs[2].semilogy([r['K'] for r in rows],[r['relative_l2'] for r in rows],'-o',label='Solution error')
    axs[2].semilogy([r['K'] for r in rows],[r['offgrid_relative_residual'] for r in rows],'-s',label='Off-grid PDE residual')
    axs[2].set(title='Independent spatial refinement',xlabel='Sine cutoff per coordinate K',ylabel='Relative L2 norm')
    axs[2].legend(fontsize=8);axs[2].grid(alpha=.2)
    fig.savefig(OUT/'elliptic_comparison.png',dpi=165)
    np.savez_compressed(OUT/'elliptic_state.npz',b=b,mn=model['mn'],centers=model['enc'].centers,
                        gamma=model['enc'].gamma,profile_weights=model['enc'].weights,
                        profile_bias=model['enc'].bias,x=x,field=field)
    print(json.dumps({key:value for key,value in result.items() if key not in ['rows','negative_control']}),flush=True)
    print(json.dumps([{key:val for key,val in row.items() if key!='residual_history'} for row in rows]),flush=True)

if __name__=='__main__':
    run()
