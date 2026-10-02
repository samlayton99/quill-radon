"""Apply current QUILL halos and measure angular/radial floors independently.

Analytic 3D anisotropic Gaussian on the closed unit ball. No training or LS.
N always counts INTERIOR centers here; n_cells=N-1 in quill_boundary.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-f19-quill-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quill_boundary import encode
from scaling import directions, targets, points

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review'

def sample(n=128, seed=431):
    x=points(3,n,seed)
    # Include a separate surface set and coordinate poles, not only interior.
    surface=points(3,32,seed+77)
    surface=surface[1:]/np.linalg.norm(surface[1:],axis=1)[:,None]
    return np.concatenate([x,surface,np.eye(3),-np.eye(3)])

def relative(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def true_fields(x,b):
    bx=x@b
    f=np.exp(-np.sum(bx*x,axis=1))
    return f,-2*bx*f[:,None],(4*np.sum(bx*bx,axis=1)-2*np.trace(b))*f

def ridge_profile(z,s,pref,derivative=0):
    t=np.asarray(z)[...,None]
    e=pref*np.exp(-t*t/s)
    if derivative==0:
        return e*(1-2*t*t/s)
    if derivative==1:
        return e*(-6*t/s+4*t**3/s**2)
    return e*(-6/s+24*t*t/s**2-8*t**4/s**3)

def cell(m,n,lam,x,b,with_plain=False):
    start=time.perf_counter()
    v,w=directions(3,m)
    s=np.einsum('mi,ij,mj->m',v,np.linalg.inv(b),v)
    pref=np.linalg.det(b)**(-.5)*s**(-1.5)
    enc=encode(lambda z:ridge_profile(z,s,pref),n-1,lam,
               halo=math.ceil(math.sqrt(n)))
    exact=true_fields(x,b)
    continuous=[np.zeros(len(x)),np.zeros_like(x),np.zeros(len(x))]
    actual=[np.full(len(x),w@enc.bias),np.zeros_like(x),np.zeros(len(x))]
    plain=np.full(len(x),w@enc.baseline_bias) if with_plain else None
    weights=enc.weights.T*w[:,None]
    for k in range(0,len(v),16):
        vv,ww=v[k:k+16],w[k:k+16]
        tt=x@vv.T
        ss,pp=s[k:k+16],pref[k:k+16]
        e=pp*np.exp(-tt*tt/ss)
        q=e*(1-2*tt*tt/ss)
        qp=e*(-6*tt/ss+4*tt**3/ss**2)
        qpp=e*(-6/ss+24*tt*tt/ss**2-8*tt**4/ss**3)
        continuous[0]+=q@ww
        continuous[1]+=(qp*ww)@vv
        continuous[2]+=qpp@ww
        z=enc.gamma*(tt[:,:,None]-enc.centers)
        tanh=np.tanh(z)
        sech2=4*np.exp(-2*np.abs(z))/(1+np.exp(-2*np.abs(z)))**2
        aa=weights[k:k+16]
        actual[0]+=np.einsum('pmi,mi->p',tanh,aa)
        per_direction=np.einsum('pmi,mi->pm',sech2,aa)*enc.gamma
        actual[1]+=per_direction@vv
        actual[2]+=np.einsum('pmi,mi->p',-2*tanh*sech2,aa)*enc.gamma**2
        if with_plain:
            plain+=np.einsum('pmi,mi->p',tanh,
                             enc.baseline_weights[:,k:k+16].T*ww[:,None])
    row=dict(M=len(v),N_interior=n,n_cells=n-1,R=enc.halo_per_side,
             per_direction=len(enc.centers),neurons=len(v)*len(enc.centers),lam=lam,
             weight_l1=float(np.sum(abs(weights))),points=len(x),
             lambda_halo_admissible=bool(lam*enc.halo_per_side>=2*np.log(2)))
    for name,a,c,f in zip(['value','gradient','laplacian'],actual,continuous,exact):
        row[name]=relative(a,f)
        row[name+'_angular']=relative(c,f)
        # Common denominator makes angular+conversion a valid upper bound.
        row[name+'_conversion']=float(np.linalg.norm(a-c)/np.linalg.norm(f))
        row[name+'_max_abs']=float(np.max(abs(a-f)))
    row['plain_value']=relative(plain,exact[0]) if with_plain else None
    row['seconds']=time.perf_counter()-start
    return row

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    b=targets(3)['anisotropic']
    x=sample()
    sweep=[]
    began=time.perf_counter()
    for n in [33,65,129,257]:
        for lam in [.12,.15,.18,.20,.25,.30,.35,.40,.50]:
            row=cell(576,n,lam,x,b,True)
            sweep.append(row)
        print(json.dumps({'phase':'lambda','N':n,'best_value':min(sweep[-9:],key=lambda r:r['value']),
                          'best_laplacian':min(sweep[-9:],key=lambda r:r['laplacian'])}),flush=True)
        (OUT/'allocation_lambda.json').write_text(json.dumps(sweep,indent=2))
    # A common lambda is required to isolate M/N floors; .25 remains the
    # conservative plateau choice unless the measured sweep contradicts it.
    lam=.25
    allocation=[]
    for m in [16,36,64,100,144,256,400,576,1024,1600]:
        for n in [33,65,129,257]:
            allocation.append(cell(m,n,lam,x,b))
        print(json.dumps({'phase':'allocation','M':m,'value_errors':[r['value'] for r in allocation[-4:]]}),flush=True)
        (OUT/'allocation_grid.json').write_text(json.dumps(allocation,indent=2))
    # Predict interior grid from independently measured axis floors.
    comparison=[]
    for r in allocation:
        if r['M']==1600 or r['N_interior']==257:
            continue
        for quantity in ['value','gradient','laplacian']:
            angular=next(q[quantity] for q in allocation if q['M']==r['M'] and q['N_interior']==257)
            center=next(q[quantity] for q in allocation if q['M']==1600 and q['N_interior']==r['N_interior'])
            prediction=max(angular,center)
            comparison.append(dict(M=r['M'],N=r['N_interior'],quantity=quantity,
                                   actual=r[quantity],predicted=prediction,
                                   ratio=r[quantity]/prediction))
    (OUT/'allocation_prediction.json').write_text(json.dumps(comparison,indent=2))
    selected=[]
    for budget in [10000,20000,50000,100000,250000]:
        allowed=[r for r in allocation if r['neurons']<=budget]
        for quantity in ['value','laplacian']:
            best=min(allowed,key=lambda r:r[quantity])
            verify=cell(best['M'],best['N_interior'],best['lam'],sample(512,9071),b,True)
            selected.append(dict(budget=budget,criterion=quantity,selection=best,validation=verify))
    (OUT/'allocation_selected.json').write_text(json.dumps(selected,indent=2))
    optimized=[]
    chosen={}
    for quantity in ['value','laplacian']:
        chosen[quantity]={}
        for n in [33,65,129,257]:
            rr=[r for r in sweep if r['N_interior']==n]
            best=min(rr,key=lambda r:r[quantity+'_conversion'])
            # Do not optimize noisy last digits of a value error already at
            # roundoff. Retain .25 if it reaches the same precision plateau.
            default=next(r for r in rr if r['lam']==.25)
            if quantity=='value' and default['value_conversion']<2e-15:
                best=default
            chosen[quantity][n]=best['lam']
            for m in [16,36,64,100,144,256,400,576,1024,1600]:
                prior=next((r for r in allocation if r['M']==m and r['N_interior']==n and r['lam']==best['lam']),None)
                row=dict(prior) if prior else cell(m,n,best['lam'],x,b)
                row['lambda_selection_criterion']=quantity
                optimized.append(row)
    (OUT/'allocation_optimized_grid.json').write_text(json.dumps(optimized,indent=2))
    optimized_selected=[]
    for budget in [10000,20000,50000,100000,250000]:
        for quantity in ['value','laplacian']:
            allowed=[r for r in optimized if r['neurons']<=budget and r['lambda_selection_criterion']==quantity]
            best=min(allowed,key=lambda r:r[quantity])
            verify=cell(best['M'],best['N_interior'],best['lam'],sample(512,9071),b,True)
            optimized_selected.append(dict(budget=budget,criterion=quantity,selection=best,validation=verify))
    (OUT/'allocation_optimized_selected.json').write_text(json.dumps(optimized_selected,indent=2))
    fig,ax=plt.subplots(1,3,figsize=(16,4.6),constrained_layout=True)
    for n in [33,65,129,257]:
        rr=[r for r in sweep if r['N_interior']==n]
        ax[0].semilogy([r['lam'] for r in rr],[r['value_conversion'] for r in rr],'-o',label=f'N={n}')
        ax[1].semilogy([r['lam'] for r in rr],[r['laplacian_conversion'] for r in rr],'-o',label=f'N={n}')
        rr=[r for r in allocation if r['N_interior']==n]
        ax[2].loglog([r['neurons'] for r in rr],[r['value'] for r in rr],'-o',label=f'N={n}')
    for a in ax:
        a.grid(alpha=.25);a.legend(fontsize=9);a.set_ylabel('Relative L2 error')
    ax[0].set(xlabel='lambda = gamma h',title='Value encoding error (angular error removed)')
    ax[1].set(xlabel='lambda = gamma h',title='Laplacian encoding error')
    ax[2].set(xlabel='Total neurons, including sqrt(N) halos',title='M / N allocation, lambda = 0.25')
    fig.suptitle('Explicit QUILL construction; current boundary correction; no least squares')
    fig.savefig(OUT/'allocation_summary.png',dpi=175)
    meta=dict(seconds=time.perf_counter()-began,target_matrix=b.tolist(),
              domain='closed unit ball in R3',N='interior centers',R='ceil(sqrt(N)) per side',
              method='Exact analytic Gaussian ridge profiles + current QUILL contour correction',
              scope='Known-function construction; not an unknown PDE solve',
              floor_prediction='empirical max rule, not an exact theorem; heldout grid excludes calibration axes',
              error_measure='empirical Euclidean norm on uniform-radius points plus sphere points and coordinate poles; not volume-integrated L2',
              validation='fresh larger point set with additional boundary checks; mixture weight of boundary points changes',
              selected_lambdas=chosen)
    (OUT/'allocation_meta.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)

if __name__=='__main__':
    main()
