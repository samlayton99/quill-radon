"""QUILL current-paper boundary correction on saved PDE Fourier states.

Encoding-only study. This script never evolves or solves a fluid equation.
N means INTERIOR CENTERS, h=2L/(N-1), R=ceil(sqrt(N)) per side.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-quill-ns-mpl')
from pathlib import Path
from functools import lru_cache
import json,time,math,argparse
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import quill_boundary as qb
from nonlinear_pde import PeriodicNS, ns_fields, burgers_solution, fourier_values, rel
qb.leggauss=lru_cache(maxsize=16)(qb.leggauss)
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/nonlinear'
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review'


class FourierProfile:
    def __init__(self,freq,coeff,mean=None):
        self.freq=np.asarray(freq); self.coeff=np.asarray(coeff)
        self.mean=np.zeros(self.coeff.shape[1]) if mean is None else np.asarray(mean)
    def __call__(self,z,derivative=0):
        z=np.asarray(z); zz=z[...,None]*self.freq
        cc=self.coeff*(1j*self.freq[:,None])**derivative
        y=2*(np.cos(zz)@cc.real-np.sin(zz)@cc.imag)
        if derivative==0:y+=self.mean
        return y


def scalar_sweep():
    x=np.r_[np.linspace(-np.pi,np.pi,1025),-np.pi+np.geomspace(1e-12,.2,40),np.pi-np.geomspace(1e-12,.2,40)]
    grid=2*np.pi*np.arange(2048)/2048
    fh=np.fft.fft(burgers_solution(grid,.8,.15)[:,:2],axis=0)/2048
    kk=np.arange(1,256);keep=np.max(np.abs(fh[kk]),axis=1)>1e-14
    profiles={'tone1':FourierProfile([1],np.array([[-.5j]])),
              'tone8':FourierProfile([8],np.array([[-.5j]])),
              'burgers':FourierProfile(kk[keep],fh[kk[keep]],np.real(fh[0]))}
    rows=[]; start=time.perf_counter()
    for name,f in profiles.items():
      truth=f(x);der=f(x,1);dd=f(x,2)
      for n in [64,96,128,192,256]:
       for lam in [.15,.2,.25,.3,.4]:
        enc=qb.encode(f,n-1,lam,interval=(-np.pi,np.pi),halo=math.ceil(math.sqrt(n)))
        for corrected in [False,True]:
         pred=enc.evaluate(x,corrected=corrected);grad=enc.evaluate(x,1,corrected);lap=enc.evaluate(x,2,corrected)
         w=enc.weights if corrected else enc.baseline_weights
         row=dict(target=name,N_interior=n,R_halo=enc.halo_per_side,total_neurons=len(enc.centers),lam=lam,
                  gamma=enc.gamma,method='corrected' if corrected else 'plain_sqrtN',
                  value_rel_l2=rel(pred[:,0],truth[:,0]),first_derivative_rel_l2=rel(grad[:,0],der[:,0]),
                  second_derivative_rel_l2=rel(lap[:,0],dd[:,0]),max_abs_value_error=float(np.max(np.abs(pred[:,0]-truth[:,0]))),
                  value_endpoint_error=float(np.max(np.abs(enc.evaluate(np.array([-np.pi,np.pi]),corrected=corrected)[:,0]-f(np.array([-np.pi,np.pi]))[:,0]))),
                  weight_l1=float(np.sum(np.abs(w[:,0]))),weight_max=float(np.max(np.abs(w[:,0]))),quadrature=enc.quadrature_order)
         rows.append(row)
      print('SCALAR',name,'done',flush=True)
    (OUT/'ns_scalar_sweep.json').write_text(json.dumps(dict(rows=rows,seconds=time.perf_counter()-start),indent=2)+'\n')
    return rows


def group_profiles(k,c):
    groups={}
    for i,kk in enumerate(k):
        div=np.gcd.reduce(np.abs(kk).astype(int)); primitive=tuple((kk/div).astype(int))
        groups.setdefault(primitive,[]).append(i)
    profiles=[];directions=[]
    for primitive,idx in groups.items():
        v=np.array(primitive,dtype=float);v/=np.linalg.norm(v)
        directions.append(v);profiles.append(FourierProfile(np.linalg.norm(k[idx],axis=1),c[idx]))
    return np.array(directions),profiles


def encode_ns(directions,profiles,mean,n,lam,per_direction=False,quad=None):
    R=math.ceil(math.sqrt(n)); all_weights=[];plain_weights=[];all_centers=[];gammas=[];bias=mean.copy();plain_bias=mean.copy()
    corrections=[]
    start=time.perf_counter()
    for v,f in zip(directions,profiles):
        L=np.pi*(np.sum(abs(v)) if per_direction else np.sqrt(3))
        enc=qb.encode(f,n-1,lam,interval=(-L,L),halo=R,quadrature_order=quad)
        all_weights.append(enc.weights);plain_weights.append(enc.baseline_weights);all_centers.append(enc.centers);gammas.append(enc.gamma)
        bias+=enc.bias;plain_bias+=enc.baseline_bias
        corrections.append(np.max(abs(enc.correction_coefficients)))
    return dict(directions=directions,centers=np.array(all_centers),gamma=np.array(gammas),
                weights=np.array(all_weights),plain_weights=np.array(plain_weights),bias=bias,plain_bias=plain_bias,
                seconds=time.perf_counter()-start,max_boundary_correction=float(max(corrections)),N=n,R=R,lam=lam,per_direction=per_direction)


def evaluate_ns(x,net,corrected=True):
    y=np.broadcast_to(net['bias'] if corrected else net['plain_bias'],(len(x),len(net['bias']))).copy()
    v=net['directions'];w=net['weights'] if corrected else net['plain_weights'];c=net['centers'];gamma=net['gamma']
    grad=np.zeros((len(x),w.shape[-1],3));lap=np.zeros_like(y)
    for j in range(0,len(v),8):
        vv=v[j:j+8];ww=w[j:j+8];gg=gamma[j:j+8]
        z=((x@vv.T)[:,:,None]-c[j:j+8][None,:,:])*gg[None,:,None]
        tan=np.tanh(z);e=np.exp(-2*abs(z));sech=4*e/(1+e)**2
        y+=np.einsum('nmc,mco->no',tan,ww,optimize=True)
        grad+=np.einsum('nmc,mco,md->nod',sech*gg[None,:,None],ww,vv,optimize=True)
        lap+=np.einsum('nmc,mco->no',-2*tan*sech*gg[None,:,None]**2,ww,optimize=True)
    return y,grad,lap


def ns_sweep():
    s=PeriodicNS(48,.05);states=np.load(OLD/'ns_spectral_states.npz')
    k,c,mean=ns_fields(s,states['state48'],1e-14)
    directions,profiles=group_profiles(k,c)
    interior=2*np.pi*(qmc.Sobol(3,scramble=True,seed=271).random_base2(7)-.5)
    corners=np.pi*np.array(np.meshgrid([-1,1],[-1,1],[-1,1],indexing='ij')).reshape(3,-1).T
    uv=2*np.pi*(qmc.Sobol(2,scramble=True,seed=292).random_base2(3)-.5)
    faces=[]
    for axis in range(3):
      for side in [-1,1]:
        points=np.zeros((len(uv),3));points[:,axis]=side*np.pi;points[:,[d for d in range(3) if d!=axis]]=uv
        faces.append(points)
    faces=np.concatenate(faces);x=np.concatenate([interior,corners,faces])
    truth,tg,tl=fourier_values(x,k,c,mean,True)
    rows=[];start=time.perf_counter();saved=None
    # Full profile grid is deliberately modest; scalar sweep is the broad screening test.
    settings=[(96,.2,False),(128,.2,False),(128,.25,False),(192,.2,False),(192,.25,False),
              (256,.2,False),(256,.25,False),(128,.2,True),(192,.2,True)]
    for n,lam,per_direction in settings:
        net=encode_ns(directions,profiles,mean,n,lam,per_direction)
        evalstart=time.perf_counter()
        for corrected in [False,True]:
            y,g,l=evaluate_ns(x,net,corrected)
            residual=y[:,3:6]+np.einsum('nd,nod->no',y[:,:3],g[:,:3,:])+g[:,6,:]-.05*l[:,:3]
            residual_ref=truth[:,3:6]+np.einsum('nd,nod->no',truth[:,:3],tg[:,:3,:])+tg[:,6,:]-.05*tl[:,:3]
            velerr=np.linalg.norm(y[:,:3]-truth[:,:3],axis=1)
            w=net['weights'] if corrected else net['plain_weights']
            yf=y[-len(faces):,:3].reshape(3,2,len(uv),3)
            row=dict(N_interior=n,R_halo=net['R'],total_centers=n+2*net['R'],directions=len(directions),
                neurons=len(directions)*(n+2*net['R']),lam=lam,per_direction_interval=per_direction,
                method='corrected' if corrected else 'plain_sqrtN',
                velocity_rel_l2=rel(y[:,:3],truth[:,:3]),interior_velocity_rel_l2=rel(y[:len(interior),:3],truth[:len(interior),:3]),
                pressure_rel_l2=rel(y[:,6],truth[:,6]),gradient_rel_l2=rel(g[:,:3,:],tg[:,:3,:]),laplacian_rel_l2=rel(l[:,:3],tl[:,:3]),
                max_velocity_error=float(np.max(velerr)),max_corner_velocity_error=float(np.max(velerr[len(interior):len(interior)+8])),
                max_face_velocity_error=float(np.max(velerr[-len(faces):])),periodic_velocity_mismatch=float(np.max(abs(yf[:,0]-yf[:,1]))),
                divergence_max=float(np.max(abs(np.trace(g[:,:3,:],axis1=1,axis2=2)))),
                physical_residual_rms=float(np.sqrt(np.mean(residual**2))),encoding_residual_difference_rms=float(np.sqrt(np.mean((residual-residual_ref)**2))),
                velocity_weight_l1=float(np.sum(abs(w[:,:,:3]))),velocity_weight_max=float(np.max(abs(w[:,:,:3]))),
                pressure_weight_max=float(np.max(abs(w[:,:,6]))),max_boundary_correction=net['max_boundary_correction'],
                encoding_seconds=net['seconds'],coefficient_storage_mb=w.nbytes/1e6)
            rows.append(row);print('NS',row,flush=True)
        if n==192 and lam==.2 and not per_direction:
            saved=net
        (OUT/'ns_quill_metrics.json').write_text(json.dumps(dict(rows=rows,seconds=time.perf_counter()-start,
            protocol='Encoding only from saved 48^3 Fourier state; no fluid solve, no training, no regression.',
            N_convention='Interior centers N; n_cells=N-1; R=ceil(sqrt(N)); total=N+2R.',
            points=dict(interior=len(interior),corners=8,faces=len(faces))),indent=2)+'\n')
    # Refine contour quadrature only for selected strongest / largest profiles, not the fluid grid.
    amplitudes=np.array([np.max(abs(p.coeff)) for p in profiles]);indices=np.argsort(amplitudes)[-12:]
    quad=[]
    for idx in indices:
        f=profiles[idx];L=np.pi*np.sqrt(3)
        a=qb.encode(f,191,.2,interval=(-L,L),halo=14)
        b=qb.encode(f,191,.2,interval=(-L,L),halo=14,quadrature_order=4*a.quadrature_order)
        z=np.linspace(-L,L,513)
        quad.append(dict(direction_index=int(idx),max_weight_change=float(np.max(abs(a.weights-b.weights))),
                     max_value_change=float(np.max(abs(a.evaluate(z)-b.evaluate(z)))),
                     max_second_derivative_change=float(np.max(abs(a.evaluate(z,2)-b.evaluate(z,2))))))
    (OUT/'ns_quadrature_check.json').write_text(json.dumps(quad,indent=2)+'\n')
    if saved:
        np.savez_compressed(OUT/'ns_quill_snapshot.npz',**{key:value for key,value in saved.items() if isinstance(value,(np.ndarray,int,float,bool))})
    return rows



def baseline_checks():
    # Match the exact same spectrum and endpoint-inclusive evaluation set.
    s=PeriodicNS(48,.05);states=np.load(OLD/'ns_spectral_states.npz')
    k,c,mean=ns_fields(s,states['state48'],1e-14)
    interior=2*np.pi*(qmc.Sobol(3,scramble=True,seed=271).random_base2(7)-.5)
    corners=np.pi*np.array(np.meshgrid([-1,1],[-1,1],[-1,1],indexing='ij')).reshape(3,-1).T
    uv=2*np.pi*(qmc.Sobol(2,scramble=True,seed=292).random_base2(3)-.5)
    faces=[]
    for axis in range(3):
      for side in [-1,1]:
        points=np.zeros((len(uv),3));points[:,axis]=side*np.pi;points[:,[d for d in range(3) if d!=axis]]=uv
        faces.append(points)
    x=np.concatenate([interior,corners,*faces]);truth,tg,tl=fourier_values(x,k,c,mean,True)
    old=np.load(OLD/'ns_snapshot48.npz')
    net=dict(directions=old['directions'],weights=old['coefficient'],bias=old['bias'],
             gamma=np.full(len(old['directions']),old['gamma']),
             centers=np.broadcast_to(old['centers'],(len(old['directions']),len(old['centers']))))
    y,g,l=evaluate_ns(x,net)
    w=net['weights'];err=np.linalg.norm(y[:,:3]-truth[:,:3],axis=1)
    baseline=dict(method='old_long_halo',neurons=len(old['directions'])*len(old['centers']),centers=len(old['centers']),
             directions=len(old['directions']),lam=.25,velocity_rel_l2=rel(y[:,:3],truth[:,:3]),
             pressure_rel_l2=rel(y[:,6],truth[:,6]),gradient_rel_l2=rel(g[:,:3,:],tg[:,:3,:]),
             laplacian_rel_l2=rel(l[:,:3],tl[:,:3]),max_velocity_error=float(np.max(err)),
             max_corner_velocity_error=float(np.max(err[128:136])),velocity_weight_l1=float(np.sum(abs(w[:,:,:3]))),
             velocity_weight_max=float(np.max(abs(w[:,:,:3]))),coefficient_storage_mb=w.nbytes/1e6)
    dirs,profiles=group_profiles(k,c);compact=encode_ns(dirs,profiles,mean,128,.25)
    np.savez_compressed(OUT/'ns_quill_compact_snapshot.npz',**{key:value for key,value in compact.items() if isinstance(value,(np.ndarray,int,float,bool))})
    # Scalar comparison also uses exactly the same finite Fourier profile.
    from nonlinear_pde import encode_fourier,eval_network
    xb=np.r_[np.linspace(-np.pi,np.pi,1025),-np.pi+np.geomspace(1e-12,.2,40),np.pi-np.geomspace(1e-12,.2,40)]
    grid=2*np.pi*np.arange(2048)/2048;fh=np.fft.fft(burgers_solution(grid,.8,.15)[:,:2],axis=0)/2048
    kk=np.arange(1,256);keep=np.max(np.abs(fh[kk]),axis=1)>1e-14
    f=FourierProfile(kk[keep],fh[kk[keep]],np.real(fh[0]));truth=f(xb)
    oldnet=encode_fourier(kk[keep,None],fh[kk[keep]],np.real(fh[0]),h=.04,halo=3)
    y,g,l=eval_network(xb[:,None],oldnet,True)
    scalar=dict(neurons=len(oldnet['centers']),value_rel_l2=rel(y[:,0],truth[:,0]),
                first_derivative_rel_l2=rel(g[:,0,0],f(xb,1)[:,0]),second_derivative_rel_l2=rel(l[:,0],f(xb,2)[:,0]))
    result=dict(navier_stokes=baseline,burgers_fourier_profile=scalar)
    (OUT/'ns_baseline_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BASELINE',result,flush=True)



def allocation_check():
    s=PeriodicNS(48,.05);states=np.load(OLD/'ns_spectral_states.npz');k,c,mean=ns_fields(s,states['state48'],1e-14)
    interior=2*np.pi*(qmc.Sobol(3,scramble=True,seed=271).random_base2(7)-.5)
    corners=np.pi*np.array(np.meshgrid([-1,1],[-1,1],[-1,1],indexing='ij')).reshape(3,-1).T
    uv=2*np.pi*(qmc.Sobol(2,scramble=True,seed=292).random_base2(3)-.5);faces=[]
    for axis in range(3):
      for side in [-1,1]:
        points=np.zeros((len(uv),3));points[:,axis]=side*np.pi;points[:,[d for d in range(3) if d!=axis]]=uv;faces.append(points)
    x=np.concatenate([interior,corners,*faces]);truth,tg,tl=fourier_values(x,k,c,mean,True)
    prior=json.loads((OUT/'ns_quill_metrics.json').read_text())['rows']
    nr={r['N_interior']:r for r in prior if r['method']=='corrected' and r['lam']==.2 and not r['per_direction_interval']}
    rows=[];curves=[];compact=None;start=time.perf_counter()
    for tol in [1e-8,1e-10,1e-12]:
        keep=np.max(abs(c[:,[0,1,2,6]]),axis=1)>tol
        dirs,profiles=group_profiles(k[keep],c[keep]);pruned,pg,pl=fourier_values(x,k[keep],c[keep],mean,True)
        mvalue=rel(pruned[:,:3],truth[:,:3]);mlap=rel(pl[:,:3],tl[:,:3])
        curves.append(dict(mode_threshold=tol,directions=len(dirs),direction_floor_velocity=mvalue,direction_floor_laplacian=mlap,
                           uniform_velocity_absolute_tail_bound=float(2*np.sum(np.linalg.norm(c[~keep,:3],axis=1)))))
        for n in [96,128,192]:
            net=encode_ns(dirs,profiles,mean,n,.2);y,g,l=evaluate_ns(x,net)
            ev=rel(y[:,:3],truth[:,:3]);el=rel(l[:,:3],tl[:,:3]);predv=max(mvalue,nr[n]['velocity_rel_l2']);predl=max(mlap,nr[n]['laplacian_rel_l2'])
            residual=y[:,3:6]+np.einsum('nd,nod->no',y[:,:3],g[:,:3,:])+g[:,6,:]-.05*l[:,:3]
            row=dict(mode_threshold=tol,directions=len(dirs),N_interior=n,R=net['R'],neurons=len(dirs)*(n+2*net['R']),lam=.2,
              velocity_rel_l2=ev,laplacian_rel_l2=el,gradient_rel_l2=rel(g[:,:3,:],tg[:,:3,:]),pressure_rel_l2=rel(y[:,6],truth[:,6]),
              physical_residual_rms=float(np.sqrt(np.mean(residual**2))),divergence_max=float(np.max(abs(np.trace(g[:,:3,:],axis1=1,axis2=2)))),
              encoding_velocity_error_common_denominator=float(np.linalg.norm(y[:,:3]-pruned[:,:3])/np.linalg.norm(truth[:,:3])),
              prediction_velocity=predv,prediction_laplacian=predl,velocity_prediction_ratio=ev/predv,laplacian_prediction_ratio=el/predl)
            rows.append(row);print('ALLOCATION',row,flush=True)
            if tol==1e-12 and n==128:compact=net
    # The old64^3reference state is not stored. Use a rigorous empirical triangle bound on the identical old128point set instead of rerunning the PDE.
    old=np.load(OLD/'ns_snapshot48.npz');oldnet=dict(directions=old['directions'],weights=old['coefficient'],bias=old['bias'],
             gamma=np.full(len(old['directions']),old['gamma']),centers=np.broadcast_to(old['centers'],(len(old['directions']),len(old['centers']))))
    oldx=2*np.pi*(qmc.Sobol(3,scramble=True,seed=1843).random_base2(9)-.5)[:128]
    yn=evaluate_ns(oldx,compact)[0][:,:3];yo=evaluate_ns(oldx,oldnet)[0][:,:3];change=rel(yn,yo)
    olderr=json.loads((OLD/'validation_metrics.json').read_text())['encode48']['velocity_rel_l2_vs_64']
    bound=olderr+change*(1+olderr)
    reference=dict(old_network_vs64_measured_error=olderr,compact_vs_old_network=change,
                    compact_vs64_upper_bound=bound,comparison='Triangle-inequality bound on identical128oldheldoutpoints; not a new measurement against64^3.')
    np.savez_compressed(OUT/'ns_quill_balanced_snapshot.npz',**{key:value for key,value in compact.items() if isinstance(value,(np.ndarray,int,float,bool))})
    result=dict(rows=rows,direction_curves=curves,center_curves={n:{k:r[k] for k in ['velocity_rel_l2','laplacian_rel_l2']} for n,r in nr.items()},
                reference64=reference,seconds=time.perf_counter()-start,protocol='Independent full-direction center floors and exact pruned-spectrum direction floors predict heldout crossed M,N settings. Norm denominators use identical full48^3state on samepoints.')
    (OUT/'ns_allocation_check.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def headline_figure():
    allrows=json.loads((OUT/'ns_quill_metrics.json').read_text())['rows']
    old=json.loads((OUT/'ns_baseline_comparison.json').read_text())['navier_stokes']
    compact=next(r for r in allrows if r['method']=='corrected' and not r['per_direction_interval'] and r['N_interior']==128 and r['lam']==.25)
    derivative=next(r for r in allrows if r['method']=='corrected' and not r['per_direction_interval'] and r['N_interior']==192 and r['lam']==.2)
    allocation=json.loads((OUT/'ns_allocation_check.json').read_text())
    balanced=next(r for r in allocation['rows'] if r['N_interior']==128 and r['mode_threshold']==1e-12)
    rows=[old,compact,derivative,balanced]
    names=['Earlier\nlong halo','Compact\nN128, λ=.25','Derivatives\nN192, λ=.20','Balanced M,N\nN128, λ=.20']
    colors=['#888888','#2865b0','#26855b','#b05b24']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    xx=np.arange(4)
    axes[0].bar(xx,[r['neurons']/1000 for r in rows],color=colors)
    for j,r in enumerate(rows):axes[0].text(j,r['neurons']/1000+14,f"{r['neurons']:,}",ha='center',fontsize=9)
    axes[0].set(ylim=(0,1150),ylabel='Neurons (thousands)',title='Same saved48³ flow; explicit readouts')
    for key,style,label in [('velocity_rel_l2','o-','Velocity'),('laplacian_rel_l2','s--','Velocity Laplacian'),('pressure_rel_l2','^:','Pressure')]:
        axes[1].semilogy(xx,[r[key] for r in rows],style,label=label)
    axes[1].set(ylabel='Relative error vs full Fourier state',title='Encoding / pruning error only');axes[1].legend(fontsize=9)
    for j,r in enumerate(allocation['direction_curves']):
        rr=[a for a in allocation['rows'] if a['mode_threshold']==r['mode_threshold']]
        axes[2].loglog([a['neurons'] for a in rr],[a['velocity_rel_l2'] for a in rr],'o-',label=f"M={r['directions']}")
        axes[2].loglog([a['neurons'] for a in rr],[a['prediction_velocity'] for a in rr],':',color=axes[2].lines[-1].get_color())
    axes[2].axhline(3.66e-11,c='.5',ls='--',label='Earlier48³→64³ error (context)')
    axes[2].set(xlabel='Total neurons M(N+2⌈√N⌉)',ylabel='Velocity error',title='Separate floors predict crossed settings')
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:ax.set_xticks(xx,names,fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('N = interior centers; R = ⌈√N⌉ per side; current-paper boundary correction; no fluid re-solve',fontsize=12)
    fig.savefig(OUT/'ns_quill_headline.png',dpi=180);plt.close(fig)


def plots(scalar,ns):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,4.6),constrained_layout=True)
    for method,style in [('plain_sqrtN','--o'),('corrected','-o')]:
        r=[r for r in scalar if r['target']=='burgers' and r['lam']==.2 and r['method']==method]
        axes[0].loglog([q['total_neurons'] for q in r],[q['value_rel_l2'] for q in r],style,label=method)
    axes[0].set(xlabel='Total neurons',ylabel='Relative value error',title='Burgers profile: sqrt(N) halo, λ=.2');axes[0].legend(fontsize=9)
    for lam in [.2,.25]:
      for method,style in [('plain_sqrtN','--o'),('corrected','-o')]:
        r=[r for r in ns if r['lam']==lam and r['method']==method and not r['per_direction_interval']]
        axes[1].loglog([q['neurons'] for q in r],[q['velocity_rel_l2'] for q in r],style,label=f'{method}, λ={lam}')
    matched=json.loads((OUT/'ns_baseline_comparison.json').read_text())['navier_stokes']['velocity_rel_l2']
    axes[1].axhline(matched,c='.5',ls=':',label='Earlier long-halo baseline (same points)')
    axes[1].set(xlabel='Total neurons',ylabel='Velocity encoding relative error',title='Saved 3D NS state; endpoints included');axes[1].legend(fontsize=8)
    for method,style in [('plain_sqrtN','--o'),('corrected','-o')]:
        r=[r for r in ns if r['lam']==.2 and r['method']==method and not r['per_direction_interval']]
        axes[2].loglog([q['neurons'] for q in r],[q['gradient_rel_l2'] for q in r],style,label=f'{method}: gradient')
        axes[2].loglog([q['neurons'] for q in r],[q['laplacian_rel_l2'] for q in r],style,label=f'{method}: Laplacian',alpha=.55)
    axes[2].set(xlabel='Total neurons',ylabel='Derivative encoding relative error',title='Physics derivatives, λ=.2');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'ns_quill_comparison.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','scalar','ns','baseline','allocation','headline','plot'],default='all');args=parser.parse_args()
    if args.stage in ['all','scalar']:scalar_sweep()
    if args.stage in ['all','ns']:ns_sweep()
    if args.stage in ['all','baseline']:baseline_checks()
    if args.stage in ['all','allocation']:allocation_check()
    if args.stage in ['all','headline']:headline_figure()
    if args.stage in ['all','plot']:
        scalar=json.loads((OUT/'ns_scalar_sweep.json').read_text())['rows'];ns=json.loads((OUT/'ns_quill_metrics.json').read_text())['rows'];plots(scalar,ns)
if __name__=='__main__':main()
