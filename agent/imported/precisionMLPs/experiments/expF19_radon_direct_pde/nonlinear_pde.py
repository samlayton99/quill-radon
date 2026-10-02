"""No least-squares nonlinear PDE experiments and Fourier-to-tanh export.

Numerical 3D NS evolution is a classical Fourier Galerkin method, not a direct
closed-form PDE solution. Neural snapshots are explicitly encoded afterward.
The Burgers branch uses its Cole--Hopf reduction from the prescribed IC.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-nonlinear-mpl')
from pathlib import Path
import json, time, argparse
import numpy as np
from scipy.special import ive
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/nonlinear'


def rel(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def canonical(k):
    nonzero=np.flatnonzero(k)
    return bool(len(nonzero) and k[nonzero[0]]>0)


def fourier_values(x,k,c,mean=None,derivatives=False):
    """Real field from one member of each conjugate Fourier pair."""
    if mean is None: mean=np.zeros(c.shape[1])
    y=np.broadcast_to(mean,(len(x),len(mean))).copy()
    if derivatives:
        grad=np.zeros((len(x),c.shape[1],x.shape[1])); lap=np.zeros_like(y)
    for j in range(0,len(k),256):
        kk=k[j:j+256]; cc=c[j:j+256]
        e=np.exp(1j*x@kk.T)
        y+=2*np.real(e@cc)
        if derivatives:
            grad+=2*np.real(np.einsum('nm,mo,md->nod',e,cc,1j*kk))
            lap+=2*np.real(e@(-np.sum(kk*kk,axis=1)[:,None]*cc))
    return (y,grad,lap) if derivatives else y


def encode_fourier(k,c,mean,h=.04,gamma=None,domain_halfwidth=np.pi,halo=3.,corrected=True):
    """Direct geometry/readout formula; no sampled targets or linear solves.

    f(x)=mean+sum_g q_g(v_g dot x), q_g(s)=2Re sum a_k exp(i |k| s).
    tanh coefficients=h Re sum i a_k sinh(A |k|)/A exp(i |k| c), A=pi/(2 gamma).
    """
    if gamma is None: gamma=.25/h
    A=np.pi/(2*gamma)
    groups={}
    for i,kk in enumerate(k):
        g=np.gcd.reduce(np.abs(kk).astype(int))
        primitive=tuple((kk/g).astype(int)); groups.setdefault(primitive,[]).append(i)
    bound=domain_halfwidth*np.sqrt(k.shape[1])+halo
    centers=h*np.arange(-int(np.ceil(bound/h)),int(np.ceil(bound/h))+1)
    directions=[]; weights=[]
    for primitive,indices in groups.items():
        idx=np.array(indices); v=np.array(primitive,dtype=float); v/=np.linalg.norm(v)
        rho=np.linalg.norm(k[idx],axis=1)
        multiplier=np.sinh(A*rho)/A if corrected else rho
        coeff=h*np.real(np.exp(1j*centers[:,None]*rho[None,:]) @ (1j*multiplier[:,None]*c[idx]))
        directions.append(v); weights.append(coeff)
    directions=np.array(directions); weights=np.array(weights)
    atzero=mean+2*np.real(np.sum(c,axis=0))
    bias=atzero-np.einsum('mco,c->o',weights,np.tanh(-gamma*centers))
    return dict(directions=directions,centers=centers,gamma=gamma,coefficient=weights,bias=bias)


def eval_network(x,net,derivatives=False):
    v=net['directions']; centers=net['centers']; gamma=net['gamma']; w=net['coefficient']
    y=np.broadcast_to(net['bias'],(len(x),len(net['bias']))).copy()
    if derivatives: grad=np.zeros((len(x),w.shape[-1],x.shape[1])); lap=np.zeros_like(y)
    for j in range(0,len(v),8):
        vv=v[j:j+8]; ww=w[j:j+8]
        t=np.tanh(gamma*(x@vv.T)[:,:,None]-gamma*centers[None,None,:])
        y+=np.einsum('nmc,mco->no',t,ww,optimize=True)
        if derivatives:
            first=gamma*(1-t*t)
            grad+=np.einsum('nmc,mco,md->nod',first,ww,vv,optimize=True)
            second=-2*gamma*t*first
            lap+=np.einsum('nmc,mco->no',second,ww,optimize=True)
    return (y,grad,lap) if derivatives else y


class PeriodicNS:
    """Rotational-form Fourier Galerkin NS; 2/3 dealiasing and RK4."""
    def __init__(self,n,nu=.05):
        self.n=n; self.nu=nu
        one=np.fft.fftfreq(n,1/n)
        self.k=np.array(np.meshgrid(one,one,one,indexing='ij'))
        self.k2=np.sum(self.k*self.k,axis=0)
        self.den=np.where(self.k2>0,self.k2,1.)
        # Strict inequality removes equality-edge alias triads when n is divisible by 3.
        self.mask=np.max(np.abs(self.k),axis=0)<n/3
        axis=2*np.pi*np.arange(n)/n
        x,y,z=np.meshgrid(axis,axis,axis,indexing='ij')
        u=np.array([np.sin(x)*np.cos(y)*np.cos(z),-np.cos(x)*np.sin(y)*np.cos(z),np.zeros_like(x)])
        self.u0=np.fft.fftn(u,axes=(1,2,3))/n**3
    def project(self,h): return h-self.k*np.sum(self.k*h,axis=0)[None,...]/self.den
    def physical(self,h): return np.fft.ifftn(h*self.n**3,axes=(1,2,3)).real
    def rotational(self,h):
        u=self.physical(h)
        curlh=1j*np.cross(self.k,h,axisa=0,axisb=0,axisc=0)
        curl=self.physical(curlh)
        rotational=np.cross(u,curl,axisa=0,axisb=0,axisc=0)
        return np.fft.fftn(rotational,axes=(1,2,3))/self.n**3
    def rhs(self,h): return self.project(self.rotational(h))*self.mask-self.nu*self.k2*h
    def run(self,end=.5,dt=.005):
        start=time.perf_counter(); steps=int(np.ceil(end/dt)); dt=end/steps; h=self.u0.copy()
        history=[]
        for it in range(steps):
            a=self.rhs(h); b=self.rhs(h+dt*a/2); c=self.rhs(h+dt*b/2); d=self.rhs(h+dt*c)
            h+=dt*(a+2*b+2*c+d)/6
            if it%max(1,steps//20)==0 or it==steps-1:
                energy=.5*np.sum(np.abs(h)**2)
                history.append(dict(t=(it+1)*dt,energy=float(energy),div_max=float(np.max(np.abs(np.sum(self.k*h,axis=0))))))
        return h,dict(grid=self.n,dt=dt,steps=steps,elapsed_seconds=time.perf_counter()-start,history=history)
    def modes(self,h,threshold=1e-15):
        k=self.k.reshape(3,-1).T.astype(int); c=h.reshape(h.shape[0],-1).T
        keep=np.array([canonical(kk) for kk in k]) & (np.max(np.abs(c),axis=1)>threshold)
        return k[keep],c[keep],np.real(h[:,0,0,0])


def pad_spectrum(h,new_n):
    old=h.shape[-1]; result=np.zeros((h.shape[0],new_n,new_n,new_n),complex)
    oldk=np.fft.fftfreq(old,1/old).astype(int); idx=oldk%new_n
    result[:,idx[:,None,None],idx[None,:,None],idx[None,None,:]]=h
    return result


def ns_fields(sim,h,threshold=1e-15):
    """u, u_t and physical pressure; pressure products evaluated without aliases."""
    n=sim.n; big=PeriodicNS(2*n,sim.nu); hp=pad_spectrum(h,2*n)
    rot=big.rotational(hp)
    bigu=big.physical(hp)
    kinetic=np.fft.fftn(.5*np.sum(bigu*bigu,axis=0))/(2*n)**3
    pressure=-1j*np.sum(big.k*rot,axis=0)/big.den-kinetic
    pressure[0,0,0]=0
    utp=pad_spectrum(sim.rhs(h),2*n)
    stacked=np.concatenate([hp,utp,pressure[None,...]],axis=0)
    return big.modes(stacked,threshold)


def burgers_solution(x,t,nu=.15,nterms=80):
    """IC sin(x), psi(0)=exp(cos(x)/(2nu)), heat evolved by Bessel Fourier series."""
    k=np.arange(nterms+1)
    a=ive(k,1/(2*nu))*np.exp(-nu*k*k*t)  # harmless positive common scaling
    e=np.exp(1j*np.asarray(x)[:,None]*k[None,:]); factors=np.where(k==0,1.,2.)
    psi=np.real(e@(factors*a))
    px=np.real(e@(factors*a*1j*k)); pxx=np.real(e@(factors*a*(-k*k)))
    pxxx=np.real(e@(factors*a*(-1j*k**3)))
    u=-2*nu*px/psi
    ux=-2*nu*(pxx/psi-(px/psi)**2)
    uxx=-2*nu*(pxxx/psi-3*px*pxx/psi**2+2*(px/psi)**3)
    ut=-2*nu**2*(pxxx/psi-px*pxx/psi**2)
    return np.stack([u,ut,ux,uxx],axis=1)


def burgers_rk_reference(n=256,t=.8,nu=.15,dt=.0005):
    x=2*np.pi*np.arange(n)/n; k=np.fft.fftfreq(n,1/n); mask=np.abs(k)<n/3
    h=np.fft.fft(np.sin(x))/n
    def rhs(h):
        u=np.fft.ifft(h*n).real
        return -.5j*k*np.fft.fft(u*u)/n*mask-nu*k*k*h
    steps=int(round(t/dt)); dt=t/steps
    for _ in range(steps):
        a=rhs(h); b=rhs(h+dt*a/2); c=rhs(h+dt*b/2); d=rhs(h+dt*c)
        h+=dt*(a+2*b+2*c+d)/6
    return h


def run_burgers():
    nu=.15; t=.8; x=np.linspace(-np.pi,np.pi,1025); truth=burgers_solution(x,t,nu)
    train_grid=2*np.pi*np.arange(2048)/2048
    field=burgers_solution(train_grid,t,nu)
    hhat=np.fft.fft(field[:,:2],axis=0)/len(train_grid)
    k_all=np.arange(1,256); keep=np.max(np.abs(hhat[k_all]),axis=1)>1e-14
    k=k_all[keep,None]; c=hhat[k[:,0]]; mean=np.real(hhat[0])
    rows=[]; networks=[]
    for h in [.16,.08,.04,.02]:
        net=encode_fourier(k,c,mean,h=h,halo=3.)
        pred,grad,lap=eval_network(x[:,None],net,True)
        residual=pred[:,1]+pred[:,0]*grad[:,0,0]-nu*lap[:,0]
        row=dict(h=h,gamma=.25/h,centers=len(net['centers']),neurons=len(net['centers']),directions=1,
                 velocity_rel_l2=rel(pred[:,0],truth[:,0]),physical_residual_rms=float(np.sqrt(np.mean(residual**2))),
                 derivative_rel_l2=rel(grad[:,0,0],truth[:,2]),second_derivative_rel_l2=rel(lap[:,0],truth[:,3]))
        print('BURGERS',row,flush=True); rows.append(row); networks.append(net)
    ref=burgers_rk_reference()
    rk=fourier_values(x[:,None],np.arange(1,128)[:,None],ref[1:128,None],np.real(ref[:1]))[:,0]
    info=dict(nu=nu,t=t,initial_condition='sin(x)',modes=len(k),rows=rows,
              cole_hopf_vs_independent_rk_rel_l2=rel(truth[:,0],rk),
              cole_hopf_physical_residual_rms=float(np.sqrt(np.mean((truth[:,1]+truth[:,0]*truth[:,2]-nu*truth[:,3])**2))))
    np.savez_compressed(OUT/'burgers.npz',x=x,truth=truth,**networks[-1])
    return info,x,truth,networks


def run_ns():
    end=.5; nu=.05; x=2*np.pi*(qmc.Sobol(3,scramble=True,seed=1843).random_base2(9)-.5)
    configs=[(16,.01),(24,.01),(32,.01),(32,.005),(32,.0025),(48,.0025)]
    solves=[]; states={}; preds={}
    for n,dt in configs:
        sim=PeriodicNS(n,nu); state,meta=sim.run(end,dt)
        k,c,mean=sim.modes(state,1e-17)
        preds[(n,dt)]=fourier_values(x,k,c,mean)
        states[(n,dt)]=(sim,state)
        print('NS EVOLVE',n,dt,meta['elapsed_seconds'],flush=True)
        solves.append(meta)
    reference=preds[(48,.0025)]
    for meta in solves:
        key=(meta['grid'],meta['dt']); meta['rel_l2_vs_48']=rel(preds[key],reference)
        print('NS CONVERGENCE',meta['grid'],meta['dt'],meta['rel_l2_vs_48'],flush=True)
    encoding=[]; selected_net=None; field_records=None
    # Only evolution uses FFTs; this evaluation computes the actual tanh MLP.
    for n in [16,24,32]:
        sim,state=states[(n,.0025 if n==32 else .01)]
        k,c,mean=ns_fields(sim,state,threshold=1e-14)
        started=time.perf_counter(); net=encode_fourier(k,c,mean,h=.04,halo=3.)
        pred,grad,lap=eval_network(x[:128],net,True)
        spectral,sg,sl=fourier_values(x[:128],k,c,mean,True)
        residual=pred[:,3:6]+np.einsum('nd,nod->no',pred[:,:3],grad[:,:3,:])+grad[:,6,:]-nu*lap[:,:3]
        sr=spectral[:,3:6]+np.einsum('nd,nod->no',spectral[:,:3],sg[:,:3,:])+sg[:,6,:]-nu*sl[:,:3]
        div=np.trace(grad[:,:3,:],axis1=1,axis2=2)
        row=dict(grid=n,directions=len(net['directions']),centers=len(net['centers']),
                 neurons=len(net['directions'])*len(net['centers']),
                 retained_conjugate_mode_pairs=len(k),
                 velocity_rel_l2_vs_spectral=rel(pred[:,:3],spectral[:,:3]),
                 velocity_rel_l2_vs_48=rel(pred[:,:3],reference[:128]),
                 velocity_gradient_rel_l2=rel(grad[:,:3,:],sg[:,:3,:]),
                 pressure_rel_l2=rel(pred[:,6],spectral[:,6]),
                 divergence_max=float(np.max(np.abs(div))),physical_residual_rms=float(np.sqrt(np.mean(residual**2))),
                 spectral_physical_residual_rms=float(np.sqrt(np.mean(sr**2))),
                 encode_and_128_eval_seconds=time.perf_counter()-started,
                 coefficient_storage_mb=net['coefficient'].nbytes/1e6)
        print('NS ENCODE',row,flush=True); encoding.append(row)
        if n==32: selected_net=net; field_records=(k,c,mean)
    sim,h=states[(32,.0025)]
    energy0=.5*np.sum(np.abs(sim.u0)**2); energy=.5*np.sum(np.abs(h)**2)
    u3_share=float(np.sum(np.abs(h[2])**2)/np.sum(np.abs(h)**2))
    initial_support=np.sum(np.abs(sim.u0)**2,axis=0)>1e-20
    new_energy=float(np.sum(np.abs(h[:,~initial_support])**2)/np.sum(np.abs(h)**2))
    net=selected_net
    np.savez_compressed(OUT/'ns_snapshot.npz',**net,t=end,nu=nu)
    np.savez_compressed(OUT/'ns_spectral_states.npz',state32=h,state48=states[(48,.0025)][1])
    info=dict(nu=nu,t=end,domain='periodic [-pi,pi]^3',initial_condition='(sin x cos y cos z, -cos x sin y cos z, 0)',
              protocol='Unforced IVP. Fourier Galerkin evolution, no interior solution labels, training, or least squares. Explicit neural snapshots, not a fixed four-input network.',
              solves=solves,encoding=encoding,initial_energy=float(energy0),final_energy=float(energy),
              energy_fraction_in_third_velocity=u3_share,energy_fraction_in_new_modes=new_energy)
    return info,x,preds,net,field_records


def plot_results(burgers,xb,tb,nets,ns,x,preds,net,field_records):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(16,4.4),constrained_layout=True)
    ax[0].plot(xb,np.sin(xb),c='.6',label='Initial condition sin x')
    ax[0].plot(xb,tb[:,0],c='black',lw=2,label='Cole–Hopf evolution')
    yn=eval_network(xb[::16,None],nets[-1])[:,0]
    ax[0].plot(xb[::16],yn,'o',ms=3,mfc='none',c='#2865b0',label='Explicit tanh network')
    ax[0].set(xlabel='x',ylabel='u(x,0.8)',title='Nonlinear Burgers, ν = 0.15'); ax[0].legend(fontsize=9)
    br=burgers['rows']; ax[1].loglog([r['neurons'] for r in br],[r['velocity_rel_l2'] for r in br],'o-',label='Burgers solution')
    ax[1].loglog([r['neurons'] for r in br],[r['physical_residual_rms'] for r in br],'s--',label='Burgers PDE residual RMS')
    ax[1].set(xlabel='Neurons (one direction)',ylabel='Error',title='No loss minimization'); ax[1].legend(fontsize=9)
    enc=ns['encoding']; ax[2].semilogy([r['grid'] for r in enc],[r['velocity_rel_l2_vs_48'] for r in enc],'o-',label='3D NS solution vs refined grid')
    ax[2].semilogy([r['grid'] for r in enc],[r['velocity_rel_l2_vs_spectral'] for r in enc],'o-',label='Tanh encoding alone')
    ax[2].semilogy([r['grid'] for r in enc],[r['physical_residual_rms'] for r in enc],'s--',label='Physical PDE residual RMS')
    ax[2].set(xlabel='FFT grid per dimension',ylabel='Error',title='3D Taylor–Green, t = 0.5'); ax[2].legend(fontsize=8)
    for a in ax:a.grid(alpha=.2)
    fig.savefig(OUT/'nonlinear_summary.png',dpi=180); plt.close(fig)
    # A genuinely generated third velocity component: zero initially, nonzero later.
    axis=np.linspace(-np.pi,np.pi,49); xx,yy=np.meshgrid(axis,axis,indexing='ij')
    points=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,np.pi/4)]
    k,c,mean=field_records; truth=fourier_values(points,k,c,mean)[:,2].reshape(xx.shape)
    sample=points[::8]; neural=eval_network(sample,net)[:,2]; reference=fourier_values(sample,k,c,mean)[:,2]
    fig,ax=plt.subplots(1,3,figsize=(16,4.4),constrained_layout=True)
    mesh=ax[0].pcolormesh(axis,axis,truth.T,shading='auto',cmap='RdBu_r');fig.colorbar(mesh,ax=ax[0],label='u₃')
    ax[0].set(xlabel='x',ylabel='y',title='Generated u₃ at z = π/4 (initially zero)')
    ax[1].plot(reference,neural,'.',ms=3,c='#2865b0');lim=np.max(np.abs(reference));ax[1].plot([-lim,lim],[-lim,lim],'k--',lw=1)
    ax[1].set(xlabel='Fourier evolution',ylabel='Explicit tanh network',title='Fresh slice points: actual network')
    for meta in ns['solves']:
        if meta['grid'] in [16,24,32] and meta['dt']==.01:
            hist=meta['history'];ax[2].plot([v['t'] for v in hist],[v['energy'] for v in hist],label=f"{meta['grid']}³ grid")
    ax[2].set(xlabel='Time',ylabel='Kinetic energy / volume',title='Unforced viscous energy decay');ax[2].legend()
    for a in ax:a.grid(alpha=.15)
    fig.savefig(OUT/'navier_stokes_3d.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    b,xb,tb,nets=run_burgers()
    (OUT/'burgers_metrics.json').write_text(json.dumps(b,indent=2)+'\n')
    ns,x,preds,net,records=run_ns()
    (OUT/'ns_metrics.json').write_text(json.dumps(ns,indent=2)+'\n')
    plot_results(b,xb,tb,nets,ns,x,preds,net,records)
    (OUT/'metrics.json').write_text(json.dumps(dict(burgers=b,navier_stokes=ns,elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')
    print('FINISHED',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
