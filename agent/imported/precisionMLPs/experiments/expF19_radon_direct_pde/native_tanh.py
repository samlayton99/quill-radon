"""Native QUILL dynamics: theta'=E A F(S theta), no solution-oracle input.

S evaluates the ACTUAL corrected tanh network and its analytic derivatives.
A is explicit real trigonometric quadrature. E is QUILL's explicit encoding
of fixed entire sin/cos functions, built before dynamics, with no linear solve.
The method is a modified spectral collocation scheme, not a claimed new class
of numerical PDE algorithms. No FFT, least squares or optimizer in native loop.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-native-quill-mpl')
from pathlib import Path
import json,time
import numpy as np
from quill_boundary import encode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/native_dynamics'

def ic(x):return np.sin(x)+.2*np.cos(2*x)

def modes(x,k):
    x=np.asarray(x)
    z=x[...,None]*np.arange(1,k+1)
    return np.concatenate([np.ones(x.shape+(1,)),np.cos(z),np.sin(z)],axis=-1)

def features(x,enc):
    z=enc.gamma*(np.asarray(x)[:,None]-enc.centers)
    t=np.tanh(z)
    r=np.exp(-2*np.abs(z));s=4*r/(1+r)**2
    return (np.column_stack([t,np.ones(len(x))]),
            np.column_stack([enc.gamma*s,np.zeros(len(x))]),
            np.column_stack([-2*enc.gamma**2*t*s,np.zeros(len(x))]))

def construct(k,n,lam=.25):
    began=time.perf_counter()
    # N is interior centers incl endpoints of [-pi,pi]; halos are additional.
    enc=encode(lambda z:modes(z,k),n-1,lam,interval=(-np.pi,np.pi),
               halo=int(np.ceil(np.sqrt(n))))
    e=np.vstack([enc.weights,enc.bias])
    q=4*k+4
    x=-np.pi+2*np.pi*np.arange(q)/q
    a=modes(x,k).T/q;a[1:]*=2
    phi,dx,dxx=features(x,enc)
    c=e@a
    theta=c@ic(x)
    return dict(enc=enc,E=e,A=a,C=c,phi=phi,dx=dx,dxx=dxx,x=x,
                theta0=theta,k=k,n=n,lam=lam,Q=q,
                setup_seconds=time.perf_counter()-began)

def rk4(rhs,u,tend,dt):
    steps=int(np.ceil(tend/dt));dt=tend/steps
    for _ in range(steps):
        a=rhs(u);b=rhs(u+.5*dt*a);c=rhs(u+.5*dt*b);d=rhs(u+dt*c)
        u=u+dt/6*(a+2*b+2*c+d)
    return u,steps,dt

def native(model,tend=.5,nu=.05,dt=.001):
    phi,dx,dxx,c=(model[key] for key in ['phi','dx','dxx','C'])
    def rhs(theta):
        u=phi@theta
        return c@(-u*(dx@theta)+nu*(dxx@theta))
    began=time.perf_counter()
    theta,steps,dt=rk4(rhs,model['theta0'].copy(),tend,dt)
    elapsed=time.perf_counter()-began
    return theta,dict(seconds=elapsed,steps=steps,dt=dt)

def quadrature_baseline(model,tend=.5,nu=.05,dt=.001):
    """Same real trig quadrature, actual sin/cos derivatives instead of QUILL."""
    x,k,a=model['x'],model['k'],model['A']
    w=np.arange(1,k+1);z=x[:,None]*w
    p=modes(x,k)
    d=np.column_stack([np.zeros(len(x)),-w*np.sin(z),w*np.cos(z)])
    dd=np.column_stack([np.zeros(len(x)),-w*w*np.cos(z),-w*w*np.sin(z)])
    rhs=lambda b:a@(-(p@b)*(d@b)+nu*(dd@b))
    began=time.perf_counter()
    b,steps,dt=rk4(rhs,a@ic(x),tend,dt)
    return b,time.perf_counter()-began

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    x=np.linspace(-np.pi,np.pi,4097)
    histories=[];models=[]
    for k,n in [(8,129),(16,257),(32,513),(64,1025)]:
        model=construct(k,n)
        theta,stats=native(model)
        phi,dx,dxx=features(x,model['enc'])
        values=phi@theta
        histories.append(dict(k=k,N_interior=n,R=model['enc'].halo_per_side,
                              neurons=len(model['enc'].centers),Q=model['Q'],lam=.25,
                              setup_seconds=model['setup_seconds'],**stats))
        models.append((model,theta,values))
        print(json.dumps(histories[-1]),flush=True)
    # References are computed only after ALL native solutions are complete.
    from native_burgers import fft_baseline,eval_fourier
    reference,meta=fft_baseline(768,end=.5,nu=.05,dt=5e-5)
    truth=eval_fourier(reference,x)
    for row,(model,theta,values) in zip(histories,models):
        base,btime=quadrature_baseline(model)
        exact_modes=modes(x,model['k'])@base
        fft_ref,fft_meta=fft_baseline(model['Q'],end=.5,nu=.05,dt=.001)
        fft_values=eval_fourier(fft_ref,x)
        error=values-truth
        f,fx,fxx=features(x,model['enc'])
        nodal_u=model['phi']@theta
        nodal_rhs=-nodal_u*(model['dx']@theta)+.05*(model['dxx']@theta)
        theta_dot=model['C']@nodal_rhs
        physical_rhs=-values*(fx@theta)+.05*(fxx@theta)
        residual=f@theta_dot-physical_rhs
        heat_eigenvalues=np.linalg.eigvals(.05*model['A']@model['dxx']@model['E'])
        row.update(relative_l2=float(np.linalg.norm(error)/np.linalg.norm(truth)),
                   max_abs=float(np.max(abs(error))),
                   same_modes_gap=float(np.linalg.norm(values-exact_modes)/np.linalg.norm(truth)),
                   same_modes_baseline_error=float(np.linalg.norm(exact_modes-truth)/np.linalg.norm(truth)),
                   same_modes_baseline_seconds=btime,
                   fft_baseline_meta=fft_meta,
                   fft_baseline_error=float(np.linalg.norm(fft_values-truth)/np.linalg.norm(truth)),
                   periodic_mismatch=float(abs(values[0]-values[-1])),
                   energy_initial=float(np.mean(ic(model['x'])**2)/2),
                   energy_final=float(np.mean((model['phi']@theta)**2)/2),
                   offgrid_pde_residual_rms=float(np.sqrt(np.mean(residual**2))),
                   offgrid_pde_residual_relative=float(np.linalg.norm(residual)/np.linalg.norm(physical_rhs)),
                   max_heat_eigenvalue_real=float(np.max(heat_eigenvalues.real)),
                   most_negative_heat_eigenvalue_real=float(np.min(heat_eigenvalues.real)))
    model,theta,values=models[-1]
    refined,stats=native(model,dt=.0005)
    row=histories[-1]
    phi=features(x,model['enc'])[0]
    row['time_halving_difference']=float(np.linalg.norm(phi@(refined-theta))/np.linalg.norm(truth))
    row['refined_time_error']=float(np.linalg.norm(phi@refined-truth)/np.linalg.norm(truth))
    # Verify the native loop invokes none of these solvers/transforms.
    import contextlib
    @contextlib.contextmanager
    def guard():
        replaced=[]
        def forbidden(*a,**kw):raise RuntimeError('Forbidden native solver/FFT')
        for module,names in [(np.linalg,['solve','lstsq','pinv','svd','inv']),
                             (np.fft,['fft','ifft','rfft','irfft','fftn','ifftn'])]:
            for name in names:
                replaced.append((module,name,getattr(module,name)));setattr(module,name,forbidden)
        try:yield
        finally:
            for module,name,old in replaced:setattr(module,name,old)
    with guard():
        native(models[0][0],tend=.002)
    result=dict(rows=histories,nu=.05,T=.5,IC='sin(x)+0.2*cos(2*x)',domain=[-np.pi,np.pi],
                reference=meta,method='theta_dot=E A F(S theta); actual tanh inference each stage',
                no_linalg_or_fft_in_native_loop=True,
                quadrature='Q=4K+4, real trigonometric analysis, sufficient for quadratic spectral dealiasing',
                note='Neural tails outside retained K introduce a small additional projection error; this is a modified spectral method, not a speedup claim.')
    (OUT/'tanh_metrics.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'tanh_trajectory_endpoints.npz',x=x,truth=truth,
                        outputs=np.array([item[2] for item in models]),
                        final_weights=theta[:-1],bias=theta[-1],centers=model['enc'].centers,
                        gamma=model['enc'].gamma)
    plot_saved()
    print(json.dumps(result),flush=True)


def plot_saved():
    result=json.loads((OUT/'tanh_metrics.json').read_text())
    histories=result['rows']
    data=np.load(OUT/'tanh_trajectory_endpoints.npz')
    x,truth,outputs=data['x'],data['truth'],data['outputs']
    fig,ax=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    ax[0].plot(x,ic(x),'--',color='.6',label='Initial condition')
    ax[0].plot(x,truth,'k',lw=2,label='Independent reference at t=.5')
    ax[0].plot(x,outputs[-1],color='#277da8',label='Native QUILL evolution')
    ax[0].set(title='Nonlinear Burgers: evolve network from IC',xlabel='x',ylabel='u(x,t)')
    for row,val in zip(histories,outputs):
        ax[1].semilogy(x,np.maximum(abs(val-truth),1e-17),label=f"K={row['k']}, P={row['neurons']}")
    ax[1].set(title='Off-grid absolute error',xlabel='x',ylabel='Absolute error')
    ax[2].loglog([r['k'] for r in histories],[r['relative_l2'] for r in histories],'-o',label='Native QUILL')
    ax[2].loglog([r['k'] for r in histories],[r['same_modes_baseline_error'] for r in histories],'--x',label='Same retained modes, classical')
    ax[2].set(title='Refinement (RK4 dt=.001)',xlabel='Retained Fourier cutoff K (same for both)',ylabel='Relative L2 error')
    for a in ax:
        a.grid(alpha=.2)
        a.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2))
    fig.savefig(OUT/'tanh_native.png',dpi=175)

if __name__=='__main__':run()
