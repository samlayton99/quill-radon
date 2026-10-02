"""IC/operator -> explicit tanh readouts: no fitting or optimization.

Advection/Airy are fixed two-input tanh MLPs. Heat examples are spatial
ridge networks whose readouts follow an explicit time law, NOT fixed
space-time tanh MLPs. Exact solutions are used only by verification routines.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-f19-linear-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/linear'

def rel(y,t): return float(np.linalg.norm(y-t)/np.linalg.norm(t))
def rms(y): return float(np.sqrt(np.mean(np.asarray(y)**2)))
def tder(z,n):
    t=np.tanh(z)
    if n==0:return t
    s=1-t*t
    if n==1:return s
    if n==2:return -2*t*s
    if n==3:return -2*s*(1-3*t*t)
    raise ValueError(n)

def profile_encode(fun,derivative,h,lo,hi,corrected=True):
    gamma=.25/h; a=np.pi/(2*gamma)
    # 25 inverse widths: suppress truncation for nondecaying sine profiles.
    centers=h*np.arange(np.floor((lo-25/gamma)/h),np.ceil((hi+25/gamma)/h)+1)
    density=np.imag(fun(centers+1j*a))/a if corrected else derivative(centers)
    coef=h/2*density
    bias=float(fun(0)-coef@np.tanh(-gamma*centers))
    return dict(centers=centers,gamma=gamma,coef=coef,bias=bias)

def profile_eval(s,p,n=0):
    return (p['bias'] if n==0 else 0)+p['gamma']**n*tder(p['gamma']*(s[:,None]-p['centers']),n)@p['coef']

def two_input_tests():
    xy=qmc.Sobol(2,scramble=True,seed=5719).random_base2(11)
    x=2*xy[:,0]-1;t=xy[:,1]
    xi=np.linspace(-1,1,1001)
    rows=[]; saved={}
    for name in ['advection','airy','heat']:
      for h in [.12,.08,.05,.03,.02]:
       for corrected in [False,True]:
        start=time.monotonic();y=np.zeros(len(x));ut=y.copy();ux=y.copy();uxx=y.copy();uxxx=y.copy();ic=np.zeros(len(xi));models=[]
        if name=='advection':
            f=lambda s:np.exp(-12*(s+.5)**2)
            fp=lambda s:-24*(s+.5)*f(s)
            p=profile_encode(f,fp,h,-2,1,corrected);models=[p]
            y=profile_eval(x-t,p);ux=profile_eval(x-t,p,1);ut=-ux
            ic=profile_eval(xi,p);truth=f(x-t);ictruth=f(xi);res=ut+ux;scale=rms(ux)
            arch='fixed two-input tanh MLP, one direction'
        elif name=='airy':
            truth=np.zeros(len(x));ictruth=np.zeros(len(xi))
            for amp,k in [(1.,1.5),(.5,2.2)]:
                f=lambda s,a=amp:a*np.sin(s)
                fp=lambda s,a=amp:a*np.cos(s)
                p=profile_encode(f,fp,h,-k,k+k**3,corrected);models.append(p)
                s=k*x+k**3*t;y+=profile_eval(s,p);ut+=k**3*profile_eval(s,p,1);uxxx+=k**3*profile_eval(s,p,3)
                ic+=profile_eval(k*xi,p);truth+=amp*np.sin(s);ictruth+=amp*np.sin(k*xi)
            res=ut+uxxx;scale=rms(ut);arch='fixed two-input tanh MLP, two directions'
        else:
            truth=np.zeros(len(x));ictruth=np.zeros(len(xi))
            for amp,k in [(1.,np.pi),(.4,2*np.pi)]:
                f=lambda s,a=amp,k=k:a*np.sin(k*s)
                fp=lambda s,a=amp,k=k:a*k*np.cos(k*s)
                p=profile_encode(f,fp,h,-1,1,corrected);models.append(p)
                decay=np.exp(-.15*k*k*t);piece=decay*profile_eval(x,p)
                y+=piece;ut+=-.15*k*k*piece;uxx+=decay*profile_eval(x,p,2)
                ic+=profile_eval(xi,p);truth+=amp*decay*np.sin(k*x);ictruth+=amp*np.sin(k*xi)
            res=ut-.15*uxx;scale=rms(ut);arch='spatial tanh network with explicit exponential time readouts'
        row=dict(problem=name,h=h,gamma=.25/h,corrected=corrected,directions=len(models),neurons=sum(len(p['centers']) for p in models),rel_l2=rel(y,truth),max_abs=float(np.max(np.abs(y-truth))),ic_rel_l2=rel(ic,ictruth),pde_rms=rms(res),pde_scaled_rms=rms(res)/max(scale,1e-30),architecture=arch,seconds=time.monotonic()-start)
        rows.append(row);print(row,flush=True)
        if h==.02 and corrected:
            saved[name]=dict(x=x,t=t,pred=y,truth=truth,residual=res)
            np.savez_compressed(OUT/f'{name}_network.npz',**{f'{key}_{i}':val for i,p in enumerate(models) for key,val in p.items()})
    return rows,saved

def sphere_rule(nz):
    z,wz=leggauss(nz);ph=2*np.pi*np.arange(2*nz)/(2*nz)
    zz,pp=np.meshgrid(z,ph,indexing='ij');r=np.sqrt(1-zz*zz)
    v=np.stack([r*np.cos(pp),r*np.sin(pp),zz],axis=-1).reshape(-1,3)
    w=np.broadcast_to(wz[:,None]/(4*nz),zz.shape).ravel();keep=v[:,2]>0
    return v[keep],2*w[keep]

def hermite(n,x):
    h0=np.ones_like(x)
    if n==0:return h0
    h1=2*x
    for k in range(1,n):h0,h1=h1,2*x*h1-2*k*h0
    return h1

def heat_profile(s,v,t,components,D,n=0):
    """Filtered Radon profile q and nth offset derivative, evolved from IC.
    R u_t = (v^T D v) (R u)_ss, hence q obeys the same one-dimensional law.
    No solved interior field is used here.
    """
    out=np.zeros(np.broadcast_shapes(s.shape,(len(v),1)),dtype=np.result_type(s,float))
    for amp,A,mu in components:
        cov=np.linalg.inv(A)+4*t*D
        sv=np.einsum('mi,ij,mj->m',v,cov,v)[:,None]
        y=s-(v@mu)[:,None];z=y/np.sqrt(sv)
        fac=-amp/(2*np.sqrt(np.linalg.det(A))*np.sqrt(sv))*((-1)**n)*sv**(-(n+2)/2)
        out+=fac*hermite(n+2,z)*np.exp(-z*z)
    return out

def heat_encode(v,w,c,gamma,t,components,D,corrected=True):
    h=c[1]-c[0];a=np.pi/(2*gamma);dv=np.einsum('mi,ij,mj->m',v,D,v)
    if corrected:
        density=np.imag(heat_profile(c[None,:]+1j*a,v,t,components,D))/a
        dt_density=dv[:,None]*np.imag(heat_profile(c[None,:]+1j*a,v,t,components,D,2))/a
    else:
        density=heat_profile(c[None,:],v,t,components,D,1)
        dt_density=dv[:,None]*heat_profile(c[None,:],v,t,components,D,3)
    coef=h/2*w[:,None]*density;dtcoef=h/2*w[:,None]*dt_density
    b=w@heat_profile(np.zeros((len(v),1)),v,t,components,D)[:,0]-np.sum(coef*np.tanh(-gamma*c))
    bt=(w*dv)@heat_profile(np.zeros((len(v),1)),v,t,components,D,2)[:,0]-np.sum(dtcoef*np.tanh(-gamma*c))
    return coef,float(b),dtcoef,float(bt)

def heat_eval(x,v,c,gamma,coef,b,dtcoef,bt,D):
    y=np.full(len(x),b);yt=np.full(len(x),bt);lap=np.zeros(len(x));dv=np.einsum('mi,ij,mj->m',v,D,v)
    for st in range(0,len(v),8):
        sl=slice(st,st+8);s=x@v[sl].T
        z=gamma*(s[:,:,None]-c[None,None,:]);phi=tder(z,0)
        y+=np.einsum('nmc,mc->n',phi,coef[sl]);yt+=np.einsum('nmc,mc->n',phi,dtcoef[sl])
        lap+=gamma**2*np.einsum('nmc,mc,m->n',tder(z,2),coef[sl],dv[sl])
    return y,yt,lap

def heat_exact(x,t,components,D):
    # Verification only: physical-space Gaussian heat semigroup.
    y=np.zeros(len(x));yt=y.copy()
    for amp,A,mu in components:
        cov=np.linalg.inv(A)+4*t*D;B=np.linalg.inv(cov);r=x-mu
        part=amp/np.sqrt(np.linalg.det(A)*np.linalg.det(cov))*np.exp(-np.einsum('ni,ij,nj->n',r,B,r))
        y+=part;yt+=part*(-2*np.trace(B@D)+4*np.einsum('ni,ij,jk,kl,nl->n',r,B,D,B,r))
    return y,yt

def heat3d_tests():
    # A non-axis-aligned SPD diffusivity and two nonradial IC components.
    D=np.array([[.12,.025,-.01],[.025,.08,.015],[-.01,.015,.16]])
    A=np.array([[3.,.7,.2],[.7,5.,-.4],[.2,-.4,2.]])
    components=[(1.,A,np.array([.12,-.15,.09])),(-.3,2.2*np.eye(3),np.array([-.21,.17,-.08]))]
    x=1.2*(qmc.Sobol(3,scramble=True,seed=5731).random_base2(7)-.5)
    rows=[];times=[0.,.25,1.];saved={}
    settings=[(nz,.04) for nz in [4,8,12,20,32]]+[(32,h) for h in [.16,.08,.02]]
    for nz,h in settings:
      v,w=sphere_rule(nz);gamma=.25/h
      c=h*np.arange(-np.ceil(5/h),np.ceil(5/h)+1)
      for corrected in ([False,True] if (nz,h)==(32,.04) else [True]):
       start=time.monotonic();errs=[];ress=[];tab=[]
       for t in times:
        coef,b,ct,bt=heat_encode(v,w,c,gamma,t,components,D,corrected)
        y,yt,lap=heat_eval(x,v,c,gamma,coef,b,ct,bt,D);truth,timet=heat_exact(x,t,components,D)
        errs.append(rel(y,truth));ress.append(rms(yt-lap)/max(rms(timet),1e-30));tab.append(dict(time=t,rel_l2=errs[-1],pde_rms=rms(yt-lap),pde_scaled_rms=ress[-1],time_derivative_rel=rel(yt,timet)))
       row=dict(problem='3D anisotropic heat',nz=nz,directions=len(v),centers=len(c),neurons=len(v)*len(c),h=h,gamma=gamma,corrected=corrected,max_rel_l2=max(errs),ic_rel_l2=errs[0],max_scaled_pde_residual=max(ress),times=tab,seconds=time.monotonic()-start)
       rows.append(row);print(row,flush=True)
       if nz==32 and h==.04 and corrected:
        saved=dict(v=v,w=w,c=c,gamma=gamma,D=D,components=components)
        np.savez_compressed(OUT/'heat3d_network_t1.npz',v=v,w=w,c=c,gamma=gamma,coef=coef,bias=b,dtcoef=ct,dtbias=bt,D=D,x=x,pred=y,truth=truth)
    # Independent finite differences verify implemented analytic network time derivative.
    v,w,c,gamma=saved['v'],saved['w'],saved['c'],saved['gamma'];t=.31;delta=1e-5;xx=x[:12]
    enc=heat_encode(v,w,c,gamma,t,components,D);mid=heat_eval(xx,v,c,gamma,*enc,D)
    minus=heat_eval(xx,v,c,gamma,*heat_encode(v,w,c,gamma,t-delta,components,D),D)[0]
    plus=heat_eval(xx,v,c,gamma,*heat_encode(v,w,c,gamma,t+delta,components,D),D)[0]
    check=rel((plus-minus)/(2*delta),mid[1])
    print('analytic network dt vs central finite difference',check,flush=True)
    return rows,saved,check

def figures(one,save,heat,hs):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    for a,name in zip(ax,['advection','airy','heat']):
      for cor,col,lab in [(False,'#888888','Derivative samples'),(True,'#2464ab','Width-corrected')]:
        rr=[r for r in one if r['problem']==name and r['corrected']==cor]
        a.loglog([r['neurons'] for r in rr],[r['rel_l2'] for r in rr],'o-',c=col,label=lab)
        if cor:a.loglog([r['neurons'] for r in rr],[max(r['pde_scaled_rms'],1e-17) for r in rr],'s--',c='#b64232',label='Scaled PDE residual')
      a.set(title=name.capitalize()+(' (time readouts)' if name=='heat' else ' (fixed 2-input MLP)'),xlabel='Tanh neurons',ylabel='Relative error / scaled residual');a.grid(alpha=.2);a.legend(fontsize=8)
    fig.suptitle('Initial condition + PDE → coefficients; no training, no least squares')
    fig.savefig(OUT/'linear_ivp_convergence.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    rr=[r for r in heat if r['h']==.04 and r['corrected']]
    ax[0].loglog([r['directions'] for r in rr],[r['max_rel_l2'] for r in rr],'o-',label='Max solution error over 3 times')
    ax[0].loglog([r['directions'] for r in rr],[r['max_scaled_pde_residual'] for r in rr],'s--',label='Max scaled PDE residual')
    ax[0].set(xlabel='Directions',ylabel='Relative error',title='Angular refinement, h = 0.04');ax[0].legend(fontsize=8)
    rr=sorted([r for r in heat if r['nz']==32 and r['corrected']],key=lambda r:r['centers'])
    ax[1].loglog([r['centers'] for r in rr],[r['max_rel_l2'] for r in rr],'o-',label='Solution')
    ax[1].loglog([r['centers'] for r in rr],[r['max_scaled_pde_residual'] for r in rr],'s--',label='PDE residual')
    ax[1].set(xlabel='Centers per direction',ylabel='Relative error',title='Offset refinement, 1024 directions');ax[1].legend(fontsize=8)
    v,w,c,gamma=hs['v'],hs['w'],hs['c'],hs['gamma'];z=np.linspace(-.6,.6,151);xx=np.column_stack([z,.1*np.ones_like(z),np.zeros_like(z)])
    for t,col in [(0.,'#2464ab'),(.25,'#b64232'),(1.,'#228050')]:
        y=heat_eval(xx,v,c,gamma,*heat_encode(v,w,c,gamma,t,hs['components'],hs['D']),hs['D'])[0]
        truth=heat_exact(xx,t,hs['components'],hs['D'])[0]
        ax[2].plot(z,truth,c=col,label=f't = {t:g}');ax[2].plot(z[::5],y[::5],'o',ms=3,mfc='none',c=col)
    ax[2].set(xlabel='x (y = 0.1, z = 0)',ylabel='u',title='3D anisotropic heat: lines exact, dots network');ax[2].legend(fontsize=8)
    for a in ax:a.grid(alpha=.2)
    fig.savefig(OUT/'radon_heat3d.png',dpi=170);plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    one,save=two_input_tests();heat,hs,check=heat3d_tests();figures(one,save,heat,hs)
    metrics=dict(protocol='Only PDE operator and supplied analytic initial condition enter encoding. No interior labels, least squares, or training. Exact solutions verify only.',one_dimensional_ivps=one,heat3d=heat,heat3d_network_dt_finite_difference_relative_error=check,elapsed_seconds=time.monotonic()-start)
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('DONE seconds',metrics['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
