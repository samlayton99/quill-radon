"""Numerically evolve sampled initial Radon profiles, then compile to tanh.

Only q(s,0) is sampled from the supplied initial condition. The algorithm
has no analytic q(s,t) calls after initialization: an FFT, heat semigroup
multipliers and an IFFT give the evolving coefficient density. It is a
classical one-dimensional spectral PDE solve plus explicit neural encoding.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import time,json
import numpy as np
from scipy.stats import qmc
from linear import OUT,sphere_rule,heat_profile,heat_eval,heat_exact,rel,rms

def numerical_encode(v,w,c,gamma,qhat,omega,dv,t,corrected=True):
    evolve=np.exp(-t*dv[:,None]*omega[None,:]**2)
    evolved=qhat*evolve;evdt=-dv[:,None]*omega[None,:]**2*evolved
    a=np.pi/(2*gamma);arg=a*omega
    factor=np.ones_like(arg);nz=arg!=0
    if corrected:factor[nz]=np.sinh(arg[nz])/arg[nz]
    multiplier=1j*omega*factor
    density=np.fft.ifft(evolved*multiplier,axis=1).real
    dt_density=np.fft.ifft(evdt*multiplier,axis=1).real
    qzero=np.fft.ifft(evolved,axis=1).real[:,len(c)//2]
    qtzero=np.fft.ifft(evdt,axis=1).real[:,len(c)//2]
    h=c[1]-c[0];coef=h/2*w[:,None]*density;ct=h/2*w[:,None]*dt_density
    b=w@qzero-np.sum(coef*np.tanh(-gamma*c));bt=w@qtzero-np.sum(ct*np.tanh(-gamma*c))
    return coef,b,ct,bt

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    D=np.array([[.12,.025,-.01],[.025,.08,.015],[-.01,.015,.16]])
    A=np.array([[3.,.7,.2],[.7,5.,-.4],[.2,-.4,2.]])
    components=[(1.,A,np.array([.12,-.15,.09])),(-.3,2.2*np.eye(3),np.array([-.21,.17,-.08]))]
    x=1.2*(qmc.Sobol(3,scramble=True,seed=5771).random_base2(7)-.5)
    rows=[]
    for nz in [12,20,32]:
      v,w=sphere_rule(nz);dv=np.einsum('mi,ij,mj->m',v,D,v)
      for N in [256,512,1024]:
        c=np.linspace(-8,8,N,endpoint=False);h=c[1]-c[0];gamma=.25/h
        omega=2*np.pi*np.fft.fftfreq(N,d=h)
        initial=heat_profile(c[None,:],v,0.,components,D)
        rawhat=np.fft.fft(initial,axis=1)
        # Fixed numerical precision rule, no truth-based width/cutoff selection.
        mask=np.abs(rawhat)>1e-14*np.max(np.abs(rawhat),axis=1,keepdims=True)
        qhat=np.where(mask,rawhat,0)
        for t in [0.,.25,1.]:
            enc=numerical_encode(v,w,c,gamma,qhat,omega,dv,t)
            y,yt,lap=heat_eval(x,v,c,gamma,*enc,D)
            truth,td=heat_exact(x,t,components,D)
            row=dict(nz=nz,directions=len(v),centers=N,neurons=N*len(v),h=h,gamma=gamma,time=t,rel_l2=rel(y,truth),scaled_pde_residual=rms(yt-lap)/rms(td),retained_frequency_max=float(np.max(np.abs(omega[np.any(mask,axis=0)]))))
            rows.append(row);print(row,flush=True)
    metrics=dict(protocol='Sample only initial filtered Radon profiles; FFT discretization; exact heat multiplier of discretized operator; explicit tanh conversion; never read analytic interior profiles.',profile_period=16.,cutoff_relative=1e-14,rows=rows,seconds=time.monotonic()-start)
    (OUT/'numerical_evolution_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('DONE',metrics['seconds'],flush=True)
if __name__=='__main__':main()
