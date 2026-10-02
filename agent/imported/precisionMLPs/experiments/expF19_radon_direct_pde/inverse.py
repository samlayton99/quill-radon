"""Small physical inverse problems using explicitly constructed neural forward maps.

Only six diffusion coefficients are optimized; every neural readout is prescribed
by the initial condition and heat equation. A second test exposes backward-heat
instability and uses a measurement-noise discrepancy rule, not truth, to truncate.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-f19-inverse-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from linear import sphere_rule, profile_encode, profile_eval

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/inverse'
PAIRS = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
B0 = np.array([[3., .7, .2],[.7, 5., -.4],[.2, -.4, 2.]])
MU = np.array([.12, -.15, .09])
DTRUE = np.array([[.12,.025,-.01],[.025,.08,.015],[-.01,.015,.16]])

def matrix(p):
    d = np.zeros((3,3))
    for value,(i,j) in zip(p, PAIRS): d[i,j] = d[j,i] = value
    return d

def parameters(d): return np.array([d[i,j] for i,j in PAIRS])
def rel(a,b):
    difference=np.asarray(a)-np.asarray(b)
    scale=float(np.max(np.abs(difference)))
    return 0. if scale==0 else float(scale*np.linalg.norm(difference/scale)/np.linalg.norm(b))

def physical_truth(x,t,d):
    covariance = np.linalg.inv(B0)+4*t*d
    y=x-MU
    return np.exp(-np.einsum('ni,ij,nj->n',y,np.linalg.inv(covariance),y))/np.sqrt(np.linalg.det(B0)*np.linalg.det(covariance))

class ConstructedHeat:
    def __init__(self, x, times, nz=20, h=.05):
        self.v,self.w = sphere_rule(nz)
        self.centers = h*np.arange(-np.ceil(4/h),np.ceil(4/h)+1)
        self.gamma = .25/h
        self.shift = np.pi/(2*self.gamma)
        self.z = self.centers[None,:]+1j*self.shift-(self.v@MU)[:,None]
        self.z0 = -(self.v@MU)[:,None]
        self.s0 = np.einsum('mi,ij,mj->m',self.v,np.linalg.inv(B0),self.v)
        self.ds = np.array([self.v[:,i]*self.v[:,j]*(1 if i==j else 2) for i,j in PAIRS]).T
        self.groups=[]
        for t in np.unique(times):
            ids=np.flatnonzero(times==t)
            phi=np.tanh(self.gamma*((x[ids]@self.v.T)[:,:,None]-self.centers))
            phi-=np.tanh(-self.gamma*self.centers)[None,None,:]
            kernel=phi*(h/(2*self.shift))*self.w[None,:,None]
            self.groups.append((float(t),ids,kernel))
        self.n=len(x)
        self.last=None

    def q(self,z,s):
        r=z*z/s
        base=np.exp(-r)/(np.sqrt(np.linalg.det(B0))*s**1.5)
        return base*(1-2*r),base/s*(-1.5+6*r-2*r*r)

    def both(self,p):
        if self.last is not None and np.array_equal(p,self.last[0]): return self.last[1:]
        dv=self.ds@p
        ans=np.empty(self.n); jac=np.empty((self.n,6))
        for t,ids,kernel in self.groups:
            s=(self.s0+4*t*dv)[:,None]
            q,qs=self.q(self.z,s); q0,q0s=self.q(self.z0,s)
            ans[ids]=np.einsum('pmc,mc->p',kernel,np.imag(q))+self.w@q0[:,0]
            partial=np.einsum('pmc,mc->pm',kernel,np.imag(qs))+(self.w*q0s[:,0])[None,:]
            jac[ids]=partial@(4*t*self.ds)
        self.last=(p.copy(),ans,jac)
        return ans,jac

def tensor_inverse():
    x=1.8*(qmc.Sobol(3,scramble=True,seed=1901).random_base2(6)[:36]-.5)
    times=np.repeat([.1,.35,.7],12)
    true=np.concatenate([physical_truth(x[times==t],t,DTRUE) for t in np.unique(times)])
    rows=[]; saved={}; started=time.monotonic()
    # Independent positive-definiteness guarantee: diagonal >= .065 and
    # each off-diagonal <= .03 implies strict diagonal dominance.
    bounds=(np.array([.065]*3+[-.03]*3),np.array([.3]*3+[.03]*3))
    init=np.array([.1]*3+[0.]*3)
    for nz in [12,20]:
        forward=ConstructedHeat(x,times,nz)
        # Analytic sensitivity validated with an independent centered difference.
        _,j=forward.both(init)
        eps=1e-5
        jfd=np.column_stack([(forward.both(init+eps*np.eye(6)[k])[0]-forward.both(init-eps*np.eye(6)[k])[0])/(2*eps) for k in range(6)])
        for noise in [0.,.01]:
            for seed in (range(8) if noise else [0]):
                rng=np.random.default_rng(1902+seed)
                y=true+noise*np.sqrt(np.mean(true**2))*rng.normal(size=len(true))
                start=time.monotonic()
                fit=least_squares(lambda p:forward.both(p)[0]-y,init,jac=lambda p:forward.both(p)[1],bounds=bounds,xtol=1e-12,ftol=1e-12,gtol=1e-12)
                row=dict(directions=len(forward.v),centers=len(forward.centers),neurons=len(forward.v)*len(forward.centers),noise_relative_rms=noise,seed=seed,recovered_tensor=matrix(fit.x).tolist(),tensor_relative_error=rel(matrix(fit.x),DTRUE),measurement_relative_fit_error=rel(fit.fun+y,y),function_evaluations=fit.nfev,seconds=time.monotonic()-start,sensitivity_check_relative_error=rel(j,jfd),success=bool(fit.success))
                rows.append(row)
                if nz==20 and seed==0:saved[noise]=(fit.x.copy(),y.copy(),forward.both(fit.x)[0].copy())
    xe=1.8*(qmc.Sobol(3,scramble=True,seed=1903).random_base2(8)-.5)
    te=np.repeat([.15,.3,.55,.8],64)
    verifier=ConstructedHeat(xe,te,20)
    ye=np.concatenate([physical_truth(xe[te==t],t,DTRUE) for t in np.unique(te)])
    heldout={str(noise):rel(verifier.both(p)[0],ye) for noise,(p,_,_) in saved.items()}
    result=dict(protocol='36 scalar measurements at three times; known Gaussian initial field; unknown six-entry SPD diffusion tensor. Six-parameter nonlinear least squares only. All neural readouts and sensitivities constructed, never fitted.',true_tensor=DTRUE.tolist(),rows=rows,heldout_solution_errors=heldout,elapsed_seconds=time.monotonic()-started)
    (OUT/'tensor_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'tensor_measurements.npz',x=x,t=times,truth=true,noisy=saved[.01][1],noisy_prediction=saved[.01][2],true_tensor=DTRUE)
    fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    pos=np.arange(6); width=.25
    for shift,label,p,color in [(-width,'True tensor',parameters(DTRUE),'#222222'),(0.,'Noiseless recovery',saved[0.][0],'#3274a1'),(width,'1% noisy observations',saved[.01][0],'#dd8452')]:
        ax[0].bar(pos+shift,p,width,label=label,color=color)
    ax[0].set_xticks(pos,['xx','yy','zz','xy','xz','yz']);ax[0].set(title='Infer six physical numbers',ylabel='Diffusion tensor entry');ax[0].legend(fontsize=8)
    ax[1].plot(true,saved[.01][1],'o',ms=4,label='Noisy measurements',color='#dd8452');ax[1].plot(true,saved[.01][2],'.',label='Constructed neural prediction',color='#3274a1');ax[1].plot([0,1],[0,1],':',c='k')
    ax[1].set(xlabel='Independent exact verifier',ylabel='Measured / predicted value',title='36 observations; no readout fit');ax[1].legend(fontsize=8)
    for noise,color in [(0.,'#3274a1'),(.01,'#dd8452')]:
        vals=[r['tensor_relative_error'] for r in rows if r['directions']==400 and r['noise_relative_rms']==noise]
        ax[2].scatter(np.full(len(vals),noise),vals,color=color)
    ax[2].set_yscale('log');ax[2].set_xticks([0,.01],['No noise','1% RMS noise']);ax[2].set(ylabel='Relative diffusion-tensor error',title='Noise test: eight realizations')
    for a in ax:a.grid(alpha=.2)
    fig.savefig(OUT/'inverse_diffusion.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(inverse_tensor=result),indent=2),flush=True)
    return result

def backward_heat():
    n=128;x=2*np.pi*np.arange(n)/n
    initial=np.sin(x)+.3*np.cos(3*x)+.15*np.sin(6*x)
    nu=.1;t=1.;k=np.fft.fftfreq(n,1/n)
    modes=np.fft.fft(initial)/n; transfer=np.exp(-nu*t*k*k)
    terminal=np.real(np.fft.ifft(n*modes*transfer))
    rng=np.random.default_rng(1930)
    sigma=1e-3*np.sqrt(np.mean(terminal**2))
    noise=sigma*rng.normal(size=n)
    measured=terminal+noise; measured_modes=np.fft.fft(measured)/n
    raw_modes=measured_modes/transfer
    raw=np.real(np.fft.ifft(n*raw_modes))
    # Morozov-style discrepancy threshold: use the prescribed observation-noise
    # standard deviation, never the realized noise or unknown initial field.
    # Choose least retained bandwidth consistent
    # with measurement precision; Fourier tail is the data-space residual.
    threshold=1.1*sigma*np.sqrt(n)
    candidates=[]
    for cutoff in range(1,n//2):
        retained=np.abs(k)<=cutoff
        unexplained=np.real(np.fft.ifft(n*measured_modes*(~retained)))
        candidates.append((cutoff,float(np.linalg.norm(unexplained))))
    cutoff=next(c for c,r in candidates if r<=threshold)
    filtered_modes=np.where(np.abs(k)<=cutoff,raw_modes,0.)
    repaired=np.real(np.fft.ifft(n*filtered_modes))
    # Explicit real Fourier profile continued analytically to complex centers.
    def f(s):
        ans=np.zeros_like(np.asarray(s),dtype=np.result_type(s,float))+filtered_modes[0].real
        for j in range(1,cutoff+1):ans+=2*(filtered_modes[j].real*np.cos(j*s)-filtered_modes[j].imag*np.sin(j*s))
        return ans
    def fp(s):
        ans=np.zeros_like(np.asarray(s),dtype=np.result_type(s,float))
        for j in range(1,cutoff+1):ans+=-2*j*(filtered_modes[j].real*np.sin(j*s)+filtered_modes[j].imag*np.cos(j*s))
        return ans
    net=profile_encode(f,fp,.02,0,2*np.pi,True)
    xx=np.linspace(0,2*np.pi,1001);truth=np.sin(xx)+.3*np.cos(3*xx)+.15*np.sin(6*xx)
    nn=profile_eval(xx,net)
    result=dict(protocol='Periodic 1D backward heat: 128 noisy final-time values, known diffusivity and final time. FFT inversion with discrepancy cutoff 1.1*sigma*sqrt(n), using the specified noise standard deviation, not realized noise or true initial field; explicit tanh readout, no fitting.',nu=nu,time=t,observations=n,noise_standard_deviation=float(sigma),discrepancy_threshold=float(threshold),noise_relative_rms=float(np.linalg.norm(noise)/np.linalg.norm(terminal)),cutoff=cutoff,raw_initial_relative_error=rel(raw,initial),regularized_initial_relative_error=rel(repaired,initial),neural_initial_relative_error=rel(nn,truth),neural_conversion_relative_error=rel(nn,f(xx)),neurons=len(net['centers']),noise_amplification_at_k6=float(1/transfer[6]),noise_amplification_at_k20=float(1/transfer[20]))
    (OUT/'backward_heat_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    ax[0].plot(x,terminal,label='True final state');ax[0].plot(x,measured,'.',ms=2,label='0.1% noisy measurements');ax[0].set(title='Almost indistinguishable observations',xlabel='x');ax[0].legend(fontsize=8)
    ax[1].semilogy(np.arange(n//2),1/transfer[:n//2]);ax[1].axvline(cutoff,ls='--',c='#dd8452',label=f'Noise-selected cutoff {cutoff}');ax[1].set(title='Backward heat amplifies noise',xlabel='Fourier frequency',ylabel='Inverse amplification');ax[1].legend(fontsize=8)
    ax[2].plot(xx,truth,c='#222222',label='True initial condition');ax[2].plot(xx,nn,'--',c='#3274a1',label='Regularized neural reconstruction');ax[2].set(title='Information loss remains',xlabel='x');ax[2].legend(fontsize=8)
    for a in ax:a.grid(alpha=.2)
    fig.savefig(OUT/'inverse_backward_heat.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(backward_heat=result),indent=2),flush=True)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    tensor_inverse();backward_heat()

if __name__=='__main__':main()
