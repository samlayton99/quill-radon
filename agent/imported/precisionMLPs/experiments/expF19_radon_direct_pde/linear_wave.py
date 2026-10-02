"""A fixed four-input tanh MLP for the full-space 3D wave IVP.

Initial displacement -> initial filtered Radon profiles -> traveling profiles
-> explicit fixed spacetime input weights. No training, no time-dependent
readout, no interior solution samples, and no least-squares problem.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
import numpy as np
from scipy.stats import qmc
from linear import OUT,sphere_rule,heat_encode,heat_profile,tder,rel,rms
import matplotlib.pyplot as plt

def wave_eval(x,t,v,c,gamma,coef,b,speed):
    # Equivalent weights: gamma*(v_1,v_2,v_3,+speed) and
    # gamma*(v_1,v_2,v_3,-speed); each has half the initial readout.
    u=np.full(len(x),b);ut=np.zeros(len(x));utt=ut.copy();lap=ut.copy()
    for start in range(0,len(v),8):
        sl=slice(start,start+8);s=x@v[sl].T
        for sign in [-1,1]:
            z=gamma*(s[:,:,None]+sign*speed*t-c[None,None,:])
            u+=.5*np.einsum('nmc,mc->n',tder(z,0),coef[sl])
            ut+=.5*sign*speed*gamma*np.einsum('nmc,mc->n',tder(z,1),coef[sl])
            zz=tder(z,2)
            utt+=.5*speed**2*gamma**2*np.einsum('nmc,mc->n',zz,coef[sl])
            lap+=.5*gamma**2*np.einsum('nmc,mc,m->n',zz,coef[sl],np.sum(v[sl]**2,axis=1))
    return u,ut,utt,lap

def physical_initial(x,components):
    y=np.zeros(x.shape[:-1]);grad=np.zeros(x.shape)
    for amp,A,mu in components:
        z=x-mu;u=amp*np.exp(-np.einsum('...i,ij,...j->...',z,A,z))
        y+=u;grad+=-2*u[...,None]*np.einsum('ij,...j->...i',A,z)
    return y,grad

def kirchhoff(x,t,speed,components,nz):
    # Independent physical-space sphere integration, not Radon profiles.
    hemi,w=sphere_rule(nz);v=np.vstack([hemi,-hemi]);w=np.r_[w,w]/2
    ans=np.zeros(len(x))
    for start in range(0,len(v),128):
        sl=slice(start,start+128);pts=x[:,None,:]+speed*t*v[None,sl,:]
        f,g=physical_initial(pts,components)
        ans+=(f+speed*t*np.einsum('nmi,mi->nm',g,v[sl]))@w[sl]
    return ans

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();speed=.7
    A=np.array([[3.,.7,.2],[.7,5.,-.4],[.2,-.4,2.]])
    comps=[(1.,A,np.array([.12,-.15,.09])),(-.3,2.2*np.eye(3),np.array([-.21,.17,-.08]))]
    x=1.2*(qmc.Sobol(3,scramble=True,seed=5791).random_base2(7)-.5)
    times=[0.,.25,.5,1.];refs={t:kirchhoff(x,t,speed,comps,48) for t in times}
    refcheck=max(rel(kirchhoff(x,t,speed,comps,64),refs[t]) for t in times)
    c=.04*np.arange(-125,126);gamma=.25/.04;rows=[]
    for nz in [4,8,12,20,32]:
      v,w=sphere_rule(nz)
      for corrected in ([False,True] if nz==32 else [True]):
        coef,b,_,_=heat_encode(v,w,c,gamma,0.,comps,np.eye(3),corrected)
        table=[]
        for t in times:
            y,ut,utt,lap=wave_eval(x,t,v,c,gamma,coef,b,speed)
            table.append(dict(time=t,rel_l2=rel(y,refs[t]),max_abs=float(np.max(np.abs(y-refs[t]))),pde_rms=rms(utt-speed**2*lap),pde_scaled_rms=rms(utt-speed**2*lap)/max(rms(utt),1e-30),ic_velocity_rms=rms(ut) if t==0 else None))
        row=dict(directions_spatial=len(v),directions_spacetime=2*len(v),centers=len(c),neurons=2*len(v)*len(c),corrected=corrected,times=table,max_rel_l2=max(r['rel_l2'] for r in table),max_scaled_residual=max(r['pde_scaled_rms'] for r in table))
        rows.append(row);print(row,flush=True)
        if nz==32 and corrected:
            np.savez_compressed(OUT/'wave3d_fixed_4input_network.npz',v=v,angular_weights=w,centers=c,gamma=gamma,speed=speed,coefficient=coef/2,bias=b,architecture='Each v has two spacetime directions (v,+speed),(v,-speed); shared centers; readout coef/2.')
            final=(v,coef,b)
    metrics=dict(protocol='Full-space 3D wave IVP u_tt=c^2 Lap u; initial Gaussian mixture, zero initial velocity. Fixed 4-input tanh MLP. Independent Kirchhoff physical-space quadrature for verification.',speed=speed,lambda_value=.25,profile_h=.04,reference_refinement_relative_difference=refcheck,rows=rows,seconds=time.monotonic()-start)
    (OUT/'wave_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    rr=[r for r in rows if r['corrected']]
    ax[0].loglog([r['neurons'] for r in rr],[r['max_rel_l2'] for r in rr],'o-',label='Max solution error, 4 times')
    ax[0].loglog([r['neurons'] for r in rr],[r['max_scaled_residual'] for r in rr],'s--',label='Scaled PDE residual')
    ax[0].set(xlabel='Tanh neurons',ylabel='Relative error',title='Fixed four-input network');ax[0].legend(fontsize=8)
    z=np.linspace(-.6,.6,151);xx=np.column_stack([z,np.zeros_like(z),np.zeros_like(z)])
    for t,col in [(0.,'#2464ab'),(.5,'#b64232'),(1.,'#228050')]:
        truth=kirchhoff(xx,t,speed,comps,48);y=wave_eval(xx,t,final[0],c,gamma,final[1],final[2],speed)[0]
        ax[1].plot(z,truth,c=col,label=f't = {t:g}');ax[1].plot(z[::5],y[::5],'o',mfc='none',ms=3,c=col)
        ax[2].semilogy(z,np.maximum(np.abs(y-truth),1e-17),c=col)
    ax[1].set(xlabel='x (y = z = 0)',ylabel='u',title='Kirchhoff solution (lines), network (dots)');ax[1].legend(fontsize=8)
    ax[2].set(xlabel='x (y = z = 0)',ylabel='Absolute error',title='Independent physical-space reference')
    for a in ax:a.grid(alpha=.2)
    fig.savefig(OUT/'radon_wave3d_fixed.png',dpi=170);plt.close(fig)
    print('DONE',metrics['seconds'],refcheck,flush=True)
if __name__=='__main__':main()
