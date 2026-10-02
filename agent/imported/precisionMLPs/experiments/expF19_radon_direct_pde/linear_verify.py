"""Independent dense standard-MLP/autograd checks for compiled PDE networks."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
import numpy as np
import torch
from linear import OUT,sphere_rule,heat_encode,profile_encode,profile_eval,rel,rms
from linear_wave import wave_eval

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 A=np.array([[3.,.7,.2],[.7,5.,-.4],[.2,-.4,2.]])
 comps=[(1.,A,np.array([.12,-.15,.09])),(-.3,2.2*np.eye(3),np.array([-.21,.17,-.08]))]
 v,w=sphere_rule(8);cent=.04*np.arange(-125,126);gamma=6.25;speed=.7
 coef,b,_,_=heat_encode(v,w,cent,gamma,0.,comps,np.eye(3),True)
 W=np.concatenate([np.repeat(np.column_stack([v,np.full(len(v),sign*speed)]),len(cent),axis=0)*gamma for sign in [-1,1]])
 B=np.tile(-gamma*cent,2*len(v));V=np.tile(coef.ravel()/2,2)
 rng=np.random.default_rng(24);xyz=rng.uniform(-.5,.5,(8,3));T=.37
 X=torch.tensor(np.column_stack([xyz,np.full(8,T)]),requires_grad=True)
 y=b+torch.tanh(X@torch.tensor(W).T+torch.tensor(B))@torch.tensor(V)
 g=torch.autograd.grad(y.sum(),X,create_graph=True)[0]
 dd=[torch.autograd.grad(g[:,k].sum(),X,retain_graph=True)[0][:,k] for k in range(4)]
 res=dd[3]-speed**2*sum(dd[:3]);analytic=wave_eval(xyz,T,v,cent,gamma,coef,b,speed)[0]
 out=dict(wave_dense_autograd=dict(neurons=len(V),dense_vs_factorized_relative=rel(y.detach().numpy(),analytic),autograd_pde_rms=rms(res.detach().numpy()),input_dimension=4))
 # A separate dense two-input Airy MLP and third autograd derivatives.
 h=.12;ws=[];bs=[];vs=[];bias=0.;ref=np.zeros(8);x=rng.uniform(-1,1,8);t=rng.uniform(0,1,8)
 for amp,k in [(1.,1.5),(.5,2.2)]:
  p=profile_encode(lambda s,a=amp:a*np.sin(s),lambda s,a=amp:a*np.cos(s),h,-k,k+k**3,True)
  ws.append(np.tile([k,k**3],(len(p['centers']),1))*p['gamma']);bs.append(-p['gamma']*p['centers']);vs.append(p['coef']);bias+=p['bias']
  ref+=profile_eval(k*x+k**3*t,p)
 X=torch.tensor(np.column_stack([x,t]),requires_grad=True)
 y=bias+torch.tanh(X@torch.tensor(np.vstack(ws)).T+torch.tensor(np.concatenate(bs)))@torch.tensor(np.concatenate(vs))
 g=torch.autograd.grad(y.sum(),X,create_graph=True)[0]
 g2=torch.autograd.grad(g[:,0].sum(),X,create_graph=True)[0][:,0]
 g3=torch.autograd.grad(g2.sum(),X)[0][:,0]
 out['airy_dense_autograd']=dict(neurons=sum(len(v) for v in vs),dense_vs_factorized_relative=rel(y.detach().numpy(),ref),autograd_pde_rms=rms((g[:,1]+g3).detach().numpy()),input_dimension=2)
 (OUT/'independent_autograd_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
