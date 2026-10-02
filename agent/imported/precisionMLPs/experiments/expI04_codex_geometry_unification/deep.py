"""CPU-only two-hidden-layer regression and controlled latent-ridge placement audit.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/expI04_codex_geometry_unification/deep.py
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex_deep_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,copy,argparse
import numpy as np
import scipy.linalg as sla
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/deep'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
CONFIG=dict(width=48,seeds=[0,1],train_n=513,val_n=1024,test_n=4096,adam_steps=10000,adam_lr=.002,adam_end_lr=.0001,lbfgs_max_iterations=6000,lbfgs_loss_scale=1e6,target='sin(2*pi*x)+0.4*exp(-80*(x-0.35)^2)',dtype='float64',lstsq_cutoffs=[1e-10,1e-12,1e-14,1e-15])
def target(x):return np.sin(2*np.pi*x)+.4*np.exp(-80*(x-.35)**2)
def target_first(x):return 2*np.pi*np.cos(2*np.pi*x)-64*(x-.35)*np.exp(-80*(x-.35)**2)
def target_second(x):return -(2*np.pi)**2*np.sin(2*np.pi*x)+.4*(25600*(x-.35)**2-160)*np.exp(-80*(x-.35)**2)
def rel(p,y):return float(np.linalg.norm(p-y)/np.linalg.norm(y))
class Net(nn.Module):
 def __init__(self,w=48):
  super().__init__();self.l1=nn.Linear(1,w);self.l2=nn.Linear(w,w);self.head=nn.Linear(w,1)
  for l in [self.l1,self.l2,self.head]:nn.init.xavier_normal_(l.weight);nn.init.zeros_(l.bias)
 def forward(self,x):return self.head(torch.tanh(self.l2(torch.tanh(self.l1(x)))))
def arrays(net):return {k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
def features(p,x):
 h=np.tanh(x[:,None]*p['l1.weight'][:,0]+p['l1.bias']);dh=(1-h*h)*p['l1.weight'][:,0]
 z=h@p['l2.weight'].T+p['l2.bias'];dz=dh@p['l2.weight'].T
 return h,np.tanh(z),z,dh,dz
def pred(p,x):return features(p,x)[1]@p['head.weight'][0]+p['head.bias'][0]
def train(seed):
 path=OUT/f'seed{seed}.npz'
 if path.exists():
  d=np.load(path);p={k:d[k] for k in d.files};log=json.loads((OUT/f'seed{seed}_training.json').read_text())
  if log[-1]['phase']=='L-BFGS scaled':return p,log
  net=Net(CONFIG['width']);net.load_state_dict({k:torch.tensor(v) for k,v in p.items()});return polish(seed,net,log)
 torch.manual_seed(seed);net=Net(CONFIG['width']);initial=arrays(net);np.savez(OUT/f'seed{seed}_initial.npz',**initial)
 x=np.linspace(-1,1,CONFIG['train_n']);tx=torch.tensor(x[:,None]);ty=torch.tensor(target(x)[:,None]);log=[];start=time.time()
 opt=torch.optim.Adam(net.parameters(),lr=CONFIG['adam_lr']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CONFIG['adam_steps'],eta_min=CONFIG['adam_end_lr'])
 for step in range(CONFIG['adam_steps']):
  opt.zero_grad();loss=((net(tx)-ty)**2).mean();loss.backward();opt.step();sched.step()
  if step%1000==0 or step+1==CONFIG['adam_steps']:
   err=float(torch.linalg.norm(net(tx)-ty).detach()/torch.linalg.norm(ty));log.append(dict(step=step+1,phase='Adam',rel_l2=err,seconds=time.time()-start));print(seed,log[-1],flush=True)
 adam=arrays(net);np.savez(OUT/f'seed{seed}_adam.npz',**adam)
 return polish(seed,net,log)

def polish(seed,net,log):
 x=np.linspace(-1,1,CONFIG['train_n']);tx=torch.tensor(x[:,None]);ty=torch.tensor(target(x)[:,None]);start=time.time();previous=log[-1]['seconds'];path=OUT/f'seed{seed}.npz'
 opt=torch.optim.LBFGS(net.parameters(),lr=1,max_iter=100,history_size=100,line_search_fn='strong_wolfe',tolerance_grad=1e-14,tolerance_change=1e-16)
 calls=0
 def closure():
  nonlocal calls
  calls+=1;opt.zero_grad();loss=CONFIG['lbfgs_loss_scale']*((net(tx)-ty)**2).mean();loss.backward();return loss
 for block in range(CONFIG['lbfgs_max_iterations']//100):
  opt.step(closure);err=float(torch.linalg.norm(net(tx)-ty).detach()/torch.linalg.norm(ty));log.append(dict(step=log[-1]['step']+100,phase='L-BFGS scaled',rel_l2=err,seconds=previous+time.time()-start,evaluations=calls));print(seed,log[-1],flush=True)
  if err<3e-7:break
 p=arrays(net);np.savez(path,**p);(OUT/f'seed{seed}_training.json').write_text(json.dumps(log,indent=2));return p,log

def pca_analysis(p,x):
 h1,h2,z,dh,dz=features(p,x);result={};pca={}
 for name,h in [('first',h1),('second',h2)]:
  mu=h.mean(0);u,s,vt=sla.svd(h-mu,full_matrices=False);frac=s*s/(s*s).sum();cum=np.cumsum(frac);coords=(h-mu)@vt.T;pca[name]=coords
  vals={'pc_variance_fraction':frac.tolist(),'dims_99':int(np.searchsorted(cum,.99)+1),'dims_999':int(np.searchsorted(cum,.999)+1),'dims_9999':int(np.searchsorted(cum,.9999)+1),'linear_decode':{}}
  train=np.arange(len(x))%2==0
  for n in [1,2,3,6,12,48]:
   a=np.c_[np.ones(len(x)),coords[:,:n]];fits={}
   for label,y in [('x',x),('target',target(x))]:
    w=sla.lstsq(a[train],y[train],cond=1e-12)[0];fits[label]=rel(a[~train]@w,y[~train])
   vals['linear_decode'][str(n)]=fits
  result[name]=vals
 return result,pca

def transitions(p,x):
 h,h2,z,dh,dz=features(p,x);centers=[];gammas=[];counts=[];all_roots=[]
 for j in range(z.shape[1]):
  ids=np.where(z[:-1,j]*z[1:,j]<=0)[0];roots=x[ids]-z[ids,j]*(x[ids+1]-x[ids])/(z[ids+1,j]-z[ids,j]+1e-300);all_roots.append(roots);counts.append(len(roots))
  if len(roots):
   g=np.interp(roots,x,np.abs(dz[:,j]));best=np.argmax(g);centers.append(float(roots[best]));gammas.append(float(g[best]))
  else:
   k=np.argmax(np.abs((1-h2[:,j]**2)*dz[:,j]));centers.append(float(x[k]));gammas.append(float(abs(dz[k,j])))
 return np.array(centers),np.array(gammas),np.array(counts),all_roots

def fit_readout(p,xtrain,xval,xtest):
 a=np.c_[np.ones(len(xtrain)),features(p,xtrain)[1]];av=np.c_[np.ones(len(xval)),features(p,xval)[1]];at=np.c_[np.ones(len(xtest)),features(p,xtest)[1]]
 y=target(xtrain);yv=target(xval);yt=target(xtest);candidates=[]
 for cutoff in CONFIG['lstsq_cutoffs']:
  w,res,rank,s=sla.lstsq(a,y,cond=cutoff,lapack_driver='gelsd');candidates.append(dict(cutoff=cutoff,rank=int(rank),train=rel(a@w,y),validation=rel(av@w,yv),test=rel(at@w,yt),coefficient_norm=float(np.linalg.norm(w)),w=w,singulars=s))
 best=min(candidates,key=lambda c:c['validation']);q={**p,'head.bias':best['w'][:1],'head.weight':best['w'][1:][None,:]};m={k:v for k,v in best.items() if k not in ['w','singulars']};m['cutoff_sweep']=[{k:v for k,v in c.items() if k not in ['w','singulars']} for c in candidates];return m,q

def place(p,x,kind,alpha=1,eligible_only=False,preserve_width=False):
 centers,gamma,counts,roots=transitions(p,x);h,h2,z,dh,dz=features(p,x);idx=np.where(counts==1)[0] if eligible_only else np.arange(len(centers));idx=idx[np.argsort(centers[idx])];m=len(idx);t=(np.arange(m)+.5)/m
 if kind=='uniform_input':new=-1+2*t
 elif kind=='uniform_arc':
  speed=np.linalg.norm(dh,axis=1);s=np.cumsum(np.r_[0,.5*(speed[1:]+speed[:-1])*np.diff(x)]);new=np.interp(t,s/s[-1],x)
 elif kind=='curvature_density':
  rho=(np.abs(target_second(x))**2+1e-8)**.2;s=np.cumsum(np.r_[0,.5*(rho[1:]+rho[:-1])*np.diff(x)]);new=np.interp(t,s/s[-1],x)
 elif kind.startswith('random'):
  rng=np.random.default_rng(int(kind.split('_')[-1]));new=np.sort(rng.uniform(-1,1,m))
 else:raise ValueError(kind)
 new=(1-alpha)*centers[idx]+alpha*new;q={k:v.copy() for k,v in p.items()};hnew=features(p,new)[0];q['l2.bias'][idx]=-np.einsum('ij,ij->i',hnew,p['l2.weight'][idx])
 if preserve_width:
  dhnew=features(p,new)[3];new_gamma=np.abs(np.einsum('ij,ij->i',dhnew,p['l2.weight'][idx]));scale=gamma[idx]/np.maximum(new_gamma,1e-12);q['l2.weight'][idx]*=scale[:,None];q['l2.bias'][idx]*=scale
 return q

def tangent_qi(p,x,gamma=8):
 q={k:v.copy() for k,v in p.items()};m=len(p['l2.bias']);halo=6;n=m-2*halo;step=2/(n-1);centers=-1+(np.arange(m)-halo)*step;h,h2,z,dh,dz=features(p,centers);directions=gamma*dh/np.sum(dh**2,axis=1)[:,None];q['l2.weight']=directions;q['l2.bias']=-np.sum(directions*h,axis=1);return q

def run_analysis(seed,p,log):
 x=np.linspace(-1,1,4097);xt=np.linspace(-1,1,CONFIG['train_n']);xv=-1+2*(np.arange(CONFIG['val_n'])+.5)/CONFIG['val_n'];rng=np.random.default_rng(123456);xe=np.sort(rng.uniform(-1,1,CONFIG['test_n']));c,g,n,roots=transitions(p,x);pca,coords=pca_analysis(p,x)
 metrics=dict(seed=seed,live=dict(train=rel(pred(p,xt),target(xt)),validation=rel(pred(p,xv),target(xv)),test=rel(pred(p,xe),target(xe))),transitions=dict(zero_crossings_hist={str(k):int(np.sum(n==k)) for k in np.unique(n)},centers=c.tolist(),effective_gamma=g.tolist()),pca=pca,interventions={})
 initial=np.load(OUT/f'seed{seed}_initial.npz');metrics['pca_initial']=pca_analysis({k:initial[k] for k in initial.files},x)[0]
 models={};m,q=fit_readout(p,xt,xv,xe);metrics['interventions']['refit_only']=m;models['refit_only']=q
 for eligible in [False,True]:
  for kind in ['uniform_input','uniform_arc','curvature_density']:
   for alpha in [.25,.5,1.]:
    name=f'{kind}_a{alpha:g}'+('_single_zero_only' if eligible else '');pp=place(p,x,kind,alpha,eligible);m,q=fit_readout(pp,xt,xv,xe);m['zero_crossings_hist']={str(k):int(np.sum(transitions(q,x)[2]==k)) for k in np.unique(transitions(q,x)[2])};metrics['interventions'][name]=m;models[name]=q
 for kind in ['uniform_input','uniform_arc']:
  for alpha in [.25,.5,1.]:
   name=f'{kind}_preserve_width_a{alpha:g}';m,q=fit_readout(place(p,x,kind,alpha,preserve_width=True),xt,xv,xe);metrics['interventions'][name]=m;models[name]=q
 for r in range(10):
  name=f'random_{r}';m,q=fit_readout(place(p,x,name),xt,xv,xe);metrics['interventions'][name]=m;models[name]=q
 for label,pp in [('drop_non_crossing',p),('uniform_arc_then_drop_non_crossing',place(p,x,'uniform_arc',1,eligible_only=True))]:
  pp={k:v.copy() for k,v in pp.items()};pp['l2.weight'][n==0]=0;pp['l2.bias'][n==0]=0;m,q=fit_readout(pp,xt,xv,xe);metrics['interventions'][label]=m;models[label]=q
 for r in range(10):
  name=f'random_single_zero_only_{r}';m,q=fit_readout(place(p,x,f'random_{r}',eligible_only=True),xt,xv,xe);metrics['interventions'][name]=m;models[name]=q
 for kind in ['uniform_input','uniform_arc']:
  name=f'{kind}_preserve_width_single_zero_only_a1';m,q=fit_readout(place(p,x,kind,1,eligible_only=True,preserve_width=True),xt,xv,xe);metrics['interventions'][name]=m;models[name]=q
 for gamma in [1,2,4,8,12,20]:
  name=f'tangent_qi_gamma{gamma}';m,q=fit_readout(tangent_qi(p,x,gamma),xt,xv,xe);metrics['interventions'][name]=m;models[name]=q
 # A direct one-layer reference uses the same number of readout features, including halos.
 mwidth=CONFIG['width'];halo=6;ni=mwidth-2*halo;step=2/(ni-1);qc=-1+(np.arange(mwidth)-halo)*step
 for gamma in [1,2,4,8,12,20]:
  # Embed raw input as arctanh of first units is unnecessary: direct basis fit.
  a=np.c_[np.ones(len(xt)),np.tanh(gamma*(xt[:,None]-qc))];av=np.c_[np.ones(len(xv)),np.tanh(gamma*(xv[:,None]-qc))];at=np.c_[np.ones(len(xe)),np.tanh(gamma*(xe[:,None]-qc))];best=None
  for cutoff in CONFIG['lstsq_cutoffs']:
   w,_,rank,s=sla.lstsq(a,target(xt),cond=cutoff,lapack_driver='gelsd');item=dict(train=rel(a@w,target(xt)),validation=rel(av@w,target(xv)),test=rel(at@w,target(xe)),gamma=gamma,cutoff=cutoff,rank=int(rank),coefficient_norm=float(np.linalg.norm(w)))
   if best is None or item['validation']<best['validation']:best=item
  metrics['interventions'][f'direct_qi_gamma{gamma}']=best
 # Select each family solely by validation, then expose independent test result.
 metrics['selected']={}
 for family in ['uniform_input','uniform_arc','curvature_density','tangent_qi','direct_qi']:
  choices=[(k,v) for k,v in metrics['interventions'].items() if k.startswith(family) and 'single_zero_only' not in k and 'preserve_width' not in k];key,value=min(choices,key=lambda kv:kv[1]['validation']);metrics['selected'][family]={'name':key,**value}
 metrics['first_layer_pca_ablation']={}
 ht=features(p,xt)[0];mu=ht.mean(0);_,ss,vt=sla.svd(ht-mu,full_matrices=False)
 for rank in [1,2,3,6,12]:
  proj=vt[:rank].T@vt[:rank];pp={k:v.copy() for k,v in p.items()};pp['l2.weight']=p['l2.weight']@proj;pp['l2.bias']=p['l2.bias']+p['l2.weight']@(mu-proj@mu);mm,qq=fit_readout(pp,xt,xv,xe);metrics['first_layer_pca_ablation'][str(rank)]={'explained_variance':float((ss[:rank]**2).sum()/(ss**2).sum()),**mm}
 metrics['qi_readout_prediction']=coefficient_prediction(p,models['refit_only'],x)
 metrics['random_test_median']=float(np.median([v['test'] for k,v in metrics['interventions'].items() if k.startswith('random_') and 'single_zero_only' not in k]))
 metrics['random_single_zero_test_median']=float(np.median([v['test'] for k,v in metrics['interventions'].items() if k.startswith('random_single_zero_only')]))
 (OUT/f'seed{seed}_metrics.json').write_text(json.dumps(metrics,indent=2))
 np.savez(OUT/f'seed{seed}_analysis.npz',x=x,centers=c,effective_gamma=g,crossings=n,pc_first=coords['first'][:,:3],pc_second=coords['second'][:,:3],target=target(x),live=pred(p,x),refit=pred(models['refit_only'],x),**{k:pred(models[k],x) for k in ['uniform_input_a1','uniform_arc_a1']})
 make_plots(seed,p,metrics,models,coords,x,log)
 return metrics

def coefficient_prediction(p,refit,x):
 c,g,n,roots=transitions(p,x);ids=np.where(n==1)[0];ids=ids[np.argsort(c[ids])];cs=c[ids]
 if len(ids)<3:return {'eligible':len(ids)}
 h=np.r_[cs[1]-cs[0],(cs[2:]-cs[:-2])/2,cs[-1]-cs[-2]]
 dz=features(p,cs)[4];sgn=np.sign(dz[np.arange(len(ids)),ids]);vpred=.5*h*target_first(cs)
 signed_dphi=(1-features(p,x)[1][:,ids]**2)*features(p,x)[4][:,ids]*sgn;neg=np.trapezoid(np.maximum(-signed_dphi,0),x,axis=0);total=np.trapezoid(np.abs(signed_dphi),x,axis=0)
 result={'negative_derivative_mass_fraction':(neg/total).tolist(),'max_negative_fraction':float(np.max(neg/total)),'eligible':len(ids),'total':len(c),'fraction_eligible':float(len(ids)/len(c)),'centers':cs.tolist(),'spacing':h.tolist(),'predicted':vpred.tolist()}
 for name,q in [('trained',p),('refit',refit)]:
  actual=q['head.weight'][0,ids]*sgn;interior=np.abs(cs)<.8;result[name]={'relative_coefficient_error':rel(vpred,actual),'pearson':float(np.corrcoef(vpred,actual)[0,1]),'interior_relative_error':rel(vpred[interior],actual[interior]),'actual':actual.tolist()}
 return result

def make_plots(seed,p,m,models,coords,x,log):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,3,figsize=(16,8.5),layout='constrained');h,h2,z,dh,dz=features(p,x);c=np.array(m['transitions']['centers']);g=np.array(m['transitions']['effective_gamma']);cnt=transitions(p,x)[2];v=p['head.weight'][0]*np.sign(features(p,c)[4].diagonal())
 ax=axs[0,0];ax.plot(x,target(x),'k',label='Target');ax.plot(x,pred(p,x),color='#bd3a33',ls='--',label='Trained network');ax.set(title=f'Seed {seed}: two-layer tanh fit',xlabel='Input x',ylabel='Function');ax.legend()
 ax=axs[0,1];a=coords['first'];sc=ax.scatter(a[:,0],a[:,1],c=x,s=3,cmap='viridis');ax.set(title=f'First hidden curve; {m["pca"]["first"]["dims_99"]} PCs cover 99%',xlabel='PC 1',ylabel='PC 2');fig.colorbar(sc,ax=ax,label='Input x')
 ax=axs[0,2];a=coords['second'];sc=ax.scatter(a[:,0],a[:,1],c=x,s=3,cmap='viridis');ax.set(title=f'Second hidden curve; {m["pca"]["second"]["dims_99"]} PCs cover 99%',xlabel='PC 1',ylabel='PC 2');fig.colorbar(sc,ax=ax,label='Input x')
 ax=axs[1,0];sc=ax.scatter(c,v,c=g,cmap='plasma',s=28);ax.axhline(0,color='.7',lw=.8);ax.set(title='Readout vs strongest transition center',xlabel='Input location of transition',ylabel='Orientation-canonical readout');fig.colorbar(sc,ax=ax,label='Effective inverse width')
 ax=axs[1,1];sc=ax.scatter(c,g,c=cnt,cmap='viridis',s=30,vmin=0,vmax=max(cnt));ax.set(title='Second-layer effective widths',xlabel='Input location of strongest transition',ylabel='|d preactivation / dx| at center');fig.colorbar(sc,ax=ax,label='Zero crossings / neuron')
 ax=axs[1,2];ax.semilogy(x,np.maximum(np.abs(pred(p,x)-target(x)),1e-16),label='Trained');ax.semilogy(x,np.maximum(np.abs(pred(models['refit_only'],x)-target(x)),1e-16),label='Same geometry, refit head');ax.set(title='Pointwise held-domain error',xlabel='Input x',ylabel='Absolute error');ax.legend()
 fig.suptitle('Learned hidden curves and the second-layer transitions placed along them',fontsize=14);fig.savefig(OUT/f'seed{seed}_geometry.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
 ax=axs[0];names=['Trained','Refit only','Uniform input','Uniform arc','Tangent QI','Direct QI'];values=[m['live']['test'],m['interventions']['refit_only']['test'],m['selected']['uniform_input']['test'],m['selected']['uniform_arc']['test'],m['selected']['tangent_qi']['test'],m['selected']['direct_qi']['test']];ax.bar(np.arange(len(names)),values,color=['#777777','#333333','#b14536','#d77736','#277ca3','#28855c']);ax.set_yscale('log');ax.set_xticks(np.arange(len(names)),names,rotation=35,ha='right');ax.set(title='Independent test error (validation selection)',ylabel='Relative L2 error');
 for i,val in enumerate(values):ax.text(i,val*1.35,f'{val:.1e}',ha='center',fontsize=8)
 ax=axs[1]
 for family,color in [('uniform_input','#b14536'),('uniform_arc','#d77736'),('curvature_density','#277ca3')]:
  alphas=[0,.25,.5,1];vals=[m['interventions']['refit_only']['test']]+[m['interventions'][f'{family}_a{alpha:g}']['test'] for alpha in alphas[1:]];ax.semilogy(alphas,vals,'o-',label=family.replace('_',' '),color=color)
 ax.axhline(m['random_test_median'],color='.5',ls=':',label='Random placement median');ax.set(title='Bias relocation with fixed directions',xlabel='Fraction moved to proposed placement',ylabel='Independent test relative L2');ax.legend(fontsize=8)
 ax=axs[2];speed1=np.linalg.norm(dh,axis=1);speed2=np.linalg.norm((1-h2*h2)*dz,axis=1);ax.plot(x,speed1/np.mean(speed1),label='First-layer arclength density');ax.plot(x,speed2/np.mean(speed2),label='Second-layer arclength density');rho=(np.abs(target_second(x))**2+1e-8)**.2;ax.plot(x,rho/np.mean(rho),ls='--',label='Linear-spline optimum density');ax.set(title='Uniform in one coordinate is not uniform in another',xlabel='Input x',ylabel='Density / mean');ax.legend(fontsize=8)
 fig.suptitle(f'Seed {seed}: interventions change actual second-layer weights/biases, then solve the head',fontsize=13);fig.savefig(OUT/f'seed{seed}_placement.png',dpi=170);plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,nargs='*',default=CONFIG['seeds']);ap.add_argument('--extra-polish',type=int,default=0);ap.add_argument('--loss-scale',type=float,default=1e6);ap.add_argument('--reproduce-final',action='store_true');args=ap.parse_args();CONFIG['lbfgs_loss_scale']=args.loss_scale;CONFIG['extra_polish_requested']=args.extra_polish;(OUT/'config.json').write_text(json.dumps(CONFIG,indent=2));allm=[]
 for seed in args.seed:
  p,log=train(seed)
  if args.reproduce_final:
   adam=dict(np.load(OUT/f'seed{seed}_adam.npz'));net=Net(CONFIG['width']);net.load_state_dict({k:torch.tensor(v) for k,v in adam.items()});log=[r for r in log if r['phase']=='Adam']
   stages=([(3000,1.)] if seed==0 else [])+[(6000,1e6),(30000,1e6),(10000,1e12)]
   for iterations,scale in stages:
    CONFIG['lbfgs_max_iterations']=iterations;CONFIG['lbfgs_loss_scale']=scale;p,log=polish(seed,net,log)
  if args.extra_polish:
   np.savez(OUT/f'seed{seed}_before_extra_polish_scale{args.loss_scale:g}.npz',**p);net=Net(CONFIG['width']);net.load_state_dict({k:torch.tensor(v) for k,v in p.items()});CONFIG['lbfgs_max_iterations']=args.extra_polish;p,log=polish(seed,net,log)
  m=run_analysis(seed,p,log);allm.append(m);print('FINAL',seed,m['live'],{k:v['test'] for k,v in m['selected'].items()},flush=True)
 (OUT/'summary.json').write_text(json.dumps(allm,indent=2))
if __name__=='__main__':main()
