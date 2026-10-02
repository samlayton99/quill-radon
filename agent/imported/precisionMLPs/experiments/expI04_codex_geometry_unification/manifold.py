"""Noiseless 3D manifold in R24: unsupervised chart, held-out ridge-bank fits.
All fitted normalizations and geometry use training inputs only. Target values never
enter the encoder loss. Validation selects readout cutoff and bandwidth; test is final.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex_i04_mpl')
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.stats import qmc
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/expH01_highdim_suite'))
from h01suite.baseline import even_directions
OUT=ROOT/'results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/manifold'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

def targets(u):
    return np.column_stack([np.sin(np.pi*u[:,0])*np.cos(np.pi*u[:,1]),
       np.exp(.5*np.sin(np.pi*u[:,0])+.3*u[:,1]*u[:,2]),
       np.exp(-2*np.sum((u-np.array([.2,-.15,.1]))**2,axis=1))])

def embed(u, curved=True):
    if not curved: raw=np.column_stack([u,np.zeros((len(u),21))])
    else:
        raw=np.column_stack([u,.65*u*u,.65*u[:,[0,1,2]]*u[:,[1,2,0]],
         .65*np.sin(np.pi*u),.35*np.cos(np.pi*u),.25*np.sin(2*np.pi*u),
         .35*u[:,[0,1,2]]**2*u[:,[1,2,0]],.25*np.cos(np.pi*(u[:,[0,1,2]]+u[:,[1,2,0]]))])
    rot=np.linalg.qr(np.random.default_rng(92).normal(size=(24,24)))[0]
    return raw@rot

class AE(nn.Module):
    def __init__(self,pca):
        super().__init__(); self.register_buffer('pca',torch.tensor(pca))
        self.enc=nn.Sequential(nn.Linear(24,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,3))
        self.dec=nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,24))
        nn.init.zeros_(self.enc[-1].weight); nn.init.zeros_(self.enc[-1].bias)
        nn.init.zeros_(self.dec[-1].weight); nn.init.zeros_(self.dec[-1].bias)
    def encode(self,x): return x@self.pca+self.enc(x)
    def forward(self,x):
        z=self.encode(x); return z@self.pca.T+self.dec(z)

def rel(y,p): return np.linalg.norm(y-p,axis=0)/np.linalg.norm(y,axis=0)

def data():
    us=[2*qmc.Sobol(3,scramble=True,seed=s).random_base2(k)-1 for s,k in [(11,12),(22,10),(33,12)]]
    xs=[embed(u) for u in us]; mu=xs[0].mean(0); xs=[x-mu for x in xs]
    _,sv,vt=la.svd(xs[0],full_matrices=False,check_finite=False)
    return us,xs,vt[:3].T,sv,mu

def train(steps,seed):
    us,xs,pca,sv,mu=data(); torch.manual_seed(seed); model=AE(pca)
    tx=torch.tensor(xs[0]); vx=torch.tensor(xs[1]); optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    hist=[]; best=float('inf'); start=time.time()
    for k in range(steps+1):
        optimizer.zero_grad(); loss=((model(tx)-tx)**2).mean(); loss.backward(); optimizer.step()
        if k%250==0 or k==steps:
            with torch.no_grad(): ve=float(((model(vx)-vx)**2).mean().sqrt())
            hist.append([k,float(loss.detach().sqrt()),ve,time.time()-start]); print('AE',seed,hist[-1],flush=True)
            if ve<best:
                best=ve; torch.save(model.state_dict(),OUT/f'encoder_s{seed}.pt')
        if k in [steps//2,3*steps//4]:
            for g in optimizer.param_groups:g['lr']*=.3
    model.load_state_dict(torch.load(OUT/f'encoder_s{seed}.pt',weights_only=True))
    optimizer=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=100,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-15)
    for k in range(40):
        def closure():
            optimizer.zero_grad(); loss=((model(tx)-tx)**2).mean();loss.backward();return loss
        optimizer.step(closure)
        with torch.no_grad(): tr=float(((model(tx)-tx)**2).mean().sqrt());ve=float(((model(vx)-vx)**2).mean().sqrt())
        hist.append([steps+100*(k+1),tr,ve,time.time()-start]);print('AE LBFGS',seed,hist[-1],flush=True)
        if ve<best:best=ve;torch.save(model.state_dict(),OUT/f'encoder_s{seed}.pt')
    np.save(OUT/f'encoder_history_s{seed}.npy',np.array(hist))

class RidgeBank:
    def __init__(self,x,dirs=24,centers=24,lam=.25):
        self.v=even_directions(x.shape[1],dirs)
        q=x@self.v.T; lo=q.min(0);hi=q.max(0);self.mid=(lo+hi)/2
        T=1.25*(hi-lo)/2;self.h=2*T/centers
        self.c=(-1+(np.arange(centers)+.5)*2/centers)[None,:]*T[:,None]
        self.gam=lam/self.h; self.centers=centers
    def features(self,x):
        t=x@self.v.T-self.mid
        f=np.tanh(self.gam[None,:,None]*(t[:,:,None]-self.c[None,:,:])).reshape(len(x),-1)
        return np.column_stack([f,np.ones(len(x))])

def fit_eval(label, zs, ys, dirs=24,centers=24):
    # Coordinatewise centering/scaling from unlabeled training inputs, applied identically.
    mid=(zs[0].max(0)+zs[0].min(0))/2;scale=(zs[0].max(0)-zs[0].min(0))/2
    zs=[(z-mid)/np.maximum(scale,1e-12) for z in zs]
    rows=[]; best=[None]*ys[0].shape[1]
    for lam in [.15,.25,.4]:
        bank=RidgeBank(zs[0],dirs,centers,lam);a,b,c=[bank.features(z) for z in zs]
        start=time.time();U,s,Vt=la.svd(a,full_matrices=False,check_finite=False,lapack_driver='gesdd')
        rhs=U.T@ys[0]
        for rc in [1e-8,1e-10,1e-12,1e-14]:
            keep=s>rc*s[0];w=Vt[keep].T@(rhs[keep]/s[keep,None]);pv=b@w;pt=c@w
            ev=rel(ys[1],pv);et=rel(ys[2],pt);tr=rel(ys[0],a@w)
            row=dict(label=label,lambda_value=lam,rcond=rc,rank=int(keep.sum()),neurons=dirs*centers,
                train=tr.tolist(),validation=ev.tolist(),test=et.tolist(),coefficient_norm=np.linalg.norm(w,axis=0).tolist())
            rows.append(row)
            for j in range(len(best)):
                if best[j] is None or ev[j]<best[j]['validation']:
                    best[j]=dict(target=j,validation=float(ev[j]),test=float(et[j]),train=float(tr[j]),lambda_value=lam,rcond=rc,rank=int(keep.sum()),neurons=dirs*centers)
        print('FIT',label,lam,'seconds',time.time()-start,'best',best,flush=True)
    return dict(label=label,best=best,sweep=rows)

def evaluate(seed,large=False):
    us,xs,pca,sv,mu=data();model=AE(pca);model.load_state_dict(torch.load(OUT/f'encoder_s{seed}.pt',weights_only=True));model.eval()
    with torch.no_grad():
        zs=[model.encode(torch.tensor(x)).numpy() for x in xs]
        recon=[model(torch.tensor(x)).numpy() for x in xs]
    ys=[targets(u) for u in us]
    size=(128,16) if large else (24,24)
    cases={'oracle intrinsic coordinates':us,'PCA 3D':[x@pca for x in xs],'learned AE 3D':zs,'ambient 24D':xs}
    records=[]
    for label,z in cases.items():records.append(fit_eval(label,z,ys,*size))
    # Is latent representation an injective chart? Ground-truth forward Jacobian
    # differentiation is diagnostic only, never used in encoder or target fitting.
    ut=torch.tensor(us[2][:256],requires_grad=True)
    rot=torch.tensor(np.linalg.qr(np.random.default_rng(92).normal(size=(24,24)))[0])
    raw=torch.cat([ut,.65*ut*ut,.65*ut[:,[0,1,2]]*ut[:,[1,2,0]],.65*torch.sin(torch.pi*ut),
        .35*torch.cos(torch.pi*ut),.25*torch.sin(2*torch.pi*ut),.35*ut[:,[0,1,2]]**2*ut[:,[1,2,0]],
        .25*torch.cos(torch.pi*(ut[:,[0,1,2]]+ut[:,[1,2,0]]))],1)
    zz=model.encode(raw@rot-torch.tensor(mu))
    jac=torch.stack([torch.autograd.grad(zz[:,j].sum(),ut,retain_graph=True)[0] for j in range(3)],1).detach().numpy()
    sval=np.linalg.svd(jac,compute_uv=False);det=np.linalg.det(jac)
    pz=(raw@rot-torch.tensor(mu))@torch.tensor(pca)
    pjac=torch.stack([torch.autograd.grad(pz[:,j].sum(),ut,retain_graph=True)[0] for j in range(3)],1).detach().numpy()
    pdet=np.linalg.det(pjac)
    # Linear decoder of intrinsic coordinates is a diagnostic, not used for labels.
    az=np.column_stack([zs[0],np.ones(len(zs[0]))]);bz=np.column_stack([zs[2],np.ones(len(zs[2]))])
    decode=la.lstsq(az,us[0],cond=1e-12)[0]
    info=dict(seed=seed,train_points=len(xs[0]),validation_points=len(xs[1]),test_points=len(xs[2]),
       latent_dim=3,ambient_dim=24,pca_explained_fraction=float(np.sum(sv[:3]**2)/np.sum(sv**2)),
       pca_test_reconstruction_rel=float(np.linalg.norm(xs[2]-xs[2]@pca@pca.T)/np.linalg.norm(xs[2])),
       ae_test_reconstruction_rel=float(np.linalg.norm(xs[2]-recon[2])/np.linalg.norm(xs[2])),
       ae_latent_linear_decode_rel=rel(us[2],bz@decode).tolist(),jacobian_min_singular=float(sval[:,-1].min()),
       jacobian_condition_median=float(np.median(sval[:,0]/sval[:,-1])),jacobian_positive_det_fraction=float(np.mean(det>0)),
       pca_jacobian_det_min=float(pdet.min()),pca_jacobian_det_max=float(pdet.max()),
       pca_jacobian_positive_det_fraction=float(np.mean(pdet>0)),
       records=records)
    suffix=f's{seed}'+('_large' if large else '')
    (OUT/f'metrics_{suffix}.json').write_text(json.dumps(info,indent=2))
    np.savez(OUT/f'coordinates_{suffix}.npz',u=us[2],x=xs[2],pca=xs[2]@pca,z=zs[2],reconstruction=recon[2],jac=jac)
    fig,axs=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
    names=['Product wave','Composition','Gaussian bump'];colors=['#666666','#db8c20','#147d92','#8958b3']
    for j,ax in enumerate(axs):
        vals=[r['best'][j]['test'] for r in records]
        ax.bar(np.arange(4),vals,color=colors);ax.set_yscale('log');ax.set_xticks(range(4),['True 3D\nchart','PCA\n3D','Learned\n3D chart','Original\n24D']);ax.set_title(names[j]);ax.grid(axis='y',alpha=.2);ax.set_ylabel('Held-out relative L2')
        for i,v in enumerate(vals):ax.text(i,v*1.25,f'{v:.1e}',ha='center',fontsize=10)
    fig.suptitle(f'Noiseless 3D manifold in 24D — same {size[0]*size[1]} tanh ridge features; seed {seed}',fontsize=13)
    fig.savefig(OUT/f'function_errors_{suffix}.png',dpi=180);plt.close(fig)
    fig=plt.figure(figsize=(13,4.7))
    for i,(z,name) in enumerate([(us[2],'True coordinates'),(xs[2]@pca,'PCA coordinates'),(zs[2],'Unsupervised learned chart')]):
        ax=fig.add_subplot(1,3,i+1,projection='3d');ax.scatter(*z[:900].T,c=us[2][:900,0],s=3,cmap='coolwarm',alpha=.6);ax.set_title(name);ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_zticklabels([])
    fig.suptitle('Same unseen points, colored by true coordinate 1 (color not used in learning)')
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.86,wspace=.05)
    fig.savefig(OUT/f'charts_{suffix}.png',dpi=180);plt.close(fig)
    print('SUMMARY',json.dumps({k:v for k,v in info.items() if k!='records'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);p.add_argument('--steps',type=int,default=3000);p.add_argument('--seed',type=int,default=0);p.add_argument('--large',action='store_true');args=p.parse_args()
    config=dict(task='unsupervised manifold compression then frozen ridge bank',dtype='float64',device='cpu',seed=args.seed,steps=args.steps,train=4096,validation=1024,test=4096,encoder='PCA residual 24-64-64-3',decoder='3-64-64-24 residual',targets=['product_wave','composition','gaussian'],bandwidths=[.15,.25,.4],rconds=[1e-8,1e-10,1e-12,1e-14])
    if args.stage=='train':
        config['lbfgs_blocks']=40
        config['lbfgs_max_iterations_per_block']=100
        (OUT/f'config_s{args.seed}.json').write_text(json.dumps(config,indent=2))
        train(args.steps,args.seed)
    else:evaluate(args.seed,args.large)
