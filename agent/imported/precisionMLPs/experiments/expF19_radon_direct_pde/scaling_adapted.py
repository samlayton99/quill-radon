"""Analytic Gaussian geometry adaption versus target-independent sphere sampling.

This uses the known target precision matrix, not learned/inferred directions.
It is an importance-geometry control, not evidence of discovery from PDE data.
"""
import json,time,resource
import numpy as np
from scipy.special import hyp1f1
from scaling import OUT,points,targets,directions,evaluate,relative
import matplotlib.pyplot as plt


def adapted(d,m,seed):
    started=time.perf_counter()
    u,w=directions(d,m,seed,structured=False)
    b=targets(d)['anisotropic']
    eig,q=np.linalg.eigh(b); root=(q*np.sqrt(eig))@q.T
    z=u@root; scales=np.linalg.norm(z,axis=1);v=z/scales[:,None]
    centers=np.linspace(-4,4,201);h=centers[1]-centers[0];gamma=.25/h;a=np.pi/(2*gamma)
    coef=np.empty((m,len(centers)))
    for k in range(0,m,512):
        qq=hyp1f1(d/2,.5,-(scales[k:k+512,None]*(centers[None,:]+1j*a))**2)
        coef[k:k+512]=.5*h*w[k:k+512,None]*qq.imag/a
    bias=1-np.sum(coef*np.tanh(-gamma*centers))
    x=points(d);truth=np.exp(-np.einsum('ni,ij,nj->n',x,b,x));continuous=np.zeros(len(x))
    for k in range(0,m,512):
        continuous+=w[k:k+512]@hyp1f1(d/2,.5,-(x@z[k:k+512].T).T**2)
    y=evaluate(x,v,centers,gamma,coef[:,:,None],np.array([bias]))[:,0]
    return dict(d=d,directions=m,centers=201,neurons=m*201,scramble=seed,tanh_rel_l2=relative(y,truth),angular_rel_l2=relative(continuous,truth),conversion_rel_l2=relative(y,continuous),seconds=time.perf_counter()-started,process_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)


def main():
    rows=[]
    for d in [4,8,16,32,64]:
        for seed in [0,1]:
            r=adapted(d,16384,seed);rows.append(r)
            (OUT/'adapted.json').write_text(json.dumps(rows,indent=2));print(json.dumps(r),flush=True)
    original=json.loads((OUT/'dimension.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    ds=[4,8,16,32]
    for data,label,color in [(original,'Target-independent sphere directions','#c56345'),(rows,'Analytically adapted directions','#2768a4')]:
        med=[];lo=[];hi=[]
        for d in ds:
            a=[r['tanh_rel_l2'] for r in data if r['d']==d and r['directions']==16384 and r.get('target','anisotropic')=='anisotropic']
            med.append(np.median(a));lo.append(min(a));hi.append(max(a))
        ax.semilogy(ds,med,'o-',color=color,label=label);ax.fill_between(ds,lo,hi,color=color,alpha=.15)
    ax.set(xlabel='Input dimensions',ylabel='Relative L2 error',title='Same 3.29 million neurons: angular geometry matters');ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'adapted_geometry.png',dpi=170)


if __name__=='__main__':main()
