"""Structured d=4,5 sphere cubature beyond the old dense-solve limit."""
import json
import numpy as np
from scipy.special import roots_jacobi
from scaling import OUT, cell, points
import matplotlib.pyplot as plt


def full_sphere(d,n):
    if d==2:
        theta=np.arange(2*n)*np.pi/n
        return np.column_stack([np.cos(theta),np.sin(theta)]),np.full(2*n,1/(2*n))
    z,w=roots_jacobi(n,(d-3)/2,(d-3)/2);w/=sum(w)
    prev,pw=full_sphere(d-1,n)
    v=np.empty((n,len(prev),d));v[:,:,0]=z[:,None]
    v[:,:,1:]=np.sqrt(1-z*z)[:,None,None]*prev[None,:,:]
    return v.reshape(-1,d),(w[:,None]*pw[None,:]).ravel()


def sphere(d,n):
    z,w=roots_jacobi(n,(d-3)/2,(d-3)/2);w/=sum(w)
    keep=z>0;z=z[keep];w=2*w[keep]
    prev,pw=full_sphere(d-1,n)
    v=np.empty((len(z),len(prev),d));v[:,:,0]=z[:,None]
    v[:,:,1:]=np.sqrt(1-z*z)[:,None,None]*prev[None,:,:]
    return v.reshape(-1,d),(w[:,None]*pw[None,:]).ravel()


def main():
    rows=[]
    for d in [4,5]:
        for order in ([4,8,12,16,24,32] if d==4 else [4,8,12,16,24]):
            v,w=sphere(d,order)
            rr,_=cell(d,len(v),201,vw=(v,w),rule='Gauss-Jacobi sphere')
            for r in rr:r['cubature_order']=order
            rows.extend(rr)
            (OUT/'structured.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(rr),flush=True)
    # Independent denser validation points on the finest dictionaries.
    validation=[]
    for d,order in [(4,32),(5,16)]:
        v,w=sphere(d,order)
        rr,_=cell(d,len(v),201,vw=(v,w),rule='Gauss-Jacobi sphere',x=points(d,1024,seed=512))
        validation.extend(rr)
        (OUT/'structured_validation.json').write_text(json.dumps(validation,indent=2))
        print(json.dumps(rr),flush=True)
    plot(rows)


def plot(rows):
    fig,ax=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for a,target in zip(ax,['isotropic','anisotropic']):
        for d,color in [(4,'#2666a8'),(5,'#d25c35')]:
            r=[r for r in rows if r['d']==d and r['target']==target]
            a.loglog([t['neurons'] for t in r],[t['tanh_rel_l2'] for t in r],'o-',label=f'd={d}, tensor sphere rule',color=color)
            a.loglog([t['neurons'] for t in r],[t['conversion_rel_l2'] for t in r],':',color=color,alpha=.7)
        a.set(title=target.capitalize()+' Gaussian',xlabel='Actual tanh neurons',ylabel='Relative L2 error');a.grid(alpha=.2);a.legend(fontsize=9)
    fig.suptitle('Direct 4D/5D construction: angular cubature quality is decisive; dotted = tanh conversion error')
    fig.savefig(OUT/'structured_dimensions.png',dpi=170)


if __name__=='__main__':main()
