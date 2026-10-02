"""Fixed budget and leading-vs-corrected controls, no fitting."""
import json,time
import numpy as np
from scipy.special import hyp1f1
from scaling import OUT,cell,directions,targets,points,evaluate,profile,relative
import matplotlib.pyplot as plt


def main():
    rows=[]
    for budget in [52000,208000]:
        for order in [8,12,16,20,24,32,40,48,64]:
            m=order**2;n=2*int((budget/m-1)/2)+1
            if n<21:continue
            rr,_=cell(3,m,n)
            for r in rr:r['budget']=budget
            rows.extend(rr)
            (OUT/'equal_budget.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rr),flush=True)
    control=[]
    for d,m in [(2,64),(3,1024)]:
        v,w=directions(d,m);x=points(d);c=np.linspace(-4,4,201);h=c[1]-c[0];gamma=.25/h
        for name,b in targets(d).items():
            s=np.einsum('mi,ij,mj->m',v,np.linalg.inv(b),v)[:,None]
            pref=np.exp(-.5*np.linalg.slogdet(b)[1]-.5*d*np.log(s))
            qprime=pref*(-2*d*c[None,:]/s)*hyp1f1(d/2+1,1.5,-c[None,:]**2/s)
            a=.5*h*w[:,None]*qprime
            bias=w@profile(np.zeros((len(v),1)),v,b)[:,0]-np.sum(a*np.tanh(-gamma*c))
            y=evaluate(x,v,c,gamma,a[:,:,None],np.array([bias]))[:,0]
            truth=np.exp(-np.einsum('ni,ij,nj->n',x,b,x))
            control.append(dict(d=d,directions=m,centers=201,target=name,plain_rel_l2=relative(y,truth)))
    (OUT/'plain_control.json').write_text(json.dumps(control,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for a,budget in zip(ax,[52000,208000]):
        for target,color in [('isotropic','#2768a4'),('anisotropic','#c36342')]:
            r=[r for r in rows if r['budget']==budget and r['target']==target]
            a.loglog([t['directions'] for t in r],[t['tanh_rel_l2'] for t in r],'o-',color=color,label=target)
            if target=='anisotropic':
                for t in r:a.annotate('N='+str(t['centers']),(t['directions'],t['tanh_rel_l2']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
        a.set(title=f'≤{budget:,} neurons',xlabel='Directions M (centers decrease)',ylabel='Relative L2 error');a.grid(alpha=.2);a.legend(fontsize=9)
    fig.suptitle('3D fixed-budget test: allocate to directions until the center floor takes over')
    fig.savefig(OUT/'equal_budget.png',dpi=170)


if __name__=='__main__':main()
