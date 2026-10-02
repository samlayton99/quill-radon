"""Analytic angular reconstruction exported as an ordinary flat tanh MLP.

This is a KNOWN-FUNCTION representation/derivative audit in 2--4 dimensions,
not a PDE solve. No fitting, SVD, or polynomial forward evaluator enters the
exported network. The source identity is (2.7) of Kerkyacharian et al. (2010):
E_omega[C_n^(d/2)(omega.x) C_n^(d/2)(omega.nu)] = C_n^(d/2)(nu.x).
Here omega and nu are unit vectors and E uses normalized sphere measure.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import eval_gegenbauer,roots_jacobi,poch
from quill_boundary import encode

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/pure_mlp_repair'


def sphere_rule(dimension,degree,fold=True):
    """Positive product cubature; exact through degree 2*degree in arithmetic.

    Folding antipodes preserves EVEN integrands only. Every angular product
    in the stated same-degree reconstruction identity is even.
    """
    if dimension<2 or int(dimension)!=dimension or degree<0 or int(degree)!=degree:
        raise ValueError('Integer dimension>=2 and degree>=0 required')
    q=degree+1
    def full(d):
        if d==2:
            angle=np.arange(2*q)*np.pi/q
            return np.column_stack((np.cos(angle),np.sin(angle))),np.full(2*q,1/(2*q))
        t,w=roots_jacobi(q,(d-3)/2,(d-3)/2);w=w/w.sum()
        rest,weights=full(d-1)
        vectors=np.concatenate([np.column_stack((np.full(len(rest),v),np.sqrt(1-v*v)*rest)) for v in t])
        return vectors,np.concatenate([a*weights for a in w])
    if not fold:return full(dimension)
    def half(d):
        # Fold the symmetric quadrature indices, not rounded floating-point
        # vectors. Rounded antipodes sometimes differed in their last digit,
        # creating duplicate directions and invalidating resource preflights.
        if d==2:
            angle=np.arange(q)*np.pi/q
            return np.column_stack((np.cos(angle),np.sin(angle))),np.full(q,1/q)
        t,w=roots_jacobi(q,(d-3)/2,(d-3)/2);w=w/w.sum()
        rest,weights=full(d-1)
        vectors=[];out_weights=[]
        for index in range((q+1)//2,q):
            v=t[index]
            vectors.append(np.column_stack((np.full(len(rest),v),np.sqrt(1-v*v)*rest)))
            out_weights.append((w[index]+w[q-1-index])*weights)
        if q%2:
            equator,equator_weights=half(d-1)
            vectors.append(np.column_stack((np.zeros(len(equator)),equator)))
            out_weights.append(w[q//2]*equator_weights)
        return np.concatenate(vectors),np.concatenate(out_weights)
    return half(dimension)


def ball_points(dimension,count,seed):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(count,dimension));x/=np.linalg.norm(x,axis=1)[:,None]
    return x*rng.random(count)[:,None]**(1/dimension)


def profile(t,dimension,degree,order=0):
    if order>degree:return np.zeros_like(t)
    lam=dimension/2
    return (2**order*poch(lam,order)*eval_gegenbauer(degree-order,lam+order,t)
            /eval_gegenbauer(degree,lam,1.))


def compile_network(dimension,degree,centers=129,lam=.2,angular_degree=None):
    directions,q=sphere_rule(dimension,degree if angular_degree is None else angular_degree)
    nu=np.arange(1,dimension+1,dtype=float);nu/=np.linalg.norm(nu)
    angular=q*eval_gegenbauer(degree,dimension/2,directions@nu)
    bank=encode(lambda t:profile(t,dimension,degree),centers-1,lam=lam,
                halo=math.ceil(math.sqrt(centers)))
    first=bank.gamma*np.repeat(directions,len(bank.centers),axis=0)
    bias=-bank.gamma*np.tile(bank.centers,len(directions))
    readout=(angular[:,None]*bank.weights[None,:]).ravel()
    output_bias=float(angular.sum()*bank.bias)
    model=torch.nn.Sequential(torch.nn.Linear(dimension,len(readout),dtype=torch.float64),
                              torch.nn.Tanh(),torch.nn.Linear(len(readout),1,dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(first));model[0].bias.copy_(torch.from_numpy(bias))
        model[2].weight.copy_(torch.from_numpy(readout[None,:]));model[2].bias.fill_(output_bias)
    model.requires_grad_(False)
    return model,nu,directions,q,bank


def model_derivative(model,points,order):
    x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    y=model(x)[:,0]
    for axis,multiplicity in enumerate(order):
        for _ in range(multiplicity):
            y=torch.autograd.grad(y.sum(),x,create_graph=True)[0][:,axis]
    return y.detach().numpy()


def audit(dimension,degree,centers,lam,angular_degree=None):
    start=time.perf_counter()
    model,nu,directions,q,bank=compile_network(dimension,degree,centers,lam,angular_degree)
    interior=ball_points(dimension,97,1179+dimension+degree)
    points=np.r_[interior,nu[None,:],-nu[None,:],np.zeros((1,dimension))]
    dot=points@directions.T
    angular=q*eval_gegenbauer(degree,dimension/2,directions@nu)
    angular_only=profile(dot,dimension,degree)@angular
    reference=profile(points@nu,dimension,degree)
    orders=[(0,)*dimension]
    for k in (1,2):
        for axis in range(dimension):
            a=[0]*dimension;a[axis]=k;orders.append(tuple(a))
    a=[0]*dimension;a[0]=a[1]=1;orders.append(tuple(a))
    checks=[]
    # Derivatives use a smaller independent set and include ridge endpoints.
    jet_points=np.r_[ball_points(dimension,17,6211+degree),points[-3:]]
    for order in orders:
        query=points if sum(order)==0 else jet_points
        factor=float(np.prod(nu**np.array(order)))
        truth=factor*profile(query@nu,dimension,degree,sum(order))
        values=model_derivative(model,query,order)
        error=values-truth
        scale=max(1.,float(abs(truth).max()))
        checks.append(dict(order=order,max_absolute=float(abs(error).max()),
                           max_scaled=float(abs(error).max()/scale),
                           rms=float(np.sqrt(np.mean(error**2)))))
    # Serialize the literal ordinary MLP; no source-only custom evaluator needed.
    stem=f'ridge_frame_d{dimension}_p{degree}_n{centers}_a{angular_degree if angular_degree is not None else degree}'
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(),dimension=dimension,
                    hidden_width=model[0].out_features,activation='tanh',scope='Known-function capacity only'),OUT/(stem+'.pt'))
    row=dict(dimension=dimension,degree=degree,interior_centers=centers,
             halo_per_side=bank.halo_per_side,lam=lam,directions=len(directions),
             neurons=model[0].out_features,architecture=['Linear','Tanh','Linear'],
             angular_degree=degree if angular_degree is None else angular_degree,
             angular_only_max_error=float(abs(angular_only-reference).max()),
             derivative_checks=checks,seconds=time.perf_counter()-start,
             scope='Known-function analytic reconstruction, no fitting; NOT a PDE solve',
             forward_contains_polynomials=False,forward_contains_product_gates=False,
             export=str(OUT/(stem+'.pt')))
    (OUT/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimensions',type=int,nargs='+',default=[2,3,4])
    parser.add_argument('--degrees',type=int,nargs='+',default=[4,6])
    parser.add_argument('--centers',type=int,nargs='+',default=[129,257])
    parser.add_argument('--lam',type=float,default=.2)
    args=parser.parse_args();torch.set_num_threads(1)
    rows=[audit(d,p,n,args.lam) for d in args.dimensions for p in args.degrees for n in args.centers]
    rows.append(audit(4,6,257,args.lam,angular_degree=3))
    (OUT/'ridge_frame_dimensions.json').write_text(json.dumps(rows,indent=2)+'\n')
