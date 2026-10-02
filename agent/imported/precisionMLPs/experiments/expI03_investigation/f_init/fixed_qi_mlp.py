"""A compact QI MLP with fixed midpoint banks and data-calibrated initialization.

Each stage computes p = affine(z), then tanh(gamma * (p - centers)).
The shared affine projections are learned; bank centers and gamma are buffers.
Channels and center counts may vary by stage, supporting arbitrary depth.
There is no normalization in forward, no SVD in the default initializer, and
no readout solve during training. The optional polynomial initializer is only
an experimental control reproducing I02's degree-3 raw-profile recipe.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class FixedBank(nn.Module):
    def __init__(self, n: int, lam: float = .25):
        super().__init__()
        if n < 1 or lam <= 0:
            raise ValueError("Positive bank size and lambda are required")
        h=2./n
        self.register_buffer("centers",-1.+(torch.arange(n)+.5)*h)
        self.register_buffer("gamma",torch.tensor(lam/h))
        self.n,self.h,self.lam=n,h,lam

    def forward(self,p):
        return torch.tanh(self.gamma*(p.unsqueeze(-1)-self.centers))


def polynomial_profiles(n, count, generator, dtype, device, lam=.25):
    """I02 degree-3 random Chebyshev profiles fit over [-.8,.8], rcond1e-10.

    Uses one common SVD for the bank, rather than refactoring the identical bank
    for every profile. The same raw coefficient distribution results.
    """
    x=torch.linspace(-1.,1.,4001,dtype=torch.float64)
    x=x[x.abs()<=.8]
    centers=-1.+(torch.arange(n,dtype=torch.float64)+.5)*(2./n)
    B=torch.tanh((lam*n/2)*(x[:,None]-centers))
    angle=torch.arccos((x/.8).clamp(-1,1))
    basis=torch.stack([torch.cos(k*angle)/(k+1) for k in range(4)],1)
    xi=torch.randn(count,4,generator=generator,dtype=torch.float64)
    g=basis@xi.T
    g=(g-g.mean(0))/g.std(0)
    U,s,Vh=torch.linalg.svd(B,full_matrices=False)
    keep=s>s[0]*1e-10
    weights=(Vh[keep].T@((U[:,keep].T@g)/s[keep,None])).T
    return weights.to(device=device,dtype=dtype)


class FixedQIMLP(nn.Module):
    def __init__(self,d_in,channels=(12,12),centers=(11,11),d_out=1,lam=.25):
        super().__init__()
        if not channels or len(channels)!=len(centers):
            raise ValueError("channels and centers must be nonempty and equally long")
        self.channels=tuple(channels)
        self.center_counts=tuple(centers)
        self.projections=nn.ModuleList()
        self.banks=nn.ModuleList()
        for m,n in zip(channels,centers):
            self.projections.append(nn.Linear(d_in,m))
            self.banks.append(FixedBank(n,lam))
            d_in=m*n
        self.head=nn.Linear(d_in,d_out)

    def features(self,x):
        for projection,bank in zip(self.projections,self.banks):
            x=bank(projection(x)).flatten(1)
        return x

    def forward(self,x):
        return self.head(self.features(x))

    @torch.no_grad()
    def initialize_(self,x,mixer_init="random",calibration="range",seed=0,quantile=.999,max_rows=4096):
        """Only training inputs are used; calibration runs once.

        Directions/mixer rows start as independent unit Gaussian directions.
        Each row is divided by quantile(|projection|,.999), matching F04's
        occupied projection range. Thus gamma is bank-defined, not data-defined.
        The optional 'std' control centers projection means and sets sd=.4.
        Intermediate 'polynomial' rows are independent smooth I02 profiles over
        each incoming channel, then calibrated using the same selected rule.
        Initial head uses standard Linear uniform bounds with its own seed.
        """
        if mixer_init not in ("random","polynomial") or calibration not in ("range","std"):
            raise ValueError("Unknown initializer or calibration")
        x=x[:max_rows].to(self.head.weight)
        diagnostics=[]
        for i,(projection,bank) in enumerate(zip(self.projections,self.banks)):
            gen=torch.Generator().manual_seed(seed+100*i)
            if mixer_init=="polynomial" and i>0:
                m_prev,n_prev=self.channels[i-1],self.center_counts[i-1]
                w=polynomial_profiles(n_prev,self.channels[i]*m_prev,gen,projection.weight.dtype,projection.weight.device,self.banks[i-1].lam)
                w=w.reshape(self.channels[i],m_prev*n_prev)/math.sqrt(m_prev)
            else:
                w=torch.randn(projection.weight.shape,generator=gen,dtype=torch.float64).to(projection.weight)
                w=w/w.norm(dim=1,keepdim=True)
            projection.weight.copy_(w)
            projection.bias.zero_()
            p=projection(x)
            if calibration=="range":
                scale=p.abs().quantile(quantile,dim=0).clamp_min(torch.finfo(p.dtype).eps).reciprocal()
                mean=torch.zeros_like(scale)
            else:
                mean=p.mean(0)
                scale=.4/p.std(0,unbiased=False).clamp_min(torch.finfo(p.dtype).eps)
            projection.weight.mul_(scale[:,None])
            projection.bias.copy_(-mean*scale)
            p=projection(x)
            diagnostics.append(dict(stage=i,mean=p.mean(0).tolist(),std=p.std(0,unbiased=False).tolist(),
                                    abs_quantile=p.abs().quantile(quantile,dim=0).tolist(),
                                    outside_fraction=(p.abs()>1).double().mean().item(),
                                    weight_norm=projection.weight.norm().item()))
            x=bank(p).flatten(1)
        gen=torch.Generator().manual_seed(seed+10000)
        bound=1/math.sqrt(self.head.in_features)
        for p in (self.head.weight,self.head.bias):
            v=torch.empty(p.shape,dtype=torch.float64).uniform_(-bound,bound,generator=gen)
            p.copy_(v)
        return diagnostics

    @torch.no_grad()
    def expanded_mlp(self):
        """Equivalent ordinary dense tanh MLP, useful for verification only."""
        modules=[]
        for projection,bank in zip(self.projections,self.banks):
            layer=nn.Linear(projection.in_features,projection.out_features*bank.n).to(projection.weight)
            layer.weight.copy_((bank.gamma*projection.weight).repeat_interleave(bank.n,0))
            layer.bias.copy_((bank.gamma*(projection.bias[:,None]-bank.centers)).flatten())
            modules.extend((layer,nn.Tanh()))
        head=nn.Linear(self.head.in_features,self.head.out_features).to(self.head.weight)
        head.load_state_dict(self.head.state_dict());modules.append(head)
        return nn.Sequential(*modules)
