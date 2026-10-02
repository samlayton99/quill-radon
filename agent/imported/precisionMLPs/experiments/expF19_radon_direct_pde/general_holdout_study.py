"""Additional equations declared after freezing the shared default policy."""
import argparse
import math
import numpy as np
import torch
from general_residual_study import Case,exact_jets,spatial_dirichlet,value_block
from general_adaptive_study import run
from solver.general_domains import ResidualDomain
from solver.general_residual import ResidualBlock


def cases():
    domain=ResidualDomain([[-1,1],[-1,1]])
    exact=lambda x:.3*torch.sin(1.5*x[:,0])*torch.cos(1.2*x[:,1])
    def plap(x,j,p):
        a,b=j[(1,0)],j[(0,1)];aa,ab,bb=j[(2,0)],j[(1,1)],j[(0,2)]
        return -(1+a*a+b*b)*(aa+bb)-2*(a*a*aa+2*a*b*ab+b*b*bb)+j[(0,0)]**3
    first=Case('quasilinear_gradient_diffusion',domain,exact,plap,
        ((0,0),(1,0),(0,1),(2,0),(1,1),(0,2)),spatial_dirichlet(domain,exact),
        manufactured=True,note='Gradient-dependent elliptic operator; default common solver unchanged.')
    robin_exact=lambda x:torch.exp(.2*x[:,0])*torch.cos(1.1*x[:,1])
    robin_op=lambda x,j,p:-j[(2,0)]-j[(0,2)]+j[(0,0)]+j[(0,0)]**3
    def robin_constraints(n,seed):
        bd=domain.boundary(n,seed);isleft=np.isclose(bd.points[:,0],-1)
        points=bd.points[~isleft];normals=torch.tensor(bd.normals[~isleft],dtype=torch.float64)
        jt=exact_jets(robin_exact,points,[(0,0),(1,0),(0,1)])
        target=2*jt[(0,0)]+normals[:,0:1]*jt[(1,0)]+normals[:,1:2]*jt[(0,1)]
        return [value_block('left_dirichlet',bd.points[isleft],robin_exact,2),
                ResidualBlock('robin',points,lambda x,j,p:2*j[(0,0)]+normals[:,0:1]*j[(1,0)]+normals[:,1:2]*j[(0,1)]-target,
                              ((0,0),(1,0),(0,1)))]
    robin=Case('mixed_robin',domain,robin_exact,robin_op,((0,0),(2,0),(0,2)),
                robin_constraints,manufactured=True)
    coupled_exact=lambda x:torch.stack((torch.sin(1.2*x[:,0])+.4*torch.cos(.8*x[:,1]),
                                       torch.exp(.3*x[:,0])*torch.cos(1.4*x[:,1])),axis=1)
    def coupled_op(x,j,p):
        u,v=j[(0,0)][:,0],j[(0,0)][:,1];lap=j[(2,0)]+j[(0,2)]
        return torch.stack((-lap[:,0]+u+u*v*v,-2*lap[:,1]+v+v*u*u),axis=1)
    coupled=Case('coupled_reaction_diffusion',domain,coupled_exact,coupled_op,
        ((0,0),(2,0),(0,2)),spatial_dirichlet(domain,coupled_exact),fields=2,manufactured=True)
    interval=ResidualDomain([[0,1]]);epsilon=.01
    layer_exact=lambda x:(torch.exp((x[:,0]-1)/epsilon)-math.exp(-1/epsilon))/(1-math.exp(-1/epsilon))
    layer=Case('convection_boundary_layer',interval,layer_exact,
        lambda x,j,p:-epsilon*j[(2,)]+j[(1,)],((1,),(2,)),
        spatial_dirichlet(interval,layer_exact),note='Unforced convection-diffusion; epsilon=.01, no residual reweighting.')
    # Sharp smooth coefficient change, with well-defined smooth solution data.
    contrast_exact=lambda x:torch.sin(1.2*x[:,0])*torch.cos(.9*x[:,1])
    def contrast_op(x,j,p):
        s=torch.sigmoid(8*x[:,0:1]);a=1+999*s;ax=999*8*s*(1-s)
        return -a*(j[(2,0)]+j[(0,2)])-ax*j[(1,0)]+j[(0,0)]**3
    contrast=Case('high_contrast_diffusion',domain,contrast_exact,contrast_op,
        ((0,0),(1,0),(2,0),(0,2)),spatial_dirichlet(domain,contrast_exact),manufactured=True,
        note='Diffusion varies smoothly from about1 to1000; defaults unchanged.')
    return {c.name:c for c in (first,robin,coupled,layer,contrast)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cases',nargs='+',default=['all'])
    parser.add_argument('--backend',choices=['quill','polynomial'],default='quill')
    args=parser.parse_args();all_cases=cases()
    for name in (list(all_cases) if args.cases==['all'] else args.cases):run(all_cases[name],args.backend)
