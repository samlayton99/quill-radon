"""Cross-equation tests of one constructed-feature residual solver.

All equation knowledge lives in declarations here, not the solver. Analytic
solutions are used for boundary/initial data, manufactured forcing where noted,
and separate validation. No interior reference labels enter forward cases.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from solver.general_domains import ResidualDomain
from solver.general_features import ConstructedFeatures
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual


OUT = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual'


def exact_jets(exact, points, derivatives):
    """Analytic Torch derivatives used only to declare data/validate a study."""
    x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    y = exact(x)
    if y.ndim == 1: y = y[:, None]
    d = x.shape[1]
    cache = {(0,)*d: y}
    def get(alpha):
        if alpha not in cache:
            axis = next(i for i,v in enumerate(alpha) if v)
            parent = list(alpha); parent[axis] -= 1
            z = get(tuple(parent))
            columns=[]
            for c in range(z.shape[1]):
                if z[:,c].requires_grad:
                    g=torch.autograd.grad(z[:,c].sum(), x, create_graph=True,
                                          retain_graph=True, allow_unused=True)[0]
                    columns.append(torch.zeros_like(x[:,axis]) if g is None else g[:,axis])
                else: columns.append(torch.zeros_like(x[:,axis]))
            cache[alpha]=torch.stack(columns,axis=1)
        return cache[alpha]
    return {tuple(a):get(tuple(a)).detach() for a in derivatives}


def value_block(name, points, exact, dimension, component=None, weight=1.):
    key=(0,)*dimension
    target=exact_jets(exact,points,[key])[key]
    if component is not None:target=target[:,component:component+1]
    def residual(x,j,p):
        z=j[key] if component is None else j[key][:,component:component+1]
        return z-target
    return ResidualBlock(name,points,residual,(key,),weight=weight)


@dataclass
class Case:
    name:str
    domain:ResidualDomain
    exact:object
    operator:object
    derivatives:tuple
    constraints:object
    fields:int=1
    manufactured:bool=False
    parameter_initial:tuple=()
    parameter_truth:tuple=()
    parameter_bounds:object=None
    note:str=''

    def blocks(self, n, seed):
        x=self.domain.interior(n,seed)
        if self.manufactured:
            data=self.operator(torch.tensor(x,dtype=torch.float64),
                               exact_jets(self.exact,x,self.derivatives),
                               torch.zeros((len(x),0),dtype=torch.float64)).detach()
        else:data=0.
        def residual(z,j,p):return self.operator(z,j,p)-data
        blocks=[ResidualBlock('PDE',x,residual,self.derivatives)]
        blocks.extend(self.constraints(max(80,int(np.sqrt(n))*8),seed+13))
        return blocks


def spatial_dirichlet(case_domain, exact, fields=None):
    def constraints(n,seed):
        b=case_domain.boundary(n,seed)
        if fields is None:
            return [value_block('Dirichlet',b.points,exact,case_domain.dimension)]
        return [value_block('Dirichlet_'+str(k),b.points,exact,case_domain.dimension,k) for k in fields]
    return constraints


def space_time_constraints(domain,exact,initial_derivative=False):
    def constraints(n,seed):
        left=domain.interior(n//2,seed);left[:,0]=domain.bounds[0,0]
        right=domain.interior(n-n//2,seed+104729);right[:,0]=domain.bounds[0,1]
        x=np.concatenate([left,right])
        initial=domain.interior(n,seed+1);initial[:,1]=domain.bounds[1,0]
        blocks=[value_block('spatial_boundary',x,exact,2),value_block('initial',initial,exact,2)]
        if initial_derivative:
            target=exact_jets(exact,initial,[(0,1)])[(0,1)]
            blocks.append(ResidualBlock('initial_time_derivative',initial,
                lambda x,j,p:j[(0,1)]-target,((0,1),)))
        return blocks
    return constraints


def cases():
    box=ResidualDomain([[-1,1],[-1,1]])
    truth=lambda x:torch.exp(.3*x[:,0])*torch.cos(1.3*x[:,1])+.1*x[:,0]*x[:,1]
    op=lambda x,j,p:-j[(2,0)]-j[(0,2)]+j[(0,0)]**3
    cubic=Case('nonlinear_diffusion',box,truth,op,((0,0),(2,0),(0,2)),
               spatial_dirichlet(box,truth),manufactured=True)

    # Nonconvex implicit domain with offset circular hole; same sampler/solver.
    def level(x):
        outer=(x[:,0]/1.0)**2+(x[:,1]/.8)**2-1
        hole=(x[:,0]-.25)**2+(x[:,1]+.08)**2-.22**2
        return outer*hole
    perforated=ResidualDomain([[-1.05,1.05],[-.85,.85]],level)
    def variable(x,j,p):
        xx=x[:,0:1]; yy=x[:,1:2]
        return -((1+xx*xx)*j[(2,0)]+2*xx*j[(1,0)]
                 +(2+yy*yy)*j[(0,2)]+2*yy*j[(0,1)]+.2*j[(1,1)])+.5*j[(0,0)]**3
    irregular=Case('irregular_variable_diffusion',perforated,truth,variable,
        ((0,0),(1,0),(0,1),(2,0),(0,2),(1,1)),spatial_dirichlet(perforated,truth),
        manufactured=True,note='Implicit perforated ellipse; variable anisotropic diffusion, cubic reaction.')

    plate_truth=lambda x:(1-x[:,0]**2)**2*(1-x[:,1]**2)**2
    plate_op=lambda x,j,p:j[(4,0)]+2*j[(2,2)]+j[(0,4)]
    def plate_constraints(n,seed):
        b=box.boundary(n,seed);normal=torch.tensor(b.normals,dtype=torch.float64)
        jt=exact_jets(plate_truth,b.points,[(1,0),(0,1)])
        target=normal[:,0:1]*jt[(1,0)]+normal[:,1:2]*jt[(0,1)]
        return [value_block('clamped_value',b.points,plate_truth,2),
                ResidualBlock('clamped_normal_derivative',b.points,
                    lambda x,j,p:normal[:,0:1]*j[(1,0)]+normal[:,1:2]*j[(0,1)]-target,
                    ((1,0),(0,1)))]
    plate=Case('biharmonic_plate',box,plate_truth,plate_op,((4,0),(2,2),(0,4)),
               plate_constraints,manufactured=True,note='Fourth-order equation and clamped constraints.')

    st=ResidualDomain([[-1,1],[0,1]])
    nu=.1;alpha=.5
    def burgers_truth(x):
        a=alpha*torch.exp(-nu*math.pi**2*x[:,1])
        return 2*nu*math.pi*a*torch.sin(math.pi*x[:,0])/(1+a*torch.cos(math.pi*x[:,0]))
    burgers_op=lambda x,j,p:j[(0,1)]+j[(0,0)]*j[(1,0)]-nu*j[(2,0)]
    burgers=Case('burgers_ivp',st,burgers_truth,burgers_op,((0,0),(1,0),(0,1),(2,0)),
        space_time_constraints(st,burgers_truth),note='Unforced nonlinear IVP; Cole-Hopf expression only validates and supplies IC.')

    wave_domain=ResidualDomain([[-1,1],[0,.5]])
    def wave_truth(x):
        return torch.sin(math.pi*x[:,0])*torch.cos(math.pi*x[:,1])+.3/(2*math.pi)*torch.sin(2*math.pi*x[:,0])*torch.sin(2*math.pi*x[:,1])
    wave=Case('wave_ivp',wave_domain,wave_truth,lambda x,j,p:j[(0,2)]-j[(2,0)],
        ((0,2),(2,0)),space_time_constraints(wave_domain,wave_truth,True),note='Second-order time derivative; no final-time solution labels.')

    ns_domain=ResidualDomain([[0,1],[-.5,.5]])
    reynolds=20.;decay=reynolds/2-math.sqrt(reynolds**2/4+4*math.pi**2)
    def ns_truth(x):
        e=torch.exp(decay*x[:,0]);s=torch.sin(2*math.pi*x[:,1]);c=torch.cos(2*math.pi*x[:,1])
        return torch.stack((1-e*c,decay/(2*math.pi)*e*s,(1-e*e)/2),axis=1)
    def ns_op(x,j,p):
        u,v=j[(0,0)][:,0],j[(0,0)][:,1]
        dx,dy=j[(1,0)],j[(0,1)];lap=j[(2,0)]+j[(0,2)]
        return torch.stack((u*dx[:,0]+v*dy[:,0]+dx[:,2]-lap[:,0]/reynolds,
                            u*dx[:,1]+v*dy[:,1]+dy[:,2]-lap[:,1]/reynolds,
                            dx[:,0]+dy[:,1]),axis=1)
    def ns_constraints(n,seed):
        result=spatial_dirichlet(ns_domain,ns_truth,[0,1])(n,seed)
        result.append(value_block('pressure_gauge',np.array([[0.,0.]]),ns_truth,2,2))
        return result
    ns=Case('steady_navier_stokes',ns_domain,ns_truth,ns_op,
        ((0,0),(1,0),(0,1),(2,0),(0,2)),ns_constraints,fields=3,
        note='Unforced coupled velocity-pressure residual; no Fourier pressure projector or NS-specific solve.')

    ball=ResidualDomain([[-1,1],[-.8,.8],[-.6,.6]],
        lambda x:x[:,0]**2+(x[:,1]/.8)**2+(x[:,2]/.6)**2-1)
    t3=lambda x:torch.exp(.3*x[:,0])*torch.cos(.7*x[:,1])*torch.sin(.8*x[:,2])
    op3=lambda x,j,p:-j[(2,0,0)]-j[(0,2,0)]-j[(0,0,2)]+j[(0,0,0)]**3
    three=Case('three_dimensional_ellipsoid',ball,t3,op3,
        ((0,0,0),(2,0,0),(0,2,0),(0,0,2)),spatial_dirichlet(ball,t3),manufactured=True)

    heat_truth=lambda x:torch.sin(math.pi*x[:,0])*torch.exp(-.2*math.pi**2*x[:,1])
    heat_op=lambda x,j,p:j[(0,1)]-p[:,0:1]*j[(2,0)]
    def inverse_constraints(n,seed):
        result=space_time_constraints(st,heat_truth)(n,seed)
        observations=st.interior(24,817)
        result.append(value_block('observations',observations,heat_truth,2))
        return result
    inverse=Case('inverse_diffusivity',st,heat_truth,heat_op,
        ((0,1),(2,0)),inverse_constraints,parameter_initial=(.1,),
        parameter_truth=(.2,),parameter_bounds=((.02,),(1.,)),
        note='24 noiseless measurements; joint coefficient/physical-parameter solve.')
    return {c.name:c for c in [cubic,irregular,plate,burgers,wave,ns,three,inverse]}


def evaluate_case(case,solution,n=1009):
    x=case.domain.interior(n,9183)
    target=case.exact(torch.tensor(x,dtype=torch.float64)).detach().numpy()
    if target.ndim==1:target=target[:,None]
    predicted=solution.evaluate(x)
    relative=float(np.linalg.norm(predicted-target)/np.linalg.norm(target))
    component=[float(np.linalg.norm(predicted[:,i]-target[:,i])/max(np.linalg.norm(target[:,i]),1e-30)) for i in range(case.fields)]
    errors={};measurements={}
    for block in case.blocks(641,91837):
        jets={a:torch.tensor(solution.evaluate(block.points,derivative=a),dtype=torch.float64) for a in block.derivatives}
        pars=torch.tensor(np.broadcast_to(solution.parameters,(len(block.points),len(solution.parameters))).copy(),dtype=torch.float64)
        r=block.function(torch.tensor(block.points,dtype=torch.float64),jets,pars).detach().numpy()
        destination=measurements if block.name=='observations' else errors
        destination[block.name]=float(np.sqrt(np.mean(r*r)))
    return {'relative_l2':relative,'component_relative_l2':component,'independent_residual_rms':errors,
            'measurement_fit_rms':measurements,
            'parameter_relative_error':None if not case.parameter_truth else float(np.linalg.norm(solution.parameters-case.parameter_truth)/np.linalg.norm(case.parameter_truth))}


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def run(case,degree,backend,iterations,points=None):
    start=time.perf_counter()
    order=max(map(sum,case.derivatives))
    features=ConstructedFeatures(case.domain.bounds,degree,backend=backend,max_derivative=order)
    n=points or max(256,5*features.size)
    blocks=case.blocks(n,113)
    problem=ResidualProblem(features,blocks,fields=case.fields,
        parameter_initial=case.parameter_initial,parameter_bounds=case.parameter_bounds)
    before=time.perf_counter()
    solution=solve_residual(problem,max_iterations=iterations,tolerance=1e-9)
    elapsed=time.perf_counter()-before
    result={'case':case.name,'degree':degree,'backend':backend,'fields':case.fields,
            'coefficients':features.size*case.fields,'interior_points':n,
            'setup_seconds':before-start,'solve_seconds':elapsed,
            'features':features.metrics,'solver':solution.metrics,'history':solution.history,
            'validation':evaluate_case(case,solution),'manufactured_forcing':case.manufactured,
            'note':case.note}
    OUT.mkdir(parents=True,exist_ok=True)
    stem=f'{case.name}_{backend}_p{degree}'
    np.savez_compressed(OUT/(stem+'.npz'),coefficients=solution.coefficients,parameters=solution.parameters)
    (OUT/(stem+'.json')).write_text(json.dumps(clean(result),indent=2))
    print(json.dumps({'case':case.name,'degree':degree,'backend':backend,
        'seconds':elapsed,'validation':result['validation'],'solver':solution.metrics}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['nonlinear_diffusion'])
    parser.add_argument('--degrees',nargs='+',type=int,default=[4,8,12])
    parser.add_argument('--backend',choices=['quill','polynomial'],default='quill')
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--points',type=int)
    args=parser.parse_args();all_cases=cases()
    selected=list(all_cases) if args.cases==['all'] else args.cases
    for name in selected:
        for degree in args.degrees:run(all_cases[name],degree,args.backend,args.iterations,args.points)
