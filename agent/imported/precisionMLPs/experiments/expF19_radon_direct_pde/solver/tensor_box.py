"""Shared QUILL feature banks with product gates for a 2D Dirichlet box.

This explicitly changes the architecture from a flat sum of ridge neurons.
The same constructed 1D bank is reused for all resolved sine modes. The
nonlinear PDE residual is evaluated with actual neural values/Laplacians.
"""
from dataclasses import dataclass
from typing import Callable
import time
import numpy as np
from quill_boundary import encode


@dataclass
class BoxProblem:
    forcing:Callable
    bounds:tuple=((0.,np.pi),(0.,np.pi))
    diffusivity:float=1.
    beta:float=5.


class TensorBasis:
    def __init__(self,bounds,k,n=None,lam=.2,backend='quill'):
        self.bounds=np.asarray(bounds,float);self.k=k;self.backend=backend
        self.length=self.bounds[:,1]-self.bounds[:,0]
        if self.bounds.shape!=(2,2) or np.any(self.length<=0):raise ValueError('A 2D increasing box is required')
        self.freq=np.arange(1,k+1)
        self.neurons=0
        if backend=='quill':
            n=n or 16*k+1
            self.enc=encode(lambda z:np.sin(np.pi/2*(np.asarray(z)[...,None]+1)*self.freq),
                            n-1,lam,interval=(-1,1),halo=int(np.ceil(np.sqrt(n))))
            self.end=self.enc.evaluate(np.array([-1.,1.]))
            self.neurons=2*len(self.enc.centers)
        elif backend!='classical':raise ValueError('Backend must be quill or classical')

    def banks(self,xy,second=False):
        xy=np.atleast_2d(np.asarray(xy,float))
        unit=(xy-self.bounds[:,0])/self.length
        if np.any(unit<-1e-12) or np.any(unit>1+1e-12):raise ValueError('Query outside box')
        values=[];d2=[]
        for axis in [0,1]:
            t=unit[:,axis]
            if self.backend=='classical':
                val=np.sin(np.pi*t[:,None]*self.freq)
                dd=-val*(np.pi*self.freq/self.length[axis])**2
            else:
                val=self.enc.evaluate(2*t-1)-(1-t[:,None])*self.end[0]-t[:,None]*self.end[1]
                dd=self.enc.evaluate(2*t-1,2)*(2/self.length[axis])**2 if second else None
            values.append(val)
            if second:d2.append(dd)
        return values,d2

    def matrices(self,xy):
        (x,y),(xx,yy)=self.banks(xy,True)
        tensor=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
        return tensor(x,y),tensor(xx,y)+tensor(x,yy)


class BoxSolution:
    def __init__(self,problem,basis,b,metrics):
        self.problem,self.basis,self.coefficients,self.metrics=problem,basis,b,metrics
        self.status='converged' if metrics['converged'] else 'iteration_limit'

    def evaluate(self,xy,derivatives=False):
        if derivatives:
            p,lap=self.basis.matrices(xy)
            return dict(value=p@self.coefficients,laplacian=lap@self.coefficients)
        (x,y),_=self.basis.banks(xy)
        return np.sum((x@self.coefficients.reshape(self.basis.k,self.basis.k))*y,axis=1)

    def residual(self,xy):
        f=self.evaluate(xy,True)
        return -self.problem.diffusivity*f['laplacian']+self.problem.beta*f['value']**3-self.problem.forcing(xy)


def solve_box(problem:BoxProblem,modes=12,n_centers=None,lam=.2,shift=5.,
              tolerance=2e-12,max_iterations=1000,backend='quill',spatial_tolerance=1e-5):
    if not isinstance(modes,(int,np.integer)) or modes<1 or not isinstance(max_iterations,(int,np.integer)) or max_iterations<1:
        raise ValueError('modes and max_iterations must be positive integers')
    if modes<1 or problem.diffusivity<=0 or problem.beta<0 or shift<0:
        raise ValueError('Require positive modes/diffusion and nonnegative beta/shift')
    started=time.perf_counter()
    basis=TensorBasis(problem.bounds,modes,n_centers,lam,backend)
    q=4*modes+1
    node=np.arange(1,q+1)/(q+1)
    unit=np.stack(np.meshgrid(node,node,indexing='ij'),axis=-1).reshape(-1,2)
    xy=basis.bounds[:,0]+unit*basis.length
    p,lap=basis.matrices(xy)
    exactx=np.sin(np.pi*unit[:,0,None]*basis.freq)
    exacty=np.sin(np.pi*unit[:,1,None]*basis.freq)
    a=4*(exactx[:,:,None]*exacty[:,None,:]).reshape(len(xy),-1).T/(q+1)**2
    eig=((np.pi*basis.freq[:,None]/basis.length[0])**2+(np.pi*basis.freq[None,:]/basis.length[1])**2).ravel()
    force=np.asarray(problem.forcing(xy),float)
    if force.shape!=(len(xy),) or not np.all(np.isfinite(force)):raise ValueError('Invalid forcing values')
    setup=time.perf_counter()-started
    t=time.perf_counter();b=np.zeros(modes*modes);history=[];converged=False;updates=0
    scale=max(np.linalg.norm(a@force),1e-30)
    for _ in range(max_iterations):
        value=p@b
        if not np.all(np.isfinite(value)) or np.max(abs(value))>1e8:break
        r=a@(-problem.diffusivity*(lap@b)+problem.beta*value**3-force)
        error=float(np.linalg.norm(r)/scale);history.append(error)
        if error<tolerance:converged=True;break
        b-=r/(problem.diffusivity*eig+shift);updates+=1
    metrics=dict(backend=backend,representation='shared 1D QUILL banks + exact affine boundary correction + product gates' if backend=='quill' else 'classical sine products',
                 modes=modes,independent_coefficients=modes*modes,neurons=basis.neurons,
                 quadrature_points=len(xy),setup_seconds=setup,solve_seconds=time.perf_counter()-t,
                 total_seconds=time.perf_counter()-started,converged=converged,updates=updates,
                 projected_residual=history[-1] if history else float('inf'),history=history,
                 cached_bytes=sum(z.nbytes for z in [p,lap,a]),
                 no_readout_fit=True,scope='2D box, constantpositive diffusion, cubicnonlinearity, homogeneousDirichlet; explicit iteration can fail')
    solution=BoxSolution(problem,basis,b,metrics)
    # Discrete residual convergence can hide an unresolved forcing/solution.
    check=np.random.default_rng(673).uniform(size=(521,2))*basis.length+basis.bounds[:,0]
    physical=float(np.linalg.norm(solution.residual(check))/max(np.linalg.norm(problem.forcing(check)),1e-30))
    metrics['independent_relative_residual']=physical
    metrics['spatial_residual_tolerance']=spatial_tolerance
    metrics['spatial_verification']='independent sampled residual; not a continuum error certificate'
    metrics['spatial_residual_check_passed']=bool(physical<=spatial_tolerance)
    solution.status=('residual_checks_passed' if physical<=spatial_tolerance else 'underresolved') if converged else 'iteration_limit'
    metrics['status']=solution.status
    metrics['total_with_check_seconds']=time.perf_counter()-started
    return solution


def solve_box_refined(problem,tolerance=1e-6,starting_modes=4,maximum_modes=24,**options):
    previous=None;history=[];latest=None
    bounds=np.asarray(problem.bounds,float)
    probe=bounds[:,0]+np.random.default_rng(186).uniform(size=(701,2))*(bounds[:,1]-bounds[:,0])
    nonlinear_tolerance=options.pop('nonlinear_tolerance',2e-12)
    for k in range(starting_modes,maximum_modes+1,4):
        current=solve_box(problem,modes=k,tolerance=nonlinear_tolerance,spatial_tolerance=tolerance,**options)
        latest=current;value=current.evaluate(probe)
        row=dict(modes=k,status=current.status,seconds=current.metrics['total_with_check_seconds'],
                 independent_relative_residual=current.metrics['independent_relative_residual'])
        if previous is not None:
            change=float(np.linalg.norm(value-previous)/max(np.linalg.norm(value),1e-30))
            row['relative_refinement_discrepancy']=change
            if current.status=='residual_checks_passed' and change<tolerance:
                history.append(row);current.metrics['refinement_history']=history
                current.metrics['all_refinements_seconds']=sum(r['seconds'] for r in history)
                return current
        history.append(row);previous=value
    if latest is None:raise ValueError('Refinement budget contains no modes')
    latest.status='refinement_limit';latest.metrics['status']=latest.status
    latest.metrics['refinement_history']=history
    return latest
