"""Routine interfaces for explicitly supported equation classes.

These are numerical solvers with constructed neural representations, not a
universal nonlinear PDE inverse. Unsupported structures raise rather than
silently substituting a different mathematical problem.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Callable
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import chi2
from quill_boundary import encode
from native_tanh import modes,features


@dataclass
class Resolution:
    modes:int=32
    centers:int|None=None
    lam:float|None=None
    lambda_candidates:tuple=(.15,.20,.25,.30)

    def __post_init__(self):
        if not isinstance(self.modes,(int,np.integer)) or isinstance(self.modes,bool):
            raise ValueError('modes must be an integer')
        if self.centers is not None and (not isinstance(self.centers,(int,np.integer)) or isinstance(self.centers,bool)):
            raise ValueError('centers must be an integer')
        if self.modes<2 or (self.centers is not None and self.centers<5):
            raise ValueError('Need at least 2 modes and 5 interior centers')
        if self.lam is not None and self.lam<=0:
            raise ValueError('lambda must be positive')
        if not self.lambda_candidates or any(not np.isfinite(v) or v<=0 for v in self.lambda_candidates):
            raise ValueError('lambda_candidates must contain positive finite values')


@dataclass
class PeriodicEvolution:
    """Scalar 1D u_t=rhs(t,x,u,ux,uxx,parameters), periodic at domain endpoints."""
    initial:Callable
    rhs:Callable
    domain:tuple=(-np.pi,np.pi)
    parameters:dict=field(default_factory=dict)
    stiff:bool=False
    name:str='periodic_evolution'


@dataclass
class Conservation1D:
    """Conservative, entropy-dissipating backend for supported scalar fluxes."""
    initial:Callable
    domain:tuple=(-1.,1.)
    boundary:str='periodic'
    flux:str='burgers'


@dataclass
class ReactionDiffusion1D:
    """Periodic stiff reaction problem with optional physical invariant checks."""
    initial:Callable
    domain:tuple=(-np.pi,np.pi)
    diffusivity:float=.1
    reaction:str='allen_cahn'
    reaction_rate:float=100.


@dataclass
class RidgeLinear:
    """Whole-space constant-coefficient heat/advection; ridge IC supplied explicitly."""
    profiles:list
    directions:np.ndarray
    diffusivity:float=0.
    velocity:np.ndarray|None=None
    period:float=2.


class QuillBasis1D:
    def __init__(self,domain,resolution:Resolution):
        started=time.perf_counter()
        self.lo,self.hi=map(float,domain)
        if self.hi<=self.lo:raise ValueError('Domain endpoints must increase')
        self.scale=2*np.pi/(self.hi-self.lo)
        self.k=resolution.modes
        n=resolution.centers or 16*self.k+1
        q=6*self.k+4
        z=-np.pi+2*np.pi*np.arange(q)/q
        self.x=self.lo+(z+np.pi)/self.scale
        self.A=modes(z,self.k).T/q;self.A[1:]*=2
        candidates=[resolution.lam] if resolution.lam is not None else resolution.lambda_candidates
        checks=np.linspace(-np.pi,np.pi,513)
        w=np.arange(1,self.k+1);phase=checks[:,None]*w
        second=np.column_stack([np.zeros(len(checks)),-np.cos(phase)*w*w,-np.sin(phase)*w*w])
        encodings=[];sweep=[]
        for lam in candidates:
            enc=encode(lambda z:modes(z,self.k),n-1,lam,interval=(-np.pi,np.pi),halo=int(np.ceil(np.sqrt(n))))
            value_error=np.linalg.norm(enc.evaluate(checks)-modes(checks,self.k))/np.linalg.norm(modes(checks,self.k))
            derivative_error=np.linalg.norm(enc.evaluate(checks,2)-second)/np.linalg.norm(second)
            sweep.append(dict(lam=float(lam),value_error=float(value_error),d2_error=float(derivative_error)))
            encodings.append(enc)
        best=int(np.argmin([r['d2_error'] for r in sweep]))
        self.enc=encodings[best];self.E=np.vstack([self.enc.weights,self.enc.bias])
        p,dx,dxx=features(z,self.enc)
        self.U=p@self.E;self.Ux=self.scale*dx@self.E;self.Uxx=self.scale**2*dxx@self.E
        self.metadata=dict(modes=self.k,reduced_coordinates=2*self.k+1,interior_centers=n,
                           halo_per_side=self.enc.halo_per_side,neurons=len(self.enc.centers),
                           lam=self.enc.lam,lambda_sweep=sweep,quadrature_points=q,
                           cached_bytes=sum(a.nbytes for a in [self.A,self.E,self.U,self.Ux,self.Uxx]),
                           setup_seconds=time.perf_counter()-started,
                           geometry='uniform centers, corrected sqrt(N) halos; actual QUILL derivatives')

    def matrix(self,x,derivative=0):
        x=np.atleast_1d(np.asarray(x,float))
        if derivative not in [0,1,2]:raise ValueError('Only derivatives 0,1,2 supported')
        if np.any(x<self.lo-1e-12) or np.any(x>self.hi+1e-12):
            raise ValueError('Query lies outside the constructed domain')
        out=[];block=max(1,1000000//len(self.enc.centers))
        for start in range(0,len(x),block):
            z=(x[start:start+block]-self.lo)*self.scale-np.pi
            out.append(features(z,self.enc)[derivative]@self.E*self.scale**derivative)
        return np.concatenate(out,axis=0)

    def tail_ratio(self,b):
        begin=max(1,int(.75*self.k))
        idx=np.r_[np.arange(1+begin,1+self.k),np.arange(1+self.k+begin,1+2*self.k)]
        return float(np.linalg.norm(b[idx])/max(np.linalg.norm(b),1e-30))


class EvolutionSolution:
    def __init__(self,basis,result,problem,metrics,status):
        self.basis,self.result,self.problem=basis,result,problem
        self.metrics,self.status=metrics,status
        self.time_interval=(float(result.t[0]),float(result.t[-1]))

    def evaluate(self,x,t,derivative=0):
        t=np.asarray(t,float)
        if np.any(t<self.time_interval[0]) or np.any(t>self.time_interval[1]+1e-14):
            raise ValueError('Time query outside valid trajectory; extrapolation is disabled')
        values=self.basis.matrix(x,derivative)@self.result.sol(t)
        return values if t.ndim==0 else values.T

    def readouts(self,t):
        if t<self.time_interval[0] or t>self.time_interval[1]:raise ValueError('Invalid time')
        weights=self.basis.E@self.result.sol(t)
        centers=self.basis.lo+(self.basis.enc.centers+np.pi)/self.basis.scale
        return dict(centers=centers,gamma=self.basis.enc.gamma*self.basis.scale,
                    weights=weights[:-1],bias=weights[-1])

    def residual(self,x,t):
        b=self.result.sol(t);q=self.basis
        nodal=self.problem.rhs(t,q.x,q.U@b,q.Ux@b,q.Uxx@b,self.problem.parameters)
        ut=q.matrix(x)@(q.A@nodal)
        return ut-self.problem.rhs(t,np.asarray(x),self.evaluate(x,t),self.evaluate(x,t,1),
                                  self.evaluate(x,t,2),self.problem.parameters)


def solve_periodic(problem:PeriodicEvolution,t_end,resolution=None,rtol=1e-8,atol=1e-10,
                   max_step=np.inf,amplitude_limit=1e4,tail_limit=.01,initial_tolerance=1e-6):
    if t_end<=0:raise ValueError('t_end must be positive')
    q=QuillBasis1D(problem.domain,resolution or Resolution())
    y0=np.asarray(problem.initial(q.x),float)
    if y0.shape!=q.x.shape or not np.all(np.isfinite(y0)):
        raise ValueError('Initial condition must return one finite scalar per point')
    endpoints=np.asarray(problem.initial(np.asarray(problem.domain)),float)
    if abs(endpoints[1]-endpoints[0])>1e-8*max(1,np.max(abs(y0))):
        raise ValueError('Periodic backend received incompatible endpoint initial values')
    b0=q.A@y0
    checks=np.linspace(q.lo,q.hi,259)
    ic_values=np.asarray(problem.initial(checks),float)
    ic_error=float(np.linalg.norm(q.matrix(checks)@b0-ic_values)/max(np.linalg.norm(ic_values),1e-30))
    if initial_tolerance is not None and ic_error>initial_tolerance:
        raise ValueError(f'Initial condition is underresolved: relative check error {ic_error:.3g}; increase modes')
    def rhs(t,b):
        return q.A@np.asarray(problem.rhs(t,q.x,q.U@b,q.Ux@b,q.Uxx@b,problem.parameters),float)
    def large(t,b):return amplitude_limit-float(np.max(abs(q.U@b)))
    def unresolved(t,b):return 1. if tail_limit is None else tail_limit-q.tail_ratio(b)
    large.terminal=unresolved.terminal=True;large.direction=unresolved.direction=-1
    if large(0,b0)<=0 or unresolved(0,b0)<=0:raise ValueError('Initial state exceeds validity threshold; refine or change limit')
    started=time.perf_counter()
    result=solve_ivp(rhs,(0,t_end),b0,method='BDF' if problem.stiff else 'DOP853',rtol=rtol,atol=atol,
                     max_step=max_step,dense_output=True,events=(large,unresolved))
    elapsed=time.perf_counter()-started
    status='reached_t_end' if result.success else 'integrator_failure'
    if len(result.t_events[0]):status='amplitude_limit'
    if len(result.t_events[1]):status='underresolved'
    metrics={**q.metadata,'solve_seconds':elapsed,'nfev':result.nfev,'implicit_factorizations':result.nlu,
             'initial_projection_relative_error':ic_error,
             'accepted_steps':len(result.t)-1,'final_time':float(result.t[-1]),'tail_ratio':q.tail_ratio(result.y[:,-1]),
             'time_method':'BDF' if problem.stiff else 'DOP853','status':status,
             'scope':'scalar smooth periodic 1D, actual neural RHS; no solution labels or neural readout fit',
             'residual_note':'Spatial semidiscrete residual; time error is controlled separately by integrator/refinement.',
             'validity_note':'Tail/amplitude events are diagnostics, not certified PDE error or blow-up proofs.'}
    return EvolutionSolution(q,result,problem,metrics,status)


def _field_agreement_control(problem,t_end,tolerance=1e-6,starting_modes=8,maximum_modes=128,**options):
    """Historical negative-control algorithm; NOT the public adaptive solver.

    Agreement alone can miss the same unresolved forcing at two resolutions.
    Retained only to reproduce the measured failure in verification_study.py.
    """
    if tolerance<=0 or maximum_modes<starting_modes:raise ValueError('Invalid refinement budget')
    history=[];previous=None;latest=None;k=starting_modes
    initial_tolerance=options.pop('initial_tolerance',tolerance)
    x=np.linspace(*problem.domain,389)
    times=np.linspace(0,t_end,5)
    while k<=maximum_modes:
        try:
            current=solve_periodic(problem,t_end,Resolution(modes=k),initial_tolerance=initial_tolerance,**options)
        except ValueError as exc:
            if 'underresolved' not in str(exc) and 'validity threshold' not in str(exc):raise
            history.append(dict(modes=k,status='initial_underresolved',detail=str(exc)))
            k*=2;continue
        latest=current
        row=dict(modes=k,status=current.status,setup_seconds=current.metrics['setup_seconds'],
                 solve_seconds=current.metrics['solve_seconds'])
        if current.status=='reached_t_end':
            values=current.evaluate(x,times)
            if previous is not None:
                discrepancy=float(np.linalg.norm(values-previous)/max(np.linalg.norm(values),1e-30))
                row['relative_refinement_discrepancy']=discrepancy
                if discrepancy<tolerance:
                    history.append(row);current.metrics['refinement_history']=history
                    current.metrics['requested_spatial_tolerance']=tolerance
                    current.metrics['spatial_tolerance_met_empirically']=True
                    current.metrics['all_refinements_seconds']=sum(r.get('setup_seconds',0)+r.get('solve_seconds',0) for r in history)
                    return current
            previous=values
        elif current.status=='amplitude_limit':
            history.append(row);current.metrics['refinement_history']=history
            return current
        history.append(row);k*=2
    if latest is None:raise RuntimeError('Resolution budget exhausted before an acceptable initial representation')
    latest.status='refinement_limit';latest.metrics['status']=latest.status
    latest.metrics['refinement_history']=history
    latest.metrics['spatial_tolerance_met_empirically']=False
    latest.metrics['requested_spatial_tolerance']=tolerance
    return latest


def solve_refined(problem,t_end,tolerance=1e-6,starting_modes=8,maximum_modes=128,**options):
    """Adaptive spatial solve with independent PDE and time-refinement checks."""
    from .verification import solve_verified
    solution=solve_verified(problem,t_end,tolerance=tolerance,starting_modes=starting_modes,
                            maximum_modes=maximum_modes,**options)
    v=solution.metrics['verification']
    solution.metrics.update(spatial_tolerance_met_empirically=solution.status=='checks_passed',
                            requested_spatial_tolerance=tolerance,
                            refinement_history=v['spatial_history'],
                            all_refinements_seconds=v['complete_call_seconds'])
    return solution


def solve(problem,**options):
    """Dispatch a declared equation class; no silent universal-PDE claim."""
    if isinstance(problem,PeriodicEvolution):
        adaptive=options.pop('adaptive',False)
        return solve_refined(problem,**options) if adaptive else solve_periodic(problem,**options)
    if isinstance(problem,Conservation1D):
        if problem.flux!='burgers':raise NotImplementedError('Only convex Burgers flux currently implemented')
        from .hyperbolic import solve_burgers
        return solve_burgers(problem.initial,x_bounds=problem.domain,boundary=problem.boundary,**options)
    if isinstance(problem,ReactionDiffusion1D):
        from .reaction import solve_reaction_diffusion
        options.setdefault('invariant_tolerance',1e-3)
        return solve_reaction_diffusion(problem.initial,x_bounds=problem.domain,diffusivity=problem.diffusivity,
                                       reaction=problem.reaction,reaction_rate=problem.reaction_rate,**options)
    if isinstance(problem,RidgeLinear):
        from .highdim import solve_ridge_linear
        return solve_ridge_linear(problem.profiles,problem.directions,diffusivity=problem.diffusivity,
                                  velocity=problem.velocity,period=problem.period,**options)
    from .tensor_box import BoxProblem,solve_box,solve_box_refined
    if isinstance(problem,BoxProblem):
        adaptive=options.pop('adaptive',False)
        return solve_box_refined(problem,**options) if adaptive else solve_box(problem,**options)
    from .elliptic import AnnulusProblem,solve_annulus
    if isinstance(problem,AnnulusProblem):return solve_annulus(problem,**options)
    raise NotImplementedError('Declare PeriodicEvolution, ReactionDiffusion1D, Conservation1D, RidgeLinear, BoxProblem, or AnnulusProblem; arbitrary PDE parsing is not implemented')


def calibrate(forward,values,initial,bounds=(-np.inf,np.inf),sigma=1.,jacobian=None,max_nfev=100):
    """Small physical inverse problem; forward(z) supplies only measured quantities.

    sigma specifies known independent Gaussian observation standard deviations.
    Covariance is a LOCAL linearized approximation conditional on exact model
    and sigma. Rank-deficient observations return no identifiable covariance.
    """
    values=np.asarray(values,float).ravel();sigma=np.broadcast_to(np.asarray(sigma,float),values.shape)
    if np.any(sigma<=0):raise ValueError('Measurement scales must be positive; use1 for unweighted noiseless fitting')
    initial=np.asarray(initial,float)
    fun=lambda z:(np.asarray(forward(z)).ravel()-values)/sigma
    jac='3-point' if jacobian is None else lambda z:np.asarray(jacobian(z)).reshape(len(values),len(initial))/sigma[:,None]
    start=time.perf_counter()
    result=least_squares(fun,initial,jac=jac,bounds=bounds,xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=max_nfev)
    _,s,vt=np.linalg.svd(result.jac,full_matrices=False)
    rank=int(np.sum(s>(s[0]*1e-10 if len(s) else 0)))
    bounded=bool(np.any(result.active_mask))
    covariance=(vt.T/s**2)@vt if rank==len(initial) and not bounded else None
    degrees_of_freedom=len(values)-rank
    chi_squared=float(np.sum(result.fun**2))
    pvalue=float(chi2.sf(chi_squared,degrees_of_freedom)) if degrees_of_freedom>0 and not bounded else None
    return dict(parameters=result.x,success=bool(result.success),message=result.message,nfev=result.nfev,
                seconds=time.perf_counter()-start,residual=result.fun,jacobian=result.jac,rank=rank,
                active_mask=result.active_mask,uncertainty_valid_for_local_gaussian=rank==len(initial) and not bounded,
                chi_squared=chi_squared,degrees_of_freedom=degrees_of_freedom,
                reduced_chi_squared=chi_squared/degrees_of_freedom if degrees_of_freedom>0 else None,
                local_goodness_of_fit_pvalue=pvalue,
                observation_consistency='inconsistent_with_assumed_noise' if pvalue is not None and pvalue<.001 else 'not_rejected' if pvalue is not None else 'not_assessed',
                covariance=covariance,standard_deviation=None if covariance is None else np.sqrt(np.diag(covariance)),
                uncertainty_scope='local Gaussian linearization conditional on specified model/noise; unavailable at active bounds or rank deficiency; not a global posterior')
