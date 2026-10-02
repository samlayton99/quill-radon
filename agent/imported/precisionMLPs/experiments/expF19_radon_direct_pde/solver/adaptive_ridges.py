"""Adaptive periodic ridge selection with actual QUILL fields in the PDE RHS.

This is adaptive Fourier-Galerkin selection in constructed neural coordinates,
not a new theory of learned geometry. Candidate directions are the periodic
integer frequency lines inside a declared cutoff. No least squares is used.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quill_boundary import encode


@dataclass
class AdaptiveRidgeProblem:
    initial: object
    reaction: object
    diffusivity: float = .03
    max_frequency: int = 8
    name: str = "adaptive_reaction_diffusion"


class AdaptiveRidgeBasis:
    def __init__(self, cutoff, backend="quill", encoding_tolerance=1e-9,
                 center_candidates=(33,49,65,97,129,193,257,385,513),
                 lambda_candidates=(.15,.2,.25), quadrature_side=None):
        began=time.perf_counter()
        if backend not in ["quill","classical"] or cutoff<1:
            raise ValueError("Require positive cutoff and quill/classical backend")
        self.backend=backend;self.cutoff=int(cutoff);self.encoding_tolerance=encoding_tolerance
        self.center_candidates=center_candidates;self.lambda_candidates=lambda_candidates
        self.k=np.array([(i,j) for i in range(cutoff+1) for j in range(-cutoff,cutoff+1)
                         if i>0 or j>0],int)
        self.q=quadrature_side or (4*cutoff+4)
        if self.q<=3*cutoff:raise ValueError("Quadratic dealiasing requires Q > 3K")
        axis=-np.pi+2*np.pi*np.arange(self.q)/self.q
        self.x=np.stack(np.meshgrid(axis,axis,indexing="ij"),axis=-1).reshape(-1,2)
        phase=self.x@self.k.T
        self.exact=np.column_stack([np.ones(len(self.x)),np.cos(phase),np.sin(phase)])
        self.A=self.exact.T/len(self.x);self.A[1:]*=2
        self.cache={0:(np.ones(len(self.x)),np.zeros(len(self.x)))}
        self.banks={};self.mode_info={};self.encoding_checks=[]
        self.setup_seconds=time.perf_counter()-began
        self.encoding_seconds=0.

    @property
    def nmodes(self):return len(self.k)

    def indices(self, active):
        j=np.flatnonzero(active)
        return np.r_[0,1+j,1+self.nmodes+j]

    def bank(self, frequency):
        if frequency in self.banks:return self.banks[frequency]
        before=time.perf_counter();points=np.linspace(-1,1,769)
        omega=np.pi*frequency
        profile=lambda z:np.stack([np.cos(omega*z),np.sin(omega*z)],axis=-1)
        truth=profile(points);second=-omega**2*truth
        selected=None
        for n in self.center_candidates:
            best=None
            for lam in self.lambda_candidates:
                enc=encode(profile,n-1,lam,halo=math.ceil(math.sqrt(n)))
                e0=float(np.linalg.norm(enc.evaluate(points)-truth)/np.linalg.norm(truth))
                e2=float(np.linalg.norm(enc.evaluate(points,2)-second)/np.linalg.norm(second))
                emax=float(np.max(abs(enc.evaluate(points)-truth)))
                edmax=float(np.max(abs(enc.evaluate(points,2)-second))/omega**2)
                row=dict(frequency=frequency,N=n,R=enc.halo_per_side,lam=lam,value_error=e0,d2_error=e2,
                         value_linf=emax,scaled_d2_linf=edmax)
                self.encoding_checks.append(row)
                if best is None or max(e0,e2,emax,edmax)<best[0]:best=(max(e0,e2,emax,edmax),enc,row)
            if best[0]<=self.encoding_tolerance:
                selected=best;break
        if selected is None:
            raise RuntimeError(f"No validated QUILL center/bandwidth choice for frequency {frequency}")
        self.banks[frequency]=(selected[1],selected[2])
        self.encoding_seconds+=time.perf_counter()-before
        return self.banks[frequency]

    def ensure(self,active):
        before=time.perf_counter()
        for j in np.flatnonzero(active):
            if j in self.mode_info:continue
            k=self.k[j];length=np.pi*np.sum(abs(k));norm2=k@k
            if self.backend=="classical":
                val=np.column_stack([self.exact[:,1+j],self.exact[:,1+self.nmodes+j]])
                lap=-norm2*val
                self.mode_info[j]=dict(backend="classical")
            else:
                enc,info=self.bank(int(np.sum(abs(k))))
                s=(self.x@k)/length
                val=enc.evaluate(s);lap=enc.evaluate(s,2)*norm2/length**2
                self.mode_info[j]=dict(**info,primitive=tuple((k//np.gcd.reduce(abs(k))).tolist()))
            self.cache[1+j]=(val[:,0],lap[:,0])
            self.cache[1+self.nmodes+j]=(val[:,1],lap[:,1])
        return time.perf_counter()-before

    def matrices(self,active):
        self.ensure(active);idx=self.indices(active)
        return np.column_stack([self.cache[i][0] for i in idx]),np.column_stack([self.cache[i][1] for i in idx]),self.A[idx],idx

    def evaluate(self,x,b,active,laplacian=False):
        x=np.atleast_2d(np.asarray(x,float))
        if np.any(abs(x)>np.pi+1e-12):raise ValueError("Query outside [-pi,pi]^2")
        y=np.full(len(x),b[0]);lap=np.zeros(len(x))
        for j in np.flatnonzero(active):
            k=self.k[j];coef=np.array([b[1+j],b[1+self.nmodes+j]])
            phase=x@k
            if self.backend=="classical":
                val=np.column_stack([np.cos(phase),np.sin(phase)])
                dd=-(k@k)*val
            else:
                length=np.pi*np.sum(abs(k));enc,_=self.bank(int(np.sum(abs(k))))
                val=enc.evaluate(phase/length)
                dd=enc.evaluate(phase/length,2)*(k@k)/length**2 if laplacian else None
            y+=val@coef
            if laplacian:lap+=dd@coef
        return (y,lap) if laplacian else y

    def geometry(self,b,active):
        """Combine coincident corrected tanh banks; count actual shared neurons."""
        directions={};blocks={}
        for j in np.flatnonzero(active):
            k=self.k[j];g=np.gcd.reduce(abs(k));p=tuple((k//g).tolist())
            directions[p]=directions.get(p,0)+1
            if self.backend=="classical":continue
            enc,info=self.bank(int(np.sum(abs(k))))
            key=(p,info['N'],info['lam'])
            if key not in blocks:
                v=np.asarray(p,float);v/=np.linalg.norm(v)
                length=np.pi*np.sum(abs(v))
                blocks[key]=dict(direction=v,centers=length*enc.centers,gamma=enc.gamma/length,
                                 weights=np.zeros(len(enc.centers)),bias=0.,N=info['N'],R=info['R'],lam=info['lam'])
            coeff=np.array([b[1+j],b[1+self.nmodes+j]])
            blocks[key]['weights']+=enc.weights@coeff
            blocks[key]['bias']+=float(enc.bias@coeff)
        rows=list(blocks.values())
        return dict(directions=len(directions),active_modes=int(active.sum()),
                    blocks=len(rows),neurons=sum(len(r['weights']) for r in rows),
                    center_choices=sorted(set(r['N'] for r in rows)),
                    lambda_choices=sorted(set(r['lam'] for r in rows)),
                    representation_bytes=sum(r['weights'].nbytes+r['centers'].nbytes+r['direction'].nbytes+16 for r in rows),
                    rows=rows,bias=b[0])


class AdaptiveRidgeSolution:
    def __init__(self,problem,basis,b,active,metrics,history):
        self.problem,self.basis,self.b,self.active=problem,basis,b,active
        self.metrics,self.history=metrics,history
        self.status="reached_t_end"

    def evaluate(self,x,laplacian=False):return self.basis.evaluate(x,self.b,self.active,laplacian)

    def residual(self,x):
        U,L,A,idx=self.basis.matrices(self.active)
        y=U@self.b[idx];f=self.problem.diffusivity*(L@self.b[idx])+self.problem.reaction(y)
        db=np.zeros_like(self.b);db[idx]=A@f
        ut=self.basis.evaluate(x,db,self.active)
        v,lap=self.evaluate(x,True)
        return ut-self.problem.diffusivity*lap-self.problem.reaction(v)


def solve_adaptive_ridges(problem,t_end=.5,dt=.005,adapt_interval=.05,
                           tolerance=1e-5,backend="quill",adaptive=True,
                           drop_fraction=.1,encoding_tolerance=1e-9,
                           fixed_active=None,rollback=False,max_replays=8):
    """Adapt missing PDE increments; no least squares, optimizer, or oracle.

    tolerance allocates an instantaneous omitted-increment budget proportional
    to interval / t_end. It is an a posteriori selection heuristic, not a bound
    on accumulated global PDE error: nonlinear amplification and contributions
    outside the candidate cutoff require separate validation/refinement.
    """
    if min(t_end,dt,adapt_interval,tolerance)<=0 or problem.diffusivity<0:
        raise ValueError("Positive time/tolerances and nonnegative diffusion required")
    if not isinstance(max_replays,int) or max_replays<1:raise ValueError("max_replays must be a positive integer")
    started=time.perf_counter()
    basis=AdaptiveRidgeBasis(problem.max_frequency,backend,encoding_tolerance)
    y0=np.asarray(problem.initial(basis.x),float)
    if y0.shape!=(len(basis.x),) or not np.isfinite(y0).all():raise ValueError("Invalid IC")
    b=basis.A@y0;nm=basis.nmodes
    amp=np.hypot(b[1:1+nm],b[1+nm:])
    # IC support extraction is quadrature, not a fitted readout.
    active=amp>max(1e-13,tolerance*.01)*max(np.linalg.norm(b),1e-30) if adaptive else np.ones(nm,bool)
    if fixed_active is not None:active=np.asarray(fixed_active,bool).copy()
    b[np.r_[False,~active,~active]]=0
    basis.ensure(active)
    ic_check=np.random.default_rng(127).uniform(-np.pi,np.pi,(257,2))
    expected=np.asarray(problem.initial(ic_check))
    ic_error=float(np.linalg.norm(basis.evaluate(ic_check,b,active)-expected)/max(np.linalg.norm(expected),1e-30))
    if ic_error>max(tolerance*.1,encoding_tolerance*10):
        raise ValueError(f"Initial data underresolved by candidate directions/cutoff: {ic_error:.3g}")
    setup_seconds=time.perf_counter()-started
    steps=math.ceil(t_end/dt);dt=t_end/steps
    every=max(1,round(adapt_interval/dt));interval=every*dt
    history=[];added_total=dropped_total=0;drop_bound=0.;peak_modes=int(active.sum())
    discovery_seconds=0.;transfer_seconds=0.;evolution_seconds=0.
    U,L,A,idx=basis.matrices(active)
    peak_working_bytes=U.nbytes+L.nbytes+A.nbytes
    step=0;actual_steps=0;rejected_windows=0;replay_attempt=0;rollback_added=0;replaying=False
    window_b=b.copy();window_start=0
    while step<steps:
        if adaptive and step%every==0 and not replaying:
            ta=time.perf_counter()
            y=U@b[idx];increment=problem.diffusivity*(L@b[idx])+problem.reaction(y)
            all_rate=basis.A@increment
            rate=np.hypot(all_rate[1:1+nm],all_rate[1+nm:])
            norm=max(np.sqrt(b[0]**2+.5*np.sum(b[1:]**2)),1e-30)
            missing=np.flatnonzero(~active)
            scores=.5*(interval*rate[missing])**2
            order=np.argsort(scores)[::-1];remaining=float(scores.sum())
            local_budget=tolerance*norm*interval/t_end
            added=[]
            for at in order:
                if remaining<=local_budget**2:break
                j=missing[at];active[j]=True;added.append(int(j));remaining-=scores[at]
            amp=np.hypot(b[1:1+nm],b[1+nm:])
            drop_budget=drop_fraction*local_budget;spent=0.;dropped=[]
            for j in np.argsort(amp):
                if not active[j] or j in added:continue
                if amp[j]+spent>drop_budget:break
                if interval*rate[j]>.1*local_budget:continue
                active[j]=False;spent+=amp[j];dropped.append(int(j))
                b[1+j]=b[1+nm+j]=0.
            discovery_seconds+=time.perf_counter()-ta
            tt=time.perf_counter();U,L,A,idx=basis.matrices(active)
            transfer_seconds+=time.perf_counter()-tt
            peak_working_bytes=max(peak_working_bytes,U.nbytes+L.nbytes+A.nbytes)
            # Existing coefficients are unchanged when a zero-amplitude mode is
            # added. Each cos/sin error is <= encerr, so a coefficient pair's
            # error is bounded by sqrt(2)*amplitude*encerr.
            drop_bound+=spent*(1+math.sqrt(2)*encoding_tolerance)
            added_total+=len(added);dropped_total+=len(dropped)
            peak_modes=max(peak_modes,int(active.sum()))
            history.append(dict(time=step*dt,active_modes=int(active.sum()),added=len(added),dropped=len(dropped),
                                omitted_increment_relative=np.sqrt(max(remaining,0))/norm,
                                omitted_rate_times_horizon_relative=np.sqrt(max(remaining,0))/norm*t_end/interval,
                                drop_linf_bound=spent*(1+math.sqrt(2)*encoding_tolerance)))
        if step%every==0 and not replaying:
            window_b=b.copy();window_start=step;replay_attempt=0
        replaying=False
        def rhs(z):
            value=U@z
            return A@(problem.diffusivity*(L@z)+problem.reaction(value))
        te=time.perf_counter();z=b[idx]
        a=rhs(z);bb=rhs(z+dt*a/2);c=rhs(z+dt*bb/2);dd=rhs(z+dt*c)
        b[idx]=z+dt*(a+2*bb+2*c+dd)/6
        evolution_seconds+=time.perf_counter()-te
        if not np.isfinite(b).all() or np.linalg.norm(b)>1e6:raise RuntimeError("Evolution diverged")
        step+=1;actual_steps+=1
        if rollback and adaptive and (step%every==0 or step==steps):
            ta=time.perf_counter()
            # Check the newly produced field before accepting this whole time
            # window. Adding a direction at the end cannot recover its missing
            # forcing earlier in the interval, so restore the saved start.
            y=U@b[idx];all_rate=basis.A@(problem.diffusivity*(L@b[idx])+problem.reaction(y))
            rate=np.hypot(all_rate[1:1+nm],all_rate[1+nm:])
            norm=max(np.sqrt(b[0]**2+.5*np.sum(b[1:]**2)),1e-30)
            missing=np.flatnonzero(~active);scores=.5*rate[missing]**2
            remaining=float(scores.sum());budget=(tolerance*norm/t_end)**2
            new=[]
            for at in np.argsort(scores)[::-1]:
                if remaining<=budget:break
                j=missing[at];active[j]=True;new.append(int(j));remaining-=scores[at]
            discovery_seconds+=time.perf_counter()-ta
            if new:
                if replay_attempt>=max_replays:
                    raise RuntimeError("Direction replay budget exhausted; refine adaptation interval or candidate pool")
                rejected_windows+=1;replay_attempt+=1;rollback_added+=len(new);added_total+=len(new)
                history.append(dict(time=window_start*dt,active_modes=int(active.sum()),added=len(new),dropped=0,
                                    event="rejected_window_replay",replay_attempt=replay_attempt,
                                    omitted_increment_relative=np.sqrt(max(remaining,0))/norm*interval,
                                    omitted_rate_times_horizon_relative=np.sqrt(max(remaining,0))/norm*t_end,
                                    drop_linf_bound=0.))
                # Every new coefficient is zero at the old time, preserving
                # the saved physical field without any fitted transfer.
                b=window_b.copy();step=window_start;replaying=True
                tt=time.perf_counter();U,L,A,idx=basis.matrices(active)
                transfer_seconds+=time.perf_counter()-tt
                peak_working_bytes=max(peak_working_bytes,U.nbytes+L.nbytes+A.nbytes)
                peak_modes=max(peak_modes,int(active.sum()))
    geometry=basis.geometry(b,active)
    cache_bytes=basis.A.nbytes+basis.exact.nbytes+sum(a.nbytes+c.nbytes for a,c in basis.cache.values())
    metadata=dict(problem=problem.name,backend=backend,adaptive=adaptive,
                  max_frequency=problem.max_frequency,candidate_modes=nm,final_modes=int(active.sum()),peak_modes=peak_modes,
                  steps=steps,dt=dt,adapt_interval=interval,tolerance=tolerance,
                  actual_rk_steps=actual_steps,rollback=rollback,rejected_windows=rejected_windows,
                  rollback_added_modes=rollback_added,max_replays=max_replays,
                  initial_relative_error=ic_error,
                  encoding_tolerance=encoding_tolerance,quadrature_points=len(basis.x),
                  added=added_total,dropped=dropped_total,cumulative_drop_linf_bound=drop_bound,
                  geometry={k:v for k,v in geometry.items() if k not in ['rows','bias']},
                  setup_seconds=setup_seconds,evolution_seconds=evolution_seconds,
                  discovery_seconds=discovery_seconds,transfer_and_encoding_seconds=transfer_seconds,
                  encoding_calibration_seconds=basis.encoding_seconds,
                  solve_wall_seconds=time.perf_counter()-started-setup_seconds,
                  total_seconds=time.perf_counter()-started,cache_bytes=cache_bytes,
                  peak_working_arrays_bytes=peak_working_bytes,
                  estimated_core_arrays_bytes=cache_bytes+peak_working_bytes,
                  selected_encoding_checks=[info for enc,info in basis.banks.values()],
                  method="adaptive spectral/Galerkin selection; actual QUILL field/Laplacian in RHS" if backend=='quill' else "same adaptive algorithm with exact sin/cos features",
                  validity="Tolerance budgets missing candidate increments per unit time; neither omitted candidate frequencies nor global error is certified",
                  memory_note="Estimated named core arrays, not RSS or total peak allocation; all candidate analysis columns are stored even with sparse active modes")
    return AdaptiveRidgeSolution(problem,basis,b,active,metadata,history)
