"""Negative control: two agreeing spatial truncations can both be wrong."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
from solver import PeriodicEvolution,solve,solve_verified
from solver.base import _field_agreement_control

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    problem=PeriodicEvolution(initial=lambda x:np.zeros_like(x),
                              rhs=lambda t,x,u,ux,uxx,p:np.sin(x)+np.sin(17*x),name='unseen_forcing_control')
    old=_field_agreement_control(problem,t_end=.1,tolerance=1e-6,starting_modes=4,maximum_modes=16,tail_limit=None)
    new=solve_verified(problem,t_end=.1,starting_modes=4,maximum_modes=64,tail_limit=None)
    limited=solve_verified(problem,t_end=.1,starting_modes=4,maximum_modes=16,tail_limit=None)
    x=np.linspace(-np.pi,np.pi,1001);truth=.1*(np.sin(x)+np.sin(17*x))
    records=[]
    for name,s in [('field_agreement_only',old),('independent_physics_and_time',new),('insufficient_budget',limited)]:
        records.append(dict(method=name,status=s.status,relative_error=float(np.linalg.norm(s.evaluate(x,.1)-truth)/np.linalg.norm(truth)),metrics=s.metrics))
    burgers=PeriodicEvolution(lambda x:np.sin(x)+.2*np.cos(2*x),lambda t,x,u,ux,uxx,p:-u*ux+.05*uxx)
    b=solve_verified(burgers,t_end=.5,tolerance=1e-6,time_tolerance=1e-8,maximum_modes=128)
    records.append(dict(method='verified_burgers',status=b.status,metrics=b.metrics))
    oscillatory=PeriodicEvolution(lambda x:np.ones_like(x),
        lambda t,x,u,ux,uxx,p:np.sin(5*np.pi*t)**2*np.sin(17*x))
    for name,budget in [('temporal_alias_insufficient_budget',8),('temporal_alias_resolved',64)]:
        s=solve_verified(oscillatory,1.,starting_modes=4,maximum_modes=budget,tail_limit=None)
        exact=1+.5*np.sin(17*x)
        records.append(dict(method=name,status=s.status,
            relative_error=float(np.linalg.norm(s.evaluate(x,1.)-exact)/np.linalg.norm(exact)),metrics=s.metrics))
    (OUT/'verification_metrics.json').write_text(json.dumps(records,indent=2))
    print(json.dumps([{k:r[k] for k in ['method','status','relative_error'] if k in r} for r in records],indent=2))

if __name__=='__main__':main()
