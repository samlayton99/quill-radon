"""Bounded test of explicit box coordinates in the same flat ridge architecture.

Construction compares known basis values/derivatives only. No PDE solution,
target fitting, SVD, rank repair or learned warmstart enters the feature map.
"""
import json
import time
import numpy as np
import torch
from numpy.polynomial import legendre
from solver.ball_ridge_features import BallRidgeFeatures
from ridge_pinn_burgers import run,interior
from ridge_pinn_study import OUT


def main():
    torch.set_num_threads(1)
    rows=[]
    for p in [20,24]:
        started=time.perf_counter()
        features=BallRidgeFeatures([[-.5,.5],[0.,1.]],p,257,.2)
        points=interior(65,327)+0
        # Add actual corners and edge points: where polynomial conversion can
        # suffer cancellation and where PDE boundary data are enforced.
        points=np.r_[points,[[-.5,0.],[.5,0.],[-.5,1.],[.5,1.],[0.,0.],[0.,1.]]]
        normalized=(points-[0.,.5])*2
        checks=[]
        for derivative in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]:
            exact=np.ones((len(points),features.size))
            for axis,order in enumerate(derivative):
                coeff=legendre.legder(np.eye(p+1),m=order,axis=0)
                bank=legendre.legval(normalized[:,axis],coeff).T
                exact*=bank[:,features.multiindices[:,axis]]*2**order
            actual=features.evaluate(points,derivative)
            error=np.max(np.abs(actual-exact),axis=0)
            scale=np.maximum(1.,np.max(np.abs(exact),axis=0))
            checks.append(dict(derivative=derivative,maximum_absolute=float(error.max()),
                               maximum_scaled=float(np.max(error/scale))))
        worst=max(c['maximum_scaled'] for c in checks)
        row=dict(degree=p,features=features.metrics,known_basis_checks=checks,
                 preflight_seconds=time.perf_counter()-started,
                 gate='Skip PDE if maximum per-mode scaled value/derivative error exceeds 1e-3',
                 gate_passed=worst<=1e-3)
        print(json.dumps(row),flush=True)
        if worst<=1e-3:
            row['PDE_run']=run(p,backend='quill',seconds=30.,krylov=900,iterations=12,
                label='box',feature_factory=lambda degree,centers:features)
        else:
            row['status']='basis_construction_failed_preflight; no PDE solve attempted'
        row['all_preflight_solve_validation_seconds']=time.perf_counter()-started
        rows.append(row)
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'ridge_pinn_box_followup.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
