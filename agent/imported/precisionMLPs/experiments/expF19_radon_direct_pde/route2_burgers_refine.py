"""Separate endpoint-collocation refinement from representation-only encoding.

The cold PDE solve changes only collocation. The separate N513 re-encoding
uses an already PDE-solved coordinate vector, never a reference solution, and
is explicitly labelled representation-only rather than a new PDE solve.
"""
import copy
import json
from math import ceil,sqrt
import numpy as np
import torch
from quill_boundary import encode
from solver.ball_ridge_features import BallRidgeFeatures,gegenbauer_profiles
from route2_burgers_study import OUT,run,points,reference,raw_values,raw_torch_audit


def load_geometry(stem):
    data=np.load(OUT/(stem+'_geometry.npz'))
    # Restore the exact saved target-independent map without repeating its
    # arbitrary-precision construction. No neural solution labels enter.
    f=BallRidgeFeatures.__new__(BallRidgeFeatures)
    f.dimension=2;f.degree=36;f.bounds=np.array([[-.5,.5],[0.,1.]])
    f.midpoint=f.bounds.mean(axis=1);f.scale=2/np.diff(f.bounds,axis=1).ravel()/np.sqrt(2)
    for key in ['directions','angular_weights','profile_map','multiindices']:setattr(f,key,data[key])
    f.size=len(f.multiindices);f.physical_directions=f.directions*f.scale
    return f


def main():
    torch.set_num_threads(1)
    stem='route2_burgers_global_p36_n257_l0.2_s1'
    coefficients=np.load(OUT/(stem+'_slab0.npz'))['coefficients']
    base=load_geometry(stem)
    base.metrics=json.loads((OUT/(stem+'.json')).read_text())['slabs'][0]['features']
    checks=[]
    for n in [257,513]:
        f=copy.copy(base)
        f.encoding=encode(lambda z:gegenbauer_profiles(z,2,36),n-1,lam=.2,halo=ceil(sqrt(n)))
        f.encoding.evaluation_mode='anchored';f.encoding.anchor_x=-1.
        f.encoding.anchor_value=gegenbauer_profiles(np.array([-1.]),2,36)[0]
        f.metrics=dict(base.metrics,interior_centers=n,halo_per_side=ceil(sqrt(n)),tanh_count=len(f.directions)*len(f.encoding.centers))
        x=points(4096,39319);pred=raw_values(f.compile(coefficients),x);truth=reference(x)
        checks.append(dict(N=n,relative_l2=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)),
                           raw_torch=raw_torch_audit(f,coefficients,x[:257])))
        if n==257:cold_base=f
    (OUT/'route2_burgers_representation_refinement.json').write_text(json.dumps(dict(
        scope='Re-encoding of already PDE-solved coordinates only; not a new PDE solve',
        coefficients_source=stem,reference_use='heldout validation only',rows=checks),indent=2))
    print(json.dumps(checks),flush=True)
    run(36,seconds=60.,label='faces',base=cold_base,pde_faces=True)


if __name__=='__main__':main()
