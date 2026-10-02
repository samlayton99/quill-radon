"""Target-free calibration of known ridge profiles and their PDE derivatives."""
import json
from math import ceil,sqrt,factorial
import numpy as np
from scipy.special import eval_gegenbauer
from solver.ball_ridge_features import gegenbauer_profiles
from solver.general_features import _evaluate_encoding
from quill_boundary import encode
from route2_burgers_study import OUT


def run(degree=36):
    x=np.unique(np.r_[np.cos(np.pi*np.arange(401)/400),2*np.mod(np.arange(1,257)*np.sqrt(2),1)-1])
    exact=[np.column_stack([np.zeros_like(x) if n<r else (2**r*factorial(r))*eval_gegenbauer(n-r,1+r,x)
                            for n in range(degree+1)]) for r in range(3)]
    rows=[]
    for n in [129,257,513]:
        for lam in [.15,.2,.25]:
            e=encode(lambda z:gegenbauer_profiles(z,2,degree),n-1,lam=lam,halo=ceil(sqrt(n)))
            e.evaluation_mode='anchored';e.anchor_x=-1.
            e.anchor_value=gegenbauer_profiles(np.array([-1.]),2,degree)[0]
            errors=[]
            for r in range(3):
                difference=np.max(np.abs(_evaluate_encoding(e,x,r)-exact[r]),axis=0)
                scale=np.maximum(1.,np.max(np.abs(exact[r]),axis=0))
                errors.append(dict(order=r,max_scaled=float(np.max(difference/scale)),
                                   per_mode_absolute=difference.tolist()))
            rows.append(dict(interior_centers=n,halo_per_side=ceil(sqrt(n)),lam=lam,degree=degree,errors=errors))
    OUT.mkdir(parents=True,exist_ok=True)
    record=dict(scope='Known one-dimensional profiles only, no PDE solution or target reference used',
                max_order=2,degree=degree,rows=rows)
    (OUT/'route2_burgers_profile_calibration.json').write_text(json.dumps(record,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='errors'}|{'error_by_order':[e['max_scaled'] for e in r['errors']]} for r in rows],indent=2))


if __name__=='__main__':run()
