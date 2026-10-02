"""Target-free diagnostic of restricting disk-orthogonal coordinates to a box.

SVD is used ONLY to measure conditioning, never to construct readouts or solve
a PDE. This isolates one potential conditioning source; it is not a proof
that it explains every observed Burgers solve error.
"""
import json
import numpy as np
from scipy.linalg import svdvals
from scipy.stats import qmc
from solver.ridge_pinn_features import RidgePINNFeatures
from ridge_pinn_study import OUT,disk_points


def run():
    rows=[]
    for p in [12,20,28]:
        f=RidgePINNFeatures(p,backend='polynomial')
        count=4096
        square=(2*qmc.Sobol(2,scramble=True,seed=819).random_base2(12)-1)/np.sqrt(2)
        result={}
        for name,x in [('disk',disk_points(count,719)),('inscribed_square',square)]:
            values=f.evaluate(x)/np.sqrt(count)
            singular=svdvals(values,check_finite=False)
            result[name]=dict(largest=float(singular[0]),smallest=float(singular[-1]),
                              condition=float(singular[0]/singular[-1]))
        rows.append(dict(degree=p,coordinates=f.size,sample_count=count,**result))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'ridge_pinn_conditioning.json').write_text(json.dumps(dict(rows=rows,
        target_used=False,PDE_solution_used=False,method='Singular values of known basis evaluation only; no PDE solve',
        scope='A sampled function-space coordinate condition number, not the nonlinear PDE Jacobian condition'),indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':run()
