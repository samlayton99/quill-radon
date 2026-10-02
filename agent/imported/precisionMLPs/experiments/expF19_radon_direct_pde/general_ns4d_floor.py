"""Reference-only approximation floors for the 4D NS precision study.

THIS DOES NOT SOLVE A PDE. Known analytic ABC/Beltrami solution coefficients
measure the representational capacity and construction error independently of
the residual solver. Never use these coefficients as a PDE initialization.
The PDE-solve study uses only solved prior coefficients and prescribed data.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'1')
import json
import numpy as np
import mpmath as mp
import torch
from solver.general_features import ConstructedFeatures
from general_ns4d_study import DOMAIN,NU,A,B,C,reference,ns_residual,DERIVATIVES,OUT


def legendre_exp(alpha,degree):
    """Exact Legendre coefficients of exp(alpha*s), evaluated at 70 digits."""
    with mp.workdps(70):
        alpha=mp.mpc(alpha);z=-mp.j*alpha
        if alpha==0:return np.r_[1.,np.zeros(degree)].astype(complex)
        return np.array([complex((2*n+1)*mp.j**n*mp.sqrt(mp.pi/(2*z))*
                                  mp.besselj(n+mp.mpf('.5'),z))
                         for n in range(degree+1)])


def reference_coefficients(indices):
    """Analytic reference decomposition; not available to the PDE solver."""
    degree=int(np.max(indices))
    trig=legendre_exp(.5j,degree);sine=trig.imag;cosine=trig.real
    decay=np.exp(-NU/2)*legendre_exp(-NU/2,degree).real
    twice=np.exp(-NU)*legendre_exp(-NU,degree).real
    i,j,k,t=indices.T
    co=np.zeros((len(indices),4))
    co[:,0]=decay[t]*(A*sine[k]*(i==0)*(j==0)+C*cosine[j]*(i==0)*(k==0))
    co[:,1]=decay[t]*(B*sine[i]*(j==0)*(k==0)+A*cosine[k]*(i==0)*(j==0))
    co[:,2]=decay[t]*(C*sine[j]*(i==0)*(k==0)+B*cosine[i]*(j==0)*(k==0))
    co[:,3]=-twice[t]*(A*B*sine[i]*cosine[k]*(j==0)+
                       A*C*sine[k]*cosine[j]*(i==0)+
                       B*C*sine[j]*cosine[i]*(k==0))
    return co


def measure(features,points):
    coefficients=reference_coefficients(features.multiindices)
    truth=reference(torch.as_tensor(points)).numpy()
    jets={d:torch.as_tensor(features.evaluate(points,d)@coefficients) for d in DERIVATIVES}
    predicted=jets[(0,0,0,0)].numpy()
    error=predicted-truth
    residual=ns_residual(torch.as_tensor(points),jets,torch.zeros((len(points),0))).numpy()
    final=points.copy();final[:,3]=1.
    final_truth=reference(torch.as_tensor(final)).numpy()
    final_error=features.evaluate(final)@coefficients-final_truth
    return dict(backend=features.backend,degree=features.degree,features=features.size,
        nonzero_reference_rows=int(np.any(abs(coefficients)>1e-30,axis=1).sum()),
        tanh_count=features.metrics['tanh_count'],
        velocity_relative_l2=float(np.linalg.norm(error[:,:3])/np.linalg.norm(truth[:,:3])),
        pressure_relative_l2=float(np.linalg.norm(error[:,3])/np.linalg.norm(truth[:,3])),
        final_velocity_relative_l2=float(np.linalg.norm(final_error[:,:3])/np.linalg.norm(final_truth[:,:3])),
        final_pressure_relative_l2=float(np.linalg.norm(final_error[:,3])/np.linalg.norm(final_truth[:,3])),
        momentum_and_divergence_rms=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
        encoding_metrics=features.metrics)


def main():
    points=DOMAIN.interior(503,913738)
    rows=[]
    for degree in (4,6,8,10,12,14,16,18):
        for backend in ('polynomial','quill'):
            options={} if backend=='polynomial' else dict(centers=1025 if degree>=14 else 513,lam=.2)
            features=ConstructedFeatures(DOMAIN.bounds,degree,backend=backend,**options)
            row=measure(features,points);rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='encoding_metrics'}),flush=True)
            OUT.mkdir(parents=True,exist_ok=True)
            (OUT/'ns4d_reference_approximation_floor.json').write_text(json.dumps(dict(
                scope='Reference-only approximation diagnostic; NOT a PDE solve, NEVER used as PDE initialization',
                coefficient_source='Analytic Legendre expansion of validation reference, evaluated with70-digit Bessel functions',
                rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
