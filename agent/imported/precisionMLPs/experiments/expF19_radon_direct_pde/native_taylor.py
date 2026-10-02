"""Direct time-polynomial construction for native projected QUILL Navier--Stokes.

For the finite polynomial ODE b'=L b+B(b,b), construct Taylor coefficients
(n+1)b[n+1]=L b[n]+sum_j B(b[j],b[n-j]). No time stepping in construction.
This is a classical high-order Taylor method on a spatially truncated system,
not a universal closed-form solution of nonlinear PDEs. References only afterward.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/native-taylor-mpl')
from pathlib import Path
import json,time
import numpy as np
from native_ns_pilot import NeuralCoordinates,Coordinates,initial,integrate,relative
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/native_dynamics'

def construct(model,degree):
    start=time.perf_counter()
    coeff=[model.analyze(initial(model.points))]
    fields=[]
    for n in range(degree):
        fields.append(model.fields(coeff[n]))
        rhs=model.viscosity*fields[n][2]
        for j in range(n+1):
            rhs=rhs-np.einsum('nj,nij->ni',fields[j][0],fields[n-j][1])
        coeff.append(model.analyze(rhs)/(n+1))
    return coeff,time.perf_counter()-start

def evaluate(coeff,t):
    value=coeff[-1].copy()
    for c in coeff[-2::-1]:value=c+t*value
    return value

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    points=2*np.pi*(qmc.Sobol(3,scramble=True,seed=830).random_base2(9)-.5)
    model=NeuralCoordinates(5,18,.05,n_interior=257,lam=.2)
    coeff,seconds=construct(model,12)
    final_time=.05
    outputs={p:model.direct(evaluate(coeff[:p+1],final_time),points)
             for p in [2,4,6,8,10,12]}
    # Only now run numerical time-integration references for verification.
    b0=model.analyze(initial(model.points))
    same,stime=integrate(model.spectral_rhs,b0,final_time,.0005)
    same_values=model.heldout_spectral(same,points)
    classical=Coordinates(5,18,.05)
    classical.fields=lambda b:classical.synthesize_spectral(b,True)
    classic_coeff,classic_seconds=construct(classical,12)
    classic_values=classical.heldout_spectral(evaluate(classic_coeff,final_time),points)
    ref=Coordinates(9,30,.05)
    br,rtime=integrate(ref.spectral_rhs,ref.analyze(initial(ref.points)),final_time,.0005)
    truth=ref.heldout_spectral(br,points)
    rows=[dict(time_degree=p,relative_error_vs_same_spatial_cutoff=relative(y,same_values),
               relative_error_vs_cutoff9=relative(y,truth),max_absolute_error_vs_cutoff9=float(np.max(abs(y-truth))))
          for p,y in outputs.items()]
    # Recurrence residual in latent coordinates, independent of time marching.
    t=.037
    b=evaluate(coeff,t)
    derivative=evaluate([n*coeff[n] for n in range(1,len(coeff))],t)
    rhs=model.rhs(b)
    last_norms=[float(np.linalg.norm(c)*final_time**n) for n,c in enumerate(coeff)]
    data=dict(method='finite Taylor construction of b_t=A_K F(S E b)',
              recurrence='(n+1)b[n+1]=A_K[nu Lap(SE b[n])-sum_j (SE b[j] dot grad)(SE b[n-j])]',
              production_time_steps=0,production_readout_solves=0,
              build=model.build,construction_seconds=seconds,T=final_time,
              degree12_relative_coefficient_ode_residual=float(np.linalg.norm(derivative-rhs)/np.linalg.norm(rhs)),
              evaluation_time_for_residual=t,term_norms_at_T=last_norms,
              same_cutoff_reference_timing=stime,refined_reference_timing=rtime,rows=rows,
              classical_taylor_construction_seconds=classic_seconds,
              native_vs_classical_taylor_relative=relative(outputs[12],classic_values),
              limits='Local analytic series of a finite spatially projected ODE; no general infinite-PDE time-analyticity or convergence radius certified. This is numerical arithmetic without time stepping, not a new universal closed form.')
    (OUT/'taylor_metrics.json').write_text(json.dumps(data,indent=2))
    np.savez_compressed(OUT/'taylor_coefficients.npz',time_coefficients=np.array(coeff),
                        wavevectors=model.k,points=points,reference=truth,
                        native_degree12=outputs[12],T=final_time)
    fig,ax=plt.subplots(figsize=(7.5,4.3),constrained_layout=True)
    ax.semilogy([r['time_degree'] for r in rows],[r['relative_error_vs_same_spatial_cutoff'] for r in rows],'-o',label='Time-series error: same spatial cutoff')
    ax.semilogy([r['time_degree'] for r in rows],[r['relative_error_vs_cutoff9'] for r in rows],'-s',label='Total error: finer spatial reference')
    ax.set(xlabel='Degree of constructed polynomial in time',ylabel='Relative velocity error',
           title='3D nonlinear Navier–Stokes, t=.05: no time stepping')
    ax.grid(alpha=.25);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=9)
    fig.savefig(OUT/'taylor_convergence.png',dpi=180)
    print(json.dumps(data),flush=True)

if __name__=='__main__':run()
