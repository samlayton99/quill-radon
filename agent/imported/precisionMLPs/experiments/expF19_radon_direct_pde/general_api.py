"""Small reproducible examples using the common solver interface."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import json,time
import numpy as np
from solver import PeriodicEvolution,Resolution,BoxProblem,solve


def main():
    out=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver'
    out.mkdir(parents=True,exist_ok=True)
    cases=[]
    heat=PeriodicEvolution(domain=(0.,1.),initial=lambda x:.3+np.cos(2*np.pi*x)+.2*np.sin(4*np.pi*x),
         rhs=lambda t,x,u,ux,uxx,p:p['nu']*uxx-p['velocity']*ux,
         parameters={'nu':.04,'velocity':.3},name='heat_advection_scaled_domain')
    solution=solve(heat,t_end=.03,adaptive=True,tolerance=1e-8,maximum_modes=32,rtol=1e-10,atol=1e-12)
    x=np.linspace(0,1,1001)
    exact=.3+np.exp(-.04*(2*np.pi)**2*.03)*np.cos(2*np.pi*(x-.3*.03))+.2*np.exp(-.04*(4*np.pi)**2*.03)*np.sin(4*np.pi*(x-.3*.03))
    y=solution.evaluate(x,.03)
    cases.append(dict(name=heat.name,metrics=solution.metrics,relative_l2=float(np.linalg.norm(y-exact)/np.linalg.norm(exact))))
    burgers=PeriodicEvolution(initial=lambda x:np.sin(x)+.2*np.cos(2*x),
         rhs=lambda t,x,u,ux,uxx,p:-u*ux+p['nu']*uxx,parameters={'nu':.05},name='viscous_burgers')
    solution=solve(burgers,t_end=.5,adaptive=True,tolerance=1e-6,maximum_modes=128,rtol=1e-10,atol=1e-12)
    x=np.linspace(-np.pi,np.pi,1001)
    # Conventional verification is performed only after the native solve.
    from native_burgers import fft_baseline,eval_fourier
    ref,_=fft_baseline(512,end=.5,nu=.05,dt=.0001)
    exact=eval_fourier(ref,x);y=solution.evaluate(x,.5)
    res=solution.residual(x,.5)
    coeff=solution.readouts(.5)
    raw=np.tanh(coeff['gamma']*(x[:,None]-coeff['centers']))@coeff['weights']+coeff['bias']
    cases.append(dict(name=burgers.name,metrics=solution.metrics,relative_l2=float(np.linalg.norm(y-exact)/np.linalg.norm(exact)),
                      offgrid_residual_rms=float(np.sqrt(np.mean(res**2))),
                      exported_readout_equivalence=float(np.max(abs(raw-y)))))
    from native_elliptic import force,classical,classical_values
    start=time.perf_counter()
    box=solve(BoxProblem(force),adaptive=True,tolerance=1e-6,maximum_modes=20)
    elapsed=time.perf_counter()-start
    xy=np.random.default_rng(911).uniform(0,np.pi,(997,2))
    reference,mn,_=classical(28)
    truth=classical_values(xy,reference,mn)
    cases.append(dict(name='adaptive_nonlinear_dirichlet_box',metrics=box.metrics,
                      entire_adaptive_call_seconds=elapsed,
                      relative_l2=float(np.linalg.norm(box.evaluate(xy)-truth)/np.linalg.norm(truth))))
    output=dict(cases=cases,interface='Declared problem + solve(problem,t_end=...,adaptive=True,tolerance=...). Empirical refinement; no universal arbitrary-PDE parsing.')
    (out/'api_metrics.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
