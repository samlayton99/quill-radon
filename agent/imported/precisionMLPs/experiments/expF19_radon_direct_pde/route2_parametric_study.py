"""One ordinary MLP solves a continuum of diffusion-reaction BVPs.

Inputs (x,mu), x in [-1,1], mu in [1,2]. Equation -mu*u_xx+u=1,
u(-1,mu)=u(1,mu)=0. Only these equations/zero traces enter the solve.
The elementary cosh solution is heldout validation, never an initializer.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import json,time
from pathlib import Path
import numpy as np
import torch
from solver.general_domains import ResidualDomain
from solver.general_residual import ResidualBlock
from solver.general_adaptive import ResidualDeclaration
from solver.route2 import solve_route2,ordinary_jets

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/parametric'
DOMAIN=ResidualDomain([[-1.,1.],[1.,2.]])


def blocks(count,seed):
    x=DOMAIN.interior(count,seed)
    # Each physical endpoint needs the full parameter interval. Assigning
    # alternate Sobol rows to opposite endpoints aliases their leading bits.
    traces=[]
    for side in (-1.,1.):
        face=DOMAIN.interior(max(40,count//8),seed+113+int(side+1)*104729)
        face[:,0]=side;traces.append(face)
    traces=np.concatenate(traces)
    return [ResidualBlock('equation',x,lambda x,j,p:-x[:,1:2]*j[(2,0)]+j[(0,0)]-1,((0,0),(2,0))),
            ResidualBlock('boundary',traces,lambda x,j,p:j[(0,0)],((0,0),))]


def truth(x):
    if isinstance(x,np.ndarray):return 1-np.cosh(x[:,0:1]/np.sqrt(x[:,1:2]))/np.cosh(1/np.sqrt(x[:,1:2]))
    return 1-torch.cosh(x[:,0:1]/torch.sqrt(x[:,1:2]))/torch.cosh(1/torch.sqrt(x[:,1:2]))


def main():
    torch.set_num_threads(1);started=time.perf_counter()
    s=solve_route2(ResidualDeclaration(DOMAIN,blocks),degrees=(6,10,14,18,22),tolerance=1e-9,
                   max_seconds=100,maximum_neurons=30000,solver_options={'max_iterations':8})
    model=s.export();points=DOMAIN.interior(1009,51859)
    began=time.perf_counter()
    with torch.no_grad():prediction=model(torch.tensor(points)).numpy()
    inference=time.perf_counter()-began;target=truth(points)
    x=torch.tensor(points[:101],requires_grad=True,dtype=torch.float64)
    target_derivative=torch.autograd.grad(truth(x).sum(),x)[0][:,1:2].detach().numpy()
    ordinary=ordinary_jets(model,points[:101],[(0,1)])[(0,1)].numpy()
    sensitivity=float(np.linalg.norm(ordinary-target_derivative)/np.linalg.norm(target_derivative))
    curve_points=np.array([(x,mu) for mu in [1.03,1.271,1.509,1.733,1.97] for x in np.linspace(-1,1,151)])
    with torch.no_grad():curves=model(torch.tensor(curve_points)).numpy()
    curve_truth=truth(curve_points)
    row=dict(problem='-mu*u_xx+u=1; zero endpoint values; mu is input, not an inferred scalar',
        data_policy='No solution labels; source1 and zero boundary only; exact cosh formula used after solve for validation',
        status=s.status,driver=s.metrics,history=s.history,features=s.problem.features.metrics,
        ordinary_relative_l2=float(np.linalg.norm(prediction-target)/np.linalg.norm(target)),
        ordinary_maximum_error=float(abs(prediction-target).max()),
        parameter_sensitivity_relative_l2=sensitivity,inference_1009_points_seconds=inference,
        total_seconds=time.perf_counter()-started,
        scope='One network represents this smooth two-input parameter family on [1,2]; not an arbitrary function-to-function neural operator, no extrapolation claim')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'metrics.json').write_text(json.dumps(row,indent=2)+'\n')
    torch.save(dict(state_dict=model.state_dict(),hidden=model[0].out_features),OUT/'plain_mlp.pt')
    np.savez_compressed(OUT/'curves.npz',points=curve_points,prediction=curves,target=curve_truth)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,mu in enumerate([1.03,1.271,1.509,1.733,1.97]):
        sl=slice(i*151,(i+1)*151);xs=curve_points[sl,0]
        axes[0].plot(xs,curves[sl],label=f'mu={mu:g}')
        axes[0].plot(xs,curve_truth[sl],'k--',alpha=.3,lw=.8)
        axes[1].semilogy(xs,np.maximum(abs(curves[sl]-curve_truth[sl]),1e-17))
    axes[0].set(xlabel='x',ylabel='u(x,mu)',title='One flat tanh network; heldout parameters');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='x',ylabel='Absolute error',title='Ordinary exported network')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'parameter_family.png',dpi=180);plt.close(fig)
    print(json.dumps({k:row[k] for k in ['status','ordinary_relative_l2','parameter_sensitivity_relative_l2','total_seconds']},indent=2))


if __name__=='__main__':main()
