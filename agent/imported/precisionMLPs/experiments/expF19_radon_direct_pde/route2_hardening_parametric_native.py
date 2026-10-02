"""Transfer the generic native preset to a nonpolynomial parameter family.

The PDE declaration is reused unchanged from route2_parametric_study. Only
equations and zero boundary traces enter solve_route2_native; the elementary
cosh expression is called afterward for independent ordinary-MLP validation.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('KMP_USE_SHM','0')
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from route2_parametric_study import DOMAIN,blocks,truth
from solver.general_adaptive import ResidualDeclaration
from solver.route2 import solve_route2_native,ordinary_jets,raw_residual_audit

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/parametric_native'


def predict_bounded(model,points,batch_size=128):
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(points[start:start+batch_size])).numpy()
                               for start in range(0,len(points),batch_size)])


def run(tolerance=1e-13,max_seconds=180.,degrees=(6,10,14,18,22),output=OUT):
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    solution=solve_route2_native(ResidualDeclaration(DOMAIN,blocks),degrees=degrees,
        centers=257,tolerance=tolerance,max_seconds=max_seconds,check_points=513,
        export_check_points=513)
    solve_seconds=time.perf_counter()-started
    solve_peak_rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Everything below is an audit of the already solved native neural state.
    model=solution.export()
    points=DOMAIN.interior(4097,20261001)
    prediction=predict_bounded(model,points)
    target=truth(points)
    sensitivity_points=points[:257]
    x=torch.tensor(sensitivity_points,dtype=torch.float64,requires_grad=True)
    target_sensitivity=torch.autograd.grad(truth(x).sum(),x)[0][:,1:2].detach().numpy()
    predicted_sensitivity=ordinary_jets(model,sensitivity_points,[(0,1)])[(0,1)].numpy()
    residual_audit=raw_residual_audit(solution.solution,blocks(2049,20261007))
    parameters=[1.03,1.271,1.509,1.733,1.97]
    curve_points=np.array([(value,mu) for mu in parameters for value in np.linspace(-1,1,201)])
    curves=predict_bounded(model,curve_points)
    curve_targets=truth(curve_points)
    row=dict(problem='-mu*u_xx+u=1; u(-1,mu)=u(1,mu)=0; (x,mu) in [-1,1] x [1,2]',
        method='solve_route2_native with no equation-specific solver options',
        data_policy='Equation and zero traces only; no interior solution labels, external PDE solve, or oracle initialization. Cosh solution used after solve only.',
        status=solution.status,driver=solution.metrics,history=solution.history,
        solver=solution.solution.metrics,features=solution.problem.features.metrics,
        requested_tolerance=tolerance,requested_degrees=list(degrees),centers=257,
        ordinary_relative_l2=float(np.linalg.norm(prediction-target)/np.linalg.norm(target)),
        ordinary_maximum_error=float(np.max(abs(prediction-target))),
        parameter_sensitivity_relative_l2=float(np.linalg.norm(predicted_sensitivity-target_sensitivity)/np.linalg.norm(target_sensitivity)),
        parameter_sensitivity_maximum_error=float(np.max(abs(predicted_sensitivity-target_sensitivity))),
        ordinary_residual_audit=residual_audit,solve_seconds=solve_seconds,
        solve_peak_process_rss_bytes=solve_peak_rss,
        total_seconds=time.perf_counter()-started,peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        scope='Smooth two-input family, finite heldout audits, ordinary float64 Linear/Tanh/Linear. No general-dimensional or continuum certification. Time budget is soft around products/linear solves.')
    (output/'metrics.json').write_text(json.dumps(row,indent=2)+'\n')
    np.savez_compressed(output/'native_coefficients.npz',coefficients=solution.coefficients,
        parameters=solution.parameters,degree=solution.problem.features.degree,centers=257,
        lam=.2,bounds=DOMAIN.bounds,multiindices=solution.problem.features.multiindices)
    torch.save(dict(state_dict=model.state_dict(),hidden=model[0].out_features),output/'plain_mlp.pt')
    np.savez_compressed(output/'curves.npz',points=curve_points,prediction=curves,target=curve_targets)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),constrained_layout=True)
    for index,mu in enumerate(parameters):
        sl=slice(index*201,(index+1)*201)
        axes[0].plot(curve_points[sl,0],curves[sl,0],label=f'mu={mu:g}')
        axes[0].plot(curve_points[sl,0],curve_targets[sl,0],'k--',lw=.7,alpha=.35)
        axes[1].semilogy(curve_points[sl,0],np.maximum(abs(curves[sl,0]-curve_targets[sl,0]),1e-17))
    axes[0].set(xlabel='x',ylabel='u(x,mu)',title='Ordinary tanh MLP; dashed exact audit')
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel='x',ylabel='Absolute function error',title='Heldout parameter values')
    for name in ('equation','boundary'):
        errors=[next(v['maximum_scaled_rms'] for v in item['validation'] if v['name']==name) for item in solution.history]
        axes[2].semilogy([item['degree'] for item in solution.history],errors,'o-',label=name)
    axes[2].axhline(tolerance,color='k',ls=':',lw=.8)
    axes[2].set(xlabel='Coordinate degree',ylabel='Fresh scaled residual RMS',title='Native degree continuation')
    axes[2].legend(fontsize=8)
    for axis in axes:axis.grid(alpha=.2)
    fig.savefig(output/'parameter_family.png',dpi=180);plt.close(fig)
    print(json.dumps({key:row[key] for key in ('status','ordinary_relative_l2','ordinary_maximum_error',
        'parameter_sensitivity_relative_l2','ordinary_residual_audit','solve_seconds','total_seconds','peak_process_rss_bytes')},indent=2),flush=True)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tolerance',type=float,default=1e-13)
    parser.add_argument('--seconds',type=float,default=180.)
    parser.add_argument('--degrees',type=int,nargs='+',default=[6,10,14,18,22])
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    run(args.tolerance,args.seconds,tuple(args.degrees),args.output)
