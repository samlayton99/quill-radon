"""Unforced Burgers IVP, solved globally through the flat ridge readout map.

No interior/final target values and no forcing are supplied to the solve.
Cole--Hopf is only a heldout reference; the supplied initial condition is
written explicitly and the two spatial boundary values are exactly zero.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import qmc
from solver.ridge_pinn_features import RidgePINNFeatures
from solver.general_residual import ResidualBlock,ResidualProblem,solve_residual
from ridge_pinn_study import OUT


def torch_audit(f,coefficients,points,save_path=None):
    net=f.compile(coefficients)
    l1=torch.nn.Linear(2,len(net['first_bias']),dtype=torch.float64)
    l2=torch.nn.Linear(len(net['first_bias']),1,dtype=torch.float64)
    with torch.no_grad():
        l1.weight.copy_(torch.tensor(net['first_weights']));l1.bias.copy_(torch.tensor(net['first_bias']))
        l2.weight.copy_(torch.tensor(net['output_weights'].T));l2.bias.copy_(torch.tensor(net['output_bias']))
    model=torch.nn.Sequential(l1,torch.nn.Tanh(),l2)
    if save_path is not None:torch.save(model.state_dict(),save_path)
    x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    u=model(x);du=torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    dxx=torch.autograd.grad(du[:,0].sum(),x)[0][:,0:1]
    r=du[:,1:2]+u*du[:,0:1]-NU*dxx
    truth=reference(points);pred=u.detach().numpy()
    norm=np.linalg.norm(truth)
    return dict(value_relative_l2=float(np.linalg.norm(pred-truth)/norm) if norm>1e-12*np.sqrt(len(points)) else None,
                value_maximum_absolute=float(np.max(np.abs(pred-truth))),
                PDE_absolute_rms=float(np.sqrt(np.mean(r.detach().numpy()**2))))

NU=.1
AMPLITUDE=.15
K=2*np.pi


def interior(n,seed):
    q=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
    q[:,0]-=.5
    return q


def initial_data(x):
    return 2*NU*K*AMPLITUDE*np.sin(K*x)/(1+AMPLITUDE*np.cos(K*x))


def reference(points):
    a=AMPLITUDE*np.exp(-NU*K*K*points[:,1:2])
    return 2*NU*K*a*np.sin(K*points[:,0:1])/(1+a*np.cos(K*points[:,0:1]))


def run(degree=20,centers=257,backend='quill',seconds=90.,krylov=900,iterations=12,label='',feature_factory=None):
    start=time.perf_counter()
    f=(RidgePINNFeatures(degree,centers=centers,backend=backend,radius=np.sqrt(.5),
        midpoint=(0.,.5),encoding_tolerance=1e8) if feature_factory is None else feature_factory(degree,centers))
    q=interior(max(768,8*f.size),18)
    count=max(100,4*(degree+1))
    initial=interior(count,11);initial[:,1]=0.
    left=interior(count,24);left[:,0]=-.5
    right=interior(count,34);right[:,0]=.5
    target=torch.tensor(initial_data(initial[:,0:1]),dtype=torch.float64)
    blocks=[ResidualBlock('PDE',q,lambda x,j,p:j[(0,1)]+j[(0,0)]*j[(1,0)]-NU*j[(2,0)],
                           ((0,0),(1,0),(0,1),(2,0))),
        ResidualBlock('initial',initial,lambda x,j,p:j[(0,0)]-target,((0,0),)),
        ResidualBlock('boundary',np.concatenate([left,right]),lambda x,j,p:j[(0,0)],((0,0),))]
    problem=ResidualProblem(f,blocks)
    s=solve_residual(problem,max_iterations=iterations,tolerance=1e-11,max_seconds=seconds,
        high_accuracy=True,linear_refinement_steps=0,damping=1e-10,
        lsmr_max_iterations=krylov,preconditioner='auto',preconditioner_block_size=128)
    solve_end=time.perf_counter()
    held=interior(2048,19317)
    value=s.evaluate(held);truth=reference(held)
    r=s.evaluate(held,(0,1))+value*s.evaluate(held,(1,0))-NU*s.evaluate(held,(2,0))
    final=np.column_stack([np.linspace(-.5,.5,501),np.ones(501)])
    rel=float(np.linalg.norm(value-truth)/np.linalg.norm(truth))
    errors=dict(relative_l2=rel,maximum_absolute=float(np.max(np.abs(value-truth))),
                final_relative_l2=float(np.linalg.norm(s.evaluate(final)-reference(final))/np.linalg.norm(reference(final))),
                fresh_pde_rms=float(np.sqrt(np.mean(r*r))))
    OUT.mkdir(parents=True,exist_ok=True)
    stem=f'ridge_pinn_burgers{("_"+label) if label else ""}_{backend}_p{degree}_m{f.metrics["directions"]}_n{centers}'
    audit={}
    if backend=='quill':
        if hasattr(f,'evaluate_flat'):
            flat=f.evaluate_flat(held,s.coefficients)
        else:
            net=f.compile(s.coefficients)
            flat=np.concatenate([np.tanh(x@net['first_weights'].T+net['first_bias'])@net['output_weights']+net['output_bias']
                                 for x in np.array_split(held,16)])
        errors['ordinary_flat_mlp_relative_l2']=float(np.linalg.norm(flat-truth)/np.linalg.norm(truth))
        errors['flat_stable_maximum_difference']=float(np.max(np.abs(flat-value)))
        audit['interior']=torch_audit(f,s.coefficients,held[:257],OUT/(stem+'_torch_state.pt'))
        boundary=interior(258,713);boundary[:129,0]=-.5;boundary[129:,0]=.5
        audit['spatial_boundary']=torch_audit(f,s.coefficients,boundary)
        initial=interior(257,818);initial[:,1]=0.
        audit['initial']=torch_audit(f,s.coefficients,initial)
    row=dict(case='unforced Burgers IVP',features=f.metrics,solver=s.metrics,history=s.history,
        errors=errors,ordinary_torch_autograd_audit=audit,construction_solve_seconds=solve_end-start,
        all_setup_solve_evaluation_seconds=time.perf_counter()-start,
        data_policy={'initial_coefficients':'zero','forcing':'zero','interior_solution_labels':False,
                     'boundary':'zero at x=±.5','initial':'explicit rational trigonometric function',
                     'Cole_Hopf':'heldout validation only'},
        physical_domain=[[-.5,.5],[0.,1.]],nu=NU,amplitude=AMPLITUDE,
        reference_formula='2*nu*k*a(t)*sin(k*x)/(1+a(t)*cos(k*x)), a(t)=0.15*exp(-nu*k^2*t), nu=0.1, k=2*pi',
        metric_normalization={'relative_l2':'Euclidean error norm / reference norm on 2048 independent uniform space-time Sobol points',
                              'fresh_pde_rms':'absolute sqrt(mean(residual^2)), physical derivatives, no equation scaling',
                              'final_relative_l2':'relative Euclidean norm on 501 uniform x points at t=1',
                              'boundary_audit':'value absolute error is meaningful; relative boundary error divides a theoretical zero and is not interpreted'},
        limitation='Global space-time residual solve, not time marching or a closed-form neural PDE solution')
    (OUT/(stem+'.json')).write_text(json.dumps(row,indent=2))
    arrays=dict(coefficients=s.coefficients,points=held,prediction=value,target=truth,residual=r)
    if backend=='quill':arrays.update(f.compile(s.coefficients))
    np.savez_compressed(OUT/(stem+'.npz'),**arrays)
    print(json.dumps(dict(stem=stem,status=s.status,**errors,seconds=row['all_setup_solve_evaluation_seconds'])),flush=True)
    return row


if __name__=='__main__':
    torch.set_num_threads(1)
    for p in [12,20,28]:
        run(p)
