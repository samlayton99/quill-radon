"""Four-input Navier--Stokes architecture repair, with a literal flat MLP.

This is a manufactured low-degree precision control, not a difficult-flow
benchmark: all-face Dirichlet traces already determine degree-four velocity.
Unknown pressure and the nonlinear momentum/divergence equations are coupled
in the same generic residual solve. Start from zero, never a polynomial PDE
solution. Polynomial data occur solely in the declared source/IC/BC/audit.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from solver.ball_ridge_features import BallRidgeFeatures
from solver.general_residual import ResidualProblem,solve_residual
from general_ns_polynomial_precision import DOMAIN,declarations,make_blocks,validate,reference,ZERO,FIRST,SECOND,TRACE_NUANCE

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/pure_mlp_repair'


def torch_jets(model,points,orders):
    x=torch.tensor(points,dtype=torch.float64,requires_grad=True);y=model(x)
    result={ZERO:y.detach()}
    for order in orders:
        if order==ZERO:continue
        columns=[]
        for field in range(y.shape[1]):
            value=y[:,field]
            for axis,n in enumerate(order):
                for _ in range(n):value=torch.autograd.grad(value.sum(),x,create_graph=True)[0][:,axis]
            columns.append(value.detach())
        result[order]=torch.stack(columns,dim=1)
    return result


def run(degree=4,centers=257,lam=.2):
    started=time.perf_counter();torch.set_num_threads(1)
    data=declarations();f=BallRidgeFeatures(DOMAIN.bounds,degree,centers=centers,lam=lam)
    blocks,samples=make_blocks(degree,data);problem=ResidualProblem(f,blocks,fields=4)
    s=solve_residual(problem,initial_coefficients=None,max_iterations=20,tolerance=3e-13,
                     high_accuracy=True,preconditioner='block',lsmr_max_iterations=1000,max_seconds=100)
    solve_seconds=time.perf_counter()-started
    checks=validate(s,data,degree);model=f.torch_model(s.coefficients)
    points=DOMAIN.interior(1009,822041)
    with torch.no_grad():actual=model(torch.tensor(points,dtype=torch.float64)).numpy()
    truth=reference(torch.tensor(points,dtype=torch.float64)).numpy()
    exported={}
    for label,cols in [('velocity',slice(0,3)),('pressure',slice(3,4))]:
        exported[label+'_relative_l2']=float(np.linalg.norm(actual[:,cols]-truth[:,cols])/np.linalg.norm(truth[:,cols]))
    jet_points=DOMAIN.interior(37,332981)
    jets=torch_jets(model,jet_points,(ZERO,)+FIRST+SECOND)
    residual=data['equation'].function(torch.tensor(jet_points),jets,torch.empty((len(jet_points),0),dtype=torch.float64)).numpy()
    exported['raw_torch_ad_residual_rms_by_component']=np.sqrt(np.mean(residual**2,axis=0)).tolist()
    exported['raw_torch_ad_residual_maximum']=float(abs(residual).max())
    exported['stable_vs_raw_jet_maximum_by_order']={str(order):float(abs(jet.numpy()-s.evaluate(jet_points,order)).max()) for order,jet in jets.items()}
    row=dict(problem='Manufactured 3D incompressible Navier-Stokes, four spacetime inputs',
        scope='Architecture and float64 floor control; smooth degree-four manufactured solution, not hard/turbulent flow',
        trace_nuance=TRACE_NUANCE,initialization='exactly zero; no separately computed PDE solution',
        data='Declared body force, velocity on six spatial faces and initial time, pressure gauge only; no interior labels',
        architecture=['Linear(4,'+str(model[0].out_features)+')','Tanh','Linear('+str(model[0].out_features)+',4)'],
        features=f.metrics,unknown_coefficients=4*f.size,status=s.status,sample_counts=samples,
        solver=s.metrics,history=s.history,stable_validation=checks,ordinary_mlp_validation=exported,
        setup_and_solve_seconds=solve_seconds,total_seconds=time.perf_counter()-started)
    OUT.mkdir(parents=True,exist_ok=True);stem=f'flat_ridge_ns4d_p{degree}_n{centers}_lambda{lam}'
    torch.save(dict(state_dict=model.state_dict(),dimension=4,width=model[0].out_features,fields=4,activation='tanh'),OUT/(stem+'.pt'))
    np.savez_compressed(OUT/(stem+'.npz'),coefficients=s.coefficients,points=points,target=truth,prediction=actual)
    (OUT/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(dict(status=s.status,neurons=model[0].out_features,unknowns=4*f.size,seconds=row['total_seconds'],**exported)),flush=True)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--degree',type=int,default=4);parser.add_argument('--centers',type=int,default=257);parser.add_argument('--lam',type=float,default=.2)
    a=parser.parse_args();run(a.degree,a.centers,a.lam)
