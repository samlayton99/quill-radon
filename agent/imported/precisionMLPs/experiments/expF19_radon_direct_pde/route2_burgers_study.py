"""Physics-only Burgers solves in ordinary tanh MLPs, with controlled limits.

All initial readout coordinates are zero. The global solve uses one MLP;
optional predetermined time slabs use an explicitly disclosed collection of
MLPs, propagating only the preceding solved network's terminal trace. No
classical time marcher, solution fitting, or oracle warmstart is used.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from numpy.polynomial import legendre
from scipy.stats import qmc

from solver.ball_ridge_features import BallRidgeFeatures,gegenbauer_profiles
from solver.general_residual import ResidualBlock,ResidualProblem,solve_residual
from solver.general_features import _evaluate_encoding
from quill_boundary import encode

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/burgers'
NU=.1
AMPLITUDE=.15
K=2*np.pi


def initial_data(x):
    return 2*NU*K*AMPLITUDE*np.sin(K*x)/(1+AMPLITUDE*np.cos(K*x))


def reference(points):
    a=AMPLITUDE*np.exp(-NU*K*K*points[:,1:2])
    return 2*NU*K*a*np.sin(K*points[:,0:1])/(1+a*np.cos(K*points[:,0:1]))


def points(n,seed,t0=0.,t1=1.):
    z=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
    z[:,0]-=.5;z[:,1]=t0+(t1-t0)*z[:,1]
    return z


def raw_values(arrays,x):
    return np.concatenate([np.tanh(z@arrays['first_weights'].T+arrays['first_bias'])@arrays['output_weights']+arrays['output_bias']
                           for z in np.array_split(x,max(1,int(np.ceil(len(x)/128))))])


def raw_torch_audit(features,a,x):
    model=features.torch_model(a)
    predictions=[];residuals=[]
    for z in np.array_split(x,max(1,int(np.ceil(len(x)/64)))):
        z=torch.tensor(z,dtype=torch.float64,requires_grad=True)
        y=model(z);first=torch.autograd.grad(y.sum(),z,create_graph=True)[0]
        second=torch.autograd.grad(first[:,0].sum(),z)[0][:,0:1]
        residual=first[:,1:2]+y*first[:,0:1]-NU*second
        predictions.append(y.detach().numpy());residuals.append(residual.detach().numpy())
    y=np.concatenate(predictions);r=np.concatenate(residuals);truth=reference(x)
    return dict(relative_l2=float(np.linalg.norm(y-truth)/np.linalg.norm(truth)) if np.linalg.norm(truth)>1e-12 else None,
                maximum_absolute=float(np.max(np.abs(y-truth))),
                absolute_pde_rms=float(np.sqrt(np.mean(r*r))),
                absolute_pde_maximum=float(np.max(np.abs(r))))


def exact_coordinate_values(features,x,derivative=(0,0)):
    """Post-solve known-basis audit only. Never feeds coefficients to a solve."""
    normalized=2*(x-features.bounds.mean(axis=1))/(features.bounds[:,1]-features.bounds[:,0])
    out=np.ones((len(x),features.size))
    for axis,order in enumerate(derivative):
        coefficients=legendre.legder(np.eye(features.degree+1),m=order,axis=0)
        bank=legendre.legval(normalized[:,axis],coefficients).T
        out*=bank[:,features.multiindices[:,axis]]*(2/(features.bounds[axis,1]-features.bounds[axis,0]))**order
    return out


def physical_variant(base,t0,t1):
    f=copy.copy(base)
    f.bounds=np.array([[-.5,.5],[t0,t1]])
    f.midpoint=f.bounds.mean(axis=1)
    f.scale=2/(f.bounds[:,1]-f.bounds[:,0])/np.sqrt(2)
    f.physical_directions=f.directions*f.scale
    f.metrics=dict(f.metrics,physical_bounds=f.bounds.tolist(),
                   geometry_map_reused_across_time_slabs=True)
    return f


def run(degree=28,centers=257,lam=.2,intervals=1,seconds=60.,iterations=12,krylov=1500,label='global',base=None,pde_faces=False):
    start=time.perf_counter()
    if base is None:base=BallRidgeFeatures([[-.5,.5],[0.,1.]],degree,centers,lam)
    setup=time.perf_counter()-start
    OUT.mkdir(parents=True,exist_ok=True)
    stem=f'route2_burgers_{label}_p{degree}_n{centers}_l{lam:g}_s{intervals}'
    np.savez_compressed(OUT/(stem+'_geometry.npz'),profile_map=base.profile_map,
        directions=base.directions,angular_weights=base.angular_weights,multiindices=base.multiindices)
    rows=[];models=[];previous=None
    for slab in range(intervals):
        t0,t1=slab/intervals,(slab+1)/intervals
        f=physical_variant(base,t0,t1)
        n=max(768,8*f.size);count=max(120,4*(degree+1))
        q=points(n,18+slab,t0,t1)
        ic=points(count,11+slab,t0,t1);ic[:,1]=t0
        left=points(count,24+slab,t0,t1);left[:,0]=-.5
        right=points(count,34+slab,t0,t1);right[:,0]=.5
        target=initial_data(ic[:,0:1]) if previous is None else raw_values(previous,ic)
        target=torch.tensor(target,dtype=torch.float64)
        blocks=[ResidualBlock('PDE',q,lambda x,j,p:j[(0,1)]+j[(0,0)]*j[(1,0)]-NU*j[(2,0)],((0,0),(1,0),(0,1),(2,0))),
            ResidualBlock('initial',ic,lambda x,j,p:j[(0,0)]-target,((0,0),)),
            ResidualBlock('boundary',np.concatenate([left,right]),lambda x,j,p:j[(0,0)],((0,0),))]
        if pde_faces:
            traces=[]
            for name,axis,value in [('initial',1,t0),('final',1,t1),('left',0,-.5),('right',0,.5)]:
                trace=points(count,404+slab+len(traces),t0,t1);trace[:,axis]=value;traces.append(trace)
            blocks.append(ResidualBlock('PDE_on_faces',np.concatenate(traces),
                lambda x,j,p:j[(0,1)]+j[(0,0)]*j[(1,0)]-NU*j[(2,0)],((0,0),(1,0),(0,1),(2,0))))
        solved=solve_residual(ResidualProblem(f,blocks),max_iterations=iterations,tolerance=1e-11,
            max_seconds=seconds,high_accuracy=True,linear_refinement_steps=0,damping=1e-10,
            lsmr_max_iterations=krylov,preconditioner='auto',preconditioner_block_size=128)
        arrays=f.compile(solved.coefficients)
        held=points(2048,19317+slab,t0,t1)
        truth=reference(held);raw=raw_values(arrays,held);stable=solved.evaluate(held)
        check=held[:257]
        discrepancy={}
        for der in [(0,0),(1,0),(0,1),(2,0)]:
            actual=solved.evaluate(check,der)
            ideal=exact_coordinate_values(f,check,der)@solved.coefficients
            discrepancy[str(der)]=dict(maximum_absolute=float(np.max(np.abs(actual-ideal))),
                                       rms=float(np.sqrt(np.mean((actual-ideal)**2))))
        audit=raw_torch_audit(f,solved.coefficients,held[:257])
        interface_gap=None
        if previous is not None:
            interface=np.column_stack([np.linspace(-.5,.5,513),np.full(513,t0)])
            interface_gap=float(np.max(np.abs(raw_values(arrays,interface)-raw_values(previous,interface))))
        faces={}
        for name,axis,value in [('initial',1,t0),('final',1,t1),('left',0,-.5),('right',0,.5)]:
            x=points(129,751+slab,t0,t1);x[:,axis]=value
            faces[name]=raw_torch_audit(f,solved.coefficients,x)
        row=dict(slab=slab,interval=[t0,t1],features=f.metrics,solver=solved.metrics,
            history=solved.history,heldout_relative_l2=float(np.linalg.norm(raw-truth)/np.linalg.norm(truth)),
            stable_relative_l2=float(np.linalg.norm(stable-truth)/np.linalg.norm(truth)),
            heldout_maximum_absolute=float(np.max(np.abs(raw-truth))),
            export_stable_maximum_difference=float(np.max(np.abs(raw-stable))),
            ordinary_torch=audit,ordinary_torch_faces=faces,
            known_coordinate_encoding_discrepancy_at_solved_coefficients=discrepancy,
            readout_l1=float(np.sum(np.abs(arrays['output_weights']))),
            interface_maximum_value_jump=interface_gap,
            previous_initial_trace='prescribed analytic initial condition' if previous is None else 'previous actual flat MLP terminal evaluation')
        rows.append(row);models.append((t0,t1,arrays));previous=arrays
        np.savez_compressed(OUT/(stem+f'_slab{slab}.npz'),coefficients=solved.coefficients,**arrays)
        torch.save(f.torch_model(solved.coefficients).state_dict(),OUT/(stem+f'_slab{slab}_torch_state.pt'))
        print(json.dumps(dict(stem=stem,slab=slab,status=solved.status,
            raw_relative_l2=row['heldout_relative_l2'],raw_pde_rms=audit['absolute_pde_rms'],
            seconds=solved.metrics['seconds'])),flush=True)
    x=points(4096,39319);y=np.zeros((len(x),1))
    for t0,t1,a in models:
        mask=(x[:,1]>=t0)&(x[:,1]<=t1)
        y[mask]=raw_values(a,x[mask])
    truth=reference(x)
    final=np.column_stack([np.linspace(-.5,.5,513),np.ones(513)])
    final_y=raw_values(models[-1][2],final);final_truth=reference(final)
    result=dict(degree=degree,centers=centers,lam=lam,intervals=intervals,pde_faces=pde_faces,
        status='all_sampled_blocks_converged' if all(r['solver']['status']=='converged' for r in rows) else 'not_all_sampled_blocks_converged',
        architecture='one ordinary tanh MLP' if intervals==1 else f'explicit collection of {intervals} ordinary tanh MLPs on time intervals; no single-global-MLP claim',
        global_raw_relative_l2=float(np.linalg.norm(y-truth)/np.linalg.norm(truth)),
        global_raw_maximum_absolute=float(np.max(np.abs(y-truth))),
        final_raw_relative_l2=float(np.linalg.norm(final_y-final_truth)/np.linalg.norm(final_truth)),
        construction_seconds=setup,all_setup_solve_validation_seconds=time.perf_counter()-start,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_and_data_policy=dict(forcing=0,readout_initialization='zero in every slab',
            solution_labels=False,reference='heldout validation only',
            nonlinear_equation='u_t+u*u_x−0.1*u_xx=0',
            initial='2*nu*k*0.15*sin(k*x)/(1+0.15*cos(k*x)), k=2*pi',boundary='u(±.5,t)=0'),
        metric_normalization='Relative Euclidean norms on uniform Sobol space-time; absolute unscaled physical PDE residual RMS. Final relative norm uses 513 uniform spatial points.',
        slabs=rows)
    (OUT/(stem+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='slabs'}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--degrees',nargs='+',type=int,default=[24,28,32])
    parser.add_argument('--centers',type=int,default=257);parser.add_argument('--lam',type=float,default=.2)
    parser.add_argument('--intervals',type=int,default=1);parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--krylov',type=int,default=1500)
    parser.add_argument('--label',default='global');args=parser.parse_args();torch.set_num_threads(1)
    for p in args.degrees:run(p,args.centers,args.lam,args.intervals,args.seconds,args.iterations,args.krylov,args.label)
