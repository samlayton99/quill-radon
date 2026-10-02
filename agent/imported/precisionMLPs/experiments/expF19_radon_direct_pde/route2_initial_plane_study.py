"""Frozen control: native PDE solve with a prescribed initial-data branch.

Only the supplied IC is analytically encoded. No future/reference solution
is used by fitting. Every stage deploys one ordinary Linear/Tanh/Linear MLP.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from route2_initial_plane_profiles import InitialPlaneProfileOperator,concatenate_scalar_mlps
from quill_boundary import encode
from solver.general_residual import ResidualBlock,ResidualProblem,solve_residual
from solver.route2 import ordinary_jets
import route2_battletest_forward as benchmark

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/initial_plane'
PROTOCOL=dict(version=1,cases=['allen_cahn_d001','allen_cahn_d00001'],seed=0,
    degrees=[4,8,12,16,24,32,40,48],centers=257,lam=.2,
    max_seconds=600,max_iterations=20,tolerance=1e-8,batch_size=64,oversampling=6,
    inner_cap=1000,preconditioner_block_cap=512,
    source_policy='Only prescribed IC x^2*cos(pi*x) is analytically encoded; zero/native PDE corrections. No interior truth or external trajectory.',
    architecture='One ordinary Linear/Tanh/Linear model; paired correction neurons plus fixed IC neurons.',
    constraint='Correction F_q(x,t)-F_q(x,0) is symbolically zero at initial time; ordinary floating-sum and IC encoding errors audited.',
    note='Same equations as forward campaign, separate geometry/control protocol, not a runtime-isolated ablation.',
    compatibility='Published periodic IC has mismatched endpoint first derivatives at t=0. No uniform corner derivative precision claim.',
    limits='Soft budgets around Krylov work; all degrees/statuses retained. Sampled residuals are not continuum certificates.')


def initial_encoding():
    e=encode(lambda z:z*z*np.cos(np.pi*z),PROTOCOL['centers']-1,lam=PROTOCOL['lam'],
             halo=int(np.ceil(np.sqrt(PROTOCOL['centers']))))
    model=torch.nn.Sequential(torch.nn.Linear(2,len(e.centers),dtype=torch.float64),
        torch.nn.Tanh(),torch.nn.Linear(len(e.centers),1,dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.zero_();model[0].weight[:,0]=e.gamma
        model[0].bias.copy_(torch.from_numpy(-e.gamma*e.centers))
        model[2].weight.copy_(torch.from_numpy(e.weights[None,:]))
        model[2].bias.fill_(float(e.bias))
    return e,model.requires_grad_(False)


def prescribed_offset(block,encoding):
    def wrapped(x,j,p):
        shifted={order:values+(torch.as_tensor(encoding.evaluate(x[:,0].detach().numpy(),order[0]),
                    dtype=x.dtype)[:,None] if order[1]==0 else 0.) for order,values in j.items()}
        return block.function(x,shifted,p)
    return ResidualBlock(block.name,block.points,wrapped,block.derivatives,
                         weight=block.weight,scale=block.scale,aggregation=block.aggregation)


def run(name):
    if name not in PROTOCOL['cases']:raise ValueError('Outside frozen protocol')
    OUT.mkdir(parents=True,exist_ok=True)
    payload=dict(PROTOCOL,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()+
        Path(__file__).with_name('route2_initial_plane_profiles.py').read_bytes()).hexdigest())
    path=OUT/'protocol.json'
    if path.exists():
        frozen=json.loads(path.read_text());frozen.pop('implementation_sha256',None)
        if frozen!=PROTOCOL:raise RuntimeError('Frozen protocol changed')
    else:path.write_text(json.dumps(payload,indent=2)+'\n')
    torch.set_num_threads(1)
    case=benchmark.CASES[name];declaration=benchmark.make_declaration(case,0)
    encoding,anchor=initial_encoding();history=[];previous=None;began=time.perf_counter()
    for degree in PROTOCOL['degrees']:
        remaining=PROTOCOL['max_seconds']-(time.perf_counter()-began)
        if remaining<=0:break
        f=InitialPlaneProfileOperator(declaration.domain.bounds,degree,centers=PROTOCOL['centers'],
            lam=PROTOCOL['lam'],block_size=PROTOCOL['batch_size'])
        blocks=[prescribed_offset(b,encoding) for b in declaration.make_blocks(max(256,6*f.size),113+degree*1009)
                if b.name!='initial']
        problem=ResidualProblem(f,blocks,execution='streamed',batch_size=PROTOCOL['batch_size'])
        c=None
        if previous is not None:
            old=previous.problem.features;c=np.zeros((f.size,1))
            lookup={(int(n),int(k),bool(s)):i for i,(n,k,s) in enumerate(zip(f.degrees,f.harmonics,f.is_sine))}
            for i,key in enumerate(zip(old.degrees,old.harmonics,old.is_sine)):c[lookup[key]]=previous.coefficients[i]
        def checkpoint(coefficients,parameters,record):
            np.savez_compressed(OUT/(name+'_latest.npz'),coefficients=coefficients,degree=degree)
        solution=solve_residual(problem,initial_coefficients=c,max_iterations=PROTOCOL['max_iterations'],
            tolerance=PROTOCOL['tolerance'],max_seconds=max(.01,remaining),high_accuracy=True,
            preconditioner='block',preconditioner_basis='ideal',preconditioner_block_size=512,
            preconditioner_refresh=1,preconditioner_field_parity='none',linearization_basis='ideal',
            inexact_newton=True,lsmr_max_iterations=1000,linear_refinement_steps=0,
            iteration_callback=checkpoint)
        model=concatenate_scalar_mlps(anchor,f.torch_model(solution.coefficients[:,0]))
        checks=[]
        for block in declaration.make_blocks(257,987513):
            _,residual=benchmark.evaluate_raw(model,block.points,block.function,block.derivatives)
            if block.aggregation is not None:residual=block.aggregation@residual
            checks.append(dict(name=block.name,raw_rms=float(np.sqrt(np.mean(residual**2))),
                               raw_max=float(np.max(abs(residual)))))
        row=dict(degree=degree,neurons=model[0].out_features,coordinates=f.size,status=solution.status,
                 ordinary_checks=checks,solver_seconds=solution.metrics['seconds'],
                 lsmr_iterations=solution.metrics.get('lsmr_iterations'),elapsed_seconds=time.perf_counter()-began)
        history.append(row)
        torch.save(dict(state_dict=model.state_dict(),width=model[0].out_features,dimension=2),OUT/(name+'.pt'))
        np.savez_compressed(OUT/(name+'.npz'),coefficients=solution.coefficients,degree=degree)
        (OUT/(name+'.json')).write_text(json.dumps(dict(protocol=payload,history=history,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),indent=2)+'\n')
        print(json.dumps(row),flush=True)
        previous=solution
        if max(r['raw_rms'] for r in checks)<=PROTOCOL['tolerance'] and len(history)>1:break
        if solution.status=='budget_exhausted':break
    return history


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=PROTOCOL['cases'],required=True)
    run(parser.parse_args().case)
