"""Native coupled disk Navier--Stokes: actual tanh field, no-slip boundary.

Manufactured body forces are prescribed PDE input. The nonzero interior
reference is used only to construct that forcing and independently validate;
the solver sees zero velocity boundary, a pressure gauge, and the equations.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quill-hardening-mpl')
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from route2_disk_profiles import DiskProfileOperator
from route2_streamed_study import disk_points, boundary_points
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual
from solver.route2 import export_mlp, ordinary_jets

OUT = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/disk_ns'
ORDERS = ((0,0), (1,0), (0,1), (2,0), (0,2))


def exact(x):
    """Curl of (1-r^2)^2 exp(.35x-.2y), plus a nonconstant pressure."""
    a,b=.35,-.2
    xx,yy=x[:,0],x[:,1]
    q=1-xx*xx-yy*yy
    e=torch.exp(a*xx+b*yy)
    return torch.stack((e*q*(-4*yy+b*q), e*q*(4*xx-a*q),
                        .2*torch.sin(.7*xx-.4*yy)),dim=1)


def equations(jets, nu):
    value,dx,dy,dxx,dyy=(jets[o] for o in ORDERS)
    velocity=value[:,:2]
    momentum=(velocity[:,0:1]*dx[:,:2]+velocity[:,1:2]*dy[:,:2]
              -nu*(dxx[:,:2]+dyy[:,:2])+torch.stack((dx[:,2],dy[:,2]),dim=1))
    divergence=dx[:,0]+dy[:,1]
    return torch.cat((momentum,divergence[:,None]),dim=1)


def source(points, nu):
    x=torch.tensor(np.asarray(points),dtype=torch.float64,requires_grad=True)
    value=exact(x)
    first=[torch.autograd.grad(value[:,k].sum(),x,create_graph=True)[0] for k in range(3)]
    jets={(0,0):value,(1,0):torch.stack([g[:,0] for g in first],dim=1),
          (0,1):torch.stack([g[:,1] for g in first],dim=1)}
    for axis,order in ((0,(2,0)),(1,(0,2))):
        jets[order]=torch.stack([torch.autograd.grad(g[:,axis].sum(),x,retain_graph=True)[0][:,axis]
                                 for g in first],dim=1)
    return equations(jets,nu).detach().numpy()


def make_problem(degree=8,centers=257,lam=.2,interior=None,boundary=128,
                 nu=.3,batch_size=128,reflection_balanced=False):
    f=DiskProfileOperator(degree,centers=centers,lam=lam,block_size=batch_size)
    count=max(512,4*f.size) if interior is None else interior
    if reflection_balanced:
        base=disk_points((count+3)//4,1741)
        signs=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
        x=(base[:,None,:]*signs[None,:,:]).reshape(-1,2)
    else:
        x=disk_points(count,1741)
    force=torch.tensor(source(x,nu)[:,:2],dtype=torch.float64)
    def residual(x,j,p):
        result=equations(j,nu)
        return torch.cat((result[:,:2]-force,result[:,2:3]),dim=1)
    def batch(x,j,p,rows):
        result=equations(j,nu)
        return torch.cat((result[:,:2]-force[rows],result[:,2:3]),dim=1)
    blocks=[ResidualBlock('momentum_and_divergence',x,residual,ORDERS,batch_function=batch),
            ResidualBlock('no_slip',boundary_points(boundary),lambda x,j,p:j[(0,0)][:,:2],((0,0),),weight=5.),
            ResidualBlock('pressure_gauge',np.zeros((1,2)),lambda x,j,p:j[(0,0)][:,2],((0,0),))]
    return ResidualProblem(f,blocks,fields=3,execution='streamed',batch_size=batch_size)


def audit(solution,nu,count=2048):
    model=export_mlp(solution.problem.features,solution.coefficients)
    points=disk_points(count,812771)
    jets=ordinary_jets(model,points,ORDERS)
    target=exact(torch.tensor(points,dtype=torch.float64)).detach().numpy()
    pred=jets[(0,0)].numpy()
    actual=equations(jets,nu).numpy()
    forcing=source(points,nu)
    error=actual.copy();error[:,:2]-=forcing[:,:2]
    edge=ordinary_jets(model,boundary_points(257,.317),((0,0),))[(0,0)].numpy()
    stable=solution.evaluate(points)
    out=dict(velocity_relative_l2=float(np.linalg.norm(pred[:,:2]-target[:,:2])/np.linalg.norm(target[:,:2])),
             pressure_relative_l2=float(np.linalg.norm(pred[:,2]-target[:,2])/np.linalg.norm(target[:,2])),
             velocity_max_abs=float(np.max(abs(pred[:,:2]-target[:,:2]))),
             pressure_max_abs=float(np.max(abs(pred[:,2]-target[:,2]))),
             momentum_rms=float(np.sqrt(np.mean(error[:,:2]**2))),
             momentum_relative_rms=float(np.linalg.norm(error[:,:2])/np.linalg.norm(forcing[:,:2])),
             divergence_rms=float(np.sqrt(np.mean(error[:,2]**2))),
             divergence_max_abs=float(np.max(abs(error[:,2]))),
             no_slip_rms=float(np.sqrt(np.mean(edge[:,:2]**2))),
             pressure_gauge=float(model(torch.zeros((1,2),dtype=torch.float64))[0,2]),
             export_stable_max_abs=float(np.max(abs(pred-stable))),
             reference_divergence_rms=float(np.sqrt(np.mean(forcing[:,2]**2))),
             readout_l1_by_field=np.sum(abs(model[2].weight.detach().numpy()),axis=1).tolist(),
             audit_points=count,ordinary_export='Linear/Tanh/Linear; float64; Torch automatic derivatives')
    return model,out


def run(degrees=(6,10,14),centers=257,lam=.2,seconds=180.,nu=.3,
        krylov=600,preconditioner='block',block=24,interior=None,boundary=128,
        tolerance=2e-14,label='baseline',resume=None,field_parity='none',reflection_balanced=False,
        preconditioner_basis='actual',preconditioner_refresh=0):
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    previous=None;rows=[]
    for degree in degrees:
        began=time.perf_counter()
        problem=make_problem(degree,centers,lam,interior,boundary,nu,
                             reflection_balanced=reflection_balanced)
        initial=None
        if previous is None and resume is not None:
            provenance_path=Path(resume).with_suffix('.json')
            if not provenance_path.exists():
                raise ValueError('Resume requires a sibling native-run JSON provenance file')
            provenance=json.loads(provenance_path.read_text())
            if (provenance.get('config',{}).get('nu')!=nu or
                    provenance.get('source_policy',{}).get('interior_solution_labels') is not False or
                    provenance.get('source_policy',{}).get('exterior_solve') is not False):
                raise ValueError('Resume provenance must be a native disk solve with matching viscosity')
            archive=np.load(resume)
            old_coefficients=archive['coefficients']
            if old_coefficients.ndim!=2 or old_coefficients.shape[1]!=3 or len(old_coefficients)>problem.features.size:
                raise ValueError('Resume must contain a lower/equal degree three-field native disk iterate')
            initial=np.zeros((problem.features.size,3));initial[:len(old_coefficients)]=old_coefficients
        if previous is not None:
            old=previous.problem.features;new=problem.features
            keys={(int(n),int(k),bool(s)):i for i,(n,k,s) in enumerate(zip(new.degrees,new.harmonics,new.is_sine))}
            initial=np.zeros((new.size,3))
            for i,key in enumerate(zip(old.degrees,old.harmonics,old.is_sine)):
                initial[keys[tuple(key)]]=previous.coefficients[i]
        setup=time.perf_counter()-began
        checkpoints=[]
        def checkpoint(c,p,record):
            checkpoints.append(record)
            print(json.dumps(dict(event='iteration',degree=degree,**record)),flush=True)
        solved=solve_residual(problem,initial_coefficients=initial,max_iterations=20,
            tolerance=tolerance,high_accuracy=True,max_seconds=seconds,
            lsmr_max_iterations=krylov,preconditioner=preconditioner,
            preconditioner_block_size=block,preconditioner_ridge=1e-12,
            preconditioner_field_parity=field_parity,
            preconditioner_basis=preconditioner_basis,preconditioner_refresh=preconditioner_refresh,
            scaling_probes=4,linear_refinement_steps=0,validate_locality=False,
            iteration_callback=checkpoint)
        model,checks=audit(solved,nu)
        stem=f'{label}_p{degree}_n{centers}'
        row=dict(config=dict(degree=degree,centers=centers,lam=lam,nu=nu,
            max_seconds=seconds,krylov=krylov,preconditioner=preconditioner,
            preconditioner_block_size=block,tolerance=tolerance,
            field_parity=field_parity,
            reflection_balanced=reflection_balanced,
            preconditioner_basis=preconditioner_basis,preconditioner_refresh=preconditioner_refresh,
            interior=len(problem.blocks[0].points),boundary=boundary),
            source_policy=dict(initialization='zero' if initial is None else 'previous native neural PDE iterate',
                resumed_archive=str(resume) if resume is not None else None,
                exterior_solve=False,interior_solution_labels=False,boundary_velocity='identically zero',
                forcing='manufactured from analytical reference; prescribed PDE data',
                gauge='p(0)=0',reference='forcing and independent validation only'),
            features=problem.features.metrics,solver=solved.metrics,history=solved.history,
            ordinary_audit=checks,setup_seconds=setup,total_seconds=time.perf_counter()-began,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (OUT/(stem+'.json')).write_text(json.dumps(row,indent=2))
        np.savez_compressed(OUT/(stem+'.npz'),coefficients=solved.coefficients)
        torch.save(model.state_dict(),OUT/(stem+'_torch.pt'))
        print(json.dumps(dict(event='result',name=stem,status=solved.status,**checks)),flush=True)
        previous=solved;rows.append(row)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--degrees',type=int,nargs='+',default=[6,10,14])
    parser.add_argument('--centers',type=int,default=257)
    parser.add_argument('--lam',type=float,default=.2)
    parser.add_argument('--seconds',type=float,default=180.)
    parser.add_argument('--nu',type=float,default=.3)
    parser.add_argument('--krylov',type=int,default=600)
    parser.add_argument('--preconditioner',choices=['block','diagonal','auto'],default='block')
    parser.add_argument('--block',type=int,default=24)
    parser.add_argument('--field-parity',choices=['none','auto'],default='none')
    parser.add_argument('--reflection-balanced',action='store_true')
    parser.add_argument('--preconditioner-basis',choices=['actual','ideal'],default='actual')
    parser.add_argument('--preconditioner-refresh',type=int,default=0)
    parser.add_argument('--interior',type=int)
    parser.add_argument('--boundary',type=int,default=128)
    parser.add_argument('--tolerance',type=float,default=2e-14)
    parser.add_argument('--label',default='baseline')
    parser.add_argument('--resume')
    run(**vars(parser.parse_args()))
