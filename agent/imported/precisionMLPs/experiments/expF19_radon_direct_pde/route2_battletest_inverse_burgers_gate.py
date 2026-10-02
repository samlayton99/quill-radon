"""Physics-only feasibility gates for the existing Burgers inverse campaign.

No measurements or oracle values enter either native forward. The analytic
Cole--Hopf evaluator is used only after a returned state has been frozen.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(key,'1')
os.environ.setdefault('KMP_USE_SHM','0')
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from solver.general_adaptive import ResidualDeclaration
from solver.general_residual import ResidualBlock
from solver.route2 import solve_route2_native,ordinary_jets
from solver.route2_inverse import freeze_physics_parameters
from route2_battletest_inverse import DOMAIN,ORDERS,points,oracle,predict,OUT


def physics_declaration(initial_viscosity=.04):
    def blocks(count,seed):
        interior=points(count,seed);n=max(64,int(np.sqrt(count))*4)
        initial=points(n,seed+1);initial[:,1]=0
        walls=points(2*n,seed+2);walls[:n,0]=-1;walls[n:,0]=1
        return [ResidualBlock('burgers',interior,
                    lambda x,j,p:j[(0,1)]+j[(0,0)]*j[(1,0)]-p[:,:1]*j[(2,0)],ORDERS),
                ResidualBlock('initial',initial,lambda x,j,p:j[(0,0)]+torch.sin(np.pi*x[:,:1]),((0,0),)),
                ResidualBlock('walls',walls,lambda x,j,p:j[(0,0)],((0,0),))]
    return ResidualDeclaration(DOMAIN,blocks,parameter_initial=[initial_viscosity],parameter_bounds=([.001],[.5]))


def run_all(output=OUT/'burgers_reduced_forward_gates'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    settings=dict(coordinates='affine_disk',degrees=[6,10,14,18,22,26,32,40,48,64],
        centers=257,lam=.2,tolerance=1e-8,max_seconds=600.,max_iterations=60,
        maximum_working_array_mb=512,check_points=257,export_check_points=257,batch_size=128)
    source=Path(__file__).parent
    files=[Path(__file__),source/'route2_battletest_inverse.py']+[source/'solver'/p for p in ('route2.py','route2_inverse.py','general_residual.py','streamed_residual.py')]
    protocol=dict(version=1,trial_viscosities=[.04,.02],settings=settings,
        purpose='Blind physical feasibility at initial inverse guess and specified harder parameter; not an inverse estimate.',
        source_policy='Zero readouts; only PDE, known IC and walls. No sensors or oracle/reference coefficients enter fitting. Cole-Hopf quadrature occurs after solve for audit only.',
        gate='Only sampled_residual_checks_passed permits using a forward in reduced inverse. All other statuses block inverse scoring.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    text=json.dumps(protocol,indent=2,sort_keys=True)+'\n';file=output/'protocol.json'
    if file.exists() and file.read_text()!=text:raise RuntimeError('Frozen gate protocol changed')
    if not file.exists():file.write_text(text)
    for viscosity in protocol['trial_viscosities']:
        case=output/f'nu{viscosity:g}';case.mkdir(exist_ok=True)
        if (case/'metrics.json').exists():continue
        trace=[]
        def checkpoint(coefficients,parameters,row):
            if len(parameters):raise RuntimeError('Gate retained a free viscosity parameter')
            degree=int(round((np.sqrt(8*len(coefficients)+1)-3)/2))
            trace.append(dict(row,degree=degree));(case/'progress.json').write_text(json.dumps(trace,indent=2)+'\n')
            temporary=case/'latest_native.tmp.npz'
            np.savez_compressed(temporary,coefficients=coefficients,degree=degree,viscosity=viscosity,centers=257,lam=.2)
            temporary.replace(case/'latest_native.npz')
        began=time.perf_counter()
        solution=solve_route2_native(freeze_physics_parameters(physics_declaration(),[viscosity]),
            **settings,solver_options={'iteration_callback':checkpoint})
        elapsed=time.perf_counter()-began;model=solution.export();held=points(1024,90131)
        target=oracle(held,viscosity);pred=predict(model,held)
        jets=ordinary_jets(model,held[:257],ORDERS)
        residual=(jets[(0,1)]+jets[(0,0)]*jets[(1,0)]-viscosity*jets[(2,0)]).numpy()
        row=dict(viscosity=viscosity,status=solution.status,eligible_for_inverse_scoring=solution.status=='sampled_residual_checks_passed',
            degree=solution.problem.features.degree,neurons=solution.problem.features.tanh_count,
            ordinary_pde_rms=float(np.sqrt(np.mean(residual**2))),ordinary_pde_max=float(np.max(abs(residual))),
            heldout_relative_l2=float(np.linalg.norm(pred-target)/np.linalg.norm(target)),
            oracle_quadrature_max_difference=float(np.max(abs(target-oracle(held,viscosity,512)))),
            solve_seconds=elapsed,driver=solution.metrics,history=solution.history,solver=solution.solution.metrics,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        (case/'metrics.json').write_text(json.dumps(row,indent=2)+'\n')
        torch.save(dict(state_dict=model.state_dict(),hidden=model[0].out_features),case/'ordinary_mlp.pt')
        np.savez_compressed(case/'native_final.npz',coefficients=solution.coefficients,degree=solution.problem.features.degree,viscosity=viscosity,centers=257,lam=.2)
        np.savez_compressed(case/'heldout_audit.npz',points=held,prediction=pred,target=target)
        print(json.dumps({k:row[k] for k in ('viscosity','status','degree','ordinary_pde_rms','heldout_relative_l2','solve_seconds')}),flush=True)


if __name__=='__main__':run_all()
