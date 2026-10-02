"""Blind native forward gate before any wall-driven Navier--Stokes inverse fit.

No sensors, reference fields, true inverse parameters or saved flow states are
loaded here. This checks whether a zero-start native forward at the proposed
initial viscosity can pass the same feasibility gate used by reduced inverse.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('KMP_USE_SHM','0')
import json
import hashlib
import argparse
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from solver.general_domains import ResidualDomain
from solver.general_adaptive import ResidualDeclaration
from solver.general_residual import ResidualBlock
from solver.route2 import solve_route2_native,ordinary_jets
from solver.route2_inverse import freeze_physics_parameters,SensorObservations,solve_route2_inverse
from route2_physical_flow import momentum_and_divergence,boundary_velocity,ORDERS

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_forward_gate'
DOMAIN=ResidualDomain([[-1.,1.]]*2,levelset=lambda x:(x*x).sum(dim=1)-1)


def declaration(initial_viscosity=.05):
    def blocks(count,seed):
        interior=DOMAIN.interior(count,seed)
        boundary_count=max(128,4*int(np.ceil(np.sqrt(count))))
        phase=np.random.default_rng(seed+1).uniform(0,2*np.pi/boundary_count)
        theta=phase+np.arange(boundary_count)*2*np.pi/boundary_count
        boundary=np.column_stack([np.cos(theta),np.sin(theta)])
        trace=torch.tensor(boundary_velocity(boundary,'disk_stirring',1.),dtype=torch.float64)
        return [ResidualBlock('zero_force_momentum_and_divergence',interior,
                    lambda x,j,p:momentum_and_divergence(j,p[:,:1]),ORDERS),
                ResidualBlock('prescribed_tangential_wall',boundary,
                    lambda x,j,p:j[(0,0)][:,:2]-trace,((0,0),),weight=5.,
                    batch_function=lambda x,j,p,rows:j[(0,0)][:,:2]-trace[rows]),
                ResidualBlock('pressure_gauge',np.zeros((1,2)),lambda x,j,p:j[(0,0)][:,2:3],((0,0),))]
    return ResidualDeclaration(DOMAIN,blocks,fields=3,parameter_initial=[initial_viscosity],parameter_bounds=([.005],[.2]))


def run(output=OUT):
    torch.set_num_threads(1);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings=dict(coordinates='disk',degrees=[8,16,24,32,40],centers=257,lam=.2,
        tolerance=1e-8,max_seconds=1200.,maximum_working_array_mb=512,
        check_points=1025,export_check_points=1025,max_iterations=20,batch_size=128)
    protocol=dict(version=1,trial_viscosity=.05,solver='solve_route2_native',settings=settings,
        problem='Steady incompressible 2D Navier-Stokes in unit disk; zero force; prescribed tangential stirring; p(0)=0.',
        boundary='speed(theta)=1+.25*cos(2theta)+.15*sin(3theta); velocity=speed*(-y,x)',
        initialization='zero readouts followed only by native degree continuation',
        source_policy='No reference field/checkpoint/sensors generated or loaded. Trial parameter is fixed in callback closures; no free parameters in the forward solve.',
        gate='Only sampled_residual_checks_passed allows future inverse scoring; all other outcomes remain unresolved, not parameter estimates.',
        timing='Concurrent research processes may run; wall-clock times are not isolated performance comparisons.')
    serialized=json.dumps(protocol,indent=2,sort_keys=True)+'\n';path=output/'protocol.json'
    if path.exists() and path.read_text()!=serialized:raise RuntimeError('Frozen NS gate protocol differs')
    if not path.exists():path.write_text(serialized)
    history=[]
    def checkpoint(c,p,record):
        if len(p):raise RuntimeError('Blind forward retained a free viscosity parameter')
        degree=int(round((np.sqrt(8*len(c)+1)-3)/2))
        row=dict(record,degree=degree);history.append(row)
        (output/'progress.json').write_text(json.dumps(history,indent=2)+'\n')
        temporary=output/'latest_native.tmp.npz'
        np.savez_compressed(temporary,coefficients=c,parameters=p,degree=degree,trial_viscosity=.05,
                            centers=257,lam=.2,provenance='zero-start blind native PDE solve')
        temporary.replace(output/'latest_native.npz')
        print(json.dumps(dict(event='native_iteration',degree=degree,iteration=record['iteration'],
            residual=record['maximum_scaled_block_rms'],elapsed=record['elapsed_seconds'])),flush=True)
    began=time.perf_counter()
    solution=solve_route2_native(freeze_physics_parameters(declaration(),[.05]),**settings,
        solver_options={'iteration_callback':checkpoint})
    solve_seconds=time.perf_counter()-began;model=solution.export()
    held=DOMAIN.interior(2049,977131);jets=ordinary_jets(model,held,ORDERS)
    residual=momentum_and_divergence(jets,.05).numpy()
    theta=.317+np.arange(389)*2*np.pi/389;wall=np.column_stack([np.cos(theta),np.sin(theta)])
    with torch.no_grad():edge=model(torch.tensor(wall,dtype=torch.float64)).numpy()[:,:2]
    trace=boundary_velocity(wall,'disk_stirring',1.)
    row=dict(protocol=protocol,status=solution.status,inverse_estimate_emitted=False,
        eligible_for_future_inverse_scoring=solution.status=='sampled_residual_checks_passed',
        driver=solution.metrics,history=solution.history,solver=solution.solution.metrics,
        degree=solution.problem.features.degree,neurons=solution.problem.features.tanh_count,
        ordinary_momentum_rms=float(np.sqrt(np.mean(residual[:,:2]**2))),
        ordinary_momentum_max=float(np.max(abs(residual[:,:2]))),
        ordinary_divergence_rms=float(np.sqrt(np.mean(residual[:,2]**2))),
        ordinary_wall_rms=float(np.sqrt(np.mean((edge-trace)**2))),
        solve_seconds=solve_seconds,total_seconds=time.perf_counter()-began,
        process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
    (output/'metrics.json').write_text(json.dumps(row,indent=2)+'\n')
    torch.save(dict(state_dict=model.state_dict(),hidden=model[0].out_features),output/'ordinary_mlp.pt')
    np.savez_compressed(output/'native_final.npz',coefficients=solution.coefficients,parameters=solution.parameters,
                        degree=solution.problem.features.degree,trial_viscosity=.05,centers=257,lam=.2)
    print(json.dumps({k:row[k] for k in ('status','degree','ordinary_momentum_rms','ordinary_divergence_rms','solve_seconds')}),flush=True)
    return row


REFERENCE=(OUT.parents[2]/'route2_hardening/physical_flow'/
    'lowviscosity_resolution_check_disk_stirring_nu0.03_p56_n513_ordinary_mlp.pt')


def load_measurement_reference(path=REFERENCE):
    """Quarantined observation/evaluation source; never passed to the solver."""
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=torch.nn.Sequential(torch.nn.Linear(2,state['0.weight'].shape[0]),torch.nn.Tanh(),
                              torch.nn.Linear(state['0.weight'].shape[0],3)).double()
    model.load_state_dict(state);model.eval()
    return model


def batched_values(model,points,batch_size=64):
    """Bound ordinary audit activations, including the finer reference model."""
    points=np.asarray(points,float)
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(points[start:start+batch_size],dtype=torch.float64)).numpy()
                               for start in range(0,len(points),batch_size)])


def make_observations(noise):
    # Exactly sixteen velocity locations / thirty-two scalar data values.
    points=DOMAIN.interior(16,431819)
    reference=load_measurement_reference()
    values=batched_values(reference,points)[:,:2]
    values=values+noise*np.random.default_rng(812739).normal(size=values.shape)
    return SensorObservations(points,values,fields=[0,1],scale=noise if noise else 1.)


def inverse_protocol():
    settings=dict(coordinates='disk',degrees=[8,16,24,32,40,48],centers=257,lam=.2,
        tolerance=1e-8,max_seconds=1200.,maximum_working_array_mb=512,
        check_points=1025,export_check_points=1025,max_iterations=20,batch_size=128)
    outer=dict(max_seconds=3600.,max_iterations=8,max_forward_solves=36,
        difference_step=1e-3,parameter_scale=[.05],gradient_tolerance=1e-9,
        data_tolerance=1e-8,step_tolerance=1e-8,truncate_warm_starts=True)
    return dict(version=1,task='Reduced native inverse viscosity in zero-force wall-driven disk Navier-Stokes',
        case_noise=[0.,.001],initial_viscosity=.05,bounds=[.005,.2],inner=settings,outer=outer,
        sensors=dict(points=16,scalar_values=32,fields=['u','v'],seed=431819,noise_seed=812739),
        reference=dict(path=str(REFERENCE),sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            viscosity=.03,degree=56,centers=513,purpose='Observation generator and held-out audit only'),
        source_policy='Only32 sensor values/locations and known noise scale enter inverse. No reference coefficients, whole field, true viscosity or external PDE trajectory enters initialization, inner physics or parameter selection. Every case starts zero, followed by native continuation and algebraic truncation of current validated native coordinates.',
        limitations=['Same-method synthetic reference, not experimental or independent-discretization inverse validation; finer degree/N and separate samples reduce but do not eliminate inverse-crime concerns.',
            'Sampled physical and richer-resolution checks are not continuum existence/uniqueness certificates.',
            'Local scalar finite differences do not prove global parameter identifiability.',
            'Concurrent runs mean wall-clock measurements are not isolated speed comparisons.'])


def freeze_inverse_protocol(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    serialized=json.dumps(inverse_protocol(),indent=2,sort_keys=True)+'\n';path=output/'protocol.json'
    if path.exists() and path.read_text()!=serialized:raise RuntimeError('Frozen inverse NS protocol differs')
    if not path.exists():path.write_text(serialized)
    return json.loads(serialized)


def run_inverse(index,output=OUT.parent/'wall_ns_reduced_inverse'):
    torch.set_num_threads(1);output=Path(output);protocol=freeze_inverse_protocol(output)
    if index not in (0,1):raise ValueError('Inverse NS case must be0 or1')
    noise=protocol['case_noise'][index];case=output/f'noise_{noise:g}'
    case.mkdir(parents=True,exist_ok=True)
    if (case/'metrics.json').exists():raise RuntimeError('Preserve existing result; choose a new protocol/output for another run')
    observation=make_observations(noise)
    np.savez_compressed(case/'observations.npz',points=observation.points,values=observation.values,scale=observation.scale)
    forward_rows=[]
    def archive(solution,parameters,row):
        number=len(forward_rows);forward_rows.append(row)
        (case/'forward_progress.json').write_text(json.dumps(forward_rows,indent=2)+'\n')
        if solution is not None:
            np.savez_compressed(case/f'forward_{number:03d}_native.npz',coefficients=solution.coefficients,
                parameters=parameters,degree=solution.problem.features.degree,centers=257,lam=.2,
                provenance=json.dumps(row))
        print(json.dumps(dict(event='forward',noise=noise,number=number,parameters=parameters.tolist(),
            role=row['role'],status=row['status'],degree=row.get('degree'),seconds=row['seconds'],
            sensor_objective=row.get('sensor_objective'),warm_start=row.get('warm_start_provenance'))),flush=True)
    result=solve_route2_inverse(declaration(protocol['initial_viscosity']),[observation],
        inner_options=protocol['inner'],forward_callback=archive,**protocol['outer'])
    before_audit_peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    # Reference information is reintroduced only AFTER the solver returns.
    audit={};prediction=None
    if result.solution is not None:
        model=result.export();held=DOMAIN.interior(2049,882911)
        jets=ordinary_jets(model,held,ORDERS)
        residual=momentum_and_divergence(jets,float(result.parameters[0])).numpy()
        reference=load_measurement_reference()
        truth=batched_values(reference,held)
        prediction=jets[(0,0)].numpy()
        theta=.371+np.arange(389)*2*np.pi/389;wall=np.column_stack([np.cos(theta),np.sin(theta)])
        edge=batched_values(model,wall)[:,:2]
        audit=dict(heldout_velocity_relative_l2=float(np.linalg.norm(prediction[:,:2]-truth[:,:2])/np.linalg.norm(truth[:,:2])),
            ordinary_momentum_rms=float(np.sqrt(np.mean(residual[:,:2]**2))),
            ordinary_momentum_max=float(np.max(abs(residual[:,:2]))),
            ordinary_divergence_rms=float(np.sqrt(np.mean(residual[:,2]**2))),
            ordinary_wall_rms=float(np.sqrt(np.mean((edge-boundary_velocity(wall,'disk_stirring',1.))**2))),
            reference_viscosity=.03,absolute_viscosity_error=float(abs(result.parameters[0]-.03)),
            relative_viscosity_error=float(abs(result.parameters[0]-.03)/.03))
        np.savez_compressed(case/'heldout_audit.npz',points=held,reference=truth,prediction=prediction)
        torch.save(dict(state_dict=model.state_dict(),hidden=model[0].out_features),case/'ordinary_mlp.pt')
    row=dict(status=result.status,noise=noise,fitted_viscosity=float(result.parameters[0]),
        inverse_estimate_resolved=bool(result.metrics['outer_converged']),metrics=result.metrics,
        history=result.history,audit=audit,source_policy=protocol['source_policy'],limitations=protocol['limitations'],
        process_peak_rss_before_audit_bytes=before_audit_peak,
        process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
    (case/'metrics.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(dict(event='inverse_result',noise=noise,status=result.status,
        viscosity=float(result.parameters[0]),audit=audit,seconds=result.metrics['seconds'])),flush=True)
    return row


def summarize_inverse(output=OUT.parent/'wall_ns_reduced_inverse'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    rows=[json.loads(p.read_text()) for p in sorted(output.glob('noise_*/metrics.json'))]
    rows.sort(key=lambda row:row['noise'])
    if not rows:return []
    for row in rows:
        path=output/f"noise_{row['noise']:g}"/'heldout_audit.npz'
        if path.exists():
            data=np.load(path)
            row['audit']['heldout_pressure_relative_l2']=float(np.linalg.norm(data['prediction'][:,2]-data['reference'][:,2])/np.linalg.norm(data['reference'][:,2]))
        for record in row['history']:
            # Reconstruct from the actual objective, including early archived
            # solver versions whose display-only RMS had an extra sqrt(count).
            record['normalized_sensor_rms']=float(np.sqrt(2*record['sensor_objective']))
        if row['noise']>0 and row['history'] and row['history'][-1].get('response_singular_values'):
            singular=row['history'][-1]['response_singular_values'][0]
            row['audit']['local_linear_parameter_sigma']=float(.05/(np.sqrt(32)*singular)) if singular>0 else None
            row['audit']['local_uncertainty_scope']='Known-noise scalar linearization at the fitted parameter only; response includes parameter scale and mean-data normalization. Not a global identifiability, numerical-sensitivity resolution or posterior certificate.'
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    handles=[]
    for index,row in enumerate(rows):
        color=f'C{index}';label=f"Noise σ={row['noise']:g}"
        trajectory=[h['parameters'][0] for h in row['history']]
        if not trajectory or trajectory[-1]!=row['fitted_viscosity']:trajectory.append(row['fitted_viscosity'])
        line,=axes[0].plot(range(len(trajectory)),trajectory,'o-',color=color,label=label);handles.append(line)
        audit=row['audit']
        if audit:
            values=[audit['ordinary_momentum_rms'],audit['ordinary_divergence_rms'],audit['ordinary_wall_rms']]
            axes[1].semilogy(np.arange(3)+.12*(index-.5),values,'o',color=color)
            data=np.load(output/f"noise_{row['noise']:g}"/'heldout_audit.npz')
            error=np.linalg.norm(data['prediction'][:,:2]-data['reference'][:,:2],axis=1)
            axes[2].semilogx(np.maximum(np.sort(error),1e-17),np.arange(1,len(error)+1)/len(error),color=color)
    truth=axes[0].axhline(.03,color='black',linestyle='--',linewidth=1,label='Reference viscosity (audit only)');handles.append(truth)
    axes[0].set(xlabel='Accepted outer iterate',ylabel='Viscosity',title='Inverse parameter trajectory')
    axes[1].axhline(1e-8,color='gray',linestyle=':',linewidth=1)
    axes[1].set_xticks(range(3),['Momentum','Divergence','Wall velocity'])
    axes[1].set(ylabel='Ordinary MLP held-out RMS',title='Physics at the inferred parameter')
    axes[2].set(xlabel='Velocity error against synthetic reference',ylabel='Fraction of unseen points',title='Held-out field error distribution')
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('Wall-driven Navier–Stokes inverse: '+', '.join(f"σ={r['noise']:g}: {r['status']}" for r in rows),fontsize=11,y=.995)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=len(handles),frameon=False,fontsize=9)
    fig.text(.5,.015,'Finer same-method native reference supplies only32 sensor scalars; no reference-state initialization. This is not independent-discretization validation.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.065,right=.985,top=.77,bottom=.19,wspace=.36)
    fig.savefig(output/'wall_ns_inverse.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inverse-case',type=int)
    parser.add_argument('--freeze-inverse',action='store_true');parser.add_argument('--summarize-inverse',action='store_true');args=parser.parse_args()
    if args.freeze_inverse:print(json.dumps(freeze_inverse_protocol(OUT.parent/'wall_ns_reduced_inverse')),flush=True)
    elif args.summarize_inverse:summarize_inverse()
    elif args.inverse_case is not None:run_inverse(args.inverse_case)
    else:run()
