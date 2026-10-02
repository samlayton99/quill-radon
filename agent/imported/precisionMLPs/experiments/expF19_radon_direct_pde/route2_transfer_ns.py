"""Frozen transfer tests for native 3D/space-time NS and a physical 3D cavity.

All coefficients come from PDE/BC/IC/gauge residuals. No target readout, dense
reference fit, symmetry reduction, or externally evolved field initializes a
solve. Ordinary audits and plots happen after the native state is saved.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-route2-transfer-mpl')
import argparse, hashlib, itertools, json, resource, time, traceback
from pathlib import Path
import torch
import numpy as np
from route2_gap_diagnosis import ROOT,log,stamp,dump,emit
from route2_hardening_ns import make_declaration,physical_terms,orders,manufactured_jets,validation_field_torch
from solver.general_domains import ResidualDomain
from solver.general_adaptive import ResidualDeclaration
from solver.general_residual import ResidualBlock
from solver.residual_scaling import normalize_equations
from solver.route2 import solve_route2_native,ordinary_jets,footprint

OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02'
CASES={
 'N001_ns3_smooth':dict(family='trigonometric',frequency=.7,viscosity=.2,transient=False),
 'N002_ns3_bubble':dict(family='bubble',frequency=1.5,viscosity=.03,transient=False),
 'N003_ns4_transient':dict(family='trigonometric',frequency=1.5,viscosity=.05,transient=True),
 'N004_ns3_cavity':dict(family='cavity',viscosity=.05,lid_speed=.2,transient=False)}
OPTIONS=dict(coordinates='box',angular_rule='tensor',centers=257,lam=.2,
 degrees=(4,6,8,10,12,16,20),tolerance=1e-13,oversampling=6,batch_size=64,
 check_points=257,export_check_points=129,maximum_working_array_mb=1024,
 maximum_neurons=2000000,max_seconds=600.,max_iterations=8,
 solver_options=dict(inexact_newton=False,lsmr_max_iterations=4000,
                     refine_on_ideal_stall=True,damping=1e-6))


def cavity_declaration(viscosity=.05,lid_speed=.2):
    domain=ResidualDomain([[-1.,1.]]*3)
    zero,first,second,_=orders(3);ders=(zero,)+first+second
    def blocks(count,seed):
        pts=domain.interior(count,seed)
        def physics(x,j,p):
            m,div=physical_terms(j,viscosity,3)
            return torch.cat((m,div[:,None]),dim=1)
        boundary=domain.boundary(max(96,count//2),seed+17).points
        def wall(x,j,p):
            lid=torch.isclose(x[:,1],torch.ones_like(x[:,1]),atol=1e-12,rtol=0)
            speed=lid_speed*(1-x[:,0]**2)**2*(1-x[:,2]**2)**2*lid
            trace=torch.stack((speed,torch.zeros_like(speed),torch.zeros_like(speed)),dim=1)
            return j[zero][:,:3]-trace
        return [ResidualBlock('momentum_and_incompressibility',pts,physics,ders),
                ResidualBlock('velocity_boundary',boundary,wall,(zero,)),
                ResidualBlock('pressure_gauge',np.zeros((1,3)),lambda x,j,p:j[zero][:,3],(zero,))]
    return ResidualDeclaration(domain,blocks,fields=4)


def declaration(case):
    if case['family']=='cavity':return cavity_declaration(case['viscosity'],case['lid_speed'])
    return make_declaration(**case,combined_physics=True,reflection_orbits=False)


def audit(sol,case,folder):
    dimension=3+int(case['transient']);zero,first,second,dt=orders(dimension)
    domain=sol.declaration.domain if hasattr(sol,'declaration') else declaration(case).domain
    model=sol.export();architecture=[type(layer).__name__ for layer in model]
    assert architecture==['Linear','Tanh','Linear']
    value_points=domain.interior(513,734153)
    values=ordinary_jets(model,value_points,(zero,))[zero].numpy()
    dp=domain.interior(129,813733);ders=(zero,)+first+second+((dt,) if dt else ())
    jets={d:v.numpy() for d,v in ordinary_jets(model,dp,ders).items()}
    momentum,divergence=physical_terms(jets,case['viscosity'],dimension)
    expected=None
    if case['family']!='cavity':
        args={k:case[k] for k in ('family','frequency','transient')}
        force,_=physical_terms(manufactured_jets(dp,**args),case['viscosity'],dimension)
        momentum-=force
        expected=validation_field_torch(torch.tensor(value_points,dtype=torch.float64),**args).numpy()
    corners=np.array(list(itertools.product(*domain.bounds.tolist())))
    corner_values=ordinary_jets(model,corners,(zero,))[zero].numpy()
    checks={}
    for block in declaration(case).make_blocks(129,177331):
        if block.name=='momentum_and_incompressibility':continue
        bjets=ordinary_jets(model,block.points,block.derivatives)
        r=block.function(torch.tensor(block.points),bjets,torch.empty((len(block.points),0))).detach().numpy()
        checks[block.name]=dict(rms=float(np.sqrt(np.mean(r*r))),max_abs=float(np.max(abs(r))))
    grouped=sol.evaluate(value_points)
    report=dict(momentum_rms=float(np.sqrt(np.mean(momentum**2))),momentum_max=float(np.max(abs(momentum))),
        divergence_rms=float(np.sqrt(np.mean(divergence**2))),divergence_max=float(np.max(abs(divergence))),
        boundary_and_initial=checks,ordinary_vs_grouped_max=float(np.max(abs(values-grouped))),
        readout_l1_per_field=model[2].weight.detach().abs().sum(dim=1).tolist(),
        ordinary_model_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        architecture=architecture,field_points=len(value_points),derivative_points=len(dp),
        corner_count=len(corners),known_field=expected is not None,truth_used_in_fitting=False,
        velocity_relative_l2=None,pressure_relative_l2=None)
    if expected is not None:
        delta=values-expected
        corner_truth=validation_field_torch(torch.tensor(corners,dtype=torch.float64),**args).numpy()
        report.update(velocity_relative_l2=float(np.linalg.norm(delta[:,:3])/np.linalg.norm(expected[:,:3])),
            pressure_relative_l2=float(np.linalg.norm(delta[:,3])/np.linalg.norm(expected[:,3])),
            field_max_abs=float(np.max(abs(delta))),corner_max_abs=float(np.max(abs(corner_values-corner_truth))))
    plane_axis=np.linspace(-1,1,25);a,b=np.meshgrid(plane_axis,plane_axis,indexing='xy')
    plane=np.column_stack((a.ravel(),b.ravel(),np.zeros(a.size)))
    if dimension==4:plane=np.column_stack((plane,np.full(len(plane),.125)))
    plane_values=ordinary_jets(model,plane,(zero,))[zero].numpy()
    payload=dict(value_points=value_points,values=values,derivative_points=dp,momentum=momentum,
        divergence=divergence,corners=corners,corner_values=corner_values,plane_points=plane,plane_values=plane_values)
    if expected is not None:payload.update(target=expected,corner_target=corner_truth)
    np.savez_compressed(folder/'audit_arrays.npz',**payload)
    torch.save(dict(state_dict=model.state_dict(),architecture=architecture,dimension=dimension,width=model[0].out_features),folder/'ordinary_mlp.pt')
    return report


def run(run_id):
    torch.set_num_threads(1)
    case=CASES[run_id];folder=OUT/run_id;folder.mkdir(parents=True,exist_ok=False)
    dimension=3+int(case['transient'])
    forecasts=[footprint(dimension,p,OPTIONS['centers'],4,'box','tensor','streamed',64,lam=.2) for p in OPTIONS['degrees']]
    sources=[Path(__file__),Path(__file__).parent/'route2_hardening_ns.py']+[Path(__file__).parent/'solver'/n for n in ['route2.py','general_residual.py','streamed_residual.py','residual_scaling.py']]
    config=dict(run_id=run_id,case=case,options=OPTIONS,started=stamp(),dimension=dimension,
        declaration='PDE/BC/IC/gauge only; cold zero and native continuation',
        normalization='fixed zero-jet differential sensitivity on joint momentum/divergence',
        no_symmetry_or_sparse_directions=True,known_interior_target_used_only_after_fit=case['family']!='cavity',
        architecture='ordinary Linear/Tanh/Linear on original inputs',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    dump(folder/'protocol.json',config);dump(folder/'preflight.json',forecasts)
    (folder/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    log(f"\n### {run_id} — START {config['started']}\n\nNS transfer, {case}, inputs={dimension}. Frozen shared nonlinear policy: box/N257/lambda.2, exact inner, maxinner4000,maxouter8,damping1e-6,tol1e-13,600s soft budget. No interiortruth or reference coefficients in solve. Case/protocol/preflight: `{folder.relative_to(ROOT)}`.")
    def callback(c,p,row):
        record=dict(run_id=run_id,coordinate_rows=len(c),**row)
        with (folder/'iterations.jsonl').open('a') as f:f.write(json.dumps(record,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v))+'\n')
        np.savez_compressed(folder/'latest_native.npz',coefficients=c,parameters=p)
        emit('iteration',run_id=run_id,coordinate_rows=len(c),iteration=row.get('iteration'),rms=row.get('maximum_scaled_block_rms'))
    options=dict(OPTIONS,solver_options=dict(OPTIONS['solver_options'],iteration_callback=callback))
    began=time.perf_counter()
    try:
        raw=declaration(case);normalized=normalize_equations(raw,['momentum_and_incompressibility'])
        sol=solve_route2_native(normalized,**options)
        features=sol.problem.features
        np.savez_compressed(folder/'coefficients.npz',coefficients=sol.coefficients,degree=features.degree,
            centers=257,lam=.2,bounds=features.bounds)
        result=dict(run_id=run_id,status=sol.status,solver_status=sol.solution.status,dimension=dimension,
            degree=features.degree,neurons=features.tanh_count,readout_coordinates=features.size*4,
            directions=features.direction_count,solve_seconds=time.perf_counter()-began,
            stage_history=sol.history,metrics=sol.metrics,inner_history=sol.solution.history)
        dump(folder/'result_before_audit.json',result)
        result['ordinary_audit']=audit(sol,case,folder)
        result.update(seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(folder/'result.json',result);a=result['ordinary_audit']
        log(f"\n**{run_id} FINISH {stamp()} — {sol.status}.** Degree{features.degree}, neurons{features.tanh_count}; ordinary velocity relL2={a['velocity_relative_l2']}, pressure={a['pressure_relative_l2']}; raw momentum RMS={a['momentum_rms']:.6g}, divergence={a['divergence_rms']:.6g}; {result['seconds']:.1f}s, {result['peak_rss_bytes']/2**20:.1f}MiB. Unknown cavity field error remains unknown. Full validation/failure gates retained.")
        emit('result',run_id=run_id,status=result['status'],degree=result['degree'],seconds=result['seconds'],ordinary_audit=a)
    except BaseException as error:
        failure=dict(run_id=run_id,status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error),traceback=traceback.format_exc(),seconds=time.perf_counter()-began)
        dump(folder/'failure.json',failure);log(f"\n**{run_id} FINISH {stamp()} — {failure['status']}.** {error!r}. Checkpoints and failure retained.")
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_id',choices=CASES);args=p.parse_args();run(args.run_id)
