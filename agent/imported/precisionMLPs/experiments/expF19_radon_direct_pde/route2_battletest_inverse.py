"""Frozen-protocol inverse Burgers campaign; all truth stays outside the declaration.

The positive Cole--Hopf heat integral is a measurement/evaluation oracle only.
No sampled reference trajectory or target coefficients initialize native solves.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('KMP_USE_SHM','0')
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from scipy.special import roots_hermite
from scipy.stats import qmc
import torch
from solver.general_adaptive import ResidualDeclaration
from solver.general_domains import ResidualDomain
from solver.general_residual import ResidualBlock
from solver.route2 import solve_route2_native,ordinary_jets

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse'
DOMAIN=ResidualDomain([[-1.,1.],[0.,1.]])
ORDERS=((0,0),(1,0),(0,1),(2,0))
PROFILE='baseline'
SETTINGS=dict(coordinates='affine_disk',degrees=[6,10,14,18,22,26,32],centers=257,lam=.2,
    tolerance=1e-8,max_seconds=60.,maximum_working_array_mb=512,check_points=257,
    export_check_points=257,max_iterations=20,batch_size=128)


def points(count,seed):
    z=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count]
    z[:,0]=2*z[:,0]-1
    return z


def oracle(points_,viscosity,order=256):
    """Analytic heat integral, not a numerical PDE evolution or training input.

    phi0=exp(-cos(pi*x)/(2*pi*nu)); phi=E phi0(x+sqrt(2*nu*t)*Z).
    Integration by parts gives u=-E[sin(pi*Y)*phi0(Y)]/E[phi0(Y)].
    Positive quadrature avoids cancellation of the Fourier representation.
    """
    x=np.asarray(points_,float)
    nodes,weights=roots_hermite(order)
    y=x[:,0:1]+np.sqrt(4*viscosity*x[:,1:2])*nodes
    logphi=-np.cos(np.pi*y)/(2*np.pi*viscosity)
    weighted=np.exp(logphi-logphi.max(axis=1,keepdims=True))*weights
    return -np.sum(weighted*np.sin(np.pi*y),axis=1,keepdims=True)/weighted.sum(axis=1,keepdims=True)


@dataclass(frozen=True)
class Measurements:
    locations: np.ndarray
    values: np.ndarray
    noise_sigma: float


def generate_measurements(nu,noise,seed,count=32):
    locations=points(count,1000+seed)
    locations[:,1]=.05+.9*locations[:,1]
    values=oracle(locations,nu)
    values+=noise*np.random.default_rng(8000+seed).normal(size=values.shape)
    return Measurements(locations,values,float(noise))


def make_declaration(measurements,initial_viscosity):
    """Receives only measured values, their known noise scale, and initial guess."""
    locations=measurements.locations.copy()
    data=torch.tensor(measurements.values.copy(),dtype=torch.float64)
    def blocks(count,seed):
        interior=points(count,seed)
        trace_count=max(64,int(np.sqrt(count))*4)
        initial=points(trace_count,seed+1);initial[:,1]=0
        walls=points(2*trace_count,seed+2);walls[:trace_count,0]=-1;walls[trace_count:,0]=1
        def physics(x,j,p):
            return j[(0,1)]+j[(0,0)]*j[(1,0)]-p[:,:1]*j[(2,0)]
        def observed(x,j,p):return j[(0,0)]-data
        def observed_batch(x,j,p,rows):return j[(0,0)]-data[rows]
        return [ResidualBlock('burgers',interior,physics,ORDERS),
            ResidualBlock('initial',initial,lambda x,j,p:j[(0,0)]+torch.sin(np.pi*x[:,:1]),((0,0),)),
            ResidualBlock('walls',walls,lambda x,j,p:j[(0,0)],((0,0),)),
            ResidualBlock('sensors',locations,observed,((0,0),),batch_function=observed_batch)]
    return ResidualDeclaration(DOMAIN,blocks,parameter_initial=[initial_viscosity],parameter_bounds=([.001],[.5]))


def protocol():
    cases=[]
    for nu in (.1,.02):
        for noise in (0.,.001,.01):
            for seed,initial in ((0,.04),(1,.2)):
                cases.append(dict(viscosity=nu,noise=noise,seed=seed,initial=initial))
        # Same noiseless sensors, different start, for both truth regimes.
        cases.append(dict(viscosity=nu,noise=0.,seed=0,initial=.2))
    if PROFILE=='accuracy':
        cases=[case for case in cases if case['seed']==0 and case['initial']==.04 and case['noise'] in (0.,.001)]
    return dict(version={'baseline':1,'tight':2,'accuracy':3}[PROFILE],created_utc='2026-10-01',equation='u_t+u*u_x-nu*u_xx=0',
        domain=[[-1,1],[0,1]],initial='-sin(pi*x)',walls='u(-1,t)=u(1,t)=0',
        unknown='scalar viscosity nu in [.001,.5]',sensors=32,
        noise='independent Gaussian absolute standard deviation; initial amplitude is one',
        observation_tolerance='max(1e-8,1.5*known_noise_sigma); all objective weights fixed at one',
        settings=SETTINGS,cases=cases,oracle_quadrature=256,oracle_convergence_check=512,
        untouched_audit=dict(points=1024,seed=991911,ordinary_ad_points=257,curves_times=[.1,.5,.9]),
        rules=['Protocol written before first solve; cannot silently overwrite.',
            'No truth-dependent architecture, lambda, degree or initialization selection.',
            'Zero readouts for each case; only generic native degree continuation.',
            'No reference solution, true parameter, or held-out values in declaration.',
            'Every attempted case is recorded, including exceptions and budget failures.',
            'Noise does not justify machine-precision parameter claims.',
            'Same algorithm/settings across regimes; seed/start paired except noiseless crossed control.',
            'Independent audit only after solver returns; not used to select degree or best state.'],
        source_policy=dict(reference='Positive Cole-Hopf heat integral, measurement/evaluation only',
            external_pde_solver=False,target_readout_fit=False,oracle_initialization=False,
            interior_labels='Only declared 32 sensor values',parameter_truth='Generator and post-solve evaluator only',
            primary_sources=['https://maziarraissi.github.io/PINNs/','https://github.com/pdebench/PDEBench'],
            benchmark_scope='Standard nonlinear Burgers/parameter-identification task; not the published PDEBench dataset or a head-to-head baseline'))


def freeze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=json.dumps(protocol(),indent=2,sort_keys=True)+'\n'
    path=output/'protocol.json'
    if path.exists() and path.read_text()!=data:raise RuntimeError('Frozen protocol differs; create a new campaign directory')
    if not path.exists():path.write_text(data)
    digest=hashlib.sha256(data.encode()).hexdigest()
    (output/'protocol.sha256').write_text(digest+'\n')
    return digest


def case_name(case):
    return f"nu{case['viscosity']:g}_noise{case['noise']:g}_seed{case['seed']}_start{case['initial']:g}"


def predict(model,x,batch=64):
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(x[i:i+batch],dtype=torch.float64)).numpy() for i in range(0,len(x),batch)])


def run_case(case,output=OUT):
    torch.set_num_threads(1)
    digest=freeze(output);name=case_name(case);folder=Path(output)/name;folder.mkdir(exist_ok=True)
    if (folder/'metrics.json').exists():return json.loads((folder/'metrics.json').read_text())
    began=time.perf_counter();data=generate_measurements(case['viscosity'],case['noise'],case['seed'])
    np.savez(folder/'measurements.npz',locations=data.locations,values=data.values,noise_sigma=data.noise_sigma)
    declaration=make_declaration(data,case['initial'])
    result=dict(case=case,protocol_sha256=digest,settings=SETTINGS,source_policy=protocol()['source_policy'])
    try:
        solution=solve_route2_native(declaration,**SETTINGS,
            block_tolerances={'sensors':max(1e-8,1.5*data.noise_sigma)})
        solve_seconds=time.perf_counter()-began
        model=solution.export();held=points(1024,991911)
        target=oracle(held,case['viscosity']);target_finer=oracle(held,case['viscosity'],512)
        prediction=predict(model,held);jets=ordinary_jets(model,held[:257],ORDERS)
        nu=float(solution.parameters[0]);residual=(jets[(0,1)]+jets[(0,0)]*jets[(1,0)]-nu*jets[(2,0)]).numpy()
        features=solution.problem.features
        ideal={order:np.zeros((257,1)) for order in ORDERS}
        for start in range(0,features.size,64):
            columns=np.arange(start,min(start+64,features.size))
            panel=features.selected_ideal_columns(held[:257],columns,ORDERS)
            for order in ORDERS:ideal[order]+=panel[order]@solution.coefficients[columns]
        ideal_residual=ideal[(0,1)]+ideal[(0,0)]*ideal[(1,0)]-nu*ideal[(2,0)]
        encoding_gap=float(np.sqrt(np.mean((residual-ideal_residual)**2)))
        sensor_prediction=predict(model,data.locations)
        sensor_clean=oracle(data.locations,case['viscosity'])
        delta=max(1e-7,case['viscosity']*1e-4)
        # Oracle sensitivity is an audit-only local identifiability diagnostic.
        sensitivity=(oracle(data.locations,case['viscosity']+delta)-oracle(data.locations,case['viscosity']-delta))/(2*delta)
        information=float(np.sum(sensitivity**2))
        curve_points=np.array([(x,t) for t in (.1,.5,.9) for x in np.linspace(-1,1,257)])
        curve_pred=predict(model,curve_points);curve_target=oracle(curve_points,case['viscosity'])
        result.update(status=solution.status,solver_status=solution.solution.status,
            fitted_viscosity=nu,viscosity_absolute_error=abs(nu-case['viscosity']),
            viscosity_relative_error=abs(nu-case['viscosity'])/case['viscosity'],
            heldout_relative_l2=float(np.linalg.norm(prediction-target)/np.linalg.norm(target)),
            heldout_maximum_error=float(np.max(abs(prediction-target))),
            ordinary_pde_rms=float(np.sqrt(np.mean(residual**2))),actual_vs_ideal_pde_encoding_rms=encoding_gap,ordinary_pde_max=float(np.max(abs(residual))),
            train_sensor_noisy_rms=float(np.sqrt(np.mean((sensor_prediction-data.values)**2))),
            train_sensor_clean_rms=float(np.sqrt(np.mean((sensor_prediction-sensor_clean)**2))),
            oracle_quadrature_max_difference=float(np.max(abs(target-target_finer))),
            audit_local_parameter_sigma=(data.noise_sigma/np.sqrt(information) if information else None),
            audit_local_sensitivity_norm=np.sqrt(information),
            identifiability_scope='Oracle tangent local sensitivity only; assumes correct model and exact physics/IC; not fitted uncertainty or global uniqueness.',
            degree=solution.problem.features.degree,neurons=solution.problem.features.tanh_count,
            driver=solution.metrics,history=solution.history,solver=solution.solution.metrics,final_solver_history=solution.solution.history,
            solve_seconds=solve_seconds,total_seconds=time.perf_counter()-began,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        np.savez_compressed(folder/'native.npz',coefficients=solution.coefficients,parameters=solution.parameters,
            degree=solution.problem.features.degree,centers=SETTINGS['centers'],lam=SETTINGS['lam'])
        torch.save(dict(state_dict=model.state_dict(),hidden=model[0].out_features),folder/'ordinary_mlp.pt')
        np.savez_compressed(folder/'audit.npz',points=held,prediction=prediction,target=target,
            curve_points=curve_points,curve_prediction=curve_pred,curve_target=curve_target)
    except Exception as error:
        result.update(status='exception',error=repr(error),traceback=traceback.format_exc(),total_seconds=time.perf_counter()-began)
    (folder/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('case','status','degree','fitted_viscosity','heldout_relative_l2','ordinary_pde_rms','solve_seconds','error')}),flush=True)
    return result


def summarize(output=OUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter,LogLocator,NullFormatter
    from matplotlib.lines import Line2D
    output=Path(output);rows=[json.loads(p.read_text()) for p in sorted(output.glob('*/metrics.json'))]
    rows=[r for r in rows if isinstance(r.get('case'),dict) and 'viscosity' in r['case']]
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    import csv
    columns=['viscosity','noise','seed','initial','status','degree','fitted_viscosity','viscosity_relative_error','heldout_relative_l2','ordinary_pde_rms','train_sensor_noisy_rms','audit_local_parameter_sigma','solve_seconds','process_peak_rss_bytes']
    with (output/'summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for row in rows:
            flat=dict(row,**row['case']);writer.writerow({key:flat.get(key) for key in columns})
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    noises=sorted({r['case']['noise'] for r in rows});noise_positions={v:i for i,v in enumerate(noises)}
    limits={}
    for key in ('viscosity_relative_error','heldout_relative_l2'):
        values=[max(r[key],1e-16) for r in rows if key in r]
        if values:limits[key]=(10.**np.floor(np.log10(min(values))),10.**np.ceil(np.log10(max(values))))
    for index,nu in enumerate((.1,.02)):
        subset=[r for r in rows if r['case']['viscosity']==nu and 'fitted_viscosity' in r]
        for row in subset:
            noise=row['case']['noise'];xx=noise_positions[noise]
            offset=(-.12 if row['case']['seed']==0 else .12)+(.04 if row['case']['seed']==0 and row['case']['initial']==.2 else 0)
            for ax,key in zip(axes[index,:2],('viscosity_relative_error','heldout_relative_l2')):
                ax.semilogy(xx+offset,max(row[key],1e-16),'o',color='C0' if row['case']['initial']==.04 else 'C1')
            candidate=output/case_name(row['case'])/'audit.npz'
            if noise==0 and row['case']['seed']==0 and row['case']['initial']==.04:
                data=np.load(candidate)
                for i,t in enumerate((.1,.5,.9)):
                    sl=slice(i*257,(i+1)*257)
                    axes[index,2].plot(data['curve_points'][sl,0],data['curve_prediction'][sl,0],label=f't={t:g}')
                    axes[index,2].plot(data['curve_points'][sl,0],data['curve_target'][sl,0],'k--',lw=.7)
        for ax,key in zip(axes[index,:2],('viscosity_relative_error','heldout_relative_l2')):
            ax.set_xticks(range(len(noises)),[f'{v:g}' for v in noises]);ax.set_xlabel('Sensor noise standard deviation');ax.grid(alpha=.25)
            if key in limits:ax.set_ylim(*limits[key])
            ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{100*value:.3g}%'))
            ax.yaxis.set_minor_formatter(NullFormatter())
        axes[index,0].set_ylabel(f'True viscosity {nu:g}\nRelative viscosity error')
        axes[index,1].set_ylabel('Untouched field relative L2 error')
        axes[index,2].set(xlabel='x',ylabel='u');axes[index,2].grid(alpha=.25)
    for ax,title in zip(axes[0],('Viscosity identification','Field error on unseen points','Solution slices: noiseless sensors')):ax.set_title(title,fontsize=11)
    starts=sorted({r['case']['initial'] for r in rows})
    handles=[Line2D([],[],marker='o',linestyle='',color='C0' if v==.04 else 'C1',label=f'Start viscosity {v:g}') for v in starts]
    handles += [Line2D([],[],color=f'C{i}',label=f'MLP at t={t:g}') for i,t in enumerate((.1,.5,.9))]
    handles += [Line2D([],[],color='black',linestyle='--',label='Cole–Hopf reference')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=len(handles),fontsize=9,frameon=False)
    counts={}
    for row in rows:counts[row['status']]=counts.get(row['status'],0)+1
    fig.suptitle('Frozen inverse campaign: '+', '.join(f'{count} {status.replace("_"," ")} stops' for status,count in counts.items()),fontsize=11)
    fig.subplots_adjust(left=.075,right=.985,bottom=.09,top=.83,wspace=.42,hspace=.35)
    fig.savefig(output/'inverse_campaign.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return rows


def run_reduced_control(output,centers=129,physical_tolerance=2e-9):
    """A cheap native noisy-inverse control, independent of the Burgers campaign."""
    from solver.route2_inverse import SensorObservations,solve_route2_inverse
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    inner=dict(coordinates='affine_disk',degrees=[2,4,6],centers=centers,tolerance=physical_tolerance,
               max_seconds=15,check_points=31,export_check_points=33,max_iterations=12)
    document=dict(name='native reduced diffusion inverse control',
        equation='-nu*(u_xx+u_yy)=2*(2-x*x-y*y), zero boundary, square[-1,1]^2',
        unknown_bounds=[.2,2.],initial_parameter=1.1,measurement_true_parameter=.7,
        sensors=9,seed=892,noise_standard_deviations=[0.,.002],inner_options=inner,
        source_policy='True parameter used only for sensor generator and audit. All native inner solves receive physics/zero boundaries only. No target coefficients/external solver.',
        status_scope='Easy polynomial control validating constrained inverse behavior, not a hard-PDE or high-dimensional benchmark.')
    serialized=json.dumps(document,sort_keys=True,indent=2)+'\n';path=output/'protocol.json'
    if path.exists() and path.read_text()!=serialized:raise RuntimeError('Reduced-control protocol differs')
    if not path.exists():path.write_text(serialized)
    domain=ResidualDomain([[-1.,1.],[-1.,1.]])
    def blocks(count,seed):
        q=domain.interior(count,seed);wall=domain.boundary(64,seed+1).points
        def physics(x,j,p):return -p[:,:1]*(j[(2,0)]+j[(0,2)])-2*(2-x[:,:1]**2-x[:,1:2]**2)
        return [ResidualBlock('diffusion',q,physics,((2,0),(0,2))),
                ResidualBlock('zero_wall',wall,lambda x,j,p:j[(0,0)],((0,0),))]
    declaration=ResidualDeclaration(domain,blocks,parameter_initial=[1.1],parameter_bounds=([.2],[2.]))
    rows=[]
    for noise in (0.,.002):
        folder=output/f'noise{noise:g}';folder.mkdir(exist_ok=True)
        rng=np.random.default_rng(892);q=rng.uniform(-.8,.8,(9,2));g=(1-q[:,0]**2)*(1-q[:,1]**2)
        data=g/.7+noise*rng.normal(size=9)
        np.savez_compressed(folder/'measurements.npz',points=q,values=data,noise_standard_deviation=noise)
        result=solve_route2_inverse(declaration,[SensorObservations(q,data,scale=noise if noise else 1.)],
            max_seconds=90,max_iterations=8,inner_options=inner,gradient_tolerance=1e-8,step_tolerance=1e-9)
        row=dict(noise=noise,status=result.status,parameters=result.parameters.tolist(),history=result.history,metrics=result.metrics)
        if result.solution is not None:
            model=result.export();nu=float(result.parameters[0]);held=domain.interior(257,77271)
            jets=ordinary_jets(model,held,((0,0),(2,0),(0,2)))
            values=jets[(0,0)].numpy().ravel();shape=(1-held[:,0]**2)*(1-held[:,1]**2)
            residual=-nu*(jets[(2,0)]+jets[(0,2)]).numpy().ravel()-2*(2-held[:,0]**2-held[:,1]**2)
            optimum=float((g@g)/(g@data))
            row.update(statistical_optimum_audit_only=optimum,statistical_optimum_parameter_difference=abs(nu-optimum),
                true_parameter_error=abs(nu-.7),ordinary_pde_rms=float(np.sqrt(np.mean(residual**2))),
                ordinary_pde_max=float(np.max(abs(residual))),
                conditional_field_relative_error=float(np.linalg.norm(values-shape/nu)/np.linalg.norm(shape/nu)),
                true_field_relative_error=float(np.linalg.norm(values-shape/.7)/np.linalg.norm(shape/.7)),
                degree=result.solution.problem.features.degree,neurons=result.solution.problem.features.tanh_count,
                process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
            np.savez_compressed(folder/'native.npz',coefficients=result.solution.coefficients,parameters=result.parameters)
            torch.save(dict(state_dict=model.state_dict(),hidden=model[0].out_features),folder/'ordinary_mlp.pt')
        (folder/'metrics.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(json.dumps({k:row.get(k) for k in ('noise','status','parameters','statistical_optimum_audit_only','statistical_optimum_parameter_difference','ordinary_pde_rms','conditional_field_relative_error')}),flush=True)
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--case',type=int)
    parser.add_argument('--summarize',action='store_true');parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--profile',choices=['baseline','tight','accuracy'],default='baseline')
    parser.add_argument('--reduced-control',action='store_true')
    parser.add_argument('--reduced-precision',action='store_true')
    args=parser.parse_args()
    if args.reduced_control or args.reduced_precision:
        if args.reduced_precision:
            run_reduced_control(args.output or OUT/'reduced_diffusion_precision',centers=257,physical_tolerance=1e-12)
        else:run_reduced_control(args.output or OUT/'reduced_diffusion_control')
        raise SystemExit(0)
    PROFILE=args.profile
    if PROFILE=='tight':SETTINGS.update(max_iterations=60,solver_options={'inexact_newton':False})
    if PROFILE=='accuracy':SETTINGS.update(max_iterations=60,max_seconds=600.,degrees=[6,10,14,18,22,26,32,40,48,64])
    args.output=args.output or (OUT if PROFILE=='baseline' else OUT/('tight_control' if PROFILE=='tight' else 'accuracy_extension'))
    if args.freeze:print(freeze(args.output))
    elif args.summarize:summarize(args.output)
    elif args.case is not None:run_case(protocol()['cases'][args.case],args.output)
    else:
        freeze(args.output)
        for case in protocol()['cases']:run_case(case,args.output)
        summarize(args.output)
