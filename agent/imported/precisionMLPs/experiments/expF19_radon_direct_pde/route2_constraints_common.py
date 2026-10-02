"""Shared flat-tanh annulus controls; no PDE solution initializes a readout.

The manufactured profile supplies known forcing, boundary fluxes, optional
measurements and held-out truth only. All unknown fields start at zero.
The fixed feature map is constructed analytically by RidgePINNFeatures.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quill-route2-constraints-mpl')

from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import svd, lstsq
import sympy as sp
import torch

from solver.ridge_pinn_features import RidgePINNFeatures
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual, linearize_residual
from solver.general_symbolic import DifferentialResidual

OUT = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/constraints'
R0, KAPPA, FIELD_SCALE = .4,1.7,.1
ZERO,DX,DY,DXX,DYY = (0,0),(1,0),(0,1),(2,0),(0,2)
FEATURE_SETTINGS = dict(degree=10,centers=257,lam=.2,encoding_tolerance=1e-10)
SOLVER_SETTINGS = dict(max_iterations=25,tolerance=2e-13,high_accuracy=True,
                       preconditioner='block',max_seconds=60.,max_residual_evaluations=200)


def annulus_points(count,seed):
    rng = np.random.default_rng(seed)
    radius = np.sqrt(R0**2+(1-R0**2)*rng.uniform(size=count))
    angle = 2*np.pi*rng.uniform(size=count)
    return radius[:,None]*np.column_stack((np.cos(angle),np.sin(angle)))


def circle_points(count,radius,phase=.173):
    angle = 2*np.pi*(np.arange(count)+phase)/count
    return radius*np.column_stack((np.cos(angle),np.sin(angle)))


def annulus_quadrature(radial=10,angular=32):
    z,w = leggauss(radial)
    radius = np.sqrt(R0**2+(1-R0**2)*(z+1)/2)
    angle = 2*np.pi*(np.arange(angular)+.319)/angular
    points = (radius[:,None,None]*np.column_stack((np.cos(angle),np.sin(angle)))[None,:,:]).reshape(-1,2)
    weights = np.repeat(w/(2*angular),angular)
    return points,weights


def truth(points):
    x,y = points[:,0],points[:,1]
    radius2 = x*x+y*y
    return .8*(1-radius2)*(radius2-R0**2)*(1+.2*x-.1*y+.15*x*y)


def diffusion(points):
    x,y = points[:,0],points[:,1]
    return 1+.25*x+.15*y+.2*(x*x+y*y)


@lru_cache(None)
def declarations(beta,inverse,boundary_kind):
    x,y,k = sp.symbols('x y kappa',real=True)
    u = sp.Function('u')(x,y)
    r2 = x*x+y*y
    profile = sp.Rational(4,5)*(1-r2)*(r2-sp.Rational(4,25))*(1+x/5-y/10+3*x*y/20)
    a = 1+x/4+3*y/20+r2/5
    true_k = sp.Rational(17,10)
    coefficient = k if inverse else true_k
    parameters = (k,) if inverse else ()
    forcing = -true_k*(sp.diff(a*sp.diff(profile,x),x)+sp.diff(a*sp.diff(profile,y),y))+beta*profile**3
    equation = -coefficient*(sp.diff(a*sp.diff(u,x),x)+sp.diff(a*sp.diff(u,y),y))+beta*u**3-forcing
    result = dict(equation=DifferentialResidual((x,y),(u,),(equation,),parameters),
                  dirichlet=DifferentialResidual((x,y),(u,),(u,),parameters),
                  force=sp.lambdify((x,y),forcing,'numpy'),
                  profile_expression=str(profile),forcing_expression=str(forcing))
    for name,radius,sign in [('outer',sp.Integer(1),1),('inner',sp.Rational(2,5),-1)]:
        target_flux = sign*true_k*a*(x*sp.diff(profile,x)+y*sp.diff(profile,y))/radius
        flux = sign*coefficient*a*(x*sp.diff(u,x)+y*sp.diff(u,y))/radius
        reaction = 2 if boundary_kind == 'robin' and name == 'inner' else 0
        expression = flux+reaction*u-target_flux-reaction*profile
        result[name+'_flux'] = DifferentialResidual((x,y),(u,),(expression,),parameters)
    return result


def build_problem(features, *, beta=3, inverse=False,boundary_kind='robin',anchors=None,
                  parameter_initial=1.,seed=8419):
    if boundary_kind not in ('robin','dirichlet','neumann_mean'):
        raise ValueError('Boundary kind must be robin, dirichlet or neumann_mean')
    if inverse and (beta != 0 or boundary_kind != 'dirichlet'):
        raise ValueError('This inverse control declares linear diffusion with homogeneous Dirichlet data')
    declaration = declarations(beta,inverse,boundary_kind)
    interior = annulus_points(6*features.size,seed)
    outer,inner = circle_points(96,1.),circle_points(96,R0)
    blocks = [declaration['equation'].block('variable_coefficient_equation',interior)]
    if boundary_kind == 'dirichlet':
        blocks.append(declaration['dirichlet'].block('two_dirichlet_circles',np.r_[outer,inner],scale=FIELD_SCALE))
    elif boundary_kind == 'robin':
        blocks.extend((declaration['dirichlet'].block('outer_dirichlet',outer,scale=FIELD_SCALE),
                       declaration['inner_flux'].block('inner_robin',inner)))
    else:
        blocks.extend((declaration['outer_flux'].block('outer_neumann',outer),
                       declaration['inner_flux'].block('inner_neumann',inner)))
        points,weights = annulus_quadrature()
        mean = .8*(1-R0**2)**2/6
        blocks.append(ResidualBlock('integral_mean_gauge',points,
                      lambda x,j,p:j[ZERO][:,0]-mean,(ZERO,),scale=FIELD_SCALE,
                      aggregation=weights[None,:]))
    if anchors is not None:
        points,values = anchors
        prescribed = torch.as_tensor(values,dtype=torch.float64)
        blocks.append(ResidualBlock('sparse_measurements',points,
                      lambda x,j,p:j[ZERO][:,0]-prescribed,(ZERO,),scale=FIELD_SCALE))
    problem = ResidualProblem(features,blocks,parameter_initial=(parameter_initial,) if inverse else (),
                              parameter_bounds=([.25],[4.]) if inverse else None)
    return problem,declaration


def torch_model(features,coefficients):
    arrays = features.compile(coefficients)
    model = torch.nn.Sequential(torch.nn.Linear(2,len(arrays['first_bias']),dtype=torch.float64),
                                torch.nn.Tanh(),torch.nn.Linear(len(arrays['first_bias']),1,dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(arrays['first_weights']))
        model[0].bias.copy_(torch.tensor(arrays['first_bias']))
        model[2].weight.copy_(torch.tensor(arrays['output_weights'].T))
        model[2].bias.copy_(torch.tensor(arrays['output_bias']))
    return model.requires_grad_(False)


def raw_jets(model,points):
    x = torch.tensor(points,dtype=torch.float64,requires_grad=True)
    u = model(x)[:,0]
    gradient = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    xx = torch.autograd.grad(gradient[:,0].sum(),x,retain_graph=True)[0][:,0]
    yy = torch.autograd.grad(gradient[:,1].sum(),x)[0][:,1]
    return {ZERO:u.detach().numpy(),DX:gradient[:,0].detach().numpy(),
            DY:gradient[:,1].detach().numpy(),DXX:xx.numpy(),DYY:yy.numpy()}


def strong_residual(points,jets,kappa,beta,source):
    x,y = points[:,0],points[:,1]
    divergence = diffusion(points)*(jets[DXX]+jets[DYY])+(.25+.4*x)*jets[DX]+(.15+.4*y)*jets[DY]
    return -kappa*divergence+beta*jets[ZERO]**3-source(x,y)


def local_identifiability(problem,solution,noise_sigma=0.):
    """Small dense audit ONLY; never used to initialize or compute a solution.

    Rank refers to the declared finite-dimensional local model. Exact scale
    ambiguity is also available analytically for the data-free inverse case.
    This is not a global uniqueness or statistical-coverage certificate.
    """
    began = time.perf_counter()
    linear = linearize_residual(problem,solution.coefficients,solution.parameters)
    jacobian = linear.operator@np.eye(linear.operator.shape[1])
    norm = np.linalg.norm(jacobian,axis=0)
    scaled = jacobian/np.maximum(norm,np.finfo(float).tiny)
    singular = svd(scaled,compute_uv=False)
    threshold = 1e-9*singular[0]
    residual_norm = np.linalg.norm(linear.residual)
    stationarity = np.max(abs(jacobian.T@linear.residual)/
                           np.maximum(norm*residual_norm,np.finfo(float).tiny))
    result = dict(scope='Local finite-dimensional diagnostic; dense audit does not participate in solve',
                  columns=jacobian.shape[1],rank=int(np.count_nonzero(singular>threshold)),
                  relative_rank_threshold=1e-9,smallest_scaled_singular_value=float(singular[-1]),
                  largest_scaled_singular_value=float(singular[0]),
                  full_column_rank=bool(singular[-1]>threshold),
                  maximum_column_normalized_gradient=float(stationarity),
                  sampled_first_order_stationarity=bool(stationarity<1e-8),
                  stationarity_note='Maximum |J_j^T r|/(||J_j|| ||r||), threshold1e-8; meaningful for nonzero-residual noisy fits, not a global optimality guarantee',
                  audit_jacobian_bytes=jacobian.nbytes,audit_seconds=time.perf_counter()-began)
    if len(solution.parameters):
        nuisance = jacobian[:,:-1]
        parameter = jacobian[:,-1]
        correction = lstsq(nuisance,parameter,cond=1e-12)[0]
        profiled = parameter-nuisance@correction
        ratio = np.linalg.norm(profiled)/max(np.linalg.norm(parameter),np.finfo(float).tiny)
        result.update(parameter_identifiable=bool(ratio>1e-8),
                      relative_profiled_parameter_sensitivity=float(ratio),
                      absolute_profiled_parameter_sensitivity=float(np.linalg.norm(profiled)))
        if ratio>1e-8 and noise_sigma:
            observation_count = len(problem.blocks[-1].points)
            pseudo = np.linalg.pinv(jacobian,rcond=1e-12)
            standard = noise_sigma*np.linalg.norm(pseudo[-1,-observation_count:])/(FIELD_SCALE*np.sqrt(observation_count))
            result.update(linearized_parameter_noise_std=float(standard),
                          uncertainty_scope='First-order iid Gaussian observation-noise propagation with exact assumed PDE; not validated coverage or model-error uncertainty')
    return result


def run_case(label,features, *, beta=3,inverse=False,boundary_kind='robin',anchors=None,
             parameter_initial=1.,noise_sigma=0.,save_model=False):
    OUT.mkdir(parents=True,exist_ok=True)
    problem,declaration = build_problem(features,beta=beta,inverse=inverse,boundary_kind=boundary_kind,
                                      anchors=anchors,parameter_initial=parameter_initial)
    began = time.perf_counter()
    solution = solve_residual(problem,**SOLVER_SETTINGS)
    solve_seconds = time.perf_counter()-began
    began = time.perf_counter()
    held = annulus_points(1201,94717)
    expected = truth(held)
    model = torch_model(features,solution.coefficients)
    with torch.no_grad():
        predicted = model(torch.tensor(held)).numpy()[:,0]
    grouped = solution.evaluate(held)[:,0]
    kappa = float(solution.parameters[0]) if inverse else KAPPA
    jets = {d:solution.evaluate(held,d)[:,0] for d in (ZERO,DX,DY,DXX,DYY)}
    stable = strong_residual(held,jets,kappa,beta,declaration['force'])
    raw = raw_jets(model,held[:67])
    ad_residual = strong_residual(held[:67],raw,kappa,beta,declaration['force'])
    fresh,_ = build_problem(features,beta=beta,inverse=inverse,boundary_kind=boundary_kind,
                           parameter_initial=parameter_initial,seed=62377)
    constraint_checks = []
    parameters = torch.full((1,len(solution.parameters)),kappa,dtype=torch.float64)
    for block in fresh.blocks[1:]:
        # Independent angular phases and a different quadrature order/phase
        # audit the circle and global integral constraints.
        points = block.points.copy()
        aggregation = block.aggregation
        if block.aggregation is None:
            angle = .037
            points = points@np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        else:
            points,weights = annulus_quadrature(radial=13,angular=37)
            aggregation = weights[None,:]
        grouped_jets = {d:torch.tensor(solution.evaluate(points,d)) for d in block.derivatives}
        residual = block.function(torch.tensor(points),grouped_jets,parameters.expand(len(points),-1)).detach().numpy()
        if all(sum(d)==0 for d in block.derivatives):
            with torch.no_grad():
                ordinary_jets = {ZERO:model(torch.tensor(points))}
        else:
            ordinary = raw_jets(model,points)
            ordinary_jets = {d:torch.tensor(ordinary[d][:,None]) for d in block.derivatives}
        ordinary_residual = block.function(torch.tensor(points),ordinary_jets,parameters.expand(len(points),-1)).detach().numpy()
        if aggregation is not None:
            residual = aggregation@residual
            ordinary_residual = aggregation@ordinary_residual
        constraint_checks.append(dict(name=block.name,rms=float(np.sqrt(np.mean(residual**2))),maximum=float(np.max(abs(residual))),
                                      raw_ordinary_torch_ad_rms=float(np.sqrt(np.mean(ordinary_residual**2))),
                                      raw_ordinary_torch_ad_max=float(np.max(abs(ordinary_residual))),
                                      validation_points=len(points),independent_points_or_quadrature=True))
    inference_started = time.perf_counter()
    with torch.no_grad():
        model(torch.tensor(held))
    inference_seconds = time.perf_counter()-inference_started
    identification = local_identifiability(problem,solution,noise_sigma)
    traced = torch.jit.trace(model,torch.tensor(held[:3]))
    operators = sorted({node.kind() for node in traced.inlined_graph.nodes()})
    assert set(operators) <= {'prim::GetAttr','aten::linear','aten::tanh'}
    if save_model:
        torch.jit.save(traced,str(OUT/(label+'_ordinary_tanh.pt')))
    scientific_status = ('unidentifiable' if inverse and not identification['parameter_identifiable']
                         else ('noisy_fit_stationary_sampled' if identification['sampled_first_order_stationarity']
                               else 'noisy_fit_stationarity_not_verified') if noise_sigma else solution.status)
    record = dict(label=label,beta=beta,boundary_kind=boundary_kind,inverse=inverse,
                  initial_field='all zero coefficients',initial_parameter=parameter_initial if inverse else None,
                  kappa_true=KAPPA,kappa_estimate=kappa,parameter_relative_error=abs(kappa-KAPPA)/KAPPA if inverse else None,
                  parameter_bounds=[.25,4.] if inverse else None,noise_sigma_absolute=noise_sigma,
                  observation_count=0 if anchors is None else len(anchors[0]),
                  provenance='Exact profile only manufactures declared forcing/boundary/data and final audits; never a field initializer or interior label unless this case explicitly declares sparse observations',
                  features=features.metrics,solver_settings=SOLVER_SETTINGS,solver=solution.metrics,
                  scientific_status=scientific_status,identifiability=identification,
                  ordinary_forward_relative_l2=float(np.linalg.norm(predicted-expected)/np.linalg.norm(expected)),
                  ordinary_forward_max_error=float(np.max(abs(predicted-expected))),
                  ordinary_vs_grouped_max=float(np.max(abs(predicted-grouped))),
                  stable_fresh_pde_rms=float(np.sqrt(np.mean(stable**2))),
                  stable_fresh_pde_max=float(np.max(abs(stable))),
                  raw_torch_ad_pde_rms=float(np.sqrt(np.mean(ad_residual**2))),
                  raw_torch_ad_pde_max=float(np.max(abs(ad_residual))),
                  raw_torch_ad_vs_stable_jet_max={str(d):float(np.max(abs(raw[d]-jets[d][:67]))) for d in raw},
                  fresh_constraint_checks=constraint_checks,graph_operators=operators,
                  model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                  solve_including_cache_seconds=solve_seconds,heldout_audit_seconds=time.perf_counter()-began,
                  ordinary_inference_1201_points_seconds=inference_seconds,
                  exact_profile=declaration['profile_expression'],known_forcing=declaration['forcing_expression'])
    if anchors is not None:
        with torch.no_grad():
            observation_prediction = model(torch.tensor(anchors[0])).numpy()[:,0]
        record['observation_fit_rms'] = float(np.sqrt(np.mean((observation_prediction-anchors[1])**2)))
        record['observation_truth_rms'] = float(np.sqrt(np.mean((observation_prediction-truth(anchors[0]))**2)))
    (OUT/(label+'.json')).write_text(json.dumps(record,indent=2))
    np.savez_compressed(OUT/(label+'.npz'),coefficients=solution.coefficients,parameters=solution.parameters,
                        held_points=held,held_truth=expected,ordinary_prediction=predicted)
    print(json.dumps({key:record[key] for key in ('label','scientific_status','kappa_estimate','ordinary_forward_relative_l2','stable_fresh_pde_rms','raw_torch_ad_pde_rms','solve_including_cache_seconds')}),flush=True)
    return record


def new_features():
    torch.set_num_threads(1)
    return RidgePINNFeatures(**FEATURE_SETTINGS)


def summary_provenance():
    return dict(domain='annulus .4<sqrt(x^2+y^2)<1',coefficient='a(x,y)=1+.25x+.15y+.2(x^2+y^2)',
                positivity_lower_bound=.89375,features=FEATURE_SETTINGS,solver=SOLVER_SETTINGS,
                data_scale=FIELD_SCALE,initialization='Every field starts from zero; inverse kappa starts at1 unless an ambiguity probe explicitly varies it',
                boundary_freedom='Homogeneous two-circle Dirichlet admits (1-r^2)(r^2-.16)q(x,y), giving28 independent interior polynomial variations at degree10. Mixed outer-Dirichlet/inner-Robin admits (1-r^2)(r^2-.16)^2 q, giving15 such variations.',
                limits=['Smooth manufactured low-degree controls, not arbitrary irregular geometry, shocks or large-scale performance.',
                        'Fixed neural hidden layer, analytically constructed readout coordinates, numerical nonlinear coefficient/parameter solve.',
                        'Noisy data generally make zero residual impossible; stationary/iteration-limited optimization is preserved.',
                        'Identifiability audit is local finite-dimensional; the no-data scale ambiguity additionally has an exact analytical family.',
                        'All plots and reported field errors use the exported ordinary float64 tanh MLP; stable grouped jets and raw Torch AD are distinguished.'])
