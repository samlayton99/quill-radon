"""Reduced inverse problems with independently validated native neural forwards.

Physics callbacks never receive observations. Every trial parameter is frozen
in copied callbacks before a physics-only solve; it is not an inner unknown.
Only forwards passing the native driver's held-out, ordinary-MLP and richer
resolution checks can contribute a sensor objective or finite difference.

This is a small-parameter local inverse method, not a global identifiability
certificate. It uses finite differences of fully solved native networks rather
than claiming implicit sensitivities at an unresolved PDE state.
"""
from __future__ import annotations
from dataclasses import dataclass,field,replace
import time
import numpy as np
import torch
from .general_adaptive import ResidualDeclaration
from .route2 import solve_route2_native
from .general_residual import ResidualSolution


@dataclass(frozen=True)
class SensorObservations:
    points: object
    values: object
    fields: object = None
    scale: object = 1.

    def __post_init__(self):
        points=np.asarray(self.points,float).copy();values=np.asarray(self.values,float).copy()
        if values.ndim==1:values=values[:,None]
        if points.ndim!=2 or values.ndim!=2 or len(points)!=len(values) or not len(points) or values.shape[1]<1:
            raise ValueError('Sensor points and values must be finite nonempty row arrays')
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
            raise ValueError('Sensor points and values must be finite')
        fields=np.arange(values.shape[1]) if self.fields is None else np.asarray(self.fields)
        if not np.all(np.isfinite(fields)) or np.any(fields!=np.floor(fields)):
            raise ValueError('Field indices must be finite integers')
        fields=fields.astype(int)
        if fields.ndim!=1 or len(fields)!=values.shape[1] or len(set(fields))!=len(fields) or np.any(fields<0):
            raise ValueError('Provide one distinct nonnegative field index per observation column')
        scale=np.broadcast_to(np.asarray(self.scale,float),values.shape).copy()
        if np.any(scale<=0) or not np.all(np.isfinite(scale)):raise ValueError('Observation scales must be finite and positive')
        for value in (points,values,fields,scale):value.flags.writeable=False
        object.__setattr__(self,'points',points);object.__setattr__(self,'values',values)
        object.__setattr__(self,'fields',fields);object.__setattr__(self,'scale',scale)


def freeze_physics_parameters(physics,parameters):
    """Copy block callbacks and freeze trial values; inner parameter count is zero."""
    values=np.asarray(parameters,float).reshape(-1).copy()
    if len(values)!=len(physics.parameter_initial) or not np.all(np.isfinite(values)):
        raise ValueError('Trial parameter count must match the physics declaration')
    values.flags.writeable=False
    def make_blocks(count,seed):
        copied=[]
        for block in physics.make_blocks(count,seed):
            def frozen(x,j,unused,original=block.function):
                fixed=torch.tensor(values,dtype=x.dtype,device=x.device).expand(len(x),-1)
                return original(x,j,fixed)
            batched=None
            if block.batch_function is not None:
                def batched(x,j,unused,rows,original=block.batch_function):
                    fixed=torch.tensor(values,dtype=x.dtype,device=x.device).expand(len(x),-1)
                    return original(x,j,fixed,rows)
            copied.append(replace(block,function=frozen,batch_function=batched))
        return copied
    return ResidualDeclaration(physics.domain,make_blocks,fields=physics.fields,
                               parameter_initial=(),parameter_bounds=None)


@dataclass
class NativeInverseResult:
    solution: object
    parameters: np.ndarray
    status: str
    history: list=field(default_factory=list)
    metrics: dict=field(default_factory=dict)
    unresolved_solution: object=None

    def export(self):
        if self.solution is None:raise RuntimeError('No validated native forward is available')
        return self.solution.export()


def _sensor_residual(solution,observations,batch_size=128):
    # Use the ordinary deployed network even though the stable native forward
    # already passed a separate ordinary-AD audit.
    model=solution.export();parts=[]
    normalizer=np.sqrt(sum(obs.values.size for obs in observations))
    with torch.no_grad():
        for obs in observations:
            values=np.concatenate([model(torch.tensor(obs.points[start:start+batch_size],dtype=torch.float64)).numpy()
                                   for start in range(0,len(obs.points),batch_size)])[:,obs.fields]
            parts.append(((values-obs.values)/obs.scale).ravel()/normalizer)
    return np.concatenate(parts)


def _truncated_native_guess(previous,frozen_physics,ladder):
    """Copy named low-degree coordinates; never sample or fit a field.

    The previous angular rule, bandwidth and affine chart are retained. The
    resulting state is deliberately uncertified and uses the NEW parameter's
    physical callbacks. Its next solve must repeat all resolution checks.
    Unsupported coordinate families return a cold-start reason.
    """
    from route2_box_profiles import BoxProfileOperator
    from route2_disk_profiles import DiskProfileOperator
    from route2_affine_disk_profiles import AffineDiskProfileOperator
    old=previous.problem.features
    info=dict(kind='cold',source_degree=int(old.degree))
    choices=[int(p) for p in ladder if p<old.degree]
    if not choices:
        return None,dict(info,reason='no strictly coarser degree on the fixed ladder')
    if type(old) not in (BoxProfileOperator,DiskProfileOperator,AffineDiskProfileOperator):
        return None,dict(info,reason='unsupported coordinate family')
    degree=max(choices)
    # Encoding uses reference spacing 2/(N-1); recovering lambda preserves
    # the original geometry without adding new metadata to shared operators.
    lam=float(old.base.encoding.gamma)*2/(old.interior_centers-1)
    common=dict(centers=old.interior_centers,lam=lam,block_size=old.block_size)
    if type(old) is BoxProfileOperator:
        new=BoxProfileOperator(old.bounds,degree,directions=old.directions,
            angular_weights=old.angular_weights,direction_block_size=old.direction_block_size,**common)
        oldkeys=list(map(tuple,old.multiindices));newkeys=list(map(tuple,new.multiindices))
    else:
        common['direction_count']=old.direction_count
        new=(AffineDiskProfileOperator(old.bounds,degree,**common) if type(old) is AffineDiskProfileOperator
             else DiskProfileOperator(degree,midpoint=old.midpoint,scale=old.scale,**common))
        oldkeys=list(zip(old.degrees,old.harmonics,old.is_sine))
        newkeys=list(zip(new.degrees,new.harmonics,new.is_sine))
    if not np.array_equal(old.midpoint,new.midpoint) or not np.array_equal(old.scale,new.scale):
        raise RuntimeError('Native truncation changed the affine coordinate chart')
    lookup={key:index for index,key in enumerate(oldkeys)}
    coefficients=np.array(previous.coefficients[[lookup[key] for key in newkeys]],copy=True)
    # No old trial-parameter closures or validation flags survive this copy.
    blocks=frozen_physics.make_blocks(max(len(b.points) for b in previous.problem.blocks),17713)
    problem=replace(previous.problem,features=new,blocks=blocks,parameter_initial=(),parameter_bounds=None)
    info=dict(kind='algebraic_native_truncation',source_degree=int(old.degree),degree=degree,
        source_modes=int(old.size),retained_modes=int(new.size),chart_preserved=True,
        source_angular_rule_preserved=True,target_or_sensor_fit=False,validated=False,
        policy='greatest fixed-ladder degree strictly below the current validated degree')
    guess=ResidualSolution(problem,coefficients,np.empty(0),
        dict(status='uncertified_native_initial_guess',validated=False,execution=problem.execution,
             batch_size=problem.batch_size,warm_start_provenance=info))
    return guess,info


def _combine_product_counts(records):
    combined={}
    for record in records:
        for key,value in record.items():
            combined[key]=(max(combined.get(key,0),value) if key.endswith('_max')
                           else combined.get(key,0)+value)
    return combined


def _native_forward_work(solution):
    """Keep every attempted stage's work, not only the returned best field."""
    stages=[dict(degree=row.get('degree'),status=row.get('solver_status'),
                 product_counts=row.get('actual_solver_array_metrics',{}).get('product_counts') or {},
                 **row.get('solver_work',{})) for row in solution.history]
    keys=('lsmr_iterations','iterations','residual_evaluations','preconditioner_setup_seconds')
    totals={key:sum(row.get(key) or 0 for row in stages) for key in keys}
    return dict(stages=stages,totals=totals,
        product_counts=_combine_product_counts(row['product_counts'] for row in stages),
        product_counts_complete=bool(stages) and all(row['product_counts'] for row in stages),
        complete=bool(stages) and all(all(row.get(key) is not None for key in keys) for row in stages),
        scope='All attempted native degree stages in this forward; totals include unresolved stages and are not an inference-cost complexity guarantee.')


def solve_route2_inverse(physics,observations,*,inner_options=None,max_iterations=12,
                        max_forward_solves=100,max_seconds=300.,difference_step=1e-3,
                        parameter_scale=None,gradient_tolerance=1e-7,data_tolerance=1e-8,
                        step_tolerance=1e-8,rank_tolerance=1e-9,trust_radius=1.,
                        warm_starts=False,truncate_warm_starts=False,forward_callback=None,
                        check_difference_step=False,difference_check_tolerance=.1):
    """Minimize measurement error subject to validated native PDE feasibility.

    ``physics`` contains ONLY PDE, IC, BC, gauge and other physical conditions;
    ``observations`` are separate ``SensorObservations``. Observation scales may
    be known noise standard deviations (use a positive chosen scale for exact
    data). The outer objective is half the globally averaged, noise-normalized
    squared sensor error, invariant to splitting observations into sets.

    All inner settings, including the physical tolerance/resource budget, are
    exposed in ``inner_options``. Central finite differences use resolved
    native forwards; at a parameter bound a one-sided difference is used.
    Numerical response rank is not a sensitivity-resolution or identifiability
    certificate: a nonzero roundoff response has rank one in a scalar problem.
    ``check_difference_step=True`` additionally compares h and h/2 responses;
    disagreement stops with ``unresolved_finite_difference``. Agreement is a
    diagnostic, not a proof of observable parameters. An unresolved
    inner solve is reported explicitly and never assigned a data score.

    Optional warm starts copy only previous native neural coordinates: every
    inner state has zero free parameters. They are used only if a richer degree
    remains for the native driver's resolution check, otherwise the solve is
    cold. This option never fits observed/target values into an initial state.
    ``truncate_warm_starts=True`` instead retains native modes through the
    greatest fixed-ladder degree strictly below the previous validated degree.
    This allows coarser-to-richer validation afresh without artificially raising
    degree at every parameter trial. Box, disk and affine-disk streamed modes
    are supported; other coordinate families start cold. The truncated field
    is an initial guess, never a validated candidate or a sensor prediction.
    """
    observations=tuple(observations)
    if not observations or any(not isinstance(o,SensorObservations) for o in observations):
        raise ValueError('Provide at least one SensorObservations set')
    eta=np.asarray(physics.parameter_initial,float).reshape(-1).copy();k=len(eta)
    if not 1<=k<=16 or not np.all(np.isfinite(eta)):
        raise ValueError('Reduced inverse currently supports one to sixteen finite scalar parameters')
    for obs in observations:
        if obs.points.shape[1]!=physics.domain.dimension or np.any(obs.fields>=physics.fields):
            raise ValueError('Observation dimensions/fields differ from the physics declaration')
    bounds=physics.parameter_bounds
    lower=np.full(k,-np.inf) if bounds is None else np.broadcast_to(np.asarray(bounds[0],float),(k,)).copy()
    upper=np.full(k,np.inf) if bounds is None else np.broadcast_to(np.asarray(bounds[1],float),(k,)).copy()
    if np.any(lower>=upper) or np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(eta<lower) or np.any(eta>upper):
        raise ValueError('Initial parameters must satisfy ordered bounds')
    scale=np.maximum(abs(eta),1.) if parameter_scale is None else np.broadcast_to(np.asarray(parameter_scale,float),(k,)).copy()
    if not np.all(np.isfinite(scale)) or np.any(scale<=0):raise ValueError('Parameter scales must be finite and positive')
    if any(not np.isfinite(v) or v<=0 for v in (max_seconds,difference_step,gradient_tolerance,data_tolerance,step_tolerance,rank_tolerance,trust_radius)):
        raise ValueError('Positive finite tolerances and budgets required')
    if max_iterations<1 or int(max_iterations)!=max_iterations or max_forward_solves<1 or int(max_forward_solves)!=max_forward_solves:
        raise ValueError('Positive integer iteration/forward budgets required')
    options=dict(inner_options or {})
    if 'initial_solution' in options:raise ValueError('Supply no external initial_solution; inverse warm starts are managed internally')
    if not isinstance(check_difference_step,(bool,np.bool_)) or not np.isfinite(difference_check_tolerance) or difference_check_tolerance<=0:
        raise ValueError('Difference check must be boolean with positive finite tolerance')
    if forward_callback is not None and not callable(forward_callback):raise ValueError('forward_callback must be callable')
    if not isinstance(truncate_warm_starts,(bool,np.bool_)):raise ValueError('truncate_warm_starts must be boolean')
    began=time.perf_counter();runs=[];history=[];current=None;failed=None;status='iteration_limit'

    def evaluate(candidate,role,warm=None):
        nonlocal failed
        remaining=max_seconds-(time.perf_counter()-began)
        if len(runs)>=max_forward_solves or remaining<=0:return None,None,'forward_budget'
        call=dict(options);call['max_seconds']=min(float(call.get('max_seconds',120.)),remaining)
        frozen=freeze_physics_parameters(physics,candidate)
        provenance=dict(kind='cold',reason='no previous native state requested')
        if (warm_starts or truncate_warm_starts) and warm is not None:
            degree=warm.problem.features.degree
            ladder=tuple(call.get('degrees',(4,8,12,16,24,32,40,48)))
            if truncate_warm_starts:
                guess,provenance=_truncated_native_guess(warm,frozen,ladder)
                if guess is not None:
                    call['degrees']=tuple(p for p in ladder if p>=guess.problem.features.degree)
                    call['initial_solution']=guess
            elif ladder and max(ladder)>degree:
                higher=tuple(p for p in ladder if p>=degree)
                call['degrees']=(degree,)+tuple(p for p in higher if p>degree)
                call['initial_solution']=warm
                provenance=dict(kind='full_native_coordinates',source_degree=int(degree),validated=False)
            if 'initial_solution' in call:
                provenance['source_parameters']=eta.tolist()
                provenance['candidate_parameters']=candidate.tolist()
        start=time.perf_counter()
        try:
            solved=solve_route2_native(frozen,**call)
        except RuntimeError as error:
            row=dict(parameters=candidate.tolist(),role=role,status='inner_exception',error=str(error),seconds=time.perf_counter()-start,
                     warm_start_provenance=provenance)
            runs.append(row)
            if forward_callback is not None:forward_callback(None,candidate.copy(),row.copy())
            return None,None,'unresolved_forward'
        valid=solved.status=='sampled_residual_checks_passed' and bool(solved.metrics.get('validated',False))
        row=dict(parameters=candidate.tolist(),role=role,status=solved.status,validated=valid,
                 degree=solved.problem.features.degree,seconds=time.perf_counter()-start,
                 inner_parameter_count=len(solved.parameters),warm_start_used='initial_solution' in call,
                 warm_start_provenance=provenance,
                 native_work=_native_forward_work(solved),
                 validation=solved.history[-1]['validation'] if solved.history else [],
                 ordinary_validation=solved.history[-1]['ordinary_export_validation'] if solved.history else [])
        if len(solved.parameters)!=0:raise RuntimeError('Frozen native forward unexpectedly retained free parameters')
        if not valid:
            failed=solved;runs.append(row)
            if forward_callback is not None:forward_callback(solved,candidate.copy(),row.copy())
            return None,None,'unresolved_forward'
        residual=_sensor_residual(solved,observations)
        if not np.all(np.isfinite(residual)):raise FloatingPointError('Nonfinite ordinary-network sensor prediction')
        row['sensor_objective']=.5*float(residual@residual);runs.append(row)
        if forward_callback is not None:forward_callback(solved,candidate.copy(),row.copy())
        return solved,residual,None

    current,residual,error=evaluate(eta,'initial')
    if error:status='unresolved_initial_forward' if error=='unresolved_forward' else error
    else:
        for iteration in range(int(max_iterations)):
            objective=.5*float(residual@residual)
            row=dict(iteration=iteration,parameters=eta.tolist(),sensor_objective=objective,
                     # _sensor_residual already divides by sqrt(data count).
                     normalized_sensor_rms=float(np.linalg.norm(residual)))
            history.append(row)
            def response_at(step_multiplier):
                response=np.empty((len(residual),k))
                for j in range(k):
                    h=difference_step*scale[j]*step_multiplier
                    hp=min(h,upper[j]-eta[j]);hm=min(h,eta[j]-lower[j])
                    plus=minus=residual
                    if hp>np.finfo(float).eps*max(1.,abs(eta[j])):
                        parameter=eta.copy();parameter[j]+=hp
                        unused,plus,error=evaluate(parameter,f'sensitivity_plus_{j}_step{step_multiplier:g}',current)
                        del unused
                        if error:return None,error
                    else:hp=0.
                    if hm>np.finfo(float).eps*max(1.,abs(eta[j])):
                        parameter=eta.copy();parameter[j]-=hm
                        unused,minus,error=evaluate(parameter,f'sensitivity_minus_{j}_step{step_multiplier:g}',current)
                        del unused
                        if error:return None,error
                    else:hm=0.
                    if hp+hm==0:raise ValueError('Parameter bounds leave no finite-difference interval')
                    if hp and hm:
                        derivative=(hm*hm*(plus-residual)+hp*hp*(residual-minus))/(hp*hm*(hp+hm))
                    else:derivative=(plus-minus)/(hp+hm)
                    response[:,j]=derivative*scale[j]
                return response,None
            response,error=response_at(1.)
            if error:
                status='unresolved_sensitivity_forward' if error=='unresolved_forward' else error;break
            if check_difference_step:
                finer,error=response_at(.5)
                if error:
                    status='unresolved_sensitivity_forward' if error=='unresolved_forward' else error;break
                changes=np.linalg.norm(finer-response,axis=0)/np.maximum(
                    np.maximum(np.linalg.norm(finer,axis=0),np.linalg.norm(response,axis=0)),np.finfo(float).tiny)
                row['halved_step_response_relative_changes']=changes.tolist()
                if np.max(changes)>difference_check_tolerance:
                    status='unresolved_finite_difference';break
                response=finer
            singular=np.linalg.svd(response,compute_uv=False)
            rank=int(np.count_nonzero(singular>rank_tolerance*singular[0])) if singular[0]>0 else 0
            row['response_singular_values']=singular.tolist();row['local_response_rank']=rank
            if rank<k:status='rank_deficient_response';break
            gradient=response.T@residual;projected=gradient.copy()
            projected[(eta<=lower+1e-14)&(gradient>0)]=0.
            projected[(eta>=upper-1e-14)&(gradient<0)]=0.
            gradient_norm=float(np.linalg.norm(projected,np.inf))
            threshold=gradient_tolerance*max(1.,float(singular[0])*float(np.linalg.norm(residual)))
            row['scaled_projected_gradient_inf']=gradient_norm;row['gradient_threshold']=threshold
            if gradient_norm<=threshold:status='data_stationary';break
            if np.linalg.norm(residual)<=data_tolerance:status='data_tolerance';break
            step=np.linalg.lstsq(response,-residual,rcond=rank_tolerance)[0]
            norm=np.linalg.norm(step)
            if norm>trust_radius:step*=trust_radius/norm
            initial_delta=(np.clip(eta+scale*step,lower,upper)-eta)/scale
            if float(gradient@initial_delta)>=0:
                step=-projected
                step*=min(1.,trust_radius/max(np.linalg.norm(step),np.finfo(float).tiny))
                row['direction_fallback']='projected_gradient_for_bound_feasible_descent'
            accepted=False
            for backtrack in range(9):
                alpha=2.**(-backtrack);trial=np.clip(eta+alpha*scale*step,lower,upper)
                delta=(trial-eta)/scale
                if np.linalg.norm(delta)<=step_tolerance:
                    status='outer_step_limit';break
                candidate,candidate_residual,error=evaluate(trial,'line_search',current)
                if error:
                    status='unresolved_trial_forward' if error=='unresolved_forward' else error;break
                candidate_objective=.5*float(candidate_residual@candidate_residual)
                if float(gradient@delta)<0 and candidate_objective<=objective+1e-4*float(gradient@delta):
                    eta,current,residual=trial,candidate,candidate_residual
                    row['accepted_step_scale']=alpha;accepted=True;break
                del candidate
            if not accepted:
                if status=='iteration_limit':status='outer_line_search_failed'
                break
    return NativeInverseResult(current,eta,status,history,dict(
        seconds=time.perf_counter()-began,forward_runs=runs,forward_solves=len(runs),
        native_work_complete=bool(runs) and all(row.get('native_work',{}).get('complete',False) for row in runs),
        native_work_totals={key:sum(row.get('native_work',{}).get('totals',{}).get(key,0) for row in runs)
                           for key in ('lsmr_iterations','iterations','residual_evaluations','preconditioner_setup_seconds')},
        native_product_counts=_combine_product_counts(row.get('native_work',{}).get('product_counts',{}) for row in runs),
        native_product_counts_complete=bool(runs) and all(row.get('native_work',{}).get('product_counts_complete',False) for row in runs),
        validated_forward=current is not None,outer_converged=status in ('data_stationary','data_tolerance'),
        final_sensor_objective=None if current is None else .5*float(residual@residual),
        source_policy='Only physics/IC/BC/gauges in native inner solve; observations only in outer objective; no external PDE solve, target readout fit, or oracle.',
        sensitivity='Finite differences of fully validated ordinary neural forwards; not differentiation of an unresolved LS state.',
        difference_step_check=bool(check_difference_step),
        numerical_rank_scope='Rank of computed finite-difference responses only; full rank, especially scalar rank one, does not establish resolved sensitivity or identifiability.',
        uncertainty_scope='Sampled local response rank and projected-gradient checks; no global identifiability, continuum error or posterior guarantee.',
        memory_scope='Retains current/one trial neural states, bounded evaluation panels, and sensor-by-parameter response; parameter dimension capped at16.'),failed)
