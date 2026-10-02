"""Independent spatial residual and time-refinement checks for periodic evolution.

These diagnostics strengthen the public stopping rule. They are sampled tests,
not guaranteed continuum error bounds. In particular finite resolution cannot
certify arbitrary unseen frequencies or physical singularities.
"""
import time
import numpy as np
from .base import Resolution, solve_periodic


def _residual_check(solution, x, times):
    accepted=solution.result.t
    if len(accepted)>16:
        accepted=accepted[np.linspace(0,len(accepted)-1,16,dtype=int)]
    times=np.unique(np.r_[times,accepted])
    defects=[]; scales=[]
    for t in times:
        p=solution.problem
        rhs=p.rhs(t,x,solution.evaluate(x,t),solution.evaluate(x,t,1),
                  solution.evaluate(x,t,2),p.parameters)
        defects.append(solution.residual(x,t))
        scales.append(rhs)
    absolute=float(np.sqrt(np.mean(np.asarray(defects)**2)))
    scale=float(np.sqrt(np.mean(np.asarray(scales)**2)))
    per_time_rms=np.sqrt(np.mean(np.asarray(defects)**2,axis=1))
    per_time_scale=np.sqrt(np.mean(np.asarray(scales)**2,axis=1))
    return dict(rms=absolute,scale=scale,relative=absolute/max(scale,1e-12),
                evaluation_times=times.tolist(),per_time_rms=per_time_rms.tolist(),
                per_time_scale=per_time_scale.tolist())


def _residual_pass(check,absolute_tolerance,relative_tolerance):
    return bool(np.all(np.asarray(check['per_time_rms']) <=
                       absolute_tolerance+relative_tolerance*np.asarray(check['per_time_scale'])))


def solve_verified(problem,t_end,tolerance=1e-6,time_tolerance=1e-7,
                   residual_tolerance=1e-5,residual_absolute_tolerance=1e-10,
                   starting_modes=8,maximum_modes=128,max_time_refinements=3,
                   **options):
    """Refine space until field agreement AND independent PDE checks pass.

    Then repeat the complete accepted spatial solve at tighter integration
    tolerances and smaller maximum steps. All work/validation is timed.
    """
    if min(tolerance,time_tolerance,residual_tolerance,residual_absolute_tolerance)<=0:
        raise ValueError('Check tolerances must be positive')
    if starting_modes<2 or maximum_modes<starting_modes or max_time_refinements<1:
        raise ValueError('Invalid refinement budget')
    started=time.perf_counter()
    rng=np.random.default_rng(95032)
    lo,hi=problem.domain
    x=lo+(hi-lo)*rng.uniform(size=317)
    # Equally spaced checks can all hit zeros of an ordinary periodic force.
    # Stratified random times avoid that systematic alias; accepted-step
    # samples are included separately in each physical residual check.
    temporal_seed=95033
    temporal_rng=np.random.default_rng(temporal_seed)
    times=np.sort(np.r_[0.,t_end,t_end*(np.arange(20)+temporal_rng.uniform(size=20))/20])
    initial_tolerance=options.pop('initial_tolerance',tolerance)
    history=[];previous=None;latest=None;accepted=None;k=starting_modes
    while k<=maximum_modes:
        try:
            current=solve_periodic(problem,t_end,Resolution(modes=k),
                                   initial_tolerance=initial_tolerance,**options)
        except ValueError as exc:
            if 'underresolved' not in str(exc) and 'validity threshold' not in str(exc):raise
            history.append(dict(modes=k,status='initial_underresolved',detail=str(exc)))
            k*=2;continue
        latest=current
        row=dict(modes=k,status=current.status)
        if current.status=='reached_t_end':
            values=current.evaluate(x,times)
            check=_residual_check(current,x,times)
            row['independent_physical_residual']=check
            row['residual_pass']=_residual_pass(check,residual_absolute_tolerance,residual_tolerance)
            if previous is not None:
                difference=float(np.linalg.norm(values-previous)/max(np.linalg.norm(values),1e-12))
                row['field_refinement_difference']=difference
                if row['residual_pass'] and difference<=tolerance:
                    accepted=current;history.append(row);break
            previous=values
        elif current.status in ['amplitude_limit','integrator_failure']:
            history.append(row);break
        history.append(row);k*=2
    if latest is None:raise RuntimeError('No initial representation passed within the spatial budget')
    time_history=[]
    if accepted is None:
        if latest.status not in ['amplitude_limit','integrator_failure']:
            latest.status='spatial_resolution_limit'
    else:
        latest=accepted
        time_options=dict(options)
        for _ in range(max_time_refinements):
            time_options['rtol']=max(2.3e-14,time_options.get('rtol',1e-8)*.1)
            time_options['atol']=max(1e-15,time_options.get('atol',1e-10)*.1)
            time_options['max_step']=min(time_options.get('max_step',np.inf)*.5,t_end/20)
            fine=solve_periodic(problem,t_end,Resolution(modes=k),initial_tolerance=initial_tolerance,**time_options)
            row=dict(rtol=time_options['rtol'],atol=time_options['atol'],max_step=time_options['max_step'],status=fine.status)
            if fine.status!='reached_t_end':
                latest=fine;time_history.append(row);break
            a=fine.evaluate(x,times);b=latest.evaluate(x,times)
            difference=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))
            row['field_difference']=difference
            latest=fine;time_history.append(row)
            if difference<=time_tolerance:
                check=_residual_check(fine,x,times)
                row['independent_physical_residual']=check
                latest.status='checks_passed' if _residual_pass(check,residual_absolute_tolerance,residual_tolerance) else 'spatial_resolution_limit'
                break
        else:latest.status='time_resolution_limit'
    latest.metrics.update(status=latest.status,verification=dict(
        spatial_history=history,time_history=time_history,
        field_tolerance=tolerance,time_tolerance=time_tolerance,
        relative_residual_tolerance=residual_tolerance,
        absolute_residual_tolerance=residual_absolute_tolerance,
        temporal_validation_seed=temporal_seed,comparison_times=times.tolist(),
        complete_call_seconds=time.perf_counter()-started,
        scope='Independent sampled spatial checks and integration refinement; no continuum certificate'))
    return latest
