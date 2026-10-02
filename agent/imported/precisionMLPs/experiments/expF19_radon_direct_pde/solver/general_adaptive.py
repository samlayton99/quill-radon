"""Equation-independent degree continuation with independent residual checks.

The declaration factory receives only a sample count and seed, and supplies
the same mathematical equation/constraints on fresh points. No reference
solution is queried for stopping or continuation. This is empirical validation,
not a certified PDE error estimator.
"""
from dataclasses import dataclass, field
import time
import numpy as np
import torch
from .general_features import ConstructedFeatures
from .general_residual import ResidualProblem, solve_residual


@dataclass
class ResidualDeclaration:
    domain: object
    make_blocks: object
    fields: int = 1
    parameter_initial: object = ()
    parameter_bounds: object = None


@dataclass
class AdaptiveResidualSolution:
    solution: object
    status: str
    history: list=field(default_factory=list)
    metrics: dict=field(default_factory=dict)

    @property
    def coefficients(self):return self.solution.coefficients
    @property
    def parameters(self):return self.solution.parameters
    @property
    def problem(self):return self.solution.problem
    def evaluate(self,points,derivative=None):
        return self.solution.evaluate(points,derivative)


def check_residuals(solution,blocks):
    results=[]
    original={b.name:b for b in solution.problem.blocks}
    for block in blocks:
        x=torch.tensor(block.points,dtype=torch.float64)
        jets={d:torch.tensor(solution.evaluate(block.points,d),dtype=torch.float64)
              for d in block.derivatives}
        parameters=torch.tensor(np.broadcast_to(solution.parameters,
            (len(x),len(solution.parameters))).copy(),dtype=torch.float64)
        raw=block.function(x,jets,parameters).detach().numpy()
        if raw.ndim==1:raw=raw[:,None]
        if block.aggregation is not None:raw=block.aggregation@raw
        relation=getattr(block,'relation','eq')
        if relation=='le':raw=np.maximum(raw,0)
        if relation=='ge':raw=np.minimum(raw,0)
        if not np.all(np.isfinite(raw)):
            maximum=np.inf
        else:maximum=float(np.max(np.sqrt(np.mean((raw/block.scale)**2,axis=0))))
        training=original.get(block.name)
        reused=training is not None and np.array_equal(training.points,block.points)
        results.append(dict(name=block.name,maximum_scaled_rms=maximum,
                            reuses_training_locations=bool(reused),points=len(x)))
    return results


def solve_declared(declaration, *, tolerance=1e-6,
                   degrees=tuple(range(4,65,4)), backend='quill',
                   oversampling=6,check_points=1009,max_iterations=40,
                   max_seconds=180,maximum_basis_cache_mb=512,
                   solver_options=None,feature_options=None):
    if tolerance<=0 or not np.isfinite(tolerance):raise ValueError('Positive tolerance required')
    if not degrees or any(int(p)!=p or p<0 for p in degrees) or any(b<=a for a,b in zip(degrees,degrees[1:])):
        raise ValueError('degrees must be a strictly increasing sequence of nonnegative integers')
    if oversampling<1 or check_points<1 or max_seconds<=0 or maximum_basis_cache_mb<=0:
        raise ValueError('Sampling and resource budgets must be positive')
    started=time.perf_counter();trace=[];previous=None;last=None
    feature_settings=dict(feature_options or {})
    if any(key in feature_settings for key in ('bounds','degree','backend','max_derivative')):
        raise ValueError('feature_options may set encoding controls, not domain/degree/backend/order')
    if tolerance<1e-9 and backend=='quill':
        feature_settings.setdefault('evaluation','anchored')
        # A worst-mode calibration gate is not a field error certificate.
        # Actual nonlinear residuals and field agreement remain mandatory.
        feature_settings.setdefault('encoding_tolerance',max(128*np.finfo(float).eps,tolerance*.1))
    previous_score=np.inf;best_solution=None;best_score=np.inf
    validation=declaration.make_blocks(check_points,92173)
    max_order=max(sum(d) for b in validation for d in b.derivatives)
    probe=declaration.domain.interior(check_points,91381)
    parameters=np.asarray(declaration.parameter_initial,dtype=float)
    status='resolution_limit'
    pending=list(degrees);attempted=[]
    while pending:
        degree=pending.pop(0);attempted.append(degree)
        remaining=max_seconds-(time.perf_counter()-started)
        if remaining<=0:status='time_budget';break
        try:
            features=ConstructedFeatures(declaration.domain.bounds,degree,backend=backend,
                                         max_derivative=max_order,**feature_settings)
        except ValueError as error:
            if last is None or not str(error).startswith('QUILL basis derivative encoding'):
                raise
            status='construction_limit';break
        points=max(256,int(oversampling*features.size))
        blocks=declaration.make_blocks(points,113+degree*1009)
        estimate=sum(len(b.points)*len(b.derivatives)*features.size*8 for b in blocks)
        if estimate>maximum_basis_cache_mb*1024**2:
            lower=trace[-1]['degree'] if trace else 0
            if degree-lower>1:
                # A coarse degree ladder can skip a useful affordable space.
                # Bisect the proposed jump before declaring memory exhaustion.
                midpoint=(degree+lower)//2
                pending.insert(0,midpoint)
                continue
            status='memory_budget';break
        initial=None
        if previous is not None:
            old=previous.problem.features
            mapping={tuple(a):i for i,a in enumerate(features.multiindices)}
            initial=np.zeros((features.size,declaration.fields))
            for i,a in enumerate(old.multiindices):initial[mapping[tuple(a)]]=previous.coefficients[i]
            parameters=previous.parameters.copy()
        problem=ResidualProblem(features,blocks,fields=declaration.fields,
            parameter_initial=parameters,parameter_bounds=declaration.parameter_bounds)
        # Near roundoff, an extra decade of requested inner accuracy can fit
        # arithmetic noise rather than improve the field. The public target
        # remains unchanged and is checked independently afterward.
        inner_tolerance=tolerance if tolerance<1e-10 else tolerance*.1
        options=dict(max_iterations=max_iterations,tolerance=inner_tolerance,
                     gradient_tolerance=min(1e-12,tolerance*1e-5),
                     max_seconds=max(.01,remaining),preconditioner='auto',
                     high_accuracy=tolerance<1e-9)
        options.update(solver_options or {})
        candidate=solve_residual(problem,initial_coefficients=initial,**options)
        checks=check_residuals(candidate,validation)
        worst=max(c['maximum_scaled_rms'] for c in checks)
        agreement=None
        if previous is not None:
            values=candidate.evaluate(probe);old_values=previous.evaluate(probe)
            rms=np.sqrt(np.mean(values*values))
            agreement=float(np.sqrt(np.mean((values-old_values)**2))/max(1.,rms))
        row=dict(degree=int(degree),coefficients=features.size*declaration.fields,
                 tanh_count=features.metrics['tanh_count'],interior_samples=points,
                 estimated_basis_cache_bytes=estimate,validation=checks,
                 maximum_validation_rms=worst,successive_field_difference=agreement,
                 solver=candidate.metrics,elapsed_seconds=time.perf_counter()-started)
        trace.append(row);last=candidate
        if best_solution is None or worst<best_score:best_solution=candidate;best_score=worst
        if worst<=tolerance and agreement is not None and agreement<=tolerance:
            status='sampled_checks_passed';break
        # Do not discard an earlier better solution to warm-start a later
        # refinement from a numerically degraded fit.
        if previous is None or worst<=previous_score:
            previous=candidate;previous_score=worst
        if candidate.status=='budget_exhausted':status='time_budget';break
    if last is None:
        raise RuntimeError('Budget insufficient for the first declared resolution')
    best=min(range(len(trace)),key=lambda i:trace[i]['maximum_validation_rms'])
    selected=last if status=='sampled_checks_passed' else best_solution
    return AdaptiveResidualSolution(selected,status,trace,
        dict(seconds=time.perf_counter()-started,tolerance=tolerance,
             status=status,validated=status=='sampled_checks_passed',
             stages=len(trace),best_residual_stage=best,
             attempted_degrees=attempted,
             reference_solution_used=False,
             validation_scope='Fresh equation/constraint residuals and successive-field agreement; empirical, not continuum certification',
             equations_are_declarations=True,named_equation_dispatch=False,
             memory_budget_scope='Cached basis jet estimate only; not whole-process RSS'))
