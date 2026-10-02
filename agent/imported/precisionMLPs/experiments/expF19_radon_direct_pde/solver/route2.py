"""Equation-independent route-2 driver: actual flat tanh networks throughout.

Build analytically prescribed readout coordinates, solve declared physics,
refine degree, and check fresh residuals plus ordinary exported Torch AD.
No reference solution callback is accepted. Passing sampled checks is NOT a
continuum error certificate, an identifiability certificate or an UQ claim.
"""
from dataclasses import dataclass,field
from math import comb,ceil,sqrt
import time
import numpy as np
import torch
from scipy import sparse
from .ball_ridge_features import BallRidgeFeatures
from .ridge_pinn_features import RidgePINNFeatures
from .general_residual import ResidualProblem,ResidualSolution,solve_residual,_make_engine,_initial_vector,_NonfiniteResidual
from .general_adaptive import check_residuals as _cached_check_residuals
from route2_disk_profiles import DiskProfileOperator
from route2_affine_disk_profiles import AffineDiskProfileOperator
from route2_box_profiles import BoxProfileOperator
from route2_directional_operator import jet_batch_plan


def footprint(dimension,degree,centers,fields=1,coordinates='box',angular_rule='auto',
              execution='cached',batch_size=128,lam=.2):
    if execution not in ('cached','streamed'):
        raise ValueError('execution must be cached or streamed')
    if coordinates not in ('box','disk','affine_disk') or dimension<2 or (coordinates!='box' and dimension!=2):
        raise ValueError('Box coordinates require dimension>=2; disk and affine_disk require dimension=2')
    if coordinates=='affine_disk' and execution!='streamed':
        raise ValueError('affine_disk coordinates require execution=streamed')
    if angular_rule not in ('auto','tensor','sparse'):
        raise ValueError('angular_rule must be auto, tensor or sparse')
    if angular_rule=='sparse' and (coordinates!='box' or degree>3):
        raise ValueError('Sparse angular construction supports box coordinates and degree <=3')
    tensor=degree+1 if coordinates!='box' else (degree+1)**(dimension-1)
    sparse=(dimension**2 if degree<=2 else
            dimension+2*comb(dimension,2)+4*comb(dimension,3)) if degree<=3 else None
    chosen='tensor'
    if coordinates=='box' and sparse is not None and (angular_rule=='sparse' or (angular_rule=='auto' and sparse<tensor)):
        chosen='sparse'
    directions=sparse if chosen=='sparse' else tensor
    count=comb(degree+dimension,dimension)
    neurons=directions*(centers+2*ceil(sqrt(centers)))
    powers=comb(degree+dimension-1,dimension-1)
    # Same entire-profile contour geometry as quill_boundary.encode. Include
    # its dense quadrature setup and complex profile panels BEFORE allocation.
    halo=ceil(sqrt(centers));h=2/(centers-1);tau=np.pi*h/(2*lam)
    delta=max(8*tau,(halo+.5)*h)
    contour_k=max(1,ceil(delta/(8*tau)-1e-14));poles=(halo+1)//2
    quadrature=max(96,8*poles*contour_k+32)
    per_direction=centers+2*halo
    encoder_arrays=(8*(6*quadrature**2+4*per_direction*(degree+1)+8*poles*(degree+1))+
                    16*(8*(per_direction+quadrature)*(degree+1)+4*poles*quadrature))
    result=dict(dimension=dimension,degree=degree,directions=directions,angular_rule=chosen,
                readout_coordinates=count,neurons=neurons,
                ordinary_network_bytes=8*(neurons*(dimension+1+fields)+fields),
                encoder_quadrature_order=quadrature,encoder_peak_array_estimate_bytes=encoder_arrays,
                profile_map_bytes=8*directions*(degree+1)*count,
                map_workspace_bytes=int(np.dtype(np.longdouble).itemsize)*powers*(count+2*directions) if coordinates=='box' else 0)
    if execution=='streamed':
        per_direction=centers+2*ceil(sqrt(centers))
        result.update(profile_map_bytes=0,map_workspace_bytes=0,
            encoding_bank_bytes=8*per_direction*(degree+1),
            angular_table_bytes=16*directions*(degree+1) if coordinates!='box' else 8*directions*(dimension+1),
            mode_label_bytes=25*count if coordinates!='box' else 8*(dimension+1)*count,
            prepared_state_bytes=8*fields*(neurons+directions),
            scalar_basis_panel_bytes=8*batch_size*per_direction,
            coordinate_vector_bytes=8*fields*count,
            coordinate_workspace_bytes=0 if coordinates!='box' else 8*(degree+1)*count,
            angular_panel_bytes=0 if coordinates!='box' else 8*min(16,directions)*count,
            coordinate_transform_state_bytes=0 if coordinates!='box' else
                8*(5*dimension*count+(degree+1)**2+(degree+1)**3))
    result['execution']=execution
    return result


def export_mlp(features,coefficients):
    """The deployable model contains exactly Linear/Tanh/Linear."""
    if isinstance(features,(DiskProfileOperator,BoxProfileOperator)):
        coefficients=np.asarray(coefficients,float)
        if coefficients.ndim!=2 or coefficients.shape[0]!=features.size:
            raise ValueError('Streamed export expects coefficient-by-field arrays')
        first=first_bias=None
        out=np.empty((coefficients.shape[1],features.tanh_count))
        bias=np.empty(coefficients.shape[1])
        for field in range(coefficients.shape[1]):
            arrays=features.compile(coefficients[:,field])
            if first is None:
                first=np.asarray(arrays['first_weights']);first_bias=np.asarray(arrays['first_bias'])
            out[field]=np.asarray(arrays['output_weights']).reshape(-1)
            bias[field]=np.asarray(arrays['output_bias']).reshape(-1)[0]
    else:
        arrays=features.compile(coefficients)
        first=np.asarray(arrays['first_weights']);first_bias=np.asarray(arrays['first_bias'])
        out=np.asarray(arrays['output_weights']).T;bias=np.asarray(arrays['output_bias'])
    model=torch.nn.Sequential(torch.nn.Linear(first.shape[1],first.shape[0],dtype=torch.float64),
        torch.nn.Tanh(),torch.nn.Linear(first.shape[0],out.shape[0],dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.copy_(torch.as_tensor(first));model[0].bias.copy_(torch.as_tensor(first_bias))
        model[2].weight.copy_(torch.as_tensor(out));model[2].bias.copy_(torch.as_tensor(bias))
    return model.requires_grad_(False)


def _streamed_solver_workspace(cost,blocks,fields,parameters,batch_size,centers,
                               coordinates,solver_options):
    """Named bounded-panel storage, distinguished by the basis actually used.

    Ideal panels contain recurrence tables and selected coordinate jets, not
    directional profiles or neuron readouts. Factors remain O(unknowns*block).
    Callback equation counts are estimates until the callback has been run.
    """
    options=solver_options or {}
    unknowns=cost['readout_coordinates']*fields+parameters
    jets=max(len(block.derivatives) for block in blocks)
    equations=max(max(fields,np.size(block.scale)) for block in blocks)
    axis_orders=max(sum(len({order[axis] for order in block.derivatives})
                        for axis in range(cost['dimension'])) for block in blocks)
    def ideal_workspace(width):
        # Box: retained univariate tables, Jacobi evaluation temporaries and z.
        # Disk: bounded Cartesian/Jacobi selected-column intermediates.
        recurrence=(8*batch_size*((axis_orders+4)*(cost['degree']+1)+cost['dimension'])
                    if coordinates=='box' else 8*batch_size*width*12)
        return recurrence+8*batch_size*width*(jets+2)
    width=min(int(options.get('preconditioner_block_size',64)),max(1,unknowns//2))
    parts=dict(basis=options.get('preconditioner_basis','actual'),block_width=width,
               directional_panel_bytes=0,ideal_coordinate_panel_bytes=0,
               local_physics_panel_bytes=0,factor_bytes=0,factorization_workspace_bytes=0)
    if options.get('preconditioner','auto')!='diagonal':
        if parts['basis']=='ideal':
            parts['ideal_coordinate_panel_bytes']=ideal_workspace(width)
        else:
            # Bounded angular profiles plus one direction's neuron panel.
            parts['directional_panel_bytes']=8*width*(cost['directions']*(cost['degree']+1)+
                                                       centers+2*ceil(sqrt(centers)))
            parts['local_physics_panel_bytes']+=8*batch_size*width*jets
        # Aggregation accumulator, feature-sensitivity product and active panel.
        parts['local_physics_panel_bytes']+=8*batch_size*width*equations*4
        parts['factor_bytes']=8*unknowns*width
        parts['factorization_workspace_bytes']=8*6*width**2
    preconditioner=sum(value for key,value in parts.items() if key.endswith('_bytes'))
    correction=(ideal_workspace(min(64,cost['readout_coordinates']))+
                8*batch_size*fields*(jets+equations)
                if options.get('linearization_basis','actual')=='ideal' else 0)
    tensor_plan=None
    if coordinates=='box' and options.get('linearization_basis','actual')=='ideal':
        from route2_box_tensor import tensor_product_plan
        orders=tuple(dict.fromkeys(order for block in blocks for order in block.derivatives))
        tensor_plan=tensor_product_plan(cost['dimension'],cost['degree'],fields,batch_size,orders)
        if tensor_plan['enabled']:
            correction=tensor_plan['workspace_bytes']+8*tensor_plan['rows']*fields*equations
    return dict(preconditioner_workspace_bytes=preconditioner,
                ideal_correction_workspace_bytes=correction,preconditioner_components=parts,
                ideal_tensor_plan=tensor_plan)


def _memory_bounded_preconditioner(cost,blocks,fields,parameters,batch_size,centers,
                                  coordinates,solver_options,available_bytes):
    """Largest requested local block that fits the named-array remainder."""
    options=dict(solver_options or {})
    def workspace(width):
        return _streamed_solver_workspace(cost,blocks,fields,parameters,batch_size,centers,
            coordinates,dict(options,preconditioner_block_size=width))
    requested=int(options.get('preconditioner_block_size',64))
    current=workspace(requested)
    def total(value):
        return value['preconditioner_workspace_bytes']+value['ideal_correction_workspace_bytes']
    chosen=requested
    if options.get('preconditioner','auto')!='diagonal' and total(current)>available_bytes:
        lo,hi=1,min(requested,max(1,(cost['readout_coordinates']*fields+parameters)//2))
        # If even size one does not fit, return that honest estimate so the
        # driver refuses before construction; never silently exceed the cap.
        while lo<hi:
            middle=(lo+hi+1)//2
            if total(workspace(middle))<=available_bytes:lo=middle
            else:hi=middle-1
        chosen=lo
        options['preconditioner_block_size']=chosen
        current=workspace(chosen)
    current['block_budget_selection']=dict(requested=requested,selected=chosen,
        available_bytes=max(0,int(available_bytes)),changed=chosen!=requested,
        scope='Named arrays only; smaller blocks can require more Krylov iterations')
    return options,current


def check_residuals(solution,blocks):
    """Fresh checks honor streaming, including row-indexed measured data."""
    if solution.metrics.get('execution',solution.problem.execution)!='streamed':
        return _cached_check_residuals(solution,blocks)
    problem=ResidualProblem(solution.problem.features,blocks,fields=solution.problem.fields,
        parameter_initial=solution.parameters,
        parameter_bounds=(solution.problem.parameter_lower,solution.problem.parameter_upper),
        execution='streamed',batch_size=solution.metrics.get('batch_size',solution.problem.batch_size))
    try:
        _,metrics=_make_engine(problem,False).evaluate(_initial_vector(problem,solution.coefficients))
    except _NonfiniteResidual:
        metrics=[dict(name=b.name,maximum_scaled_rms=float('inf')) for b in blocks]
    original={b.name:b for b in solution.problem.blocks}
    return [dict(name=row['name'],maximum_scaled_rms=row['maximum_scaled_rms'],points=len(block.points),
        reuses_training_locations=bool(block.name in original and
                                       np.array_equal(block.points,original[block.name].points)))
        for block,row in zip(blocks,metrics)]


def ordinary_jets(model,points,orders,activation_budget_bytes=32*1024**2):
    """Standard Torch AD, bounded batches; no stable custom derivative rules."""
    points=np.asarray(points,float);width=model[0].out_features
    block=max(1,min(128,activation_budget_bytes//max(1,8*width*6)))
    collected={tuple(d):[] for d in orders}
    for start in range(0,len(points),block):
        for order in collected:
            x=torch.tensor(points[start:start+block],dtype=torch.float64,requires_grad=True)
            y=model(x)
            if sum(order):
                columns=[]
                for column in range(y.shape[1]):
                    value=y[:,column]
                    for axis,n in enumerate(order):
                        for _ in range(n):
                            value=torch.autograd.grad(value.sum(),x,create_graph=True)[0][:,axis]
                    columns.append(value)
                y=torch.stack(columns,dim=1)
            collected[order].append(y.detach())
    return {order:torch.cat(values,dim=0) for order,values in collected.items()}


def raw_residual_audit(solution,blocks):
    model=export_mlp(solution.problem.features,solution.coefficients)
    rows=[]
    for block in blocks:
        # Captured point-specific sources and integral quadrature must retain
        # every original location. Memory is bounded inside ordinary_jets.
        points=block.points
        streamed=solution.metrics.get('execution',solution.problem.execution)=='streamed'
        batch_size=solution.metrics.get('batch_size',solution.problem.batch_size) if streamed else len(points)
        raw=None
        for start in range(0,len(points),batch_size):
            stop=min(start+batch_size,len(points));batch=points[start:stop]
            jets=ordinary_jets(model,batch,block.derivatives)
            parameters=torch.tensor(np.broadcast_to(solution.parameters,(len(batch),len(solution.parameters))).copy(),dtype=torch.float64)
            with torch.no_grad():
                values=(block.batch_function(torch.tensor(batch),jets,parameters,np.arange(start,stop))
                        if streamed and block.batch_function is not None else
                        block.function(torch.tensor(batch),jets,parameters)).detach().numpy()
            if values.ndim==1:values=values[:,None]
            if values.ndim!=2 or len(values)!=len(batch):
                raise ValueError(f'{block.name}: raw audit callback must return one row per point batch')
            if raw is None:raw=np.empty((len(points),values.shape[1]))
            raw[start:stop]=values
        if block.aggregation is not None:raw=block.aggregation@raw
        if block.relation=='le':raw=np.maximum(raw,0)
        if block.relation=='ge':raw=np.minimum(raw,0)
        finite=bool(np.all(np.isfinite(raw)))
        rows.append(dict(name=block.name,points=len(points),finite=finite,
                         maximum_scaled_rms=float(np.max(np.sqrt(np.mean((raw/block.scale)**2,axis=0)))) if finite else float('inf'),
                         maximum_absolute=float(np.max(abs(raw))) if finite else float('inf')))
    return rows


@dataclass
class Route2Solution:
    solution:object
    status:str
    history:list=field(default_factory=list)
    metrics:dict=field(default_factory=dict)
    @property
    def coefficients(self):return self.solution.coefficients
    @property
    def parameters(self):return self.solution.parameters
    @property
    def problem(self):return self.solution.problem
    def evaluate(self,points,derivative=None):return self.solution.evaluate(points,derivative)
    def export(self):return export_mlp(self.problem.features,self.coefficients)


def solve_route2(declaration,*,degrees=(4,6,8,10,12),centers=257,lam=.2,
                 coordinates='box',angular_rule='auto',disk_radius=None,disk_midpoint=None,
                 tolerance=1e-8,block_tolerances=None,oversampling=6,
                 check_points=257,max_seconds=120,maximum_working_array_mb=512,
                 maximum_neurons=300000,max_iterations=20,solver_options=None,
                 execution='cached',batch_size=128,export_check_points=None,initial_solution=None,
                 adaptive_preconditioner_blocks=False):
    """Refine only from actual neural PDE solutions; never fit known solutions.

    `block_tolerances` supplies explicit per-block scaled RMS tolerances, e.g.
    noise-level checks for observations. It does not automatically infer a
    noise model or rebalance the optimization objective. The resource bound
    estimates named arrays, not whole-process RSS or total Python overhead.

    ``execution='streamed'`` uses analytical independent disk or box modes
    and bounded feature/callback batches. Box conversion factors the ball
    moments without a stored global angular map. Point-specific captured data need
    ``ResidualBlock.batch_function`` to select their original row indices.
    Ordinary-network validation uses independently seeded export_check_points
    locations, defaulting to the same count as check_points.
    ``initial_solution`` optionally resumes an existing ResidualSolution or
    Route2Solution using its named coordinates. It never fits sampled values;
    the caller is responsible for the supplied state's provenance. The affine
    coordinate chart must match. Changing bandwidth/center count is allowed,
    but changes the neural encoding and is rechecked against the actual PDE.
    ``adaptive_preconditioner_blocks`` reduces the requested local block cap
    only when needed to fit the named-array budget, before feature allocation.
    ``coordinates='affine_disk'`` (2D, streamed only) maps the declared bounding
    box into the reference disk and uses its stable angular/Jacobi coordinates.
    It avoids box monomial conversion without assuming a circular PDE domain.
    The ordinary network and all boundary conditions remain in physical space.
    """
    if coordinates not in ('box','disk','affine_disk'):
        raise ValueError('coordinates must be box, disk, or affine_disk')
    if execution not in ('cached','streamed'):raise ValueError('execution must be cached or streamed')
    if coordinates=='affine_disk':
        if execution!='streamed':raise ValueError('affine_disk coordinates require execution=streamed')
        if disk_radius is not None or disk_midpoint is not None:
            raise ValueError('affine_disk derives its chart from domain bounds; omit disk_radius and disk_midpoint')
    if not isinstance(adaptive_preconditioner_blocks,(bool,np.bool_)):
        raise ValueError('adaptive_preconditioner_blocks must be boolean')
    if int(batch_size)!=batch_size or batch_size<1:raise ValueError('batch_size must be a positive integer')
    batch_size=int(batch_size)
    if solver_options and ('execution' in solver_options and solver_options['execution']!=execution or
                           'batch_size' in solver_options and solver_options['batch_size']!=batch_size):
        raise ValueError('Set execution and batch_size on solve_route2, not conflicting solver_options')
    d=declaration.domain.dimension
    if d<2 or (coordinates!='box' and d!=2):
        raise ValueError('Box requires d>=2; disk and affine_disk require dimension=2')
    if not degrees or any(int(p)!=p or p<0 for p in degrees) or any(b<=a for a,b in zip(degrees,degrees[1:])):
        raise ValueError('degrees must be strictly increasing nonnegative integers')
    if isinstance(initial_solution,Route2Solution):initial_solution=initial_solution.solution
    if initial_solution is not None:
        if not isinstance(initial_solution,ResidualSolution):
            raise ValueError('initial_solution must be an existing neural ResidualSolution or Route2Solution')
        old=initial_solution.problem.features
        if (old.dimension!=d or initial_solution.problem.fields!=declaration.fields or
                len(initial_solution.parameters)!=len(declaration.parameter_initial)):
            raise ValueError('Initial solution fields, dimension, or parameter count mismatch')
        if degrees[0]<old.degree:
            raise ValueError('First resumed degree must include all initial solution coordinates')
        if (coordinates!='box')!=isinstance(old,(DiskProfileOperator,RidgePINNFeatures)):
            raise ValueError('Initial solution coordinate family mismatch')
        if coordinates!='box' and ((execution=='streamed')!=isinstance(old,DiskProfileOperator)):
            raise ValueError('Disk resume requires the same coordinate basis; cached and streamed bases differ')
        if coordinates=='box':
            if not np.array_equal(getattr(old,'bounds',None),declaration.domain.bounds):
                raise ValueError('Initial solution affine coordinate chart differs: box bounds must match')
        elif coordinates=='affine_disk':
            bounds=np.asarray(declaration.domain.bounds)
            midpoint=bounds.mean(axis=1)
            scale=2/(sqrt(2)*(bounds[:,1]-bounds[:,0]))
            if not np.array_equal(old.midpoint,midpoint) or not np.array_equal(old.scale,scale):
                raise ValueError('Initial solution affine coordinate chart differs: affine disk bounds must match')
        else:
            bounds=np.asarray(declaration.domain.bounds)
            radius=float(disk_radius if disk_radius is not None else np.max(np.diff(bounds,axis=1))/2)
            midpoint=bounds.mean(axis=1) if disk_midpoint is None else np.asarray(disk_midpoint)
            old_scale=old.scale if isinstance(old,DiskProfileOperator) else 1/old.radius
            if (not np.array_equal(old.midpoint,midpoint) or radius<=0 or
                    not np.all(np.asarray(old_scale)==1/radius)):
                raise ValueError('Initial solution affine coordinate chart differs: disk midpoint and radius must match')
    if centers<3 or int(centers)!=centers or not np.isfinite(lam) or lam<=0 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Invalid center, bandwidth or tolerance setting')
    if min(oversampling,check_points,max_seconds,maximum_working_array_mb,maximum_neurons,max_iterations)<=0:
        raise ValueError('Positive sampling and resource budgets required')
    if export_check_points is None:export_check_points=check_points
    if int(export_check_points)!=export_check_points or export_check_points<1:
        raise ValueError('export_check_points must be a positive integer')
    thresholds=dict(block_tolerances or {})
    if any(not np.isfinite(v) or v<=0 for v in thresholds.values()):raise ValueError('Positive block tolerances required')
    validation=declaration.make_blocks(check_points,521917)
    raw_validation=declaration.make_blocks(int(export_check_points),612317)
    probe=declaration.domain.interior(101,99317)
    def score(rows):
        values=np.asarray([r['maximum_scaled_rms']/thresholds.get(r['name'],tolerance) for r in rows])
        return float(np.max(values)) if len(values) and np.all(np.isfinite(values)) else float('inf')
    parameters=np.asarray(declaration.parameter_initial,float);previous=initial_solution;best=None;best_score=np.inf
    history=[];started=time.perf_counter();status='resolution_limit';rejected=[]
    for degree in degrees:
        cost=footprint(d,degree,centers,declaration.fields,coordinates,angular_rule,execution,batch_size,lam=lam)
        minimum_arrays=3*cost['profile_map_bytes']+4*cost['ordinary_network_bytes']+cost['map_workspace_bytes']
        minimum_arrays+=cost['encoder_peak_array_estimate_bytes']
        if execution=='streamed':
            minimum_arrays+=sum(cost[key] for key in ('encoding_bank_bytes','angular_table_bytes',
                'mode_label_bytes','prepared_state_bytes','scalar_basis_panel_bytes','coordinate_vector_bytes'))
            minimum_arrays+=sum(cost[key] for key in ('coordinate_workspace_bytes',
                'angular_panel_bytes','coordinate_transform_state_bytes'))
        if cost['neurons']>maximum_neurons or minimum_arrays>maximum_working_array_mb*1024**2:
            status='resource_limit';rejected.append(dict(**cost,minimum_named_arrays_bytes=minimum_arrays));break
        if time.perf_counter()-started>=max_seconds:status='time_budget';break
        count=max(256,int(oversampling*cost['readout_coordinates']))
        blocks=declaration.make_blocks(count,113+degree*1009)
        cache=(sum(len(b.points)*len(b.derivatives)*cost['readout_coordinates']*8 for b in blocks)
               if execution=='cached' else 0)
        # Retained pointwise sensitivities do not scale with coordinate count.
        # Equation count is unknown until callbacks run: use fields or the
        # declared scale-vector length here and report actual bytes after solve.
        local_estimate=(sum(8*len(b.points)*max(declaration.fields,np.size(b.scale))*
            (len(b.derivatives)*declaration.fields+len(parameters)+4) for b in blocks)
            if execution=='streamed' else 0)
        supplied_arrays={}
        for block in tuple(blocks)+tuple(validation)+tuple(raw_validation):
            supplied_arrays[id(block.points)]=block.points
            if block.aggregation is not None:supplied_arrays[id(block.aggregation)]=block.aggregation
        supplied_arrays[id(probe)]=probe
        supplied_bytes=sum(value.data.nbytes+value.indices.nbytes+value.indptr.nbytes
            if sparse.issparse(value) else value.nbytes for value in supplied_arrays.values())
        workspace=0
        if execution=='streamed':
            jet_orders=max(len({sum(order) for order in block.derivatives}) for block in blocks)
            _,_,workspace=jet_batch_plan(centers+2*ceil(sqrt(centers)),batch_size,
                jet_orders,declaration.fields,cost['directions'] if declaration.fields>1 else 1)
        degree_solver_options=dict(solver_options or {})
        fixed_arrays=cache+minimum_arrays+local_estimate+supplied_bytes+workspace
        if execution=='streamed' and adaptive_preconditioner_blocks:
            degree_solver_options,solver_workspace=_memory_bounded_preconditioner(cost,blocks,
                declaration.fields,len(parameters),batch_size,centers,coordinates,degree_solver_options,
                maximum_working_array_mb*1024**2-fixed_arrays)
        else:
            solver_workspace=(_streamed_solver_workspace(cost,blocks,declaration.fields,len(parameters),
                batch_size,centers,coordinates,degree_solver_options) if execution=='streamed' else
                dict(preconditioner_workspace_bytes=0,ideal_correction_workspace_bytes=0,
                     preconditioner_components={}))
        preconditioner_workspace=solver_workspace['preconditioner_workspace_bytes']
        arrays=(cache+minimum_arrays+local_estimate+supplied_bytes+workspace+preconditioner_workspace+
                solver_workspace['ideal_correction_workspace_bytes'])
        if cost['neurons']>maximum_neurons or arrays>maximum_working_array_mb*1024**2:
            status='resource_limit';rejected.append(dict(**cost,estimated_named_arrays_bytes=arrays,
                                                        solver_workspace=solver_workspace));break
        remaining=max_seconds-(time.perf_counter()-started)
        if remaining<=0:status='time_budget';break
        if coordinates=='affine_disk':
            features=AffineDiskProfileOperator(declaration.domain.bounds,degree,centers,lam,block_size=batch_size)
            keys=list(zip(features.degrees,features.harmonics,features.is_sine))
        elif coordinates=='disk':
            bounds=np.asarray(declaration.domain.bounds);radius=float(disk_radius if disk_radius is not None else np.max(np.diff(bounds,axis=1))/2)
            midpoint=bounds.mean(axis=1) if disk_midpoint is None else disk_midpoint
            if not np.isfinite(radius) or radius<=0:raise ValueError('disk_radius must be finite and positive')
            if execution=='streamed':
                features=DiskProfileOperator(degree,centers=centers,lam=lam,
                    midpoint=midpoint,scale=1/radius,block_size=batch_size)
                keys=list(zip(features.degrees,features.harmonics,features.is_sine))
            else:
                try:
                    features=RidgePINNFeatures(degree,centers=centers,lam=lam,radius=radius,midpoint=midpoint,encoding_tolerance=1e-3)
                except ValueError as error:
                    if best is None or not str(error).startswith('Known-profile derivative encoding failed'):
                        raise
                    status='construction_limit';rejected.append(dict(**cost,reason=str(error)));break
                keys=features.mode_pairs
        elif execution=='streamed':
            angular={}
            if cost['angular_rule']=='sparse':
                from route2_scaling_features import fourth_order_sphere_rule,sixth_order_sphere_rule
                rule=sixth_order_sphere_rule if degree==3 else fourth_order_sphere_rule
                directions,weights=rule(d)
                angular=dict(directions=directions,angular_weights=weights)
            features=BoxProfileOperator(declaration.domain.bounds,degree,centers,lam,
                block_size=batch_size,**angular)
            keys=map(tuple,features.multiindices)
        elif cost['angular_rule']=='sparse':
            from route2_scaling_features import LowDegreeSparseRidgeFeatures
            features=LowDegreeSparseRidgeFeatures(declaration.domain.bounds,degree,centers,lam)
            keys=map(tuple,features.multiindices)
        else:
            features=BallRidgeFeatures(declaration.domain.bounds,degree,centers,lam);keys=map(tuple,features.multiindices)
        keys=list(keys);initial=None
        if previous is not None:
            old=previous.problem.features
            old_keys=(zip(old.degrees,old.harmonics,old.is_sine) if isinstance(old,DiskProfileOperator) else
                      old.mode_pairs if coordinates=='disk' else map(tuple,old.multiindices))
            mapping={tuple(k):i for i,k in enumerate(keys)};initial=np.zeros((features.size,declaration.fields))
            for i,k in enumerate(old_keys):initial[mapping[tuple(k)]]=previous.coefficients[i]
            parameters=previous.parameters.copy()
        problem=ResidualProblem(features,blocks,declaration.fields,parameters,declaration.parameter_bounds,
                                execution=execution,batch_size=batch_size)
        options=dict(tolerance=tolerance,high_accuracy=True,preconditioner='auto',
                     max_iterations=max_iterations,lsmr_max_iterations=1000,linear_refinement_steps=0)
        options.update(degree_solver_options)
        options['max_seconds']=max(.01,max_seconds-(time.perf_counter()-started))
        candidate=solve_residual(problem,initial_coefficients=initial,**options)
        checks=check_residuals(candidate,validation);scaled=score(checks)
        values=candidate.evaluate(probe)
        agreement=None if previous is None else float(np.sqrt(np.mean((values-previous.evaluate(probe))**2))/max(1.,float(np.sqrt(np.mean(values**2)))))
        richer_resolution=previous is not None and degree>previous.problem.features.degree
        raw_checks=None
        if scaled<=1:
            raw_checks=raw_residual_audit(candidate,raw_validation)
        deployable_score=max(scaled,score(raw_checks)) if raw_checks is not None else scaled
        row=dict(degree=degree,features=features.metrics,cost=cost,estimated_jet_cache_bytes=cache,
                 estimated_local_vector_bytes=local_estimate,estimated_named_arrays_bytes=arrays,
                 supplied_points_and_aggregation_bytes=supplied_bytes,estimated_feature_workspace_bytes=workspace,
                 estimated_preconditioner_workspace_bytes=preconditioner_workspace,
                 estimated_solver_workspace=solver_workspace,
                 actual_solver_array_metrics={key:candidate.metrics.get(key) for key in (
                     'basis_cache_bytes','local_derivative_bytes','input_point_bytes',
                     'supplied_aggregation_bytes','residual_vector_bytes','coefficient_vector_bytes','product_counts')},
                 validation=checks,normalized_validation_score=scaled,ordinary_export_validation=raw_checks,
                 normalized_deployable_score=deployable_score,
                 successive_field_difference=agreement,initialization='zero' if initial is None else 'previous actual neural PDE solution',
                 comparison_uses_richer_resolution=richer_resolution,
                 solver_status=candidate.status,
                 solver_work={key:candidate.metrics.get(key) for key in (
                     'lsmr_iterations','iterations','residual_evaluations','seconds',
                     'preconditioner_setup_seconds')},
                 refinement_reason=('sampled_stationarity_above_tolerance' if candidate.status=='linearized_stationary' else
                    'uncertified_approximate_model_stall' if candidate.status=='approximate_model_stalled' else
                    'heldout_residual_above_tolerance' if scaled>1 else
                    'ordinary_export_above_tolerance' if raw_checks is not None and score(raw_checks)>1 else
                    'successive_resolution_check_required'),
                 seconds=time.perf_counter()-started)
        history.append(row)
        if best is None or deployable_score<best_score:best=candidate;best_score=deployable_score
        if scaled<=1 and raw_checks is not None and score(raw_checks)<=1 and richer_resolution and agreement is not None and agreement<=tolerance:
            best=candidate;status='sampled_residual_checks_passed';break
        previous=best
        if candidate.status=='budget_exhausted':
            evaluation_limit=candidate.metrics.get('budget',{}).get('residual_evaluation_limit',np.inf)
            status=('evaluation_budget' if candidate.metrics.get('residual_evaluations',0)>=evaluation_limit
                    and time.perf_counter()-started<max_seconds else 'time_budget')
            break
    if best is None:raise RuntimeError('No affordable route2 resolution; inspect neuron/array budgets')
    return Route2Solution(best,status,history,dict(seconds=time.perf_counter()-started,coordinates=coordinates,
        execution=execution,batch_size=batch_size,
        export_check_points=int(export_check_points),
        initial_solution_supplied=initial_solution is not None,
        initialization_provenance='caller-supplied existing neural state' if initial_solution is not None else 'zero followed by native continuation',
        reference_solution_used=False,architecture='Linear/Tanh/Linear',validated=status=='sampled_residual_checks_passed',
        validation_scope='Fresh sampled equation and condition residuals, ordinary-network AD, and successive-field agreement; no continuum or identifiability guarantee',
        rejected_resolutions=rejected,resource_scope='Named-array estimate, not whole-process memory cap; streamed callback equation count is estimated from fields/scale before evaluation and actual local-array bytes are reported afterward; supplied points/aggregation are counted, callback internal workspace and external captured data are not capped'))


def solve_route2_native(declaration,**kwargs):
    """Opt-in general native preset; no PDE-name or target-specific decisions.

    Actual tanh residuals, acceptance, stopping, validation and export remain
    unchanged. Ideal coordinate jets accelerate only the linear correction and
    bounded block preconditioner. The memory budget can reduce local blocks;
    it cannot remove representation growth with dimension or ensure convergence.
    The degree ladder extends through 48, with resource preflight before each
    construction. Callers set their required tolerance and resource budgets;
    the time budget is soft around completed operator/linear-solve work.
    Every option can be overridden through the ordinary driver arguments.
    """
    options=dict(preconditioner='block',preconditioner_basis='ideal',
        linearization_basis='ideal',preconditioner_refresh=1,
        preconditioner_field_parity='auto',preconditioner_block_size=1024,
        inexact_newton=True,linear_refinement_steps=0)
    options.update(kwargs.pop('solver_options',None) or {})
    kwargs.setdefault('execution','streamed')
    kwargs.setdefault('degrees',(4,8,12,16,24,32,40,48))
    kwargs.setdefault('adaptive_preconditioner_blocks',True)
    kwargs.setdefault('maximum_neurons',float('inf'))
    return solve_route2(declaration,solver_options=options,**kwargs)
