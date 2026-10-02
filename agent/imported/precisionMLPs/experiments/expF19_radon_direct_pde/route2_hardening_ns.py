"""Native streamed 3D Navier--Stokes controls and harder manufactured tests.

Only prescribed body force, velocity traces, and a pressure gauge enter the
residual declaration. No interior truth observations, external PDE trajectory,
or fitted/analytical reference readout initializes the solver. Validation
exports Linear/Tanh/Linear and uses ordinary Torch differentiation.
An optional ideal-coordinate Jacobian proposes corrections to the actual
neural state; the neural residual still decides every accepted step.

The quadratic control is intentionally easy. The trigonometric family has
three interacting components and nonconstant pressure. The bubble family is
a curl with identically zero wall velocity, making interior forcing essential.
The optional transient version adds time as the fourth neural input.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc

# Initialize Torch's OpenMP runtime before SciPy on this macOS environment.
import torch
import numpy as np
from scipy.special import spherical_in, spherical_jn
import sympy as sp

from solver.general_adaptive import ResidualDeclaration
from solver.general_domains import ResidualDomain
from solver.general_residual import ResidualBlock, ResidualProblem, ResidualSolution
from solver.route2 import ordinary_jets, solve_route2

OUT = Path(__file__).resolve().parents[2] / 'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/ns'
FAMILIES = ('quadratic', 'trigonometric', 'bubble')


def known_trigonometric_coefficients(multiindices, frequency=.7):
    """Representation diagnostic ONLY: analytical Legendre-series truncation.

    exp(i*k*x)=sum_n (2*n+1)*i**n*j_n(k)*P_n(x). This uses the known
    manufactured target and must never initialize a physics solve. It is not
    invoked by make_declaration or run_case.
    """
    indices = np.asarray(multiindices, int)
    if indices.ndim != 2 or indices.shape[1] not in (3, 4) or np.any(indices < 0):
        raise ValueError('Expected nonnegative three- or four-dimensional multiindices')
    def plane(k, n):
        sign = (-1.)**n if k < 0 else np.ones_like(n)
        return (2*n+1)*(1j**n)*sign*spherical_jn(n, abs(k))
    output = np.zeros((len(indices), 4))
    settings = ((0, 1, 2, .3, -.2), (1, 2, 0, -.1, .4), (2, 0, 1, .2, -.3))
    for component, sine_axis, cosine_axis, sine_phase, cosine_phase in settings:
        active = indices[:, component] == 0
        sine = np.imag(np.exp(1j*sine_phase)*plane(frequency, indices[:, sine_axis]))
        cosine = np.real(np.exp(1j*cosine_phase)*plane(frequency, indices[:, cosine_axis]))
        output[:, component] = .2*sine*cosine*active
    output[:, 3] = .1*np.imag(plane(.4, indices[:, 0])*plane(.5, indices[:, 1])*plane(-.3, indices[:, 2]))
    if indices.shape[1] == 4:
        # t=(s+1)/8 on [0,.25]; exp(-a*t) has Legendre coefficients
        # exp(-a/8)*(2n+1)*(-1)^n*i_n(a/8), where i_n is spherical In.
        n = indices[:, 3]
        for field, decay in enumerate((1., 1., 1., 2.)):
            output[:, field] *= np.exp(-decay/8)*(2*n+1)*(-1.)**n*spherical_in(n, decay/8)
    return output


def orders(dimension):
    zero = (0,) * dimension
    first = tuple(tuple(int(i == j) for i in range(dimension)) for j in range(3))
    second = tuple(tuple(2 * int(i == j) for i in range(dimension)) for j in range(3))
    time_order = (0, 0, 0, 1) if dimension == 4 else None
    return zero, first, second, time_order


def _field_expressions(family, frequency, transient):
    if family not in FAMILIES:
        raise ValueError(f'Unknown family {family}')
    x, y, z, t = sp.symbols('x y z t', real=True)
    xyz = (x, y, z)
    k = sp.Float(frequency, 17)
    a = sp.Rational(1, 5)
    if family == 'quadratic':
        velocity = (a*y*z, a*z*x, a*x*y)
        pressure = sp.Rational(3, 20)*(x*y+y*z+z*x)+sp.Rational(3, 100)*(x+y+z)
    elif family == 'trigonometric':
        velocity = (a*sp.sin(k*y+sp.Rational(3, 10))*sp.cos(k*z-sp.Rational(1, 5)),
                    a*sp.sin(k*z-sp.Rational(1, 10))*sp.cos(k*x+sp.Rational(2, 5)),
                    a*sp.sin(k*x+sp.Rational(1, 5))*sp.cos(k*y-sp.Rational(3, 10)))
        pressure = sp.sin(sp.Rational(2, 5)*x+y/2-sp.Rational(3, 10)*z)/10
    else:
        bubble = sp.prod((1-r*r)**2 for r in xyz)
        vectors = ((1, 2, -1), (-2, 1, 1), (1, -1, 2))
        potential = [bubble*sp.sin(k*sum(v*r for v, r in zip(vector, xyz))/3+phase)/20
                     for vector, phase in zip(vectors, (sp.Rational(1, 5), -sp.Rational(3, 10), sp.Rational(2, 5)))]
        velocity = (sp.diff(potential[2], y)-sp.diff(potential[1], z),
                    sp.diff(potential[0], z)-sp.diff(potential[2], x),
                    sp.diff(potential[1], x)-sp.diff(potential[0], y))
        pressure = sp.sin(sp.Rational(2, 5)*x+y/2-sp.Rational(3, 10)*z)/10
    if transient:
        velocity = tuple(sp.exp(-t)*u for u in velocity)
        pressure *= sp.exp(-2*t)
    return xyz+(t,) if transient else xyz, tuple(velocity)+(pressure,)


@lru_cache(maxsize=32)
def _exact_functions(family, frequency, transient):
    variables, fields = _field_expressions(family, frequency, transient)
    zero, first, second, time_order = orders(len(variables))
    derivatives = (zero,)+first+second+((time_order,) if transient else ())
    functions = {}
    for derivative in derivatives:
        expr = []
        for field in fields:
            value = field
            for variable, count in zip(variables, derivative):
                if count:
                    value = sp.diff(value, variable, count)
            expr.append(value)
        functions[derivative] = sp.lambdify(variables, expr, 'numpy', cse=True)
    return functions


def manufactured_jets(points, family='trigonometric', frequency=.7, transient=False):
    """Analytical derivatives used for prescribed forcing and held-out truth."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3+int(transient):
        raise ValueError('Point dimension does not match steady/transient family')
    functions = _exact_functions(family, float(frequency), bool(transient))
    return {order: np.column_stack([np.broadcast_to(value, (len(points),))
                                   for value in function(*points.T)])
            for order, function in functions.items()}


def validation_field_torch(points, family='trigonometric', frequency=.7, transient=False):
    """Independent differentiable field formula; never called by a PDE solve."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xyz = points[:, :3]
    if family == 'quadratic':
        velocity = .2*torch.stack((y*z, z*x, x*y), dim=1)
        pressure = .15*(x*y+y*z+z*x)+.03*(x+y+z)
    elif family == 'trigonometric':
        k = frequency
        velocity = .2*torch.stack((torch.sin(k*y+.3)*torch.cos(k*z-.2),
                                  torch.sin(k*z-.1)*torch.cos(k*x+.4),
                                  torch.sin(k*x+.2)*torch.cos(k*y-.3)), dim=1)
        pressure = .1*torch.sin(.4*x+.5*y-.3*z)
    elif family == 'bubble':
        factors = (1-xyz*xyz)**2
        bubble = factors.prod(dim=1)
        grad_b = torch.stack([-4*xyz[:, j]*(1-xyz[:, j]**2)*
                             factors[:, (j+1) % 3]*factors[:, (j+2) % 3]
                            for j in range(3)], dim=1)
        vectors = torch.tensor(((1, 2, -1), (-2, 1, 1), (1, -1, 2)), dtype=points.dtype, device=points.device)
        vectors = frequency*vectors/3
        phases = xyz@vectors.T+torch.tensor((.2, -.3, .4), dtype=points.dtype, device=points.device)
        # dpotential[q, component, axis]
        dp = (torch.sin(phases)[:, :, None]*grad_b[:, None, :]+
              torch.cos(phases)[:, :, None]*bubble[:, None, None]*vectors[None, :, :])/20
        velocity = torch.stack((dp[:, 2, 1]-dp[:, 1, 2], dp[:, 0, 2]-dp[:, 2, 0],
                                dp[:, 1, 0]-dp[:, 0, 1]), dim=1)
        pressure = .1*torch.sin(.4*x+.5*y-.3*z)
    else:
        raise ValueError(f'Unknown family {family}')
    if transient:
        velocity = velocity*torch.exp(-points[:, 3:4])
        pressure = pressure*torch.exp(-2*points[:, 3])
    return torch.cat((velocity, pressure[:, None]), dim=1)


def physical_terms(jets, viscosity, dimension):
    """Pointwise incompressible NS: u_t + u.grad(u) - nu Lap(u) + grad(p)."""
    zero, first, second, time_order = orders(dimension)
    values = jets[zero]
    velocity = values[:, :3]
    momentum = sum(velocity[:, axis:axis+1]*jets[first[axis]][:, :3]
                   - viscosity*jets[second[axis]][:, :3] for axis in range(3))
    if isinstance(values, torch.Tensor):
        pressure_gradient = torch.stack([jets[d][:, 3] for d in first], dim=1)
    else:
        pressure_gradient = np.column_stack([jets[d][:, 3] for d in first])
    momentum = momentum+pressure_gradient
    if time_order is not None:
        momentum = momentum+jets[time_order][:, :3]
    divergence = sum(jets[first[axis]][:, axis] for axis in range(3))
    return momentum, divergence


def reflected_samples(domain, count, seed, boundary=False, spatial_axes=3):
    """Complete reflection orbits; never reflect time across an IVP boundary.

    Requested count rounds up to a multiple of 2**spatial_axes. The declared
    box and uniform face-condition type make these spatial reflections valid.
    Independent ordinary-model audits continue to use unstructured points.
    """
    orbit = np.array(list(product((-1., 1.), repeat=spatial_axes)))
    count_base = int(np.ceil(count/len(orbit)))
    base = (domain.boundary(count_base, seed).points if boundary else domain.interior(count_base, seed))
    midpoint = domain.bounds[:spatial_axes].mean(axis=1)
    rows = np.repeat(base, len(orbit), axis=0)
    rows[:, :spatial_axes] = midpoint+((base[:, None, :spatial_axes]-midpoint)*orbit[None, :, :]).reshape(-1, spatial_axes)
    return rows


def make_declaration(family='trigonometric', viscosity=.2, frequency=.7, transient=False,
                     boundary_ratio=.5, reflection_orbits=False, combined_physics=False):
    if family not in FAMILIES or viscosity <= 0 or frequency <= 0 or boundary_ratio <= 0:
        raise ValueError('Valid family and positive viscosity/frequency/boundary ratio required')
    dimension = 3+int(transient)
    domain = ResidualDomain([[-1., 1.]]*3+([[0., .25]] if transient else []))
    spatial = ResidualDomain([[-1., 1.]]*3)
    zero, first, second, time_order = orders(dimension)
    derivatives = (zero,)+first+second+((time_order,) if transient else ())

    def make_blocks(count, seed):
        points = (reflected_samples(domain, count, seed) if reflection_orbits else domain.interior(count, seed))
        exact = manufactured_jets(points, family, frequency, transient)
        source = torch.tensor(physical_terms(exact, viscosity, dimension)[0], dtype=torch.float64)
        # No exact interior field values are retained in a closure.
        del exact
        def momentum_full(x, j, p):
            return physical_terms(j, viscosity, dimension)[0]-source
        def momentum_batch(x, j, p, indices):
            return physical_terms(j, viscosity, dimension)[0]-source[indices]
        def incompressibility(x, j, p):
            return sum(j[first[axis]][:, axis] for axis in range(3))
        if combined_physics:
            def physics_full(x, j, p):
                momentum, divergence = physical_terms(j, viscosity, dimension)
                return torch.cat((momentum-source, divergence[:, None]), dim=1)
            def physics_batch(x, j, p, indices):
                momentum, divergence = physical_terms(j, viscosity, dimension)
                return torch.cat((momentum-source[indices], divergence[:, None]), dim=1)
            blocks = [ResidualBlock('momentum_and_incompressibility', points, physics_full,
                                    derivatives, batch_function=physics_batch)]
        else:
            blocks = [ResidualBlock('momentum', points, momentum_full, derivatives,
                                    batch_function=momentum_batch),
                      ResidualBlock('incompressibility', points, incompressibility, first)]
        boundary_count = max(48, int(count*boundary_ratio))
        boundary = (reflected_samples(spatial, boundary_count, seed+17, boundary=True) if reflection_orbits
                    else spatial.boundary(boundary_count, seed+17).points)
        if transient:
            times = (np.repeat(domain.interior(len(boundary)//8, seed+31)[:, 3:4], 8, axis=0)
                     if reflection_orbits else domain.interior(len(boundary), seed+31)[:, 3:4])
            boundary = np.column_stack((boundary, times))
        trace = torch.tensor(manufactured_jets(boundary, family, frequency, transient)[zero][:, :3])
        blocks.append(ResidualBlock('velocity_boundary', boundary,
            lambda x, j, p: j[zero][:, :3]-trace, (zero,),
            batch_function=lambda x, j, p, indices: j[zero][:, :3]-trace[indices]))
        if transient:
            initial_space = (reflected_samples(spatial, max(64, count//2), seed+43) if reflection_orbits
                             else spatial.interior(max(64, count//2), seed+43))
            initial = np.column_stack((initial_space, np.zeros(len(initial_space))))
            initial_values = torch.tensor(manufactured_jets(initial, family, frequency, True)[zero][:, :3])
            blocks.append(ResidualBlock('velocity_initial', initial,
                lambda x, j, p: j[zero][:, :3]-initial_values, (zero,),
                batch_function=lambda x, j, p, indices: j[zero][:, :3]-initial_values[indices]))
        gauge = np.zeros((max(4, count//16) if transient else 1, dimension))
        if transient:
            gauge[:, 3] = domain.interior(len(gauge), seed+59)[:, 3]
        blocks.append(ResidualBlock('pressure_gauge', gauge, lambda x, j, p: j[zero][:, 3], (zero,)))
        return blocks

    return ResidualDeclaration(domain, make_blocks, fields=4)


def ordinary_audit(solution, *, family, viscosity, frequency, transient, count=513, derivative_count=97):
    """Held-out actual MLP values and AD; no alternate polynomial forward."""
    dimension = 3+int(transient)
    zero, first, second, time_order = orders(dimension)
    spec = make_declaration(family, viscosity, frequency, transient)
    points = spec.domain.interior(count, 734153)
    model = solution.export()
    assert [type(layer) for layer in model] == [torch.nn.Linear, torch.nn.Tanh, torch.nn.Linear]
    values = ordinary_jets(model, points, (zero,))[zero].numpy()
    expected = validation_field_torch(torch.tensor(points), family, frequency, transient).numpy()
    independent_points = spec.domain.interior(derivative_count, 813733)
    derivative_orders = (zero,)+first+second+((time_order,) if transient else ())
    jets = {d: v.numpy() for d, v in ordinary_jets(model, independent_points, derivative_orders).items()}
    source, _ = physical_terms(manufactured_jets(independent_points, family, frequency, transient), viscosity, dimension)
    momentum, divergence = physical_terms(jets, viscosity, dimension)
    momentum = momentum-source
    velocity_error, pressure_error = values[:, :3]-expected[:, :3], values[:, 3]-expected[:, 3]
    grouped = solution.evaluate(points)
    checks = {}
    for block in spec.make_blocks(61, 177331):
        if block.name in ('momentum', 'incompressibility'):
            continue
        block_jets = ordinary_jets(model, block.points, block.derivatives)
        with torch.no_grad():
            r = block.function(torch.tensor(block.points), block_jets, torch.empty((len(block.points), 0))).numpy()
        checks[block.name] = dict(rms=float(np.sqrt(np.mean(r*r))), max_abs=float(np.max(abs(r))))
    return dict(ordinary_velocity_relative_l2=float(np.linalg.norm(velocity_error)/np.linalg.norm(expected[:, :3])),
                ordinary_pressure_relative_l2=float(np.linalg.norm(pressure_error)/np.linalg.norm(expected[:, 3])),
                ordinary_velocity_max_abs=float(np.max(abs(velocity_error))),
                ordinary_pressure_max_abs=float(np.max(abs(pressure_error))),
                ordinary_momentum_rms=float(np.sqrt(np.mean(momentum*momentum))),
                ordinary_momentum_max_abs=float(np.max(abs(momentum))),
                ordinary_divergence_rms=float(np.sqrt(np.mean(divergence*divergence))),
                ordinary_divergence_max_abs=float(np.max(abs(divergence))),
                ordinary_vs_grouped_max_abs=float(np.max(abs(values-grouped))),
                ordinary_readout_l1_per_field=model[2].weight.abs().sum(dim=1).tolist(),
                ordinary_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                value_audit_points=count, derivative_audit_points=derivative_count,
                constraints=checks, architecture='ordinary torch Linear/Tanh/Linear with standard AD',
                truth_access='Held-out values only; manufactured source and prescribed velocity traces used during solve')


def load_native_resume(path, declaration, *, family, viscosity, frequency, transient, batch_size):
    """Read this benchmark's saved native state, never its capacity controls."""
    from route2_box_profiles import BoxProfileOperator
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        coefficients = np.asarray(saved['coefficients'], float)
        indices = np.asarray(saved['multiindices'], int)
        if 'metadata' in saved:
            config = json.loads(str(saved['metadata']))
            if config.get('kind') != 'native_ns_iterate':
                raise ValueError('Resume requires an explicitly marked native NS iterate')
        else:
            record = json.loads(path.with_suffix('.json').read_text())
            if 'native actual tanh residual solve' not in record.get('provenance', ''):
                raise ValueError('Resume requires this benchmark native-solve provenance')
            config = record['config']
    for key, requested in dict(family=family, viscosity=viscosity, frequency=frequency, transient=transient).items():
        if config.get(key) != requested:
            raise ValueError(f'Resume changes {key}; use a matching saved native problem')
    degree = int(indices.sum(axis=1).max())
    features = BoxProfileOperator(declaration.domain.bounds, degree, centers=config['centers'],
                                  lam=config['lam'], block_size=batch_size)
    if not np.array_equal(indices, features.multiindices) or coefficients.shape != (features.size, 4):
        raise ValueError('Saved native coordinates do not match the declared box space')
    problem = ResidualProblem(features, declaration.make_blocks(8, 614491), fields=4,
                              execution='streamed', batch_size=batch_size)
    return ResidualSolution(problem, coefficients, np.empty(0),
                            dict(execution='streamed', batch_size=batch_size,
                                 provenance=f'Saved native PDE iterate from {path.resolve()}'))


def native_reencoding_audit(source, geometries, *, family='trigonometric', viscosity=.2,
                            frequency=.7, transient=True, count=65):
    """Compare encodings of one native iterate using physics, never target values.

    The coefficient vector is unchanged. Ideal jets evaluate that same vector
    in its coordinate basis solely to measure encoding error. They are not a
    reference solution. No target-series coefficients or interior labels are
    computed here, and no correction/fitting is performed.
    """
    from route2_box_profiles import BoxProfileOperator
    from solver.route2 import export_mlp
    spec = make_declaration(family, viscosity, frequency, transient, combined_physics=True)
    native = load_native_resume(source, spec, family=family, viscosity=viscosity,
                                frequency=frequency, transient=transient, batch_size=64)
    blocks = spec.make_blocks(count, 289173)
    coefficients = native.coefficients
    ideal_reference = []
    tensor_state = native.problem.features.prepare_ideal_tensor_forward(coefficients)
    for block in blocks:
        plan = native.problem.features.ideal_tensor_plan(4, 64, block.derivatives)
        jets = {d: np.empty((len(block.points), 4)) for d in block.derivatives}
        for start in range(0, len(block.points), plan['rows']):
            rows = slice(start, min(start+plan['rows'], len(block.points)))
            evaluated = native.problem.features.forward_ideal_tensor_jets(
                tensor_state, block.points[rows], block.derivatives)
            for derivative in block.derivatives:
                jets[derivative][rows] = evaluated[derivative]
        ideal_reference.append(jets)
    results = []
    for centers, lam in geometries:
        started = time.perf_counter()
        features = BoxProfileOperator(spec.domain.bounds, native.problem.features.degree,
                                      centers=centers, lam=lam, block_size=64)
        np.testing.assert_array_equal(features.multiindices, native.problem.features.multiindices)
        model = export_mlp(features, coefficients)
        checks = []
        for block, ideal in zip(blocks, ideal_reference):
            actual = ordinary_jets(model, block.points, block.derivatives)
            x = torch.tensor(block.points)
            parameters = torch.empty((len(block.points), 0), dtype=torch.float64)
            raw = block.function(x, actual, parameters).numpy()
            ideal_raw = block.function(x, {d: torch.tensor(j) for d, j in ideal.items()}, parameters).numpy()
            if raw.ndim == 1:
                raw, ideal_raw = raw[:, None], ideal_raw[:, None]
            checks.append(dict(name=block.name, points=len(block.points),
                actual_rms_per_equation=np.sqrt(np.mean(raw**2, axis=0)).tolist(),
                actual_max_abs_per_equation=np.max(abs(raw), axis=0).tolist(),
                actual_vs_same_coordinate_ideal_rms_per_equation=np.sqrt(np.mean((raw-ideal_raw)**2, axis=0)).tolist(),
                jet_encoding_max_abs={str(d): float(np.max(abs(actual[d].numpy()-ideal[d])))
                                      for d in block.derivatives}))
        results.append(dict(centers=centers, lam=lam, degree=features.degree,
            neurons=features.tanh_count, ordinary_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            seconds=time.perf_counter()-started, blocks=checks))
    return dict(source=str(Path(source).resolve()),
                provenance='Same saved native PDE coefficients, no fitting, no analytical target coefficients or interior target values; geometry assessed with prescribed PDE/BC/IC/gauge and encoding error only',
                rows=results)


def run_case(name, family='trigonometric', degrees=(4, 6), centers=129, lam=.2,
             viscosity=.2, frequency=.7, transient=False, max_seconds=180.,
             preconditioner='diagonal', inner_iterations=300, oversampling=3.,
             max_iterations=12, batch_size=64, tolerance=1e-13, audit_points=257,
             trace_memory=True, linear_refinement_steps=0, resume=None,
             preconditioner_block_size=32, preconditioner_field_parity='none',
             preconditioner_basis='actual', reflection_orbits=False,
             preconditioner_refresh=0, inner_tolerance=None,
             linearization_basis='actual', maximum_neurons=300000,
             maximum_working_array_mb=512, combined_physics=False):
    torch.set_num_threads(1)
    spec = make_declaration(family, viscosity, frequency, transient,
                            reflection_orbits=reflection_orbits, combined_physics=combined_physics)
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if trace_memory:
        tracemalloc.start()
    started = time.perf_counter()
    initial_solution = (load_native_resume(resume, spec, family=family, viscosity=viscosity,
                        frequency=frequency, transient=transient, batch_size=batch_size)
                        if resume is not None else None)
    OUT.mkdir(parents=True, exist_ok=True)
    progress = []
    def snapshot(coefficients, parameters, record):
        from math import comb
        from solver.general_features import _compositions
        dimension = 3+int(transient)
        degree = next(p for p in degrees if comb(p+dimension, dimension) == len(coefficients))
        indices = np.array([a for n in range(degree+1) for a in _compositions(n, dimension)])
        metadata = dict(kind='native_ns_iterate', family=family, viscosity=viscosity,
                        frequency=frequency, transient=transient, degree=degree,
                        centers=centers, lam=lam,
                        combined_physics=combined_physics,
                        initial_field=('saved native PDE iterate' if resume is not None else
                                       'zero or previous degree native PDE iterate'),
                        initial_source=None if resume is None else str(Path(resume).resolve()))
        np.savez(OUT/f'{name}_latest_native.npz', coefficients=coefficients,
                 multiindices=indices, parameters=parameters, metadata=json.dumps(metadata))
        np.savez(OUT/f'{name}_p{degree}_native.npz', coefficients=coefficients,
                 multiindices=indices, parameters=parameters, metadata=json.dumps(metadata))
        checkpoint_record = dict(coordinate_count=len(coefficients), **record,
            total_run_elapsed_seconds=time.perf_counter()-started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024))
        progress.append(checkpoint_record)
        (OUT/f'{name}_progress.json').write_text(json.dumps(progress, indent=2))
        print(json.dumps(dict(name=name, **checkpoint_record)), flush=True)
    options = dict(preconditioner=preconditioner, preconditioner_block_size=preconditioner_block_size,
            preconditioner_field_parity=preconditioner_field_parity,
            preconditioner_basis=preconditioner_basis, preconditioner_refresh=preconditioner_refresh,
            linearization_basis=linearization_basis,
            lsmr_max_iterations=inner_iterations, linear_refinement_steps=linear_refinement_steps, scaling_probes=4,
            validate_locality=False, iteration_callback=snapshot)
    if inner_tolerance is not None:
        if not np.isfinite(inner_tolerance) or inner_tolerance <= 0:
            raise ValueError('Inner correction tolerance must be positive')
        options.update(high_accuracy=False, lsmr_tolerance=inner_tolerance,
                       gradient_tolerance=1e-30, step_tolerance=1e-30, damping=1e-30)
    solution = solve_route2(spec, degrees=tuple(degrees), centers=centers, lam=lam,
        coordinates='box', execution='streamed', batch_size=batch_size,
        angular_rule='tensor', tolerance=tolerance, oversampling=oversampling,
        check_points=97, max_seconds=max_seconds, max_iterations=max_iterations,
        initial_solution=initial_solution,
        maximum_working_array_mb=maximum_working_array_mb, maximum_neurons=maximum_neurons,
        solver_options=options)
    solve_seconds = time.perf_counter()-started
    peak = None
    if trace_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    solve_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    audit_started = time.perf_counter()
    audit = ordinary_audit(solution, family=family, viscosity=viscosity, frequency=frequency,
                           transient=transient, count=audit_points)
    rss_factor = 1 if sys.platform == 'darwin' else 1024
    config = dict(name=name, family=family, degrees=list(degrees), centers=centers, lam=lam,
                  viscosity=viscosity, frequency=frequency, transient=transient,
                  max_seconds=max_seconds, preconditioner=preconditioner,
                  inner_iterations=inner_iterations, oversampling=oversampling,
                  max_iterations=max_iterations, batch_size=batch_size, tolerance=tolerance,
                  allocation_tracing=trace_memory,
                  linear_refinement_steps=linear_refinement_steps,
                  preconditioner_block_size=preconditioner_block_size,
                  preconditioner_field_parity=preconditioner_field_parity,
                  preconditioner_basis=preconditioner_basis, reflection_orbits=reflection_orbits,
                  preconditioner_refresh=preconditioner_refresh,
                  linearization_basis=linearization_basis,
                  linearization_scope=('Ideal-coordinate approximate Jacobian evaluated at actual tanh state; '
                                       'actual tanh residual, acceptance, gradient and convergence checks; '
                                       'no external PDE trajectory or target readout'
                                       if linearization_basis == 'ideal' else 'Actual tanh Jacobian'),
                  maximum_neurons=maximum_neurons,
                  maximum_working_array_mb=maximum_working_array_mb,
                  combined_physics=combined_physics,
                  inner_tolerance=inner_tolerance,
                  inner_accuracy_scope=('Bounded relative correction solves; actual physical tolerance unchanged; '
                                        'gradient/step thresholds 1e-30, initial damping1e-30'
                                        if inner_tolerance is not None else 'Default high-accuracy inner solves'),
                  resume=None if resume is None else str(Path(resume).resolve()),
                  initial_field=('saved native PDE iterate; subsequent degree continuation only from native neural PDE solves'
                                 if resume is not None else
                                 'all coefficients zero; degree continuation only from prior native neural PDE solve'))
    row = dict(config=config, status=solution.status, selected_degree=solution.problem.features.degree,
               tanh_neurons=solution.problem.features.tanh_count, coefficient_unknowns=4*solution.problem.features.size,
               solve_seconds=solve_seconds, audit_seconds=time.perf_counter()-audit_started,
               traced_solve_peak_bytes=peak, process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*rss_factor,
               process_peak_rss_after_solve_bytes=solve_peak_rss*rss_factor,
               process_peak_rss_after_audit_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*rss_factor,
               process_peak_rss_increment_bytes=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-baseline_rss)*rss_factor,
               memory_scope='Whole-process RSS includes imports and retained runtime; explicit solve and ordinary-AD audit peaks are separate. Traced solve peak is optional and excludes untraced native allocations.',
               solver=solution.solution.metrics, route2=solution.metrics, history=solution.history, audit=audit,
               provenance='No external PDE solve, no interior target labels, no fit to truth; native actual tanh residual solve',
               caveat='Manufactured smooth forced NS; this is not turbulent unforced flow or a universal precision certificate')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{name}.json').write_text(json.dumps(row, indent=2))
    actual_centers = solution.problem.features.interior_centers
    actual_lam = float(solution.problem.features.encoding.gamma*2/(actual_centers-1))
    saved_metadata = dict(kind='native_ns_iterate', family=family, viscosity=viscosity,
        frequency=frequency, transient=transient, centers=actual_centers, lam=actual_lam,
        degree=solution.problem.features.degree, combined_physics=combined_physics,
        status=solution.status, record=str((OUT/f'{name}.json').resolve()))
    np.savez(OUT/f'{name}.npz', coefficients=solution.coefficients,
             multiindices=solution.problem.features.multiindices, metadata=json.dumps(saved_metadata))
    torch.save(solution.export().state_dict(), OUT/f'{name}_ordinary_mlp.pt')
    print(json.dumps(dict(name=name, status=row['status'], degree=row['selected_degree'],
        neurons=row['tanh_neurons'], seconds=solve_seconds, lsmr=row['solver'].get('lsmr_iterations'), **audit)), flush=True)
    return row


def plot_saved_native_audits(sources, labels, output, *, title, notes):
    """Render recorded native checks; never rerun a solve or compute truth.

    Stages are categorical because both solver settings and center resolution
    can change. This figure is an experimental history, not a fitted error law.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sources = [Path(source) for source in sources]
    if len(sources) != len(labels) or not sources:
        raise ValueError('Each saved native audit needs a stage label')
    records = [json.loads(source.read_text()) for source in sources]
    audits = [record.get('audit', record) for record in records]
    output = Path(output)
    data = dict(sources=[str(source.resolve()) for source in sources],
                labels=list(labels), audits=audits, notes=list(notes),
                interpretation='Sequential native PDE checkpoints; not a controlled scaling-law fit')
    output.with_name(output.name+'_data.json').write_text(json.dumps(data, indent=2))
    with plt.rc_context({'font.size': 11, 'axes.labelsize': 11}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
        fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.29, wspace=.20)
        fig.suptitle(title, y=.965, fontsize=16)
        stages = np.arange(len(labels))
        for axis in axes:
            axis.set_yscale('log')
            axis.set_xticks(stages, labels)
            axis.grid(alpha=.20)
            axis.set_xlabel('Saved native solve stage')
        curves = (
            (0, 'ordinary_velocity_relative_l2', 'Velocity', '#286cbb'),
            (0, 'ordinary_pressure_relative_l2', 'Pressure', '#cc6510'),
            (1, 'ordinary_momentum_rms', 'Momentum residual', '#7644a4'),
            (1, 'ordinary_divergence_rms', 'Divergence', '#138c72'))
        for panel, key, label, color in curves:
            axes[panel].plot(stages, [row[key] for row in audits], 'o-',
                             color=color, label=label, lw=2)
        axes[0].axhline(1e-14, color='.5', ls=':', label=r'$10^{-14}$ reference')
        axes[0].set_ylabel(r'Held-out relative $L_2$ error')
        axes[1].set_ylabel('Held-out ordinary-AD RMS residual')
        for axis in axes:
            axis.legend(loc='lower center', bbox_to_anchor=(.5, 1.025), ncol=3, frameon=False)
        for index, note in enumerate(notes):
            fig.text(.5, .155-.036*index, note, ha='center', fontsize=10)
        fig.savefig(output.with_suffix('.png'), dpi=170)
        fig.savefig(output.with_suffix('.pdf'))
        plt.close(fig)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--family', choices=FAMILIES, default='trigonometric')
    parser.add_argument('--degrees', type=int, nargs='+', default=[4, 6])
    parser.add_argument('--centers', type=int, default=129)
    parser.add_argument('--lam', type=float, default=.2)
    parser.add_argument('--viscosity', type=float, default=.2)
    parser.add_argument('--frequency', type=float, default=.7)
    parser.add_argument('--transient', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=180.)
    parser.add_argument('--preconditioner', choices=('diagonal', 'block', 'auto'), default='diagonal')
    parser.add_argument('--inner-iterations', type=int, default=300)
    parser.add_argument('--oversampling', type=float, default=3.)
    parser.add_argument('--max-iterations', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--tolerance', type=float, default=1e-13)
    parser.add_argument('--audit-points', type=int, default=257)
    parser.add_argument('--no-trace-memory', action='store_false', dest='trace_memory',
                        help='Retain RSS/array accounting but omit allocation tracing overhead')
    parser.add_argument('--linear-refinement-steps', type=int, default=0)
    parser.add_argument('--resume', help='Saved native NS npz state from a matching benchmark; never a capacity control')
    parser.add_argument('--preconditioner-block-size', type=int, default=32)
    parser.add_argument('--preconditioner-field-parity', choices=('none', 'auto'), default='none')
    parser.add_argument('--preconditioner-basis', choices=('actual', 'ideal'), default='actual',
                        help='Select preconditioner panels independently of the correction Jacobian; PDE residual remains actual tanh')
    parser.add_argument('--reflection-orbits', action='store_true',
                        help='Train on complete spatial reflection orbits; audits stay independently unstructured')
    parser.add_argument('--preconditioner-refresh', type=int, default=0)
    parser.add_argument('--inner-tolerance', type=float,
                        help='Bound relative correction-solve accuracy; outer physical tolerance remains unchanged')
    parser.add_argument('--linearization-basis', choices=('actual', 'ideal'), default='actual',
                        help='Approximate-Jacobian correction option; residual, acceptance and export remain actual tanh')
    parser.add_argument('--maximum-neurons', type=int, default=300000,
                        help='Explicit ordinary-model neuron budget')
    parser.add_argument('--maximum-working-array-mb', type=float, default=512,
                        help='Named-array preflight cap in MiB; recorded separately from measured whole-process RSS')
    parser.add_argument('--combined-physics', action='store_true',
                        help='Share interior neural jets for four NS equations; identical objective up to row permutation')
    run_case(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
