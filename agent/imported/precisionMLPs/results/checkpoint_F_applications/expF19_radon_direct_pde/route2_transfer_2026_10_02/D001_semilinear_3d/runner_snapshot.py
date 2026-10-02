"""Frozen higher-dimensional native semilinear transfer tests.

All dimensions use the same full-rank dense quadratic family and solver policy.
Only forcing and boundary values enter fitting; interior truth is post-fit audit.
No production solver changes, external PDE solve, active axes or sparse angles.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import itertools
from math import comb
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
import torch
from route2_gap_diagnosis import ROOT, dump, emit, log, stamp
from solver.general_domains import ResidualDomain
from solver.general_adaptive import ResidualDeclaration
from solver.general_residual import ResidualBlock
from solver.residual_scaling import normalize_equations
from solver.route2 import solve_route2_native, ordinary_jets, export_mlp, footprint

OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02'
DEGREES = (4, 6, 8, 10, 12)
OPTIONS = dict(degrees=DEGREES, centers=257, lam=.2, coordinates='box',
    angular_rule='tensor', tolerance=1e-13, oversampling=6, batch_size=64,
    check_points=257, export_check_points=257, max_iterations=8, max_seconds=600.,
    maximum_working_array_mb=1024, maximum_neurons=2000000,
    solver_options=dict(lsmr_max_iterations=4000, inexact_newton=False,
        refine_on_ideal_stall=True, damping=1e-6, preconditioner='block',
        preconditioner_basis='ideal', linearization_basis='ideal',
        preconditioner_refresh=1, preconditioner_field_parity='auto',
        preconditioner_block_size=1024, linear_refinement_steps=0))


class Family:
    def __init__(self, dimension):
        self.dimension = dimension
        axis = np.arange(1, dimension+1)
        self.B = np.diag(np.linspace(.018, .034, dimension)) + .005*np.cos(axis[:, None]*axis[None, :])
        self.domain = ResidualDomain([[-1., 1.]]*dimension)
        self.value = (0,)*dimension
        self.seconds = tuple(tuple(2 if i == j else 0 for j in range(dimension)) for i in range(dimension))
        self.orders = (self.value,) + self.seconds

    def boundary(self, x):
        B = torch.tensor(self.B, dtype=x.dtype, device=x.device)
        return torch.exp(((x @ B)*x).sum(dim=1, keepdim=True))

    def forcing(self, x):
        B = torch.tensor(self.B, dtype=x.dtype, device=x.device)
        transformed = x @ B
        value = torch.exp((transformed*x).sum(dim=1, keepdim=True))
        laplace = (2*torch.trace(B)+4*(transformed**2).sum(dim=1, keepdim=True))*value
        return -laplace+value**3

    def truth(self, x):
        return np.exp(np.einsum('qi,ij,qj->q', x, self.B, x))

    def declaration(self):
        def blocks(count, seed):
            interior = self.domain.interior(count, seed)
            boundary = self.domain.boundary(max(256, count//2), seed+7919).points
            def physics(x, jets, parameters):
                return -sum(jets[d] for d in self.seconds)+jets[self.value]**3-self.forcing(x)
            return [ResidualBlock('PDE', interior, physics, self.orders),
                    ResidualBlock('Dirichlet', boundary, lambda x, j, p: j[self.value]-self.boundary(x), (self.value,))]
        return normalize_equations(ResidualDeclaration(self.domain, blocks, 1), ['PDE'])


def forecast(dimension):
    """No model allocation: the driver's initial lower bound plus exact widths.

    The driver subsequently accounts for samples, residual panels and adaptive
    local preconditioner factors; this is explicitly not the final full budget.
    """
    rows = []
    for degree in DEGREES:
        cost = footprint(dimension, degree, 257, coordinates='box', angular_rule='tensor',
                         execution='streamed', batch_size=64, lam=.2)
        lower = 4*cost['ordinary_network_bytes'] + cost['encoder_peak_array_estimate_bytes']
        lower += sum(cost[key] for key in ('encoding_bank_bytes', 'angular_table_bytes',
            'mode_label_bytes', 'prepared_state_bytes', 'scalar_basis_panel_bytes',
            'coordinate_vector_bytes', 'coordinate_workspace_bytes',
            'angular_panel_bytes', 'coordinate_transform_state_bytes'))
        row = dict(cost, minimum_named_arrays_bytes=lower,
            initial_preflight_reject=bool(cost['neurons']>OPTIONS['maximum_neurons'] or
                lower>OPTIONS['maximum_working_array_mb']*1024**2),
            full_stage_budget_checked_by_driver=True)
        rows.append(row)
    return rows


def audit(family, sol, folder):
    features = sol.problem.features
    coefficients = sol.coefficients
    seed = 573927
    points = family.domain.interior(512, seed)
    derivatives = family.domain.interior(97, seed+1)
    boundary = family.domain.boundary(257, seed+2).points
    corners = np.asarray(list(itertools.product((-1., 1.), repeat=family.dimension)))
    model = export_mlp(features, coefficients)
    values = ordinary_jets(model, points, (family.value,))[family.value].numpy().ravel()
    jets = {d: v.numpy().ravel() for d, v in ordinary_jets(model, derivatives, family.orders).items()}
    forcing = family.forcing(torch.tensor(derivatives, dtype=torch.float64)).numpy().ravel()
    residual = -sum(jets[d] for d in family.seconds)+jets[family.value]**3-forcing
    bc = ordinary_jets(model, boundary, (family.value,))[family.value].numpy().ravel()-family.truth(boundary)
    corner = ordinary_jets(model, corners, (family.value,))[family.value].numpy().ravel()-family.truth(corners)
    truth = family.truth(points)
    stable = features.forward_jets(coefficients[:, 0], points, (family.value,))[family.value].ravel()
    ideal = features.selected_ideal_columns(points, np.arange(features.size), (family.value,))[family.value] @ coefficients[:, 0]
    result = dict(ordinary_relative_l2=float(np.linalg.norm(values-truth)/np.linalg.norm(truth)),
        ordinary_field_max=float(abs(values-truth).max()),
        ordinary_pde_rms=float(np.sqrt(np.mean(residual**2))), ordinary_pde_max=float(abs(residual).max()),
        boundary_rms=float(np.sqrt(np.mean(bc**2))), boundary_max=float(abs(bc).max()),
        corner_rms=float(np.sqrt(np.mean(corner**2))), corner_max=float(abs(corner).max()),
        stable_relative_l2=float(np.linalg.norm(stable-truth)/np.linalg.norm(truth)),
        ideal_relative_l2=float(np.linalg.norm(ideal-truth)/np.linalg.norm(truth)),
        ordinary_minus_stable_rms=float(np.sqrt(np.mean((values-stable)**2))),
        readout_l1=float(model[2].weight.detach().abs().sum()),
        model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        architecture=[type(layer).__name__ for layer in model],
        field_points=512, derivative_points=97, boundary_points=257, corners=len(corners),
        interior_truth_used_only_after_solver_returned=True)
    torch.save(dict(state_dict=model.state_dict(), dimension=family.dimension,
        width=model[0].out_features, architecture='Linear/Tanh/Linear'), folder/'ordinary_mlp.pt')
    np.savez_compressed(folder/'audit.npz', points=points, truth=truth, ordinary=values,
        stable=stable, ideal=ideal, derivative_points=derivatives, raw_pde_residual=residual,
        boundary=boundary, boundary_error=bc, corners=corners, corner_error=corner)
    return result


def run(args):
    torch.set_num_threads(1)
    folder = OUT/args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    family = Family(args.dimension)
    options = dict(OPTIONS, solver_options=dict(OPTIONS['solver_options']))
    sources = [Path(__file__)] + [Path(__file__).parent/name for name in (
        'solver/route2.py', 'solver/general_residual.py', 'solver/streamed_residual.py',
        'solver/residual_scaling.py', 'solver/general_domains.py', 'route2_box_profiles.py',
        'route2_box_tensor.py', 'route2_directional_operator.py', 'quill_boundary.py')]
    protocol = dict(run_id=args.run_id, dimension=args.dimension, equation='-Delta u + u^3 = g',
        domain=family.domain.bounds.tolist(), B=family.B.tolist(),
        B_rule='diag(linspace(.018,.034,d)) + .005*cos(i*j), i,j=1,...,d',
        B_eigenvalues=np.linalg.eigvalsh(family.B).tolist(), B_rank=int(np.linalg.matrix_rank(family.B)),
        full_dense_coupling=True, target_audit='exp(x^T B x)',
        options=options, halo_rule='ceil(sqrt(N)) per side = 17 for N=257',
        normalization='fixed zero-jet PDE differential sensitivities and domain length scales; independent raw audit',
        source_policy='PDE forcing and Dirichlet values only; zero cold start then native continuation; no interior labels or reference coefficients',
        interpretation='Smooth full-rank nonlinear dimension control; not a hard real-world high-dimensional PDE claim',
        resource_scope='Single thread; 600s soft solve budget; 1024MiB named arrays, not process RSS; 2million neurons; shared host wall time',
        heldout=dict(values=512, derivatives=97, boundary=257, corners=2**args.dimension, seed=573927),
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    dump(folder/'protocol.json', protocol)
    (folder/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    estimates = forecast(args.dimension)
    dump(folder/'forecast.json', estimates)
    log(f"\n### {args.run_id} — START {stamp()}\n\nFrozen {args.dimension}D full-rank coupled semilinear control. B={family.B.tolist()}; box coordinates, tensor angular rule, N257/lambda .2/halo17, p={DEGREES}; zero native start; fixed normalization; 8 outer/4000 inner, exact inner, damping1e-6; 600s soft budget; 1024MiB named arrays, 2million neurons. Forecast rejected degrees {[r['degree'] for r in estimates if r['initial_preflight_reject']]}. Interior truth confined to post-fit audit. Artifacts: `{folder.relative_to(ROOT)}`.")
    emit('forecast', run_id=args.run_id, dimensions=args.dimension,
         stages=[{key:r[key] for key in ('degree','neurons','directions','readout_coordinates','minimum_named_arrays_bytes','initial_preflight_reject')} for r in estimates])
    start = time.perf_counter()
    def callback(c, p, record):
        degree = next(k for k in DEGREES if comb(k+args.dimension, args.dimension)==len(c))
        row = dict(degree=degree, **record)
        with (folder/'iterations.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(row, default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
        np.savez_compressed(folder/f'p{degree}_latest.npz', coefficients=c, parameters=p)
        dump(folder/f'p{degree}_latest.json', row)
        emit('iteration', run_id=args.run_id, degree=degree, iteration=record.get('iteration'),
             objective=record.get('objective'), rms=record.get('maximum_scaled_block_rms'))
    options['solver_options']['iteration_callback'] = callback
    try:
        sol = solve_route2_native(family.declaration(), **options)
        features = sol.problem.features
        report = dict(run_id=args.run_id, dimension=args.dimension, status=sol.status,
            solver_status=sol.solution.status, degree=features.degree, neurons=features.tanh_count,
            directions=features.direction_count, readout_coordinates=features.size,
            halo_per_side=features.halo_per_side, solve_seconds=time.perf_counter()-start,
            history=sol.history, metrics=sol.metrics, final_inner_history=sol.solution.history)
        np.savez_compressed(folder/'coefficients.npz', coefficients=sol.coefficients, degree=features.degree)
        dump(folder/'result.json', report)
        audit_start = time.perf_counter()
        report['ordinary_audit'] = audit(family, sol, folder)
        report['audit_seconds'] = time.perf_counter()-audit_start
        report['seconds'] = time.perf_counter()-start
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        dump(folder/'result.json', report)
        a = report['ordinary_audit']
        log(f"\n**{args.run_id} FINISH {stamp()} — {sol.status}.** {args.dimension}D, returned p{features.degree}, {features.direction_count} directions, {features.tanh_count} ordinary tanh neurons. Ordinary field relL2 {a['ordinary_relative_l2']:.6g}; raw PDE RMS {a['ordinary_pde_rms']:.6g}; boundary max {a['boundary_max']:.6g}; corner max {a['corner_max']:.6g}; readout L1 {a['readout_l1']:.5g}. {report['seconds']:.1f}s including audit; peak RSS {report['peak_rss_bytes']/1024**2:.1f}MiB. Full stage memory/work and rejected levels saved; budget/resolution exits are not precision certificates.")
        emit('result', **{k:v for k,v in report.items() if k not in ('history','metrics','final_inner_history')})
    except BaseException as error:
        report = dict(run_id=args.run_id, status='interrupted' if isinstance(error,KeyboardInterrupt) else 'exception',
            error=repr(error), traceback=traceback.format_exc(), seconds=time.perf_counter()-start)
        dump(folder/'failure.json', report)
        log(f"\n**{args.run_id} FINISH {stamp()} — {report['status']}.** {error!r}; all checkpoints retained.")
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--dimension', type=int, choices=(3,4,5), required=True)
    run(parser.parse_args())
