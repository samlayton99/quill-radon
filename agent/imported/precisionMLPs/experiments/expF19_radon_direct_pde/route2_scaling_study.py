"""Cold nonlinear PDE scaling, separate from known-function construction.

Manufactured source and boundary data define -Delta u + u^3 = s. Interior
truth is used only after solving. Every case starts at exactly zero; the
export is audited through ordinary PyTorch affine/tanh operations and AD.
Run one case per process to make peak RSS meaningful.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import json
import math
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch
from scipy.stats import qmc
from solver.ball_ridge_features import BallRidgeFeatures
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual
from route2_scaling_features import ActiveAxisRidgeFeatures, LowDegreeSparseRidgeFeatures

OUT = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/scaling'


def exact(x, axes, family):
    z = x[:, axes]
    if family == 'quadratic':
        return .5 + .1*z.sum(dim=1)/math.sqrt(len(axes)) + .2*(z*z).mean(dim=1)
    if family == 'dense_quadratic':
        s=z.sum(dim=1)/math.sqrt(len(axes))
        return .5+.1*s+.15*(z*z).mean(dim=1)+.05*s*s
    if family == 'dense_cubic':
        s=z.sum(dim=1)/math.sqrt(len(axes))
        return .5+.1*s+.15*(z*z).mean(dim=1)+.05*s*s+.02*s**3
    return .5*torch.exp(.25*z.sum(dim=1)/math.sqrt(len(axes)))


def forcing(x, axes, family):
    value = exact(x, axes, family)
    laplacian = torch.full_like(value, .4) if family in ('quadratic','dense_quadratic') else .25**2*value
    if family == 'dense_cubic':
        laplacian=.4+.12*x[:,axes].sum(dim=1)/math.sqrt(len(axes))
    return -laplacian + value**3


def sample(d, n, seed):
    return 2*qmc.Sobol(d, scramble=True, seed=seed).random_base2(math.ceil(math.log2(n)))[:n]-1


def sample_boundary(dimension, axes, count, mode='independent_faces'):
    if mode == 'legacy_interleaved':
        points=sample(dimension,count,6241)
        for j in range(count):
            points[j,axes[(j//2)%len(axes)]]=-1. if j%2==0 else 1.
        return points
    blocks=[]
    faces=2*len(axes)
    for face in range(faces):
        n=count//faces+(face<count%faces)
        if n:
            block=sample(dimension,n,6241+7919*face)
            block[:,axes[face//2]]=-1. if face%2==0 else 1.
            blocks.append(block)
    return np.concatenate(blocks)


def raw_residual(model, points, axes, family):
    x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    u = model(x)[:, 0]
    grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    lap = torch.zeros_like(u)
    for axis in range(x.shape[1]):
        lap += torch.autograd.grad(grad[:, axis].sum(), x, retain_graph=True)[0][:, axis]
    return (-lap + u**3 - forcing(x, axes, family)).detach().numpy()


def run(dimension, degree, centers=65, lam=.2, family='quadratic', active_dimension=None, sparse_angular=False, tolerance=1e-11, boundary_sampling='independent_faces'):
    torch.set_num_threads(1)
    started = time.perf_counter()
    axes = tuple(range(dimension if active_dimension is None else active_dimension))
    bounds = np.tile([-1., 1.], (dimension, 1))
    if sparse_angular and active_dimension is not None:
        raise ValueError('Keep active-axis and sparse angular comparisons separate')
    features = (LowDegreeSparseRidgeFeatures(bounds, degree, centers, lam) if sparse_angular else
                BallRidgeFeatures(bounds, degree, centers, lam) if active_dimension is None else
                ActiveAxisRidgeFeatures(bounds, axes, degree, centers, lam))
    construction_seconds = time.perf_counter()-started
    zero = (0,)*dimension
    seconds = tuple(tuple(2 if j == i else 0 for j in range(dimension)) for i in axes)
    n_interior = max(128, 4*features.size)
    n_boundary = max(64, 2*features.size)
    interior = sample(dimension, n_interior, 9821)
    boundary = sample_boundary(dimension,axes,n_boundary,boundary_sampling)
    # With a declared active-axis prior, inactive faces have homogeneous
    # Neumann conditions identically; no function labels are added there.
    def equation(x, jets, parameters):
        return -sum(jets[o][:, 0] for o in seconds) + jets[zero][:, 0]**3 - forcing(x, axes, family)
    def trace(x, jets, parameters):
        return jets[zero][:, 0]-exact(x, axes, family)
    problem = ResidualProblem(features, [
        ResidualBlock('physics', interior, equation, (zero,)+seconds),
        ResidualBlock('Dirichlet_active_faces', boundary, trace, (zero,)),
    ])
    before_solve = time.perf_counter()
    solved = solve_residual(problem, initial_coefficients=None, max_iterations=12,
                            tolerance=tolerance, high_accuracy=True, preconditioner='block',
                            lsmr_max_iterations=600, max_seconds=60)
    solve_seconds = time.perf_counter()-before_solve
    model = features.torch_model(solved.coefficients)
    test = sample(dimension, 1009, 8273)
    before_inference = time.perf_counter()
    with torch.no_grad():
        # Chunking bounds activation storage for the largest widths.
        actual = torch.cat([model(torch.tensor(x, dtype=torch.float64)) for x in np.array_split(test, 32)])[:, 0].numpy()
    inference_seconds = time.perf_counter()-before_inference
    truth = exact(torch.tensor(test), axes, family).numpy()
    ad_count = 7 if model[0].out_features > 200000 else 19
    ad_points = sample(dimension, ad_count, 17382)
    before_ad = time.perf_counter()
    residual = raw_residual(model, ad_points, axes, family)
    ad_seconds = time.perf_counter()-before_ad
    compiled = features.compile(solved.coefficients)
    dense_model_bytes = sum(v.nbytes for v in compiled.values())
    inner = getattr(features, 'inner', features)
    row = dict(dimension=dimension, active_dimension=len(axes), degree=degree,
        family=family, centers=centers, lam=lam, coordinates=features.size,
        angular_rule=('signed_sparse_degree6' if degree==3 else 'signed_sparse_degree4') if sparse_angular else 'tensor_product',
        directions=len(inner.directions), neurons=model[0].out_features,
        angular_weights_total_variation=float(abs(inner.angular_weights).sum()),
        scope='Cold nonlinear PDE, not known-function encoding',
        prior='none beyond total-degree space' if active_dimension is None else 'Declared active axes; not learned manifold',
        initial_coefficients='all zero', interior_solution_labels_used=False,
        boundary='Dirichlet on all active faces; homogeneous Neumann on inactive faces',
        boundary_sampling=boundary_sampling,
        source='Manufactured from reference; exact interior values used only for validation',
        architecture=['Linear','Tanh','Linear'],
        hardware=dict(machine=platform.machine(), host='Mac mini Apple M4, 10 cores, 16 GB', threads=1),
        construction_seconds=construction_seconds, solve_seconds=solve_seconds,
        feature_cache_seconds=solved.metrics['basis_setup_seconds'],
        cached_feature_jet_bytes=solved.metrics['basis_cache_bytes'],
        profile_map_bytes=inner.profile_map.nbytes, dense_model_bytes=dense_model_bytes,
        inference_1009_seconds=inference_seconds, ordinary_ad_seconds=ad_seconds, ordinary_ad_points=ad_count,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        interior_points=n_interior, boundary_points=n_boundary,
        status=solved.status, iterations=solved.metrics['iterations'],
        relative_l2=float(np.linalg.norm(actual-truth)/np.linalg.norm(truth)),
        maximum_error=float(abs(actual-truth).max()),
        ordinary_mlp_pde_rms=float(np.sqrt(np.mean(residual**2))),
        seconds=time.perf_counter()-started, solver=solved.metrics)
    OUT.mkdir(parents=True, exist_ok=True)
    stem=f'd{dimension}_active{len(axes)}_p{degree}_n{centers}_{family}'+('_sparse' if sparse_angular else '')
    (OUT/(stem+'.json')).write_text(json.dumps(row, indent=2)+'\n')
    # Store solved coordinates and architecture, not multi-megabyte duplicate
    # networks for every scaling point. Construction is fully reproducible.
    np.savez_compressed(OUT/(stem+'.npz'), coefficients=solved.coefficients)
    print(json.dumps({k:v for k,v in row.items() if k != 'solver'}), flush=True)
    return row


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', type=int, required=True)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--centers', type=int, default=65)
    parser.add_argument('--lam', type=float, default=.2)
    parser.add_argument('--family', choices=['quadratic','dense_quadratic','dense_cubic','exponential'], default='quadratic')
    parser.add_argument('--active-dimension', type=int)
    parser.add_argument('--sparse-angular', action='store_true')
    parser.add_argument('--tolerance', type=float, default=1e-11)
    parser.add_argument('--boundary-sampling', choices=['independent_faces','legacy_interleaved'], default='independent_faces')
    args=parser.parse_args()
    run(args.dimension, args.degree, args.centers, args.lam, args.family, args.active_dimension, args.sparse_angular, args.tolerance,args.boundary_sampling)
