"""Bounded architecture/precision audit of genuinely ordinary tanh depth.

The square modules and Legendre banks are constructed without target data.
The nonlinear PDE readout starts at zero and uses the common residual engine.
Manufactured forcing and boundary values are explicitly declared; exact
interior values are used only for the final independent audit.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quill-pure-depth-mpl')

import gc
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from numpy.polynomial import legendre
import torch

from solver.pure_mlp import PureMLPFeatures, analytic_jet, autograd_jet, architecture_metrics
from solver.general_residual import ResidualBlock, ResidualProblem, solve_residual

OUT = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/pure_mlp_depth'
EPS = np.finfo(float).eps


def polynomial_basis(features, points, derivative):
    columns = []
    for mode in features.multiindices:
        result = np.ones(len(points))
        for axis, degree in enumerate(mode):
            coefficient = np.zeros(degree+1)
            coefficient[-1] = 1.
            result *= legendre.legval(points[:,axis], legendre.legder(coefficient,m=derivative[axis]))
        columns.append(result)
    return np.array(columns).T


def product_gate_reference(features, points, derivative):
    """Audit-only matched leaf product; NEVER used by the neural solve."""
    leaves = [features.leaf_encoding.evaluate(points[:,axis], derivative=derivative[axis])
              for axis in range(features.dimension)]
    return np.column_stack([np.prod(np.column_stack([leaves[axis][:, degree] for axis,degree in enumerate(mode)]),axis=1)
                            for mode in features.multiindices])


def precision_sweep():
    points = np.r_[np.random.default_rng(381).uniform(-1,1,(73,2)),
                   np.array([[-1,-1],[-1,1],[1,-1],[1,1],[0,0]])]
    rows = []
    derivatives = [(0,0),(1,0),(2,0),(1,1)]
    for cells in (128,256,512):
        for lam in (.15,.2,.25):
            features = PureMLPFeatures([[-1,1]]*2,3,cells=cells,lam=lam)
            checks = []
            for derivative in derivatives:
                actual = features.evaluate(points, derivative)
                exact = polynomial_basis(features, points, derivative)
                product = product_gate_reference(features, points, derivative)
                raw_ad = autograd_jet(features.network,points[:7],derivative).numpy()
                checks.append(dict(derivative=list(derivative),
                    maximum_error_vs_polynomial=float(np.max(abs(actual-exact))),
                    maximum_error_vs_same_quill_product=float(np.max(abs(actual-product))),
                    raw_torch_ad_vs_stable_chain_max=float(np.max(abs(raw_ad-actual[:7])))))
            rows.append(dict(cells=cells,lam=lam,checks=checks,features=features.metrics))
            print(json.dumps(dict(event='square_precision',cells=cells,lam=lam,
                                   errors=[c['maximum_error_vs_polynomial'] for c in checks])),flush=True)
            del features
            gc.collect()
    return rows


def depth_controls():
    rows = []
    for dimension in (2,4,8):
        modes = np.array([[0]*dimension,[1]*dimension])
        features = PureMLPFeatures([[-1,1]]*dimension,dimension,cells=256,lam=.2,multiindices=modes)
        points = np.r_[np.random.default_rng(769).uniform(-1,1,(31,dimension)),np.ones((1,dimension))]
        checks = []
        for derivative in ((0,)*dimension,(1,)+(0,)*(dimension-1),
                           (2,)+(0,)*(dimension-1),(1,1)+(0,)*(dimension-2)):
            actual = features.evaluate(points,derivative)
            expected = polynomial_basis(features,points,derivative)
            checks.append(dict(derivative=list(derivative),maximum_error=float(np.max(abs(actual-expected)))))
        rows.append(dict(dimension=dimension,checks=checks,features=features.metrics,
                         scope='Only constant and the product of all coordinates, not a full polynomial bank'))
        del features
        gc.collect()
    return rows


def truth(points):
    x,y = points[:,0],points[:,1]
    return .3+.1*x+.2*y+.1*x*y+.05*(x*x+y*y)


def pde_from_zero():
    started = time.perf_counter()
    features = PureMLPFeatures([[-1,1]]*2,4,cells=256,lam=.2)
    rng = np.random.default_rng(1229)
    interior = rng.uniform(-1,1,(241,2))
    faces = []
    for axis in range(2):
        for side in (-1.,1.):
            points = rng.uniform(-1,1,(41,2))
            points[:,axis] = side
            faces.append(points)
    boundary = np.concatenate(faces)
    def equation(points,jets,parameters):
        u = jets[(0,0)][:,0]
        force = -.2+5*truth(points)**3
        return -jets[(2,0)][:,0]-jets[(0,2)][:,0]+5*u**3-force
    def boundary_equation(points,jets,parameters):
        return jets[(0,0)][:,0]-truth(points)
    blocks = [ResidualBlock('nonlinear_equation',interior,equation,((0,0),(2,0),(0,2))),
              ResidualBlock('dirichlet',boundary,boundary_equation,((0,0),))]
    problem = ResidualProblem(features,blocks)
    solution = solve_residual(problem,tolerance=2e-13,max_iterations=30,high_accuracy=True,
                              preconditioner='block',max_seconds=120)
    total = time.perf_counter()-started
    exported = features.export_readout(solution.coefficients)
    fresh = np.random.default_rng(19271).uniform(-1,1,(257,2))
    with torch.no_grad():
        predicted = exported(torch.tensor(fresh)).numpy()[:,0]
    expected = truth(fresh)
    jets = {d:solution.evaluate(fresh,d) for d in ((0,0),(2,0),(0,2))}
    stable_residual = -jets[(2,0)][:,0]-jets[(0,2)][:,0]+5*jets[(0,0)][:,0]**3-(-.2+5*expected**3)
    # Raw AD differentiates the exported nn.Sequential, never a product reference.
    ad_points = np.r_[fresh[:61], np.array([[-1,-1],[-1,1],[1,-1],[1,1]])]
    ad_value = autograd_jet(exported,ad_points,(0,0)).numpy()[:,0]
    ad_xx = autograd_jet(exported,ad_points,(2,0)).numpy()[:,0]
    ad_yy = autograd_jet(exported,ad_points,(0,2)).numpy()[:,0]
    ad_residual = -ad_xx-ad_yy+5*ad_value**3-(-.2+5*truth(ad_points)**3)
    stable_exported_residual = (-analytic_jet(exported,ad_points,(2,0)).numpy()[:,0]
                               -analytic_jet(exported,ad_points,(0,2)).numpy()[:,0]
                               +5*ad_value**3-(-.2+5*truth(ad_points)**3))
    # Fresh independent boundary points, including corners.
    line = np.linspace(-1,1,113)
    fresh_boundary = np.r_[np.c_[line,-np.ones_like(line)],np.c_[line,np.ones_like(line)],
                           np.c_[-np.ones_like(line),line],np.c_[np.ones_like(line),line]]
    with torch.no_grad():
        boundary_error = exported(torch.tensor(fresh_boundary)).numpy()[:,0]-truth(fresh_boundary)
    traced = torch.jit.trace(exported,torch.tensor(fresh[:3]))
    operators = sorted({node.kind() for node in traced.inlined_graph.nodes()})
    if not set(operators) <= {'prim::GetAttr','aten::linear','aten::tanh'}:
        raise AssertionError(f'Unexpected exported graph nodes: {operators}')
    model_path = OUT/'nonlinear_pde_tanh_mlp.pt'
    torch.jit.save(traced,str(model_path))
    loaded = torch.jit.load(str(model_path))
    with torch.no_grad():
        serialization_error = float(np.max(abs(loaded(torch.tensor(fresh[:7])).numpy()-exported(torch.tensor(fresh[:7])).numpy())))
    np.savez_compressed(OUT/'nonlinear_pde_coefficients.npz',coefficients=solution.coefficients,
                        multiindices=features.multiindices)
    grid = np.stack(np.meshgrid(np.linspace(-1,1,61),np.linspace(-1,1,61),indexing='ij'),axis=-1).reshape(-1,2)
    # Keep peak activation allocation bounded, including plotting inference.
    values = []
    with torch.no_grad():
        for first in range(0,len(grid),64):
            values.append(exported(torch.tensor(grid[first:first+64])).numpy()[:,0])
    values = np.concatenate(values)
    return dict(equation='-Delta(u)+5u^3=-0.2+5u_exact^3 on [-1,1]^2',
                exact_profile='0.3+0.1*x+0.2*y+0.1*x*y+0.05*(x^2+y^2)',
                manufactured_forcing=True,initial_coefficients='all zero',interior_labels=False,
                readout_optimization='Common matrix-free damped Gauss-Newton/LSMR; hidden affine/tanh construction fixed; no Adam/LBFGS and no solution-label initialization',
                constraints='Sampled soft Dirichlet residual on all four faces',
                genuine_interior_freedom='Degree4 contains (1-x^2)*(1-y^2), which vanishes on the entire boundary; boundary data alone do not determine the field',
                solution_metrics=solution.metrics,history=solution.history,
                setup_and_solve_seconds=total,feature_construction_seconds=features.metrics['setup_seconds'],
                architecture=architecture_metrics(exported),graph_operator_kinds=operators,
                serialized_model_bytes=model_path.stat().st_size,serialization_max_error=serialization_error,
                ordinary_forward_relative_l2=float(np.linalg.norm(predicted-expected)/np.linalg.norm(expected)),
                ordinary_forward_max_error=float(np.max(abs(predicted-expected))),
                fresh_boundary_max_error=float(np.max(abs(boundary_error))),
                stable_chain_fresh_residual_rms=float(np.sqrt(np.mean(stable_residual**2))),
                stable_chain_fresh_residual_max=float(np.max(abs(stable_residual))),
                raw_torch_ad_exported_residual_rms=float(np.sqrt(np.mean(ad_residual**2))),
                raw_torch_ad_exported_residual_max=float(np.max(abs(ad_residual))),
                stable_chain_exported_residual_rms=float(np.sqrt(np.mean(stable_exported_residual**2))),
                stable_chain_exported_residual_max=float(np.max(abs(stable_exported_residual))),
                ordinary_forward_vs_basis_readout_max=float(np.max(abs(predicted-solution.evaluate(fresh)[:,0]))),
                error_epsilon_units='Errors are absolute unless labeled relative; epsilon='+str(EPS)),(grid,values,values-truth(grid))


def four_input_cost_estimate():
    # The full degree4 space has C(4+4,4)=70 modes and needs four-factor trees.
    # Count dense affine storage without allocating any of these matrices.
    # N=257 interior centers, ceil(sqrt(N))=17 halo neurons on each side.
    width,features,dimension,arity = 291,70,4,4
    shapes = [(dimension,dimension*width),(dimension*width,features*arity)]
    hidden = [dimension*width]
    while arity > 1:
        products = features*(arity//2)
        shapes.extend(((features*arity,products*2*width),(products*2*width,products)))
        hidden.append(products*2*width)
        arity //= 2
    parameters = sum((a+1)*b for a,b in shapes)
    return dict(scope='Algebraic architecture estimate only, not built or run',feature_count=features,
                tanh_units=sum(hidden),stored_parameters_without_pde_readout=parameters,
                parameter_bytes_without_pde_readout=8*parameters,widest_layer=max(hidden),
                one_1000_point_float64_activation_array_bytes=8*1000*max(hidden),
                notes='Second-derivative propagation needs several intermediate arrays; total peak would exceed1 GB without chunking. No 4D PDE trial was run.')


def standalone_pde_resource_check():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    metrics,_ = pde_from_zero()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    result = dict(scope='Separate fresh process, nonlinear PDE construction + solve + fresh field/boundary/AD audits + TorchScript serialization + grid inference; no precision sweep',
                  peak_rss_bytes=rss,elapsed_seconds=time.perf_counter()-started,
                  setup_and_solve_seconds=metrics['setup_and_solve_seconds'],
                  parameter_bytes=metrics['architecture']['parameter_bytes'],
                  basis_cache_bytes=metrics['solution_metrics']['basis_cache_bytes'],
                  status=metrics['solution_metrics']['status'],
                  ordinary_forward_relative_l2=metrics['ordinary_forward_relative_l2'])
    (OUT/'pde_only_resources.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def run():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    sweep = precision_sweep()
    depths = depth_controls()
    pde,plot_data = pde_from_zero()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        peak_rss *= 1024
    record = dict(scope='Small fixed ordinary-tanh construction and generic nonlinear residual solve; no general speed or machine-epsilon PDE claim',
                  source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ('pure_mlp_depth_study.py','solver/pure_mlp.py','solver/general_residual.py')},
                  arithmetic='float64',machine_epsilon=EPS,precision_sweep=sweep,depth_controls=depths,
                  nonlinear_pde=pde,total_seconds=time.perf_counter()-started,process_peak_rss_bytes=peak_rss,
                  four_input_full_degree4_cost_estimate=four_input_cost_estimate(),
                  memory_scope='Whole-process peak includes Python/Torch/BLAS, construction, derivative verification and plotting preparations; architecture parameter bytes separately measure actual dense stored weights',
                  limits=['No exact multiplication by finite tanh modules; multiplication and derivatives are approximations.',
                          'No PDE solution or interior labels initialize the network; the readout is nevertheless solved iteratively.',
                          'Full multivariate feature counts and dense layer weights grow rapidly. Selected deep-product controls are not full high-dimensional polynomial spaces.',
                          'Consecutive affine bottlenecks are ordinary MLP layers; fusing them into strictly alternating layers can greatly increase stored dense weights.',
                          'Stable analytic chain-rule jets and raw Torch autograd jets are reported separately.'])
    (OUT/'metrics.json').write_text(json.dumps(record,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(13,3.7))
    for derivative,label in [((0,0),'value'),((1,0),'first derivative'),((2,0),'second derivative')]:
        selected = [row for row in sweep if row['lam'] == .2]
        axes[0].loglog([row['cells'] for row in selected],
                      [next(c['maximum_error_vs_polynomial'] for c in row['checks'] if tuple(c['derivative']) == derivative) for row in selected],
                      'o-',label=label)
    axes[0].set(xlabel='Square/leaf cells (lambda=0.2)',ylabel='Maximum error',title='Ordinary tanh product features')
    axes[0].legend(fontsize=8)
    grid,values,error = plot_data
    for axis,data,title in zip(axes[1:],(values,error),('Nonlinear PDE: MLP solution','Ordinary-forward error')):
        im=axis.imshow(data.reshape(61,61).T,origin='lower',extent=(-1,1,-1,1),aspect='equal',cmap='viridis' if axis is axes[1] else 'coolwarm')
        axis.set(xlabel='x',ylabel='y',title=title)
        fig.colorbar(im,ax=axis,shrink=.8)
    fig.tight_layout()
    fig.savefig(OUT/'precision_and_pde.png',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(event='complete',output=str(OUT),pde={k:v for k,v in pde.items() if k not in ('history','solution_metrics')},
                          solver_status=pde['solution_metrics']['status'],peak_rss_bytes=peak_rss,total_seconds=record['total_seconds'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pde-only',action='store_true',help='Fresh-process PDE resource audit without the feature/depth sweeps')
    args = parser.parse_args()
    standalone_pde_resource_check() if args.pde_only else run()
