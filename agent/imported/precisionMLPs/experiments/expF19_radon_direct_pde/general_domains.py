"""Curved, multiply connected domain checks for the reusable annular backend.

Manufactured analytic fields define known forcing and verification values.
The solver receives only the equation, coefficient, forcing, and boundary lift;
it never receives interior solution labels or an external numerical trajectory.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/general-domains-mpl')
from pathlib import Path
import json
import resource
import time
import numpy as np
from scipy.stats import qmc
from solver.elliptic import Annulus, AnnulusProblem, solve_annulus, zero_lift

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver'


def boundary_lift(domain):
    def lift(r, theta):
        w = (r-domain.inner)/(domain.outer-domain.inner)
        inner = -.1+.05*np.cos(3*theta)
        outer = .15*np.cos(theta)+.1*np.sin(2*theta)
        inner_t = -.15*np.sin(3*theta)
        outer_t = -.15*np.sin(theta)+.2*np.cos(2*theta)
        inner_tt = -.45*np.cos(3*theta)
        outer_tt = -.15*np.cos(theta)-.4*np.sin(2*theta)
        return ((1-w)*inner+w*outer, (outer-inner)/(domain.outer-domain.inner),
                (1-w)*inner_t+w*outer_t, np.zeros_like(r), (1-w)*inner_tt+w*outer_tt)
    return lift


def prescribed_jet(domain, xy, lift, mixed=False):
    """Closed analytic expression used to specify manufactured forcing."""
    r, theta = domain.polar(xy)
    width = domain.outer-domain.inner
    q = (2*r-domain.inner-domain.outer)/width
    c = .35*2/width
    mask = (domain.outer-r)/width if mixed else (r-domain.inner)*(domain.outer-r)/width**2
    dm = np.full_like(r, -1/width) if mixed else (domain.inner+domain.outer-2*r)/width**2
    d2m = 0. if mixed else -2/width**2
    e = 3*np.exp(.35*q)
    radial = e*mask
    radial_r = e*(dm+c*mask)
    radial_rr = e*(d2m+2*c*dm+c*c*mask)
    bb = 1+.15*np.sin(3*theta)
    bt = .45*np.cos(3*theta)
    btt = -1.35*np.sin(3*theta)
    angular_exp = np.exp(.2*np.cos(theta))
    angular = angular_exp*bb
    angular_t = angular_exp*(bt-.2*np.sin(theta)*bb)
    angular_tt = angular_exp*(btt-.4*np.sin(theta)*bt+(.04*np.sin(theta)**2-.2*np.cos(theta))*bb)
    g, gr, gt, grr, gtt = lift(r, theta)
    u = radial*angular+g
    ur = radial_r*angular+gr
    ut = radial*angular_t+gt
    lap = radial_rr*angular+ur/r+radial*angular_tt/r**2+grr+gtt/r**2
    ux = np.cos(theta)*ur-np.sin(theta)*ut/r
    uy = np.sin(theta)*ur+np.cos(theta)*ut/r
    return u, np.column_stack((ux, uy)), lap


def make_problem(name='homogeneous', contrast=None):
    domain = Annulus(.4, 1.2)
    kappa = .8 if contrast is None else np.log(contrast)/(2*domain.outer)
    beta = 5. if contrast is None else 50.
    lift = zero_lift if name == 'homogeneous' else boundary_lift(domain)
    mixed = name == 'mixed_robin'
    def coefficient(xy):
        return np.exp(kappa*np.asarray(xy)[:, 0])
    def coefficient_gradient(xy):
        a = coefficient(xy)
        return np.column_stack((kappa*a, np.zeros_like(a)))
    def forcing(xy):
        u, grad, lap = prescribed_jet(domain, xy, lift, mixed)
        a = coefficient(xy)
        return -a*lap-kappa*a*grad[:, 0]+beta*u**3
    problem = AnnulusProblem(domain, coefficient, coefficient_gradient, forcing, beta, lift, name)
    if mixed:
        def inner_robin(theta):
            xy = domain.inner*np.column_stack((np.cos(theta), np.sin(theta)))
            u, grad, _ = prescribed_jet(domain, xy, lift, mixed=True)
            radial_derivative = np.sum(grad*np.column_stack((np.cos(theta), np.sin(theta))), axis=1)
            alpha = 1+.2*np.cos(theta)
            return alpha, -coefficient(xy)*radial_derivative+alpha*u
        problem.inner_robin = inner_robin
    return problem, lambda xy: prescribed_jet(domain, xy, lift, mixed)[0]


def validation_points(domain, count_power=10):
    z = qmc.Sobol(2, scramble=True, seed=4195).random_base2(count_power)
    r = np.sqrt(domain.inner**2+(domain.outer**2-domain.inner**2)*z[:, 0])
    theta = 2*np.pi*z[:, 1]
    return np.column_stack((r*np.cos(theta), r*np.sin(theta)))


def boundary_points(domain):
    theta = np.linspace(0, 2*np.pi, 501, endpoint=False)
    return np.vstack([np.column_stack((r*np.cos(theta), r*np.sin(theta))) for r in (domain.inner, domain.outer)])


def validate(solution, truth, points):
    expected = truth(points)
    fields = solution.evaluate(points, True)
    boundary = boundary_points(solution.problem.domain)
    if solution.problem.inner_robin is not None:
        boundary = boundary[len(boundary)//2:]
    rb, tb = solution.problem.domain.polar(boundary)
    given = solution.problem.boundary_lift(rb, tb)[0]
    residual = solution.residual(points)
    row = dict(solution.metrics)
    row.update(relative_l2=float(np.linalg.norm(fields['value']-expected)/np.linalg.norm(expected)),
               offgrid_relative_strong_residual=float(np.linalg.norm(residual)/np.linalg.norm(solution.problem.forcing(points))),
               boundary_max_abs=float(np.max(abs(solution.evaluate(boundary)-given))),
               validation_points=len(points))
    if solution.problem.inner_robin is not None:
        theta = 2*np.pi*np.arange(503)/503
        normals = -np.column_stack((np.cos(theta), np.sin(theta)))
        inner = -solution.problem.domain.inner*normals
        field = solution.evaluate(inner, True)
        alpha, target = solution.problem.inner_robin(theta)
        residual = solution.problem.coefficient(inner)*np.sum(normals*field['gradient'], axis=1)+alpha*field['value']-target
        row['inner_robin_relative_residual'] = float(np.linalg.norm(residual)/np.linalg.norm(target))
    return row, fields['value']


def run(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    problem, truth = make_problem()
    points = validation_points(problem.domain)
    rows, solutions, states = [], [], []
    for p, m in [(4, 3), (6, 5), (8, 7), (10, 9)]:
        solution = solve_annulus(problem, p, m, n_centers=max(129, 24*max(p, m)+1))
        row, pred = validate(solution, truth, points)
        print(json.dumps(dict(domain_native={k:v for k,v in row.items() if k not in ('residual_history','cg_iterations')})), flush=True)
        rows.append(row)
        solutions.append(solution)
        states.append(pred)

    # Matching classical basis solves happen after production native solves.
    for row, solution in zip(rows, solutions):
        classical = solve_annulus(problem, solution.basis.radial_degree, solution.basis.angular_degree, backend='classical')
        matched, pred = validate(classical, truth, points)
        row['matched_classical'] = matched
        row['native_vs_classical_relative'] = float(np.linalg.norm(solution.evaluate(points)-pred)/np.linalg.norm(truth(points)))

    extras = []
    for name, contrast in [('inhomogeneous', None), ('high_contrast', 1e4), ('mixed_robin', None)]:
        equation, exact = make_problem(name, contrast)
        native = solve_annulus(equation, 10, 9, n_centers=241)
        row, pred = validate(native, exact, points)
        classical = solve_annulus(equation, 10, 9, backend='classical')
        comparison, cpred = validate(classical, exact, points)
        row['matched_classical'] = comparison
        row['native_vs_classical_relative'] = float(np.linalg.norm(pred-cpred)/np.linalg.norm(exact(points)))
        if contrast is not None:
            weak_pre = solve_annulus(equation, 10, 9, n_centers=241, preconditioner='diagonal', max_cg=30)
            row['limited_diagonal_preconditioner_control'] = weak_pre.metrics
            row['prescribed_global_contrast'] = contrast
        extras.append(row)
        print(json.dumps(dict(domain_extra={k:v for k,v in row.items() if k not in ('matched_classical','residual_history','cg_iterations','limited_diagonal_preconditioner_control')})), flush=True)

    # Independently verify manufactured derivatives by Cartesian finite differences.
    check = points[:32]
    h = 2e-4
    center, gradient, lap = prescribed_jet(problem.domain, check, problem.boundary_lift)
    fd_gradient = []
    fd_lap = np.zeros_like(center)
    for d in range(2):
        offset = np.zeros_like(check); offset[:, d] = h
        plus, minus = truth(check+offset), truth(check-offset)
        fd_gradient.append((plus-minus)/(2*h))
        fd_lap += (plus-2*center+minus)/h**2
    derivative_check = dict(gradient_relative=float(np.linalg.norm(np.array(fd_gradient).T-gradient)/np.linalg.norm(gradient)),
                            laplacian_relative=float(np.linalg.norm(fd_lap-lap)/np.linalg.norm(lap)), step=h)
    finest = solutions[-1]
    neural = finest.evaluate(check, True)
    native_fd_grad = []
    native_fd_lap = np.zeros(len(check))
    for d in range(2):
        offset = np.zeros_like(check); offset[:, d] = h
        plus, minus = finest.evaluate(check+offset), finest.evaluate(check-offset)
        native_fd_grad.append((plus-minus)/(2*h))
        native_fd_lap += (plus-2*neural['value']+minus)/h**2
    derivative_check['actual_neural_gradient_relative'] = float(np.linalg.norm(np.array(native_fd_grad).T-neural['gradient'])/np.linalg.norm(neural['gradient']))
    derivative_check['actual_neural_laplacian_relative'] = float(np.linalg.norm(native_fd_lap-neural['laplacian'])/np.linalg.norm(neural['laplacian']))
    # Quadrature refinement tests equation integration independently of basis size.
    refined = solve_annulus(problem, 10, 9, n_centers=241, quadrature_orders=(66, 102))
    quadrature_change = float(np.linalg.norm(finest.evaluate(points)-refined.evaluate(points))/np.linalg.norm(truth(points)))
    result = dict(method='Actual QUILL tensor-product Galerkin fields on a mapped annulus; damped Newton-CG with small diffusion PDE preconditioner',
                  equation='-div(a(x,y) grad u) + beta u^3 = f',
                  domain=dict(kind='annulus', inner=problem.domain.inner, outer=problem.domain.outer,
                              curved=True, convex=False, multiply_connected=True),
                  manufactured_solution='g + 3[(r-ri)(ro-r)/(ro-ri)^2] exp(.35 q + .2 cos(theta)) [1+.15 sin(3theta)]',
                  mixed_robin_manufactured_solution='Replace the radial boundary factor by (ro-r)/(ro-ri), so the inner trace is nonzero',
                  no_solution_labels=True, forcing_is_prescribed_manufactured_function=True,
                  no_neural_readout_least_squares=True, tractable_PDE_coefficient_solves=True,
                  exact_boundary_mechanism='Known boundary lift plus a radial factor vanishing on Dirichlet circles; inner Robin is imposed through its weak boundary integral',
                  exact_angular_periodicity='QUILL polynomial functions of cos(theta), with sin(theta) product factors',
                  derivative_check=derivative_check, quadrature_refinement_relative=quadrature_change,
                  rows=rows, extra_cases=extras, wall_seconds=time.perf_counter()-began,
                  peak_process_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope=['Smooth positive scalar diffusion on an annulus with known coefficient gradient',
                         'Monotone cubic reaction beta>=0',
                         'Homogeneous/inhomogeneous Dirichlet data, or inner Robin with nonnegative alpha and outer Dirichlet',
                         'Polar map and product-gated neural architecture; not a single hidden-layer tanh network',
                         'Small global PDE coefficient matrix is factored for preconditioning; no neuron readout least squares',
                         'No claim of arbitrary geometry, pure Neumann problems, shocks, high-dimensional scaling, or speed superiority'])
    (out/'domain_metrics.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(out/'domain_state.npz', coefficients=finest.coefficients,
                        radial_centers=finest.basis.enc_radial.centers, radial_gamma=finest.basis.enc_radial.gamma,
                        radial_weights=finest.basis.enc_radial.weights, radial_bias=finest.basis.enc_radial.bias,
                        angular_centers=finest.basis.enc_angular.centers, angular_gamma=finest.basis.enc_angular.gamma,
                        angular_weights=finest.basis.enc_angular.weights, angular_bias=finest.basis.enc_angular.bias,
                        validation_points=points, prediction=states[-1], exact=truth(points))
    assert all(r['converged'] for r in rows+extras)
    assert rows[-1]['relative_l2'] < rows[0]['relative_l2']/1000
    assert max(r['boundary_max_abs'] for r in rows+extras) < 1e-12
    assert derivative_check['laplacian_relative'] < 1e-5
    assert derivative_check['actual_neural_laplacian_relative'] < 1e-5
    assert quadrature_change < 1e-8
    assert extras[1]['limited_diagonal_preconditioner_control']['failure'] == 'cg_iteration_limit'
    assert extras[2]['inner_robin_relative_residual'] < 1e-8

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    re = np.linspace(problem.domain.inner, problem.domain.outer, 81)
    te = np.linspace(0,2*np.pi,161)
    rr, tt = np.meshgrid((re[:-1]+re[1:])/2, (te[:-1]+te[1:])/2, indexing='ij')
    r_edge, t_edge = np.meshgrid(re, te, indexing='ij')
    xy = np.column_stack((rr.ravel()*np.cos(tt.ravel()),rr.ravel()*np.sin(tt.ravel())))
    field = finest.evaluate(xy).reshape(rr.shape)
    image = axes[0].pcolormesh(r_edge*np.cos(t_edge),r_edge*np.sin(t_edge),field,shading='flat',cmap='viridis')
    axes[0].set(aspect='equal',title='Neural field on a nonconvex annulus',xlabel='x',ylabel='y')
    fig.colorbar(image,ax=axes[0],label='u')
    size = [r['latent_coefficients'] for r in rows]
    axes[1].loglog(size,[r['relative_l2'] for r in rows],'o-',label='Off-grid value error')
    axes[1].loglog(size,[r['offgrid_relative_strong_residual'] for r in rows],'s-',label='Off-grid PDE residual')
    axes[1].set(title='Refine the same neural PDE solver',xlabel='PDE coefficients',ylabel='Relative sampled norm')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    axes[2].semilogy(rows[-1]['residual_history'],'o-',label='Variable a, zero BC')
    for row in extras:
        axes[2].semilogy(row['residual_history'],'o-',label=row['name'])
    axes[2].set(title='Nonlinear solves from equation data',xlabel='Newton residual evaluation',ylabel='Relative weak residual')
    axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.savefig(out/'domain_comparison.png',dpi=175)
    plt.close(fig)
    result['wall_seconds'] = time.perf_counter()-began
    result['peak_process_rss_bytes_macos'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out/'domain_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(domain_summary={k:v for k,v in result.items() if k not in ('rows','extra_cases')})),flush=True)
    return result


if __name__ == '__main__':
    run()
