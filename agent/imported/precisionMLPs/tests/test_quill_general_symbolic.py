"""Symbolic differential declarations agree with independent Torch calculus."""
from pathlib import Path
import sys

import numpy as np
import pytest
import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from solver.general_symbolic import DifferentialResidual
from solver.general_features import ConstructedFeatures
from solver.general_residual import ResidualProblem, linearize_residual, residual_vector, solve_residual


def exact_jets(points, values, derivatives):
    if values.ndim == 1:
        values = values[:, None]
    dimension = points.shape[1]
    cache = {(0,)*dimension: values}
    def get(order):
        if order not in cache:
            axis = next(j for j,k in enumerate(order) if k)
            parent = list(order)
            parent[axis] -= 1
            previous = get(tuple(parent))
            columns = []
            for field in range(previous.shape[1]):
                value = previous[:, field]
                gradient = (torch.autograd.grad(value.sum(), points, create_graph=True,
                                                retain_graph=True, allow_unused=True)[0]
                            if value.requires_grad else None)
                columns.append(torch.zeros_like(points[:, axis]) if gradient is None else gradient[:, axis])
            cache[order] = torch.stack(columns, dim=1)
        return cache[order]
    return {order: get(order) for order in derivatives}


def test_variable_and_field_dependent_divergence_expands_against_autograd():
    x, y, k = sp.symbols('x y k', real=True)
    u = sp.Function('u')(x, y)
    coefficient = 1+k*x*x+sp.sin(y)/3+u*u
    equation = sp.Eq(-sp.diff(coefficient*sp.diff(u, x), x)-
                     sp.diff(coefficient*sp.diff(u, y), y)+k*sp.sin(u), x*y)
    declaration = DifferentialResidual((x, y), (u,), (equation,), parameters=(k,))
    assert set(declaration.derivatives) == {(0,0), (1,0), (0,1), (2,0), (0,2)}
    points = torch.tensor([[-.3, .1], [.2, .4], [.6, -.2]], dtype=torch.float64, requires_grad=True)
    values = torch.exp(points[:, 0]+.2*points[:, 1])+points[:, 0]*points[:, 1]**3
    jets = exact_jets(points, values, declaration.derivatives)
    parameters = torch.full((len(points), 1), .7, dtype=torch.float64)
    actual = declaration.function(points, jets, parameters)[:, 0]
    a = 1+.7*points[:, 0]**2+torch.sin(points[:, 1])/3+values**2
    dx = torch.autograd.grad((a*jets[(1,0)][:, 0]).sum(), points, retain_graph=True)[0][:, 0]
    dy = torch.autograd.grad((a*jets[(0,1)][:, 0]).sum(), points, retain_graph=True)[0][:, 1]
    expected = -dx-dy+.7*torch.sin(values)-points[:, 0]*points[:, 1]
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-13)


def test_vector_incompressibility_and_constant_equation():
    x, y = sp.symbols('x y', real=True)
    u, v = sp.Function('u')(x,y), sp.Function('v')(x,y)
    declaration = DifferentialResidual((x,y), (u,v),
        (sp.diff(u,x)+sp.diff(v,y), sp.diff(v,x)-sp.diff(u,y), sp.Integer(2)))
    points = torch.tensor([[-.3, .2], [.4, .6]], dtype=torch.float64, requires_grad=True)
    values = torch.stack((points[:,0]**2*points[:,1], -points[:,0]*points[:,1]**2), dim=1)
    jets = exact_jets(points, values, declaration.derivatives)
    actual = declaration.function(points, jets, torch.empty((len(points),0), dtype=torch.float64))
    expected = torch.stack((torch.zeros_like(points[:,0]), -points[:,0]**2-points[:,1]**2,
                            torch.full_like(points[:,0], 2.)), dim=1)
    torch.testing.assert_close(actual, expected)


def test_unknown_parameter_gradient_through_common_engine():
    x, k = sp.symbols('x k', real=True)
    u = sp.Function('u')(x)
    declaration = DifferentialResidual((x,), (u,),
        (k*sp.diff(u,x,2)+sp.exp(k)*u+sp.sin(x),), parameters=(k,))
    features = ConstructedFeatures([[-1,1]], 4, backend='polynomial')
    points = np.linspace(-.8,.8,17)[:,None]
    problem = ResidualProblem(features, [declaration.block('equation', points)], parameter_initial=[.7])
    rng = np.random.default_rng(37)
    coefficients = rng.normal(size=(features.size,1))*.2
    linear = linearize_residual(problem, coefficients)
    direction = rng.normal(size=features.size+1)
    dc, dp = direction[:-1,None], direction[-1:]
    h = 1e-6
    finite_difference = (residual_vector(problem, coefficients+h*dc, np.array([.7])+h*dp)-
                         residual_vector(problem, coefficients-h*dc, np.array([.7])-h*dp))/(2*h)
    np.testing.assert_allclose(linear.operator @ direction, finite_difference, atol=3e-9, rtol=2e-8)


def test_sixth_order_product_rule_has_no_frontend_order_ceiling():
    x = sp.Symbol('x', real=True)
    u = sp.Function('u')(x)
    declaration = DifferentialResidual((x,), (u,), (sp.diff((1+x)*sp.diff(u,x),x,5),))
    assert set(declaration.derivatives) == {(5,), (6,)}
    points = torch.tensor([[-.4], [.2], [.7]], dtype=torch.float64, requires_grad=True)
    jets = exact_jets(points, points[:,0]**7, declaration.derivatives)
    actual = declaration.function(points, jets, torch.empty((len(points),0),dtype=torch.float64))[:,0]
    torch.testing.assert_close(actual, 5040*points[:,0]+17640*points[:,0]**2, rtol=2e-13, atol=2e-11)


def test_parameter_only_equation_uses_dummy_zero_jet_and_can_solve():
    x, k = sp.symbols('x k', real=True)
    u = sp.Function('u')(x)
    declaration = DifferentialResidual((x,), (u,), (k-2,), parameters=(k,))
    assert declaration.derivatives == ((0,),)
    assert declaration.jet_entries == ()
    features = ConstructedFeatures([[-1,1]], 0, backend='polynomial')
    problem = ResidualProblem(features, [declaration.block('prior', np.array([[0.]]))], parameter_initial=[.3])
    solution = solve_residual(problem, tolerance=1e-9)
    assert solution.status == 'converged'
    np.testing.assert_allclose(solution.parameters, [2.], atol=1e-9)


def test_elementary_functions_piecewise_and_block_options():
    x = sp.Symbol('x', real=True)
    u = sp.Function('u')(x)
    expression = sp.Piecewise((x*x, x>0), (sp.sin(1), True))+sp.Abs(u)+sp.sqrt(1+x*x)+sp.Max(x,1)
    declaration = DifferentialResidual((x,), (u,), (expression,))
    points = torch.tensor([[-.3], [.6]], dtype=torch.float64)
    field = torch.tensor([[-.2], [.4]], dtype=torch.float64)
    actual = declaration.function(points, {(0,):field}, torch.empty((2,0),dtype=torch.float64))[:,0]
    expected = torch.tensor([np.sin(1), .36], dtype=torch.float64)+torch.abs(field[:,0])+torch.sqrt(1+points[:,0]**2)+1
    torch.testing.assert_close(actual, expected)
    block = declaration.block('weak_constraint', points.numpy(), weight=2., scale=3.,
                               aggregation=np.array([[.5,.5]]), relation='le')
    assert block.relation == 'le' and block.weight == 2.
    assert block.aggregation.shape == (1,2)


def test_scalar_constants_and_piecewise_elementary_functions_are_vectorized():
    x = sp.Symbol('x', real=True)
    u = sp.Function('u')(x)
    declaration = DifferentialResidual((x,), (u,),
        (sp.pi+sp.E, sp.sign(u), sp.Heaviside(u), sp.floor(u), sp.ceiling(u), sp.Min(u,1)))
    points = torch.tensor([[-1.2], [0.], [2.3]], dtype=torch.float64)
    actual = declaration.function(points, {(0,):points}, torch.empty((3,0),dtype=torch.float64))
    expected = torch.tensor([[np.pi+np.e,-1,0,-2,-1,-1.2],
                             [np.pi+np.e,0,.5,0,0,0],
                             [np.pi+np.e,1,1,2,3,1]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('kind', ['unknown_function', 'unknown_symbol', 'unknown_axis', 'unsupported_function', 'string'])
def test_rejects_undeclared_or_unsupported_symbolic_inputs(kind):
    x, z = sp.symbols('x z', real=True)
    u = sp.Function('u')(x)
    expression = {
        'unknown_function': sp.Function('g')(x),
        'unknown_symbol': u+z,
        'unknown_axis': sp.Derivative(u,z,evaluate=False),
        'unsupported_function': sp.gamma(x),
        'string': 'u(x)+1',
    }[kind]
    with pytest.raises((ValueError, TypeError)):
        DifferentialResidual((x,), (u,), (expression,))


def test_rejects_shifted_field_and_undeclared_field_arguments():
    x, y = sp.symbols('x y')
    u = sp.Function('u')
    with pytest.raises(ValueError, match='field application'):
        DifferentialResidual((x,), (u(x),), (u(x+1),))
    with pytest.raises(ValueError, match='coordinate tuple'):
        DifferentialResidual((x,y), (u(x),), (sp.diff(u(x),x),))
