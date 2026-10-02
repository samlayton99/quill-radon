"""Independent linear-adjoint, physical derivative and ordinary MLP audits."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import eval_gegenbauer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_directional_operator import DirectionalProfileOperator


@pytest.fixture(scope='module')
def setup():
    rng = np.random.default_rng(1971)
    directions = rng.normal(size=(7, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    op = DirectionalProfileOperator(directions, 6, centers=129, midpoint=[.2, -.4, .7],
                                   scale=[.7, 1.9, .4], block_size=11)
    z = rng.normal(size=(31, 3))
    z *= rng.uniform(.1, .7, len(z))[:, None]/np.linalg.norm(z, axis=1)[:, None]
    x = z/op.scale+op.midpoint
    theta = rng.normal(size=op.size)*.02
    return op, x, theta


@pytest.mark.parametrize('derivative', [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)])
def test_streamed_adjoint_dot_product(setup, derivative):
    op, x, theta = setup
    y = np.random.default_rng(75).normal(size=len(x))
    left = np.dot(op.forward(theta, x, derivative), y)
    right = np.dot(theta, op.adjoint(y, x, derivative))
    np.testing.assert_allclose(left, right, atol=3e-12, rtol=3e-12)


def test_stacked_jets_adjoint_and_block_independence(setup):
    op, x, theta = setup
    derivatives = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    jets = op.forward_jets(theta, x, derivatives)
    rng = np.random.default_rng(737)
    cotangents = {a: rng.normal(size=len(x)) for a in derivatives}
    lhs = sum(np.dot(jets[a], cotangents[a]) for a in derivatives)
    rhs = theta@op.adjoint_jets(cotangents, x)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-12, atol=2e-12)
    for a in derivatives:
        # One-row queries exercise a different reduction/block partition.
        separate = np.array([op.forward(theta, q[None, :], a)[0] for q in x])
        np.testing.assert_allclose(jets[a], separate, atol=2e-12, rtol=2e-12)


def test_actual_physical_derivatives_and_parameter_directional_difference(setup):
    op, x, theta = setup
    delta = np.zeros_like(x)
    delta[:, 0] = 2e-5
    derivative_fd = (op.forward(theta, x+delta)-op.forward(theta, x-delta))/(4e-5)
    second_fd = (op.forward(theta, x+delta, (1, 0, 0))
                 -op.forward(theta, x-delta, (1, 0, 0)))/(4e-5)
    np.testing.assert_allclose(derivative_fd, op.forward(theta, x, (1, 0, 0)), rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(second_fd, op.forward(theta, x, (2, 0, 0)), rtol=2e-7, atol=2e-7)
    direction = np.random.default_rng(729).normal(size=op.size)
    epsilon = 1e-5
    finite_difference = (op.forward(theta+epsilon*direction, x)
                         -op.forward(theta-epsilon*direction, x))/(2*epsilon)
    np.testing.assert_allclose(finite_difference, op.forward(direction, x), rtol=2e-9, atol=2e-9)


def test_plain_torch_network_value_and_ad_jets(setup):
    op, x, theta = setup
    model = op.torch_model(theta)
    assert [type(layer) for layer in model] == [torch.nn.Linear, torch.nn.Tanh, torch.nn.Linear]
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = model(tx)[:, 0]
    first = torch.autograd.grad(y.sum(), tx, create_graph=True)[0]
    second = torch.autograd.grad(first[:, 0].sum(), tx)[0]
    for a, actual in [((0, 0, 0), y), ((1, 0, 0), first[:, 0]),
                      ((2, 0, 0), second[:, 0]), ((1, 1, 0), second[:, 1])]:
        np.testing.assert_allclose(op.forward(theta, x, a), actual.detach().numpy(),
                                   atol=3e-11, rtol=3e-11)


def test_known_profile_and_anchor_constant(setup):
    op, x, _ = setup
    s = ((x-op.midpoint)*op.scale)@op.directions[3]
    for n in [1, 3, 6]:
        b = np.zeros((op.direction_count, op.degree))
        b[3, n-1] = 1.
        theta = op.pack(.17, b)
        np.testing.assert_allclose(op.forward(theta, x), .17+eval_gegenbauer(n, 1.5, s),
                                   rtol=1e-10, atol=1e-10)
    theta = np.zeros(op.size)
    theta[0] = 3.25
    assert np.all(op.forward(theta, x) == 3.25)
    assert np.all(op.forward(theta, x, (2, 0, 0)) == 0.)
    constant = DirectionalProfileOperator([[1., 0.]], 0, centers=65)
    np.testing.assert_array_equal(constant.forward([2.], [[.1, .2]]), [2.])


def test_valid_domain_and_arguments():
    op = DirectionalProfileOperator([[1., 0.], [0., 1.]], 2, centers=65)
    with pytest.raises(ValueError, match='projection band'):
        op.forward(np.zeros(op.size), [[1.1, 0.]])
    with pytest.raises(ValueError, match='projection band'):
        op.adjoint([1.], [[0., -1.1]])
    with pytest.raises(ValueError, match='unit vectors'):
        DirectionalProfileOperator([[2., 0.]], 2)
    with pytest.raises(ValueError, match='nonnegative integers'):
        op.forward(np.zeros(op.size), [[0., 0.]], [1., 0.])
    with pytest.raises(ValueError, match='finite vector'):
        op.forward(np.ones(op.size-1), [[0., 0.]])
    assert op.forward(np.zeros(op.size), np.empty((0, 2))).shape == (0,)


def test_selected_preconditioner_columns_in_arbitrary_order(setup):
    op,x,theta=setup
    indices=np.array([0,19,3,op.size-1,19,1])
    derivatives=[(0,0,0),(1,0,0),(1,1,0),(0,0,2)]
    panels=op.selected_columns(x,indices,derivatives)
    perturbation=np.array([.2,-.3,.7,.1,.5,-.4])
    sparse=np.zeros(op.size)
    np.add.at(sparse,indices,perturbation)
    eps=1e-5
    for a in derivatives:
        finite=(op.forward(theta+eps*sparse,x,a)-op.forward(theta-eps*sparse,x,a))/(2*eps)
        np.testing.assert_allclose(panels[a]@perturbation,finite,atol=2e-9,rtol=2e-9)
    for indices in [np.array([-1]),np.array([op.size]),np.array([1.2])]:
        with pytest.raises(ValueError,match='indices'):
            op.selected_columns(x,indices,derivatives)


def test_prepared_forward_and_accumulated_adjoint_preserve_partition(setup):
    op,x,theta=setup
    orders=[(0,0,0),(1,0,0),(0,1,1)]
    prepared=op.prepare_forward(theta)
    rng=np.random.default_rng(873)
    cotangents={a:rng.normal(size=len(x)) for a in orders}
    accumulator=op.prepare_adjoint()
    lhs=0.
    for rows in np.array_split(np.arange(len(x)),9):
        values=op.forward_prepared_jets(prepared,x[rows],orders)
        local={a:v[rows] for a,v in cotangents.items()}
        op.accumulate_adjoint_jets(local,x[rows],accumulator)
        lhs+=sum(values[a]@local[a] for a in orders)
    gradient=op.finish_adjoint(accumulator)
    np.testing.assert_allclose(lhs,theta@gradient,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(gradient,op.adjoint_jets(cotangents,x),rtol=2e-12,atol=2e-12)
    assert prepared['readouts'].shape==(op.direction_count,op.centers_per_direction)
    assert accumulator['neuron_gradient'].shape==prepared['readouts'].shape
