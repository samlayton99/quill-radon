"""Research checks for the new, isolated checkpoint-I probes."""
import importlib.util
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parents[1] / "experiments/expI03_investigation/probe.py"
spec = importlib.util.spec_from_file_location("expI03_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_smooth_mixer_is_an_exact_qi_reparameterization():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    D = probe.tasks.analytic("composition", 3, 128, 64)
    net = probe.qi2.make_block(3, 6, 32, 2, 64)
    probe.configure(net, D["Xtr"], "poly3", 0)
    layer = net.layers[0]
    H = layer.hidden(D["Xtr"])
    expected = torch.einsum("bmj,mjk->bk", H, layer.mixer.coeffs()) + layer.mixer.bias
    torch.testing.assert_close(layer(D["Xtr"]), expected, atol=1e-12, rtol=1e-12)
    assert layer.mixer.theta.shape == (6, 3, 2)
    assert layer.bank.gamma * layer.bank.h == .25


def test_ridge_envelope_gradient_matches_reoptimized_finite_difference():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    D = probe.tasks.analytic("composition", 3, 192, 32)
    net = probe.qi2.make_block(3, 3, 32, 1, 24)
    probe.configure(net, D["Xtr"], "poly3", 0)
    ridge = 1e-4
    y = D["Ytr"].view(-1, 1)
    for p in net.readout():
        p.requires_grad_(False)

    def objective():
        out, _ = probe.ridge_forward(net, D["Xtr"], y, ridge)
        return (out - y).square().mean() + ridge * net.layers[-1].mixer.theta.square().sum()

    objective().backward()
    parameter = net.layers[0].mixer.theta
    direction = torch.randn(parameter.shape, generator=torch.Generator().manual_seed(9))
    direction /= direction.norm()
    exact = float((parameter.grad * direction).sum())
    saved = parameter.detach().clone()
    step = 1e-5
    with torch.no_grad():
        parameter.copy_(saved + step * direction)
        plus = float(objective())
        parameter.copy_(saved - step * direction)
        minus = float(objective())
        parameter.copy_(saved)
    observed = (plus - minus) / (2 * step)
    assert abs(exact - observed) < 2e-6 * max(1., abs(exact))


def test_releasing_quadratic_coefficients_preserves_initial_function():
    torch.set_default_dtype(torch.float64)
    D = probe.tasks.analytic("fast_waves", 3, 128, 64)
    compact = probe.qi2.make_block(3, 6, 32, 2, 64, seed=4)
    released = probe.qi2.make_block(3, 6, 32, 2, 64, seed=4)
    probe.configure(compact, D["Xtr"], "quadratic", 4)
    probe.configure(released, D["Xtr"], "quadratic_free", 4)
    torch.testing.assert_close(compact.feats(D["Xte"]), released.feats(D["Xte"]),
                               rtol=1e-12, atol=1e-12)
    assert compact.layers[0].mixer.theta.shape == (6, 2, 2)
    assert released.layers[0].mixer.theta.shape == (6, 32, 2)


def test_ridge_head_solves_augmented_system_with_unpenalized_bias():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    D = probe.tasks.analytic("fast_waves", 3, 128, 32)
    net = probe.qi2.make_block(3, 3, 32, 1, 16)
    probe.configure(net, D["Xtr"], "poly3", 0)
    ridge = 1e-3
    probe.solve_head(net, D["Xtr"], D["Ytr"], ridge=ridge)
    F = net.feats(D["Xtr"]).detach()
    A = torch.cat([F, torch.ones(len(F), 1)], 1)
    penalty = torch.eye(A.shape[1]) * (ridge * len(F)) ** .5
    penalty[-1, -1] = 0
    b = torch.cat([D["Ytr"], torch.zeros(A.shape[1])])
    expected = torch.linalg.lstsq(torch.cat([A, penalty]), b, driver="gelsd").solution
    torch.testing.assert_close(net(D["Xtr"]).ravel(), A @ expected, rtol=1e-9, atol=1e-10)
