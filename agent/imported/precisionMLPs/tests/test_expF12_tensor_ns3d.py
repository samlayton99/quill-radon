"""expF12 verification: the tensor-product sech^2 basis engine matches the math.

1. Paired contraction == naive materialized Nt^4 tensor (values + all NS combos).
2. Analytic per-axis derivatives == torch autograd through the contraction.
3. Kronecker mode-wise pinv fit == dense lstsq on the materialized design matrix.
4. The Ethier-Steinman Beltrami fields satisfy the NS equations (autograd
   residual ~ machine eps) -- validates both the exact solution and the
   residual assembly conventions.
5. Supervised Kronecker fit of the Beltrami fields reaches high precision at
   modest width (the expressivity sanity check).
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "experiments" / "expF12_tensor_ns3d"))

import beltrami as bel
import tensor_basis as tb

torch.set_default_dtype(torch.float64)


def _small_net(seed=0, n_centers=4, n_out=4, train_geometry=False):
    net = tb.TensorBasisNet(n_centers=n_centers, lam=0.4, domains=bel.DOMAINS,
                            n_out=n_out, n_halo=1, train_geometry=train_geometry)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        net.A.copy_(torch.randn(net.A.shape, generator=g))
    return net


def test_contraction_matches_naive():
    net = _small_net()
    X = torch.rand(37, 4, generator=torch.Generator().manual_seed(1)) * 2 - 1
    X[:, 3] = (X[:, 3] + 1) / 2
    fast = net.evaluate(X, tb.COMBOS_NS)
    naive = net.evaluate_naive(X, tb.COMBOS_NS)
    for cb in tb.COMBOS_NS:
        err = (fast[cb] - naive[cb]).abs().max().item()
        scale = naive[cb].abs().max().item() + 1.0
        assert err / scale < 1e-13, (cb, err)


def test_analytic_derivatives_match_autograd():
    net = _small_net()
    X = (torch.rand(29, 4, generator=torch.Generator().manual_seed(2)) * 2 - 1)
    X[:, 3] = (X[:, 3] + 1) / 2
    X.requires_grad_(True)
    val = net.evaluate(X, [(0, 0, 0, 0)])[(0, 0, 0, 0)]
    for o in range(net.n_out):
        g = torch.autograd.grad(val[:, o].sum(), X, create_graph=True)[0]
        for a, cb in enumerate([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                                (0, 0, 0, 1)]):
            ana = net.evaluate(X, [cb])[cb][:, o]
            assert (g[:, a] - ana).abs().max().item() < 1e-10 * (
                ana.abs().max().item() + 1.0)
        for a, cb in enumerate([(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0)]):
            g2 = torch.autograd.grad(g[:, a].sum(), X, create_graph=True)[0][:, a]
            ana = net.evaluate(X, [cb])[cb][:, o]
            assert (g2 - ana).abs().max().item() < 1e-9 * (
                ana.abs().max().item() + 1.0)


def test_kron_fit_matches_dense_lstsq():
    """Well-conditioned config (full column rank per axis): the mode-wise pinv
    solution must equal the dense-Kronecker lstsq solution. (In degenerate
    configs the two differ -- per-factor rcond truncation is not product
    truncation -- which is why the runs log per-axis conditioning.)"""
    net = tb.TensorBasisNet(n_centers=5, lam=2.0, domains=bel.DOMAINS,
                            n_out=2, n_halo=0)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        net.A.copy_(torch.randn(net.A.shape, generator=g))
    rng = np.random.default_rng(3)
    grids = [np.linspace(lo, hi, 9) for lo, hi in bel.DOMAINS]
    Y = rng.standard_normal((9, 9, 9, 9, 2))
    tb.kron_fit(net, grids, Y)
    # dense reference: materialized Kronecker design matrix
    with torch.no_grad():
        Phis = [net.axis_features(torch.tensor(g), a, [0])[0].numpy()
                for a, g in enumerate(grids)]
    Phi = Phis[0]
    for M in Phis[1:]:
        Phi = np.kron(Phi, M)
    for o in range(2):
        a_dense = np.linalg.lstsq(Phi, Y[..., o].ravel(), rcond=1e-13)[0]
        a_kron = net.A[o].detach().numpy().ravel()
        assert np.abs(a_kron - a_dense).max() < 1e-8 * (
            np.abs(a_dense).max() + 1.0)


def test_beltrami_satisfies_navier_stokes():
    """Push the exact fields through autograd; NS residual must vanish."""
    X = (torch.rand(200, 4, generator=torch.Generator().manual_seed(4)) * 2 - 1)
    X[:, 3] = (X[:, 3] + 1) / 2
    X.requires_grad_(True)
    F = bel.fields(X)
    u, v, w, p = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
    grads, laps = [], []
    for q in (u, v, w):
        g = torch.autograd.grad(q.sum(), X, create_graph=True)[0]
        lap = sum(torch.autograd.grad(g[:, a].sum(), X, create_graph=True)[0][:, a]
                  for a in range(3))
        grads.append(g)
        laps.append(lap)
    gp = torch.autograd.grad(p.sum(), X, create_graph=True)[0]
    mom_max = 0.0
    for i, (q, g, lap) in enumerate(zip((u, v, w), grads, laps)):
        r = (g[:, 3] + u * g[:, 0] + v * g[:, 1] + w * g[:, 2]
             + gp[:, i] - bel.NU * lap)
        mom_max = max(mom_max, r.abs().max().item())
    cont = grads[0][:, 0] + grads[1][:, 1] + grads[2][:, 2]
    assert mom_max < 1e-11, mom_max
    assert cont.abs().max().item() < 1e-12


def test_supervised_kron_fit_precision():
    """Modest width already fits Beltrami to <2e-6 rel L2 at fresh points --
    the basis is expressive and the fit pipeline is wired correctly. (The
    ceiling sweep in run.py maps the full (N, lambda, rcond) landscape; the
    fp64 floor there is ~6e-9 at N=24-32, lambda=0.125, rcond=1e-15.)"""
    net = tb.TensorBasisNet(n_centers=12, lam=0.15, domains=bel.DOMAINS,
                            n_out=4, n_halo=2)
    grids = [np.linspace(lo, hi, 24) for lo, hi in bel.DOMAINS]
    tb.kron_fit(net, grids, bel.grid_targets(grids), rcond=1e-15)
    rng = np.random.default_rng(5)
    P = rng.uniform(0, 1, (4000, 4))
    for a, (lo, hi) in enumerate(bel.DOMAINS):
        P[:, a] = lo + (hi - lo) * P[:, a]
    Xe = torch.tensor(P)
    pred = net.evaluate_chunked(Xe, [(0, 0, 0, 0)])[(0, 0, 0, 0)].numpy()
    exact = bel.fields(P)
    rel = np.linalg.norm(pred[:, :3] - exact[:, :3]) / np.linalg.norm(exact[:, :3])
    assert rel < 2e-6, rel


def test_reduced_basis_derivative_consistency():
    """op_columns derivative columns match central finite differences of the
    value columns (validates the Newton-solve assembly path)."""
    import newton_solve as nsv
    rb = nsv.ReducedTensorBasis(n_centers=6, lam=0.3, K=100)
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.8, 0.8, (50, 4))
    P[:, 3] = rng.uniform(0.1, 0.9, 50)
    h = 1e-6
    for a, op in enumerate([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                            (0, 0, 0, 1)]):
        Pp, Pm = P.copy(), P.copy()
        Pp[:, a] += h
        Pm[:, a] -= h
        fd = (rb.op_columns(Pp, [((0, 0, 0, 0), 1.0)])
              - rb.op_columns(Pm, [((0, 0, 0, 0), 1.0)])) / (2 * h)
        ana = rb.op_columns(P, [(op, 1.0)])
        assert np.abs(fd - ana).max() < 1e-5 * (np.abs(ana).max() + 1.0)


def test_newton_solve_tiny_converges():
    """Tiny Gauss-Newton NS solve: the first step (Stokes) already lands well
    below the zero init, and iterating does not diverge."""
    import newton_solve as nsv
    _, _, theta, hist = nsv.solve_navier_stokes(
        n_centers=8, lam=0.2, K_vel=300, K_p=150, n_int=1200, n_ic=400,
        n_bc=400, newton_iters=2, verbose=False)
    assert hist[0]["rel_l2_v"] == 1.0  # zero init
    assert hist[1]["rel_l2_v"] < 0.05
    assert hist[-1]["rel_l2_v"] <= hist[1]["rel_l2_v"] * 1.5
