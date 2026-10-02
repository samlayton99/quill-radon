"""expI02 block prior: tests that the corrected QI block matches the research.

Fixed mesh (gamma h = lambda at every depth), coefficient coordinates as exact
reparametrizations, capped whitening, smooth init identical across coordinate
systems, the 1-D machine-precision floor through the stack, calibration onto
s_star, the band penalty, and matched-count baselines.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "expI02_block_prior"))

import qi2  # noqa: E402

torch.set_default_dtype(torch.float64)


def _ref_grid(n=2001):
    return torch.linspace(-1.0, 1.0, n)


# --------------------------------------------------------------------------- Task 1: core
def test_bank_has_fixed_mesh_with_lambda():
    b = qi2.Bank(32)
    assert b.c.shape == (32,)
    assert abs(b.gamma * b.h - qi2.LAM) < 1e-15
    assert abs(float(b.c[0]) - (-1 + 0.5 * b.h)) < 1e-15 and abs(float(b.c[-1]) - (1 - 0.5 * b.h)) < 1e-15
    H = b(torch.zeros(5, 3))
    assert H.shape == (5, 3, 32)


def test_coordinates_are_a_reparametrization():
    torch.manual_seed(0)
    M, N, K = 3, 16, 2
    H = torch.randn(7, M, N)
    for kind in ["raw", "val", "white", "val_white"]:
        mx = qi2.Mixer(M, N, K, coords=kind)
        with torch.no_grad():
            mx.theta.normal_()
        C = mx.coeffs()
        expect = torch.einsum("bmn,mnk->bk", H, C) + mx.bias
        assert torch.allclose(mx(H), expect, atol=1e-12), kind
        if kind == "val":
            th = mx.theta
            Cv = 0.5 * (th - torch.cat([torch.zeros_like(th[:, :1]), th[:, :-1]], 1))
            assert torch.allclose(C, Cv, atol=1e-14)
        if kind == "raw":
            assert torch.allclose(C, mx.theta)


def test_function_value_coordinates_are_smoothed_heights():
    """C = D theta turns the tanh steps into the bumps b_j = (t_j - t_{j+1})/2, a partition of unity on the interior;
    theta_j is then the local height up to the first-order QI smoothing (6% for cos(pi x) at N = 64)."""
    N = 64
    b = qi2.Bank(N)
    D = qi2.coord_transform(N, "val")
    x = torch.linspace(-0.5, 0.5, 201)
    B = b(x.view(-1, 1))[:, 0, :]
    ones = B @ (D @ torch.ones(N))                       # = t_0(x)/2: the sum's own constant is -a_{N-1}/2
    assert float((ones - 0.5).abs().max()) < 1e-3        # tanh tail of t_0 at x = -0.5 (gamma = 8)
    assert float((ones[-1] - 0.5).abs()) < 1e-6
    th = torch.cos(math.pi * b.c)
    f = (b(b.c.view(-1, 1))[:, 0, :] @ (D @ th))
    inner = b.c.abs() < 0.5
    d = (f - th)[inner]
    d = d - d.mean()                                     # heights up to one global constant
    assert float(d.abs().max()) < 0.1 and float(d.abs().max()) > 1e-3


def test_whitener_caps_the_spectrum():
    N, cap = 32, 1e3
    b = qi2.Bank(N)
    xr = _ref_grid(4001); xr = xr[xr.abs() <= qi2.OCCUPIED]         # the whitener's reference occupancy
    B = b(xr.view(-1, 1))[:, 0, :]
    G = B.T @ B / B.shape[0]
    s = torch.linalg.eigvalsh(G).clamp_min(0).sqrt().flip(0)
    P = qi2.coord_transform(N, "white", cap=cap)
    Gw = (B @ P).T @ (B @ P) / B.shape[0]
    ew = torch.linalg.eigvalsh(Gw).flip(0)
    expect = torch.minimum(torch.ones_like(s), (cap * s / s[0]) ** 2)
    big = expect > 1e-6
    assert torch.allclose(ew[big], expect[big], rtol=1e-6)
    assert float(ew[~big].abs().max()) < 1e-6


def test_one_layer_hits_1d_floor():
    g = torch.Generator().manual_seed(0)
    x = 2 * torch.rand(4096, 1, generator=g) - 1
    xt = 1.8 * torch.rand(4096, 1, generator=g) - 0.9
    y, yt = torch.sin(2 * math.pi * x[:, 0]), torch.sin(2 * math.pi * xt[:, 0])
    net = qi2.make_block(1, M=1, N1=128, K=1, N2=1, depth=1)
    with torch.no_grad():
        net.layers[0].V.fill_(0.8)                       # data on [-0.8, 0.8]: the 25% collar
    net.solve_readout(x, y)
    assert qi2.rel_l2(net(xt)[:, 0], yt) < 1e-11


def test_smooth_init_identical_across_coords():
    X = torch.randn(50, 3) * 0.3
    outs = [qi2.make_block(3, M=6, N1=16, K=2, N2=32, coords=k, seed=1)(X) for k in ["raw", "val", "white", "val_white"]]
    for o in outs[1:]:
        assert float((o - outs[0]).abs().max()) < 1e-8   # round trip through P (condition 1e3) on O(1) outputs
    assert float(outs[0].std()) > 0                      # nonconstant
    C = qi2.make_block(3, M=6, N1=16, K=2, N2=32, seed=1).layers[0].mixer.coeffs()
    assert float(C.abs().max()) > 0                      # never zero-initialized


def test_rank_mixer_matches_full_tensor():
    torch.manual_seed(0)
    M, N, K, S = 4, 8, 3, 2
    mx = qi2.Mixer(M, N, K, rank=S, coords="val")
    with torch.no_grad():
        mx.Phi.normal_()
    H = torch.randn(9, M, N)
    C = mx.coeffs()
    assert C.shape == (M, N, K)
    assert torch.allclose(mx(H), torch.einsum("bmn,mnk->bk", H, C) + mx.bias, atol=1e-12)


def test_counts_and_width_matching():
    w = qi2.mlp_width_for(500, d=3, key="params")
    assert qi2.MLP(3, [w, w]).counts()["params"] >= 500
    assert qi2.MLP(3, [w - 1, w - 1]).counts()["params"] < 500
    blk = qi2.make_block(3, M=6, N1=32, K=2, N2=64)
    c = blk.counts()
    assert c["units"] == 6 * 32 + 2 * 64
    assert c["params"] == sum(p.numel() for p in blk.parameters())
    assert c["flops"] == 6 * 3 + 6 * 32 + 6 * 32 * 2 + 2 * 64 + 2 * 64 * 1


def test_residual_layer_is_identity_plus_delta():
    layer = qi2.QILayer(3, 3, 16, proj=False, residual=True)
    z = torch.randn(11, 3) * 0.3
    delta = layer.mixer(layer.bank(z))
    assert torch.allclose(layer(z), z + delta, atol=1e-14)
    net = qi2.make_block(4, M=5, N1=8, K=3, N2=16, depth=4, residual=True)
    assert len(net.layers) == 4 and net.layers[1].residual and net.layers[2].residual and not net.layers[3].residual


# --------------------------------------------------------------------------- Task 2: band control
def test_calibrate_lands_on_s_star():
    net = qi2.make_block(4, M=5, N1=16, K=3, N2=32, depth=3, seed=2)
    X = torch.randn(2000, 4) + 0.3
    qi2.calibrate(net, X)
    pres = net.pres(X)
    assert len(pres) == 3
    for p in pres:
        assert float(p.mean(0).abs().max()) < 1e-9
        assert float((p.std(0, unbiased=False) - qi2.S_STAR).abs().max()) < 1e-9


def test_calibrate_residual_increment():
    net = qi2.make_block(4, M=5, N1=16, K=3, N2=32, depth=4, residual=True, seed=2)
    X = torch.randn(2000, 4)
    qi2.calibrate(net, X, s_resid=0.1)
    z = net.layers[0](X)
    for l in net.layers[1:3]:
        delta = l(z) - z
        assert float(delta.mean(0).abs().max()) < 1e-9
        assert float((delta.std(0, unbiased=False) - 0.1).abs().max()) < 1e-9
        z = l(z)


def test_band_penalty_zero_inside_positive_outside():
    net = qi2.make_block(4, M=5, N1=16, K=3, N2=32, depth=2, seed=2)
    X = torch.randn(2000, 4)
    qi2.calibrate(net, X)
    inside = float(qi2.band_penalty(net.pres(X)))
    assert inside < 0.1                                   # Gaussian tails past rho = 0.9 at sd 0.4: about 0.04
    with torch.no_grad():
        net.layers[0].V.mul_(3.0)
    outside = float(qi2.band_penalty(net.pres(X)))
    assert outside > 20 * inside
    assert float(qi2.band_penalty([])) == 0.0
    assert qi2.band_penalty(net.pres(X)).requires_grad


# --------------------------------------------------------------------------- Task 3: optimizers and the fit loop
def _toy_1d():
    g = torch.Generator().manual_seed(0)
    x = 2 * torch.rand(1024, 1, generator=g) - 1
    xt = 1.8 * torch.rand(1024, 1, generator=g) - 0.9
    f = lambda z: torch.sin(2 * math.pi * z[:, 0]) + 0.5 * torch.sin(6 * math.pi * z[:, 0])
    return x, f(x), xt, f(xt)


def test_power_iteration_positive_and_stable():
    x, y, _, _ = _toy_1d()
    net = qi2.make_block(1, M=1, N1=32, K=1, N2=1, depth=1, head_coords="white")
    with torch.no_grad():
        net.layers[0].V.fill_(0.8)
    params = net.readout()
    loss_fn = lambda: ((net(x)[:, 0] - y) ** 2).mean()
    L = qi2.power_iteration_L(loss_fn, params, iters=20)
    assert L > 0
    opt = torch.optim.SGD(params, lr=1.0 / L)
    losses = []
    for _ in range(50):
        opt.zero_grad(); l = loss_fn(); l.backward(); opt.step(); losses.append(float(l))
    assert all(b <= a * (1 + 1e-12) for a, b in zip(losses[:-1], losses[1:]))


def test_fit_logs_solved_head_gap():
    x, y, xt, yt = _toy_1d()
    net = qi2.make_block(1, M=2, N1=16, K=2, N2=32, coords="white", seed=3)
    log = qi2.fit(net, x, y, xt, yt, optimizer="adam", head="trained", steps=60, log_every=20, beta=1e-2)
    for k in ["step", "train", "test_trained", "test_solved", "band", "final", "final_trained", "time", "counts"]:
        assert k in log, k
    assert len(log["step"]) == 4 and log["step"][-1] == 59
    assert log["test_solved"][-1] <= log["test_trained"][-1] * 1.01
    assert log["final"] <= log["final_trained"] * 1.01
    assert len(log["band"][-1]) == 2                    # one entry per layer: [mean |mu|, mean sd, frac past rho]


def test_fit_momentum_and_varpro_descend():
    x, y, xt, yt = _toy_1d()
    net = qi2.make_block(1, M=2, N1=16, K=2, N2=32, coords="white", seed=3)
    e0 = qi2.refit_eval(net, x, y, xt, yt)
    log = qi2.fit(net, x, y, xt, yt, optimizer="momentum", head="varpro", steps=100, log_every=50)
    assert log["final"] < e0


def test_fit_ce_runs():
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(300, 2, generator=g) + 2, torch.randn(300, 2, generator=g) - 2])
    yl = torch.cat([torch.zeros(300, dtype=torch.long), torch.ones(300, dtype=torch.long)])
    net = qi2.make_block(2, M=4, N1=8, K=2, N2=16, d_out=2, coords="white", head_coords="white", seed=0)
    log = qi2.fit(net, X, yl, X, yl, optimizer="adam", head="trained", loss="ce", steps=100, log_every=50)
    assert log["final_trained"] < 0.2 and log["final"] < 0.2


# --------------------------------------------------------------------------- Task 4: data, targets, PDEs
def test_analytic_reproduces_expH05_cliff_cell():
    """E0b of the record: shallow ridge-QI with expH05's even directions on its r = 0.4 ball, fast waves, M = 12,
    128 offsets, solved readout: below 1e-10 (recorded 4.9e-13)."""
    import tasks
    D = tasks.analytic("fast_waves", d=2, n_train=8 * 12 * 128, seed=0)
    assert set(["Xtr", "Ytr", "Xte", "Yte", "r", "oracle"]) <= set(D)
    net = qi2.make_block(2, M=12, N1=128, K=1, N2=1, depth=1)
    with torch.no_grad():
        th = math.pi / 24 + torch.arange(12) * math.pi / 12
        net.layers[0].V.copy_(torch.stack([torch.cos(th), torch.sin(th)], 1) / (1.25 * D["r"]))
        net.layers[0].bV.zero_()
    net.solve_readout(D["Xtr"], D["Ytr"])
    assert qi2.rel_l2(net(D["Xte"])[:, 0], D["Yte"]) < 1e-10


def test_analytic_oracle_channel_is_a_quadratic():
    import tasks
    D = tasks.analytic("gauss_bump", d=3, n_train=512, seed=0)
    V, Pfn = D["oracle"]
    assert V.shape == (3, 3)
    P = Pfn(D["Xtr"])
    assert torch.allclose(torch.exp(-8 * P), D["Ytr"], atol=1e-12)     # gauss_bump = exp(-|x-a|^2 / 0.5^2) = exp(-8 P), P = |x-a|^2/2


def test_pde_manufactured_residual_zero():
    import tasks

    class Exact(torch.nn.Module):
        def __init__(self, fn):
            super().__init__(); self.fn = fn
        def forward(self, X):
            return self.fn(X).view(-1, 1)
    for name in ["poisson1d", "poisson2d"]:
        P = tasks.pde(name, seed=0, n_col=256, n_bc=32)
        r = P["residual"](Exact(P["exact"]), P["X_col"])
        assert float(r.abs().max()) < 1e-8, name
        assert P["linear"] and P["d"] == int(name[7])
        assert torch.allclose(P["u_bc"], P["exact"](P["X_bc"]), atol=1e-12)
    B = tasks.pde("burgers", seed=0, n_col=256, n_bc=32)
    assert not B["linear"] and B["d"] == 2
    ic = B["X_bc"][(B["X_bc"][:, 1] + 1).abs() < 1e-12]          # scaled time tau = 2t - 1: the IC sits at tau = -1
    assert len(ic) > 0 and float((B["exact"](ic) + torch.sin(math.pi * ic[:, 0])).abs().max()) < 1e-6


def test_structured_regression_loaders():
    import tasks
    D = tasks.friedman1(seed=0, n=400)
    assert D["Xtr"].shape[1] == 10 and D["Ytr"].dim() == 2
    assert float(D["Xtr"].mean(0).abs().max()) < 1e-6 and float((D["Xtr"].std(0) - 1).abs().max()) < 1e-6
    L = tasks.lorenz_map(seed=0, n=500)
    assert L["Xtr"].shape[1] == 3 and L["Ytr"].shape[1] == 3


def test_fashion_loader_shapes():
    import tasks
    if not tasks.fashion_cached():
        pytest.skip("Fashion-MNIST not cached")
    Xtr, ytr, Xte, yte = tasks.fashion_mnist()
    assert Xtr.shape == (60000, 784) and Xte.shape == (10000, 784) and int(ytr.max()) == 9


# --------------------------------------------------------------------------- Task 6: the legacy rung (old recipe)
def test_legacy_layer_normalizes_directions_and_tracks_range():
    """The notebook recipe: unit directions scaled to the band, level-2 input range-tracked to [-0.8, 0.8]; zero
    profile init. Kept only as the baseline rung of the A1 ladder."""
    X = torch.randn(1000, 3) * 0.3
    net = qi2.make_block(3, M=6, N1=16, K=2, N2=32, legacy=True, band_r=0.3, seed=0)
    l0, l1 = net.layers[0], net.layers[1]
    p0 = l0.pre(X)
    assert torch.allclose(p0, (X @ (l0.V / l0.V.norm(dim=1, keepdim=True)).T) / (1.25 * 0.3), atol=1e-12)
    assert float(l0.mixer.coeffs().abs().max()) == 0.0
    with torch.no_grad():
        l0.mixer.theta.normal_()
    net.update_ranges(X)
    p1 = net.pres(X)[1]
    assert abs(float(p1.abs().max()) - 0.8) < 1e-9
    assert not l1.track or l1.track and float(p1.mean(0).abs().max()) < 0.8


# --------------------------------------------------------------------------- the in-pass VarPro solve
def test_forward_pres_solve_matches_head_solve():
    x, y, xt, yt = _toy_1d()
    net = qi2.make_block(1, M=2, N1=16, K=2, N2=32, seed=3)
    qi2.calibrate(net, x)
    out, pres = net.forward_pres(x, solve=(y.view(-1, 1), 1e-14))
    assert torch.allclose(out, net(x), atol=1e-12)                    # the output was formed with the solved head
    r_fast = float((out[:, 0] - y).norm())
    net.solve_readout(x, y.view(-1, 1))
    r_ref = float((net(x)[:, 0] - y).norm())
    assert abs(r_fast - r_ref) < 1e-3 * max(r_ref, 1e-30) + 1e-10     # gelsd and QR+SVD give the same fit (truncation differs on this ill-conditioned toy)
    mlp = qi2.MLP(1, [16, 16])
    out, _ = mlp.forward_pres(x, solve=(y.view(-1, 1), 1e-14))
    assert torch.allclose(out, mlp(x), atol=1e-12)


# --------------------------------------------------------------------------- the PINN operator solve
def test_operator_solve_reaches_expF01_floor_on_poisson1d():
    """expF01: on the frozen 1-D QI geometry the linear PDE is one least squares over [L Phi; Phi_bc] and reaches the
    floor with no training. The same solve through op_solve on a one-direction block must do the same."""
    import importlib.util, tasks
    spec = importlib.util.spec_from_file_location("expi02_run", REPO / "experiments" / "expI02_block_prior" / "run.py")
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    P = tasks.pde("poisson1d", seed=0, n_col=2048)
    net = qi2.make_block(1, M=1, N1=160, K=1, N2=1, depth=1)
    with torch.no_grad():
        net.layers[0].V.fill_(0.8); net.layers[0].bV.zero_()
    err, rank = run.op_solve(net, P)
    assert err < 1e-9, err
