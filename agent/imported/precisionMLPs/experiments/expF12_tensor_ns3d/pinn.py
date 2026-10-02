"""PINN training for the 3D unsteady Navier-Stokes Beltrami problem (expF12).

Two solvers share the loss blocks and evaluation:
  - train_tensor_pinn: the tensor-product sech^2 basis net (analytic derivatives,
    paired contraction). Geometry frozen by default; only the readout tensor A
    trains. Warm start = Kronecker lstsq fit of A to the t-constant extension of
    the initial condition (uses only given IC data). Adam -> LBFGS.
  - train_mlp_pinn: standard tanh-MLP PINN baseline (autograd derivatives),
    same losses, same eval, for the benchmark plot.

Loss blocks (all mean-squared): momentum residual (3), continuity, IC (velocity
at t=0), BC (velocity on the 6 box faces, Dirichlet from the exact solution --
the standard setup for this benchmark), pressure gauge (pressure pinned at one
corner "pressure tap" over time). Everything float64 CPU.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

import beltrami as bel
import tensor_basis as tb

torch.set_default_dtype(torch.float64)


# -- sampling ------------------------------------------------------------------

def sample_interior(rng, n):
    P = rng.uniform(0.0, 1.0, (n, 4))
    for a, (lo, hi) in enumerate(bel.DOMAINS):
        P[:, a] = lo + (hi - lo) * P[:, a]
    return torch.tensor(P)


def sample_ic(rng, n):
    P = rng.uniform(-1.0, 1.0, (n, 4))
    P[:, 3] = 0.0
    return torch.tensor(P)


def sample_bc(rng, n):
    P = rng.uniform(-1.0, 1.0, (n, 4))
    P[:, 3] = rng.uniform(0.0, 1.0, n)
    face = rng.integers(0, 6, n)
    P[np.arange(n), face // 2] = np.where(face % 2 == 0, -1.0, 1.0)
    return torch.tensor(P)


def gauge_points(n=33):
    P = np.full((n, 4), -1.0)
    P[:, 3] = np.linspace(0.0, 1.0, n)
    return torch.tensor(P)


# -- evaluation ----------------------------------------------------------------

def make_eval_set(seed=123, n=20000):
    rng = np.random.default_rng(seed)
    X = sample_interior(rng, n)
    return X, torch.tensor(bel.fields(X.numpy()))


def eval_errors(predict, Xe, Fe):
    """predict: X -> [n,4]. Velocity rel L2, pressure rel L2 (gauge-adjusted by
    the mean error, since p is defined up to a constant), plus max |div u| is
    left to the caller (needs derivatives)."""
    with torch.no_grad():
        pred = predict(Xe)
    ve = torch.linalg.norm(pred[:, :3] - Fe[:, :3]) / torch.linalg.norm(Fe[:, :3])
    dp = pred[:, 3] - Fe[:, 3]
    dp = dp - dp.mean()
    pe = torch.linalg.norm(dp) / torch.linalg.norm(Fe[:, 3] - Fe[:, 3].mean())
    return float(ve), float(pe)


# -- shared loss ---------------------------------------------------------------

class LossBlocks:
    def __init__(self, weights=None):
        self.w = dict(pde=1.0, cont=1.0, ic=10.0, bc=10.0, gauge=1.0)
        if weights:
            self.w.update(weights)

    def __call__(self, residual_fn, value_fn, Xi, Xic, Xbc, Xg):
        mom, cont = residual_fn(Xi)
        ic_pred = value_fn(Xic)
        ic_true = torch.tensor(bel.fields(Xic.detach().numpy()))
        bc_pred = value_fn(Xbc)
        bc_true = torch.tensor(bel.fields(Xbc.detach().numpy()))
        g_pred = value_fn(Xg)
        g_true = torch.tensor(bel.fields(Xg.detach().numpy()))
        parts = dict(
            pde=(mom ** 2).mean(),
            cont=(cont ** 2).mean(),
            ic=((ic_pred[:, :3] - ic_true[:, :3]) ** 2).mean(),
            bc=((bc_pred[:, :3] - bc_true[:, :3]) ** 2).mean(),
            gauge=((g_pred[:, 3] - g_true[:, 3]) ** 2).mean(),
        )
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: float(v) for k, v in parts.items()}


# -- tensor-basis PINN ---------------------------------------------------------

def warm_start_from_ic(net, m_per_axis=None):
    """Kronecker-fit A to the t-constant extension of the velocity IC
    (pressure column zero). Uses only t=0 data."""
    m = m_per_axis or 2 * net.n_centers
    grids = [np.linspace(lo, hi, m) for lo, hi in bel.DOMAINS]
    P0 = np.stack(np.meshgrid(*grids[:3], indexing="ij"), axis=-1).reshape(-1, 3)
    P0 = np.concatenate([P0, np.zeros((len(P0), 1))], axis=1)
    F0 = bel.fields(P0).reshape(m, m, m, 1, 4)
    Y = np.repeat(F0, m, axis=3)
    Y[..., 3] = 0.0
    tb.kron_fit(net, grids, Y, rcond=1e-15)


def train_tensor_pinn(n_centers=16, lam=0.2, seed=0, adam_steps=1500,
                      adam_batch=2048, lr=2e-4, lbfgs_iters=300,
                      lbfgs_batch=12288, n_ic=1024, n_bc=1536,
                      eval_every=50, weights=None, warm_start=True,
                      log_path=None, train_geometry=False):
    """Note on set sizes: A is heavily overparameterized (4 Nt^4 readout
    coefficients), so the fixed LBFGS collocation set is made several times
    larger than the Adam batches -- a small fixed set can be interpolated
    without solving the PDE."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = tb.TensorBasisNet(n_centers=n_centers, lam=lam, domains=bel.DOMAINS,
                            train_geometry=train_geometry)
    if warm_start:
        warm_start_from_ic(net)
    loss_fn = LossBlocks(weights)
    Xe, Fe = make_eval_set()
    Xg = gauge_points()
    value = lambda X: net.evaluate(X, [(0, 0, 0, 0)])[(0, 0, 0, 0)]
    predict = lambda X: net.evaluate_chunked(X, [(0, 0, 0, 0)])[(0, 0, 0, 0)]
    residual = lambda X: tb.ns_residuals(net, X, bel.NU)

    history = []
    t0 = time.time()

    def record(step, phase, loss, parts):
        ve, pe = eval_errors(predict, Xe, Fe)
        rec = dict(step=step, phase=phase, wall=time.time() - t0,
                   loss=float(loss), rel_l2_v=ve, rel_l2_p=pe, **parts)
        history.append(rec)
        print(rec, flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    with torch.no_grad():
        mom, cont = residual(sample_interior(rng, 4096))
        record(-1, "init", float((mom**2).mean() + (cont**2).mean()), {})

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1))
    for step in range(adam_steps):
        Xi = sample_interior(rng, adam_batch)
        Xic, Xbc = sample_ic(rng, n_ic), sample_bc(rng, n_bc)
        loss, parts = loss_fn(residual, value, Xi, Xic, Xbc, Xg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == adam_steps - 1:
            record(step, "adam", loss.item(), parts)

    # LBFGS on fixed (large) sets
    Xi = sample_interior(rng, lbfgs_batch)
    Xic, Xbc = sample_ic(rng, 3 * n_ic), sample_bc(rng, 3 * n_bc)
    opt = torch.optim.LBFGS(params, max_iter=20, history_size=60,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=0.0, tolerance_change=0.0)
    state = {}

    def closure():
        opt.zero_grad()
        loss, parts = loss_fn(residual, value, Xi, Xic, Xbc, Xg)
        loss.backward()
        state["loss"], state["parts"] = loss.item(), parts
        return loss

    for outer in range(max(lbfgs_iters // 20, 1)):
        opt.step(closure)
        record(adam_steps + 20 * (outer + 1), "lbfgs", state["loss"],
               state["parts"])
    return net, history


# -- baseline tanh-MLP PINN ----------------------------------------------------

class MLP(torch.nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [torch.nn.Linear(4, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 4)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


def mlp_residuals(net, X, nu):
    X = X.detach().requires_grad_(True)
    out = net(X)
    u, v, w, p = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    gp = torch.autograd.grad(p.sum(), X, create_graph=True)[0]
    moms, gs = [], []
    for i, q in enumerate((u, v, w)):
        g = torch.autograd.grad(q.sum(), X, create_graph=True)[0]
        lap = sum(torch.autograd.grad(g[:, a].sum(), X, create_graph=True)[0][:, a]
                  for a in range(3))
        moms.append(g[:, 3] + u * g[:, 0] + v * g[:, 1] + w * g[:, 2]
                    + gp[:, i] - nu * lap)
        gs.append(g)
    cont = gs[0][:, 0] + gs[1][:, 1] + gs[2][:, 2]
    return torch.stack(moms, dim=1), cont


def train_mlp_pinn(width=64, depth=4, seed=0, adam_steps=6000, adam_batch=1024,
                   lr=1e-3, lbfgs_iters=200, lbfgs_batch=2048, n_ic=1024,
                   n_bc=1536, eval_every=100, weights=None, log_path=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = MLP(width=width, depth=depth)
    loss_fn = LossBlocks(weights)
    Xe, Fe = make_eval_set()
    Xg = gauge_points()
    residual = lambda X: mlp_residuals(net, X, bel.NU)
    value = net
    history = []
    t0 = time.time()

    def record(step, phase, loss, parts):
        ve, pe = eval_errors(lambda X: net(X), Xe, Fe)
        rec = dict(step=step, phase=phase, wall=time.time() - t0,
                   loss=float(loss), rel_l2_v=ve, rel_l2_p=pe, **parts)
        history.append(rec)
        print(rec, flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1))
    for step in range(adam_steps):
        Xi = sample_interior(rng, adam_batch)
        Xic, Xbc = sample_ic(rng, n_ic), sample_bc(rng, n_bc)
        loss, parts = loss_fn(residual, value, Xi, Xic, Xbc, Xg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == adam_steps - 1:
            record(step, "adam", loss.item(), parts)

    Xi = sample_interior(rng, lbfgs_batch)
    Xic, Xbc = sample_ic(rng, n_ic), sample_bc(rng, n_bc)
    opt = torch.optim.LBFGS(net.parameters(), max_iter=20, history_size=60,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=0.0, tolerance_change=0.0)
    state = {}

    def closure():
        opt.zero_grad()
        loss, parts = loss_fn(residual, value, Xi, Xic, Xbc, Xg)
        loss.backward()
        state["loss"], state["parts"] = loss.item(), parts
        return loss

    for outer in range(max(lbfgs_iters // 20, 1)):
        opt.step(closure)
        record(adam_steps + 20 * (outer + 1), "lbfgs", state["loss"],
               state["parts"])
    return net, history
