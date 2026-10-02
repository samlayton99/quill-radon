"""expF17 PINN arm (SPEC 18.8): the same single-hidden-layer tanh architecture
as the ELM/QI arms, TRAINED instead of solved.

Architecture: u(x, y) = sum_k w_k tanh(a1_k x + a2_k y - b_k) + c with
(n_ax+1)^2 - 1 hidden units + the output bias = the cell's exact column count.
fp64, PyTorch default init (nn.Linear), CPU.

Data footprint: identical to the solved arms. Oracle regime = supervised MSE on
4C shuffled uniform interior points + the perimeter set with exact values
(solve.oracle_fit's footprint); dynamic regime = PDE residual MSE on the 4C
interior points + w_mult^2 x per-block BC/IC MSE on the same point sets as
solve.build_bcs (w_mult multiplies the row weight, so it enters the loss
squared, exactly as in the lstsq arm; the sqrt(n_pde/n_b) row weight there is
the per-block mean here). Burgers trains at the target viscosity directly (the
vanishing-viscosity ladder is a solver recipe, not a vanilla-PINN one).

Optimizer (FIXED budget, not a knob): Adam at peak lr, cosine-decayed to 0
over ADAM_STEPS full-batch steps, then L-BFGS (strong-Wolfe, cap LBFGS_ITERS
iterations). lr is the declared knob, swept {3e-4, 1e-3, 3e-3} at seed 0 in
the oracle regime with the standard walk; the dynamic regime inherits lr and
sweeps w_mult (the same knob accounting as the solved arms). Wall clock is
recorded on every cell.

Certificate: every trained network also lands a frozen-feature REFIT of its
readout by the exact solver of the solved arms (oracle: truncated lstsq,
rcond 1e-15; dynamic: the same linear solve / damped Gauss-Newton warm-started
at the trained readout). The refit is the exact optimum over everything
optimizer tuning could reach given the learned geometry: it separates
'the optimizer did not converge' from 'the geometry it found cannot represent
the solution'.

Derivatives are closed-form in the hidden pre-activation (psi_o(tanh z) x
a1^ax a2^ay), so the PDE residual is exact and one autograd pass gives the
parameter gradient; this is what nested autograd computes, at a third of the
cost.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import dicts as fd, solve as sv

ADAM_STEPS = 20_000
LBFGS_ITERS = 2_000
LR_GRID = [3e-4, 1e-3, 3e-3]
LOG_EVERY = 100

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# the network (parameter layout mirrors dicts.RidgeDict)
# ---------------------------------------------------------------------------

def init_params(n_ax, seed):
    """PyTorch default init (nn.Linear) for a [2 -> n_units -> 1] tanh MLP,
    seeded as a pure function of (n_ax, seed) like dicts.dict_rng."""
    n_units = (n_ax + 1) ** 2 - 1
    with torch.random.fork_rng():
        torch.manual_seed(1_000_003 * seed + 101 * n_ax + 7_919)
        lin1 = torch.nn.Linear(2, n_units)
        lin2 = torch.nn.Linear(n_units, 1)
    a1 = lin1.weight[:, 0].detach().clone()
    a2 = lin1.weight[:, 1].detach().clone()
    b = (-lin1.bias).detach().clone()          # tanh(a1 x + a2 y - b)
    w = lin2.weight[0].detach().clone()
    c = lin2.bias.detach().clone()
    params = [p.requires_grad_(True) for p in (a1, a2, b, w, c)]
    return params


def _psi(order, t):
    if order == 0:
        return t
    if order == 1:
        return 1.0 - t * t
    if order == 2:
        return -2.0 * t * (1.0 - t * t)
    raise ValueError(order)


def net_derivs(params, P, idxs):
    """{(ax, ay): d^(ax,ay) u at P} for the requested single-axis derivative
    indices (orders <= 2, as every task needs), closed-form."""
    a1, a2, b, w, c = params
    t = torch.tanh(P[:, 0:1] * a1[None, :] + P[:, 1:2] * a2[None, :] - b[None, :])
    out = {}
    for (ax, ay) in idxs:
        fac = (a1 ** ax) * (a2 ** ay)
        val = _psi(ax + ay, t) @ (fac * w)
        if ax + ay == 0:
            val = val + c
        out[(ax, ay)] = val
    return out


def as_dict(params):
    """The learned geometry as a dicts.RidgeDict (bias column last), plus the
    trained readout in that column order."""
    a1, a2, b, w, c = [p.detach().numpy().copy() for p in params]
    d = fd.RidgeDict(a1, a2, b, dict(method="pinn", n_units=len(a1)))
    a = np.concatenate([w, c])
    return d, a


# ---------------------------------------------------------------------------
# data (the solved arms' footprints, verbatim)
# ---------------------------------------------------------------------------

def _T(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def oracle_data(task, n_ax, seed):
    cols = (n_ax + 1) ** 2
    rng = np.random.default_rng(90_000 + seed)
    P = sv.interior_points(task, sv.OVERSAMPLE * cols, rng)
    P = np.vstack([P, sv.perimeter_points(task)])
    return dict(P=_T(P), y=_T(task["exact"](P)), n_data=len(P))


def _bc_pointsets(task):
    """[(kind, points, terms, values)] -- kind 'diff' carries (PL, PR)."""
    out = []
    for blk in task["bc_blocks"]:
        where = blk["where"]
        if where == "periodic_x":
            eta = np.linspace(-1.0, 1.0, sv.N_PERIODIC)
            PL = np.stack([np.full_like(eta, -1.0), eta], axis=1)
            PR = np.stack([np.full_like(eta, 1.0), eta], axis=1)
            out.append(("diff", (PL, PR), blk["terms"], np.zeros(len(eta))))
        else:
            Pb = (sv._square() if where == "square"
                  else sv._holes() if where == "holes" else sv._edge(where))
            val = blk["value"]
            g = val(Pb) if callable(val) else np.full(len(Pb), float(val))
            out.append(("rows", Pb, blk["terms"], g))
    return out


def dynamic_data(task, n_ax, seed):
    cols = (n_ax + 1) ** 2
    rng = np.random.default_rng(90_000 + seed)
    P = sv.interior_points(task, sv.OVERSAMPLE * cols, rng)
    lin = [(idx, fd._coeff_col(c, P)) for idx, c in task["lin_terms"]]
    lin = [(idx, _T(cc).reshape(-1) if np.ndim(cc) else float(cc)) for idx, cc in lin]
    f = task["forcing"]
    fv = f(P) if callable(f) else np.full(len(P), float(f))
    idxs = sorted({i for i, _ in task["lin_terms"]} | set(task["nl"]["fields"]))
    blocks = []
    for kind, pts, terms, g in _bc_pointsets(task):
        terms_t = [(idx, (_T(fd._coeff_col(c, pts[0] if kind == "diff" else pts)).reshape(-1)
                          if callable(c) else float(c))) for idx, c in terms]
        if kind == "diff":
            blocks.append((kind, (_T(pts[0]), _T(pts[1])), terms_t, _T(g)))
        else:
            blocks.append((kind, _T(pts), terms_t, _T(g)))
    return dict(P=_T(P), P_np=P, lin=lin, fv=_T(fv), idxs=idxs, blocks=blocks,
                nl=task["nl"], n_data=len(P))


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def oracle_loss(params, data):
    u = net_derivs(params, data["P"], [(0, 0)])[(0, 0)]
    return torch.mean((u - data["y"]) ** 2)


def _block_residual(params, kind, pts, terms, g):
    idxs = [idx for idx, _ in terms]
    if kind == "diff":
        DL = net_derivs(params, pts[0], idxs)
        DR = net_derivs(params, pts[1], idxs)
        r = sum(cc * (DL[idx] - DR[idx]) for idx, cc in terms)
    else:
        D = net_derivs(params, pts, idxs)
        r = sum(cc * D[idx] for idx, cc in terms)
    return r - g


def dynamic_losses(params, data, w_mult):
    """-> (total, pde_mse, [block mse]) with total = pde + w_mult^2 sum(block)."""
    D = net_derivs(params, data["P"], data["idxs"])
    r = sum(cc * D[idx] for idx, cc in data["lin"])
    nl = data["nl"]
    if nl["fields"]:
        r = r + nl["res"](D, data["P_np"])
    r = r - data["fv"]
    pde = torch.mean(r * r)
    blk = []
    for kind, pts, terms, g in data["blocks"]:
        rb = _block_residual(params, kind, pts, terms, g)
        blk.append(torch.mean(rb * rb))
    total = pde + (w_mult ** 2) * sum(blk)
    return total, pde, blk


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(loss_fn, params, lr, adam_steps=ADAM_STEPS, lbfgs_iters=LBFGS_ITERS,
          log_every=LOG_EVERY):
    """Adam (cosine to 0) then L-BFGS polish. Returns (params, info)."""
    hist = []
    t0 = time.time()
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_steps, eta_min=0.0)
    for step in range(adam_steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(params)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == adam_steps - 1:
            hist.append((step, float(loss)))
    t_adam = time.time() - t0
    loss_adam = float(loss_fn(params))
    snapshot = [p.detach().clone() for p in params]

    t1 = time.time()
    n_eval = [0]
    if lbfgs_iters > 0:
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters,
                               max_eval=int(1.25 * lbfgs_iters), history_size=50,
                               tolerance_grad=0.0, tolerance_change=0.0,
                               line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad(set_to_none=True)
            loss = loss_fn(params)
            n_eval[0] += 1
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            lb.step(closure)
        except Exception:  # noqa: BLE001 -- a failed line search must not lose the Adam result
            pass
    loss_final = float(loss_fn(params))
    if not np.isfinite(loss_final) or loss_final > loss_adam:
        with torch.no_grad():  # the polish must never make things worse
            for p, s in zip(params, snapshot):
                p.copy_(s)
        loss_final = loss_adam
    t_lbfgs = time.time() - t1
    hist.append((adam_steps + n_eval[0], loss_final))
    info = dict(lr=lr, loss_adam=loss_adam, loss_final=loss_final, hist=hist,
                lbfgs_evals=n_eval[0], t_adam=round(t_adam, 1),
                t_lbfgs=round(t_lbfgs, 1))
    return params, info


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def _refit(task, d, a_trained, regime, seed, w_mult):
    """The frozen-feature certificate: the exact solver on the learned geometry."""
    if regime == "oracle":
        a, _ = sv.oracle_fit(task, d, seed)
        return a
    if task["key"] == "burgers":
        prob = sv.f13.make_burgers(sv.f13.NU_BURGERS)
        task = dict(task, lin_terms=prob["lin_terms"], nl=prob["nl"])
    asm = sv.assemble(task, d, seed, w_mult=w_mult)
    if not task["nl"]["fields"]:
        a, _ = sv.linear_solve(asm)
        return a
    a, _ = sv.gauss_newton(task, asm, a_trained)
    return a


def run_pinn(task, n_ax, regime, seed, lr, w_mult=1.0, adam_steps=ADAM_STEPS,
             lbfgs_iters=LBFGS_ITERS):
    """Train one cell; -> (trained metrics, refit metrics, info)."""
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.time()
    params = init_params(n_ax, seed)
    if regime == "oracle":
        data = oracle_data(task, n_ax, seed)
        loss_fn = lambda p: oracle_loss(p, data)  # noqa: E731
    else:
        data = dynamic_data(task, n_ax, seed)
        loss_fn = lambda p: dynamic_losses(p, data, w_mult)[0]  # noqa: E731
    params, info = train(loss_fn, params, lr, adam_steps=adam_steps,
                         lbfgs_iters=lbfgs_iters)
    d, a = as_dict(params)
    m_train = sv.score(task, d, a)
    a_ref = _refit(task, d, a, regime, seed, w_mult)
    m_refit = sv.score(task, d, a_ref)
    info.update(n_data=data["n_data"], t_total=round(time.time() - t0, 1),
                readout_norm=float(np.linalg.norm(a)),
                refit_readout_norm=float(np.linalg.norm(a_ref)),
                gamma_rms=float(np.sqrt(np.mean(d.a1 ** 2 + d.a2 ** 2))))
    return m_train, m_refit, info
