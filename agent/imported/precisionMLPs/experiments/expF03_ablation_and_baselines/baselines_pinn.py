"""expF03 part 3: trained-PINN baseline (torch, float64 where it matters).

Two variants per problem, per literature practice (sources in the writeup):
  lit      3 hidden layers x 128 tanh, Glorot -- the standard benchmark net
           (Expert's Guide / classic Raissi-style recipes); fixed lambda_bc = 100,
           Adam lr 1e-3 with x0.9/2000 exponential decay, then L-BFGS to
           convergence (strong Wolfe). Well-tuned expectation from the
           literature: rel L2 ~ 1e-4..1e-5 (PINNacle; Expert's Guide), with
           ~1e-7 the plain-PINN state of the art (Urban et al. 2024).
  matched  single hidden tanh layer at the part-2 deployed width W* -- the
           architecture-matched comparison to our dictionary.

Hardware probe: MPS has no float64, so the probe times (a) CPU float64
end-to-end vs (b) MPS float32 for the Adam warm phase + CPU float64 L-BFGS
polish. Whichever wins is used and recorded in the JSON.

Each training run has a hard wall-clock cap; steps completed are recorded.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from common import PART2_DIR, PART3_DIR, PART4_DIR, eval_grid, load_problem_suite, metrics
from common import interior_points_2d, _bc_points_for, DEFAULT_CFG

torch.set_default_dtype(torch.float64)

LAMBDA_BC = 100.0
ADAM_STEPS = 5000
LBFGS_MAX = 3000
TIME_CAP = 420.0        # seconds per training run
SEEDS = [0, 1, 2]


def make_net(dim, widths, seed):
    torch.manual_seed(seed)
    layers = []
    sizes = [dim] + widths + [1]
    for i in range(len(sizes) - 1):
        lin = torch.nn.Linear(sizes[i], sizes[i + 1])
        torch.nn.init.xavier_normal_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def pure_derivative(u, X, axis, order):
    """d^order u / d x_axis^order via repeated autograd (pure derivatives only)."""
    out = u
    for _ in range(order):
        g = torch.autograd.grad(out.sum(), X, create_graph=True)[0]
        out = g[:, axis]
    return out


def pde_residual(net, X, terms, forcing_vals):
    X.requires_grad_(True)
    u = net(X).squeeze(-1)
    r = -forcing_vals
    for term in terms:
        if isinstance(term[0], tuple):
            (ax, ay), coeff = term
            d = u
            if ax:
                d = pure_derivative(u, X, 0, ax)
            if ay:
                d = pure_derivative(d if ax else u, X, 1, ay)
            if ax == 0 and ay == 0:
                d = u
        else:
            o, coeff = term
            d = u if o == 0 else pure_derivative(u, X, 0, o)
        if callable(coeff):
            xnp = X.detach().cpu().numpy()
            # 1D coefficient callables expect a flat x array; (n,1) would broadcast
            # c*d to an (n,n) matrix and silently corrupt the loss
            cnp = coeff(xnp[:, 0] if xnp.shape[1] == 1 else xnp)
            c = torch.as_tensor(np.asarray(cnp).ravel(), dtype=X.dtype, device=X.device)
        else:
            c = float(coeff)
        r = r + c * d
    return r


def condition_loss(net, prob, dev, dtype):
    """Sum of MSEs over the problem's condition rows."""
    total = 0.0
    if prob["category"] == "ode1d":
        for (x0, o) in prob["conditions"]:
            X = torch.tensor([[x0]], dtype=dtype, device=dev, requires_grad=True)
            u = net(X).squeeze(-1)
            d = u if o == 0 else pure_derivative(u, X, 0, o)
            val = float(np.asarray(prob["exact_derivs"][o](np.array([x0]))).ravel()[0])
            total = total + (d - val).pow(2).mean()
        return total
    suite_f01 = load_problem_suite("f01")
    for blk in prob["bc_blocks"]:
        Pb = _bc_points_for(blk, prob, suite_f01)
        X = torch.tensor(Pb, dtype=dtype, device=dev, requires_grad=True)
        u = net(X).squeeze(-1)
        r = torch.zeros_like(u)
        for (ax, ay), coeff in blk["terms"]:
            d = u
            if ax:
                d = pure_derivative(u, X, 0, ax)
            if ay:
                d = pure_derivative(d if ax else u, X, 1, ay)
            if ax == 0 and ay == 0:
                d = u
            c = torch.as_tensor(coeff(Pb), dtype=dtype, device=dev) if callable(coeff) \
                else float(coeff)
            r = r + c * d
        g = torch.as_tensor(blk["value"](Pb), dtype=dtype, device=dev)
        total = total + (r - g).pow(2).mean()
    return total


def collocation(prob, n=4096, seed=42):
    rng = np.random.default_rng(seed)
    if prob["category"] == "ode1d":
        return rng.uniform(-1, 1, (n, 1))
    return interior_points_2d(prob["category"], n // 5, DEFAULT_CFG, rng)[:n]


# nonlinear PDE residual (expF02 zoo). The nonlinear part of each problem, keyed by
# problem key, as a pure function of the derivative-field accessor d(ax, ay); it
# mirrors the numpy nl["res"] in expF02/problems.py exactly. The linear part is read
# generically from prob["lin_terms"] and the forcing is subtracted, so the residual
# equals sum_i c_i D_i u + N(Du) - f, identical to the QI/Newton harness.
NL_PART = {
    "ode1d_o1": lambda d: 3.0 * d(0, 0) ** 2,
    "ode1d_o2": lambda d: torch.exp(d(0, 0)),
    "ode1d_o3": lambda d: 0.5 * d(0, 0) * d(2, 0),
    "pde2d_steady_o1": lambda d: d(1, 0) ** 2 + d(0, 1) ** 2,
    "pde2d_steady_o2": lambda d: d(2, 0) * d(0, 2) - d(1, 1) ** 2,
    "pde2d_steady_o3": lambda d: d(0, 0) * d(1, 0),
    "pde2d_time_o1": lambda d: d(0, 0) * d(1, 0),
    "pde2d_time_o2": lambda d: d(0, 0) * d(1, 0),
    "pde2d_time_o3": lambda d: 6.0 * d(0, 0) * d(1, 0),
}


def _mixed(u, X, ax, ay):
    """d^(ax+ay) u / dx^ax dy^ay via repeated autograd (x first, then y)."""
    g = u
    if ax:
        g = pure_derivative(g, X, 0, ax)
    if ay:
        g = pure_derivative(g, X, 1, ay)
    return g


def pde_residual_nl(net, prob, X, fvals):
    X.requires_grad_(True)
    u = net(X).squeeze(-1)
    dim = 1 if prob["category"] == "ode1d" else 2
    d = lambda ax, ay: _mixed(u, X, ax, ay if dim == 2 else 0)
    r = -fvals
    for idx, coeff in prob["lin_terms"]:
        ax, ay = (idx, 0) if dim == 1 else idx
        r = r + float(coeff) * d(ax, ay)          # f02 linear coefficients are constants
    return r + NL_PART[prob["key"]](d)


def total_loss(net, prob, X, fvals):
    if "nl" in prob:
        r = pde_residual_nl(net, prob, X, fvals)
    else:
        r = pde_residual(net, X, prob["terms"], fvals)
    return r.pow(2).mean() + LAMBDA_BC * condition_loss(net, prob, X.device, X.dtype)


def train_pinn(prob, widths, seed, device_adam="cpu"):
    """Adam (possibly on MPS/fp32) then L-BFGS (always CPU/fp64). Returns record."""
    dim = 1 if prob["category"] == "ode1d" else 2
    net = make_net(dim, widths, seed)
    P = collocation(prob, seed=42 + seed)
    f = prob["forcing"]
    fv = f(P if dim == 2 else P[:, 0]) if callable(f) else np.full(len(P), float(f))
    t0 = time.time()
    steps_done = 0

    def run_adam(dev, dtype):
        nonlocal steps_done
        netd = net.to(dtype).to(dev)
        X = torch.tensor(P, dtype=dtype, device=dev)
        fvals = torch.tensor(fv, dtype=dtype, device=dev)
        opt = torch.optim.Adam(netd.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9 ** (1 / 2000))
        for it in range(ADAM_STEPS):
            opt.zero_grad()
            loss = total_loss(netd, prob, X, fvals)
            loss.backward()
            opt.step()
            sched.step()
            steps_done += 1
            if time.time() - t0 > TIME_CAP * 0.6:
                break
        return netd.to("cpu").to(torch.float64)

    dtype = torch.float32 if device_adam == "mps" else torch.float64
    net = run_adam(device_adam, dtype)

    # L-BFGS polish, CPU float64
    X = torch.tensor(P, dtype=torch.float64)
    fvals = torch.tensor(fv, dtype=torch.float64)
    opt = torch.optim.LBFGS(net.parameters(), max_iter=LBFGS_MAX, history_size=50,
                            tolerance_grad=0, tolerance_change=0,
                            line_search_fn="strong_wolfe")
    lbfgs_evals = [0]

    def closure():
        if time.time() - t0 > TIME_CAP:
            raise TimeoutError
        opt.zero_grad()
        loss = total_loss(net, prob, X, fvals)
        loss.backward()
        lbfgs_evals[0] += 1
        return loss

    try:
        opt.step(closure)
    except TimeoutError:
        pass
    t_train = time.time() - t0

    pts, _ = eval_grid(prob["category"])
    with torch.no_grad():
        Xe = torch.tensor(pts.reshape(-1, 1) if dim == 1 else pts, dtype=torch.float64)
        u_hat = net(Xe).squeeze(-1).numpy()
    rel, linf = metrics(u_hat, prob["exact"](pts))
    with torch.enable_grad():
        final = float(total_loss(net, prob, X, fvals).item())
    n_params = sum(p.numel() for p in net.parameters())
    return dict(rel_l2=rel, linf=linf, dof=n_params, t_solve=t_train, t_assemble=0.0,
                adam_steps=steps_done, lbfgs_evals=lbfgs_evals[0], final_loss=final,
                device_adam=device_adam)


def hardware_probe():
    """Time 200 Adam steps on CPU-fp64 vs MPS-fp32 (if available) for the lit net."""
    f01 = load_problem_suite("f01")
    prob = f01.PROBLEMS["pde2d_steady_o2"]
    results = {}
    for dev, dtype in [("cpu", torch.float64)] + \
            ([("mps", torch.float32)] if torch.backends.mps.is_available() else []):
        net = make_net(2, [128, 128, 128], 0).to(dtype).to(dev)
        P = collocation(prob)
        X = torch.tensor(P, dtype=dtype, device=dev)
        fv = prob["forcing"](P)
        fvals = torch.tensor(fv, dtype=dtype, device=dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        t0 = time.time()
        for _ in range(200):
            opt.zero_grad()
            loss = total_loss(net, prob, X, fvals)
            loss.backward()
            opt.step()
        if dev == "mps":
            torch.mps.synchronize()
        results[dev] = time.time() - t0
        print(f"  probe {dev}: 200 Adam steps in {results[dev]:.1f}s", flush=True)
    use_mps = "mps" in results and results["mps"] < 0.5 * results["cpu"]
    return ("mps" if use_mps else "cpu"), results


def run_pinn():
    f01 = load_problem_suite("f01")
    summary = json.loads((PART2_DIR / "select_summary.json").read_text())
    print("hardware probe...", flush=True)
    device_adam, probe = hardware_probe()
    print(f"Adam phase on: {device_adam}", flush=True)
    rows = []
    t_all = time.time()
    for key, prob in f01.PROBLEMS.items():
        w_star = summary[f"f01:{key}"]["w_star"]
        for variant, widths, seeds in [("lit_3x128", [128, 128, 128], SEEDS),
                                       (f"matched_1x{w_star}", [min(w_star, 2304)], [0])]:
            for seed in seeds:
                t0 = time.time()
                rec = train_pinn(prob, widths, seed, device_adam=device_adam)
                rec.update(key=key, size=w_star, seed=seed, variant=variant,
                           probe=probe)
                rows.append(rec)
                print(f"  pinn {key} {variant} s{seed}: relL2={rec['rel_l2']:.1e} "
                      f"loss={rec['final_loss']:.1e} ({rec['t_solve']:.0f}s, "
                      f"{rec['adam_steps']} adam + {rec['lbfgs_evals']} lbfgs)",
                      flush=True)
        out = PART3_DIR / "baselines_pinn.json"
        out.write_text(json.dumps(rows, indent=1))
    print(f"pinn: {len(rows)} runs in {(time.time() - t_all) / 60:.0f} min", flush=True)
    return rows


def run_pinn_nonlinear(seeds=(0,)):
    """Part 4: trained-PINN baseline on the nonlinear zoo (expF02). Lit-standard
    3x128 tanh net, same schedule/loss as part 3; nonlinear residual via NL_PART.
    Smaller scale (default single seed). Writes PART4_DIR/nl_pinn.json incrementally."""
    f02 = load_problem_suite("f02")
    print("hardware probe...", flush=True)
    device_adam, probe = hardware_probe()
    print(f"Adam phase on: {device_adam}", flush=True)
    rows = []
    t_all = time.time()
    for key, prob in f02.PROBLEMS.items():
        for seed in seeds:
            rec = train_pinn(prob, [128, 128, 128], seed, device_adam=device_adam)
            rec.update(key=key, seed=seed, variant="lit_3x128", probe=probe)
            rows.append(rec)
            print(f"  pinn-nl {key} s{seed}: relL2={rec['rel_l2']:.1e} "
                  f"loss={rec['final_loss']:.1e} ({rec['t_solve']:.0f}s, "
                  f"{rec['adam_steps']} adam + {rec['lbfgs_evals']} lbfgs)", flush=True)
        (PART4_DIR / "nl_pinn.json").write_text(json.dumps(rows, indent=1))
    print(f"pinn-nl: {len(rows)} runs in {(time.time() - t_all) / 60:.0f} min", flush=True)
    return rows
