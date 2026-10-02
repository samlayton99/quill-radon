"""Standard L-BFGS refinement of saved probes, with ridge continuation.

This is a small numerical diagnostic, not a claim of a scalable precision solver.
It counts objective/gradient evaluations and retains training-monotone states.
"""
import argparse
import json
import math
import time

import torch

from probe import OUT, configure, qi2, tasks, metrics, ridge_forward, ridge_from_features, solve_head, save


def restore(path):
    checkpoint = torch.load(path, weights_only=False)
    a = checkpoint["config"]
    seed, arm, name = checkpoint["seed"], checkpoint["arm"], checkpoint["target"]
    D = tasks.analytic(name, a["d"], a["ntrain"], a["ntest"], seed=seed + 1200)
    net = qi2.make_block(a["d"], a["M"], a["N1"], a["K"], a["N2"], seed=seed)
    if arm == "mlp":
        width = qi2.mlp_width_for(net.counts()["params"], a["d"], "params")
        net = qi2.MLP(a["d"], [width, width], seed=seed)
    else:
        configure(net, D["Xtr"], arm, seed)
    net.load_state_dict(checkpoint["state"])
    return net, D, checkpoint


def run(net, D, max_eval=150, ridges=(1e-6, 1e-8, 1e-10), beta=.01):
    head = {id(p) for p in net.readout()}
    params = [p for p in net.parameters() if id(p) not in head]
    for p in net.readout():
        p.requires_grad_(False)
    curve = []
    calls = 0
    t0 = time.time()
    y = D["Ytr"].reshape(len(D["Xtr"]), -1)
    for ridge in ridges:
        # Ten (s, y) history pairs: twenty vectors, O(nonlinear parameters) state.
        opt = torch.optim.LBFGS(params, lr=1., max_iter=10, max_eval=15, history_size=10,
                               tolerance_grad=1e-13, tolerance_change=1e-15, line_search_fn="strong_wolfe")
        stage_start = calls

        def closure():
            nonlocal calls
            calls += 1
            opt.zero_grad(set_to_none=True)
            out, pres = ridge_forward(net, D["Xtr"], y, ridge)
            weight = net.readout()[0]
            objective = (out - y).square().mean() + ridge * weight.square().sum() + beta * qi2.band_penalty(pres)
            objective.backward()
            return objective

        with torch.no_grad():
            curve.append(dict(calls=calls, ridge=ridge, **metrics(net, D, ridge=ridge)))
        while calls - stage_start < max_eval:
            saved = [p.detach().clone() for p in params]
            initial = float(closure())
            opt.step(closure)
            final = float(closure())
            if not math.isfinite(final) or final > initial * (1 + 1e-10):
                with torch.no_grad():
                    for p, old in zip(params, saved):
                        p.copy_(old)
                break
            curve.append(dict(calls=calls, objective=final, ridge=ridge, **metrics(net, D, ridge=ridge)))
            if abs(initial-final) <= 1e-16 * max(1., abs(initial)):
                break
    return dict(curve=curve, final=curve[-1], precise=metrics(net, D), calls=calls, seconds=time.time()-t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="confirm_d3")
    ap.add_argument("--target", default="fast_waves")
    ap.add_argument("--seeds", default="3")
    ap.add_argument("--max-eval", type=int, default=150)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    path = OUT / f"{args.source}_refine.json"
    rows = json.loads(path.read_text()) if path.exists() else {}
    for seed in args.seeds.split(","):
        for arm in ["raw", "poly3", "quadratic", "mlp"]:
            checkpoint = OUT / f"{args.source}_{args.target}_{seed}_{arm}.pt"
            key = f"{args.target}|{seed}|{arm}"
            if not checkpoint.exists() or key in rows:
                continue
            net, D, cp = restore(checkpoint)
            result = run(net, D, max_eval=args.max_eval)
            result.update(target=args.target, seed=int(seed), arm=arm, max_eval_per_stage=args.max_eval)
            rows[key] = result
            save(path, rows)
            print(key, result["final"]["test"], result["precise"]["test"], result["calls"], flush=True)
