"""expI02 -- a first prior on what the QI blocks are good for. Studies as subcommands, resumable JSON per study.

    uv run --extra dev python experiments/expI02_block_prior/run.py a0 [--quick]
    uv run --extra dev python experiments/expI02_block_prior/run.py plot

Studies: a0 fixed-feature ladder; a1 mechanism ablation; a2 scaling; b1 analytic head-to-head; b2 structured
regression; b3 Fashion-MNIST; b4 PINNs. Results: results/checkpoint_I_depth_theory/expI02_block_prior/<study>.json,
figures under figures/. Spec: docs/superpowers/specs/2026-09-05-expI02-block-prior-design.md.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qi2      # noqa: E402
import tasks    # noqa: E402
import plots    # noqa: E402

RES = tasks.REPO / "results" / "checkpoint_I_depth_theory" / "expI02_block_prior"
QUICK = False


class Store:
    """One JSON file per study; every finished run is written immediately (atomic replace); finished keys are skipped."""
    def __init__(self, name):
        RES.mkdir(parents=True, exist_ok=True)
        self.path = RES / f"{name}.json"
        self.data = json.load(open(self.path)) if self.path.exists() else {}

    def done(self, key):
        return key in self.data

    def get(self, key):
        return self.data.get(key)

    def put(self, key, val):
        self.data[key] = val
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)


def setup(fp64=True, threads=None):
    torch.set_default_dtype(torch.float64 if fp64 else torch.float32)
    if threads:
        torch.set_num_threads(threads)


def log_row(log, **extra):
    """The part of a fit log that goes into the store."""
    row = dict(step=log["step"], train=log["train"], test=log["test_trained"], solved=log["test_solved"], band=log["band"],
               final=log["final"], final_trained=log["final_trained"], L=log["L"], time=log["time"], counts=log["counts"])
    if "final_gn" in log:
        row["final_gn"] = log["final_gn"]; row["gn_curve"] = log["gn_curve"]
    row.update(extra)
    return row


def say(*a):
    print(*a, flush=True)


# ============================================================================= A0: the fixed-feature ladder
A0_TARGETS = {
    "mixed_sine": lambda x: torch.sin(2 * math.pi * x) + 0.5 * torch.sin(6 * math.pi * x),
    "exp": lambda x: torch.exp(x),
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x ** 2),
    "gauss": lambda x: torch.exp(-((x - 0.3) / 0.25) ** 2),
}


def study_a0(store):
    """Frozen 1-D tanh bank (80 cells on [-1, 1], data on [-0.8, 0.8]); only the head is trained, from zero, for
    1000 full-batch steps; coordinates x optimizer; the solved head is the floor."""
    setup(fp64=True)
    N, n, steps = 80, 4096, (100 if QUICK else 1000)
    g = torch.Generator().manual_seed(0)
    x = 2 * torch.rand(n, 1, generator=g) - 1
    xt = 1.8 * torch.rand(n, 1, generator=g) - 0.9
    for name, fn in A0_TARGETS.items():
        y, yt = fn(x[:, 0]), fn(xt[:, 0])
        if not store.done(f"{name}|floor"):
            net = qi2.make_block(1, M=1, N1=N, K=1, N2=1, depth=1)
            with torch.no_grad():
                net.layers[0].V.fill_(0.8); net.layers[0].bV.zero_()
            net.solve_readout(x, y)
            store.put(f"{name}|floor", qi2.rel_l2(net(xt)[:, 0], yt))
        for coords in qi2.COORDS:
            for opt in ["gd", "momentum", "adam"]:
                key = f"{name}|{coords}|{opt}"
                if store.done(key):
                    continue
                net = qi2.make_block(1, M=1, N1=N, K=1, N2=1, depth=1, head_coords=coords)
                with torch.no_grad():
                    net.layers[0].V.fill_(0.8); net.layers[0].bV.zero_()
                    net.layers[0].mixer.theta.zero_()
                net.layers[0].V.requires_grad_(False); net.layers[0].bV.requires_grad_(False)
                log = qi2.fit(net, x, y, xt, yt, optimizer=opt, head="trained", steps=steps, beta=0.0, log_every=25,
                              calibrate_first=False, lr=5e-3, warmup=1)
                store.put(key, log_row(log))
                say(f"a0 {name:11s} {coords:10s} {opt:8s} trained {log['final_trained']:.2e}  solved {log['final']:.2e}  L {log['L'] if log['L'] else float('nan'):.2e}  [{log['time']:.0f}s]")


# ============================================================================= A1: the mechanism ladder
A1_TARGETS = ["gauss_bump", "fast_waves", "composition", "three_bumps"]
A1_D = 3
A1_BASE = dict(M=6, N1=32, K=2, N2=64)

# rung -> (block kwargs, init, fit kwargs); cumulative from the notebook recipe to the corrected one
A1_RUNGS = {
    "old":            (dict(legacy=True, coords="raw"), "zero",   dict(optimizer="adam", head="varpro", beta=0.0, calibrate_first=False)),
    "init":           (dict(legacy=True, coords="raw"), "smooth", dict(optimizer="adam", head="varpro", beta=0.0, calibrate_first=False)),
    "mesh":           (dict(coords="raw"),              "smooth", dict(optimizer="adam", head="varpro", beta=1e-2)),
    "val":            (dict(coords="val"),              "smooth", dict(optimizer="adam", head="varpro", beta=1e-2)),
    "white":          (dict(coords="white"),            "smooth", dict(optimizer="adam", head="varpro", beta=1e-2)),
    "val_white":      (dict(coords="val_white"),        "smooth", dict(optimizer="adam", head="varpro", beta=1e-2)),
    "white+mom":      (dict(coords="white"),            "smooth", dict(optimizer="momentum", head="varpro", beta=1e-2)),
    "white+mixed":    (dict(coords="white"),            "smooth", dict(optimizer="mixed", head="varpro", beta=1e-2)),
    "val_white+mom":  (dict(coords="val_white"),        "smooth", dict(optimizer="momentum", head="varpro", beta=1e-2)),
    "val_white+mixed": (dict(coords="val_white"),       "smooth", dict(optimizer="mixed", head="varpro", beta=1e-2)),
}
A1_LADDER = list(A1_RUNGS)
A1_CANDIDATES = A1_LADDER[2:]                       # the winner is chosen among the corrected-recipe rungs


def a1_data(name, seed, cfg=A1_BASE, d=A1_D):
    units = cfg["M"] * cfg["N1"] + cfg["K"] * cfg["N2"]
    return tasks.analytic(name, d, max(2048, 8 * units), seed=seed)


def build_block(D, cfg, block_kw, init, seed, d_out=1):
    kw = dict(block_kw)
    if kw.get("legacy"):
        kw["band_r"] = D["r"]
    net = qi2.make_block(D["d"], cfg["M"], cfg["N1"], cfg["K"], cfg["N2"], d_out=d_out, seed=seed, **kw)
    if init == "smooth" and kw.get("legacy"):
        gen = torch.Generator().manual_seed(seed + 5)
        with torch.no_grad():
            for l in net.layers[:-1]:
                qi2._init_smooth(l.mixer, gen)
    return net


def run_rung(D, cfg, rung, seed, steps, gn):
    block_kw, init, fit_kw = A1_RUNGS[rung] if isinstance(rung, str) else rung
    net = build_block(D, cfg, block_kw, init, seed)
    log = qi2.fit(net, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], steps=steps, log_every=100, gn_iters=gn, seed=seed, **fit_kw)
    return log_row(log, block=block_kw, init=init, fit=fit_kw)


@torch.no_grad()
def oracle_floor(D, cfg):
    """The factorization given: oracle directions, channel solved against the standardized known P, readout solved.
    The representational floor of the block at (N1, N2) with K = 1, nothing learned."""
    if D["oracle"] is None:
        return None
    V, Pfn = D["oracle"]
    net = qi2.make_block(D["d"], V.shape[0], cfg["N1"], 1, cfg["N2"], depth=2)
    l0 = net.layers[0]
    l0.V.copy_(V); l0.bV.zero_()
    qi2.calibrate(net, D["Xtr"])                                       # scales the oracle rows onto the band
    P = Pfn(D["Xtr"]); P = (P - P.mean()) / P.std() * qi2.S_STAR
    H1 = l0.hidden(D["Xtr"]).reshape(len(P), -1)
    C, b, _ = qi2.lstsq_bias(H1, P.view(-1, 1))
    l0.mixer.set_raw(C.view(l0.M, l0.N, 1)); l0.mixer.bias.copy_(b)
    chan = qi2.rel_l2(net.layers[0](D["Xte"])[:, 0], (Pfn(D["Xte"]) - Pfn(D["Xtr"]).mean()) / Pfn(D["Xtr"]).std() * qi2.S_STAR)
    net.solve_readout(D["Xtr"], D["Ytr"])
    return dict(err=qi2.rel_l2(net(D["Xte"])[:, 0], D["Yte"]), chan=chan, counts=net.counts())


def gmean(vals):
    return math.exp(sum(math.log(max(v, 1e-300)) for v in vals) / len(vals))


def study_a1(store):
    setup(fp64=True)
    steps, gn = (200, 3) if QUICK else (2000, 8)
    data = {name: a1_data(name, 0) for name in A1_TARGETS}
    for name, D in data.items():
        if not store.done(f"{name}|oracle"):
            store.put(f"{name}|oracle", oracle_floor(D, A1_BASE))
            o = store.get(f"{name}|oracle")
            say(f"a1 {name:12s} oracle floor {o['err'] if o else float('nan'):.2e}")
    # the ladder
    for rung in A1_LADDER:
        for name, D in data.items():
            key = f"{name}|{rung}|0"
            if store.done(key):
                continue
            row = run_rung(D, A1_BASE, rung, 0, steps, gn)
            store.put(key, row)
            say(f"a1 {name:12s} {rung:16s} final {row['final']:.2e}  +GN {row.get('final_gn', float('nan')):.2e}  [{row['time']:.0f}s]")
    # the winner among the corrected rungs: smallest geometric-mean solved error over the targets (first order, no GN)
    scores = {r: gmean([store.get(f"{n}|{r}|0")["final"] for n in A1_TARGETS]) for r in A1_CANDIDATES}
    winner = min(scores, key=scores.get)
    store.put("_winner", dict(name=winner, scores=scores))
    say(f"a1 winner: {winner}  scores {' '.join(f'{k}={v:.1e}' for k, v in scores.items())}")
    bk, init, fk = A1_RUNGS[winner]
    variants = {
        f"{winner}+trained":  (bk, init, dict(fk, head="trained")),
        f"{winner}+periodic": (bk, init, dict(fk, head="periodic")),
        f"{winner}+beta0":    (bk, init, dict(fk, beta=0.0)),
        f"{winner}+rank1":    (dict(bk, rank=1), init, fk),
        f"{winner}+rank2":    (dict(bk, rank=2), init, fk),
    }
    if "white" in bk.get("coords", ""):
        variants[f"{winner}+cap1e4"] = (dict(bk, cap=1e4), init, fk)
    for vname, spec in variants.items():
        for name, D in data.items():
            key = f"{name}|{vname}|0"
            if store.done(key):
                continue
            row = run_rung(D, A1_BASE, spec, 0, steps, gn)
            store.put(key, row)
            say(f"a1 {name:12s} {vname:22s} final {row['final']:.2e}  trained-head {row['final_trained']:.2e}  +GN {row.get('final_gn', float('nan')):.2e}  [{row['time']:.0f}s]")
    # seeds on the top three rungs (ladder plus variants)
    every = {r: gmean([store.get(f"{n}|{r}|0")["final"] for n in A1_TARGETS]) for r in A1_CANDIDATES + list(variants)}
    top3 = sorted(every, key=every.get)[:3]
    store.put("_top3", top3)
    for rung in top3:
        spec = A1_RUNGS.get(rung) or variants[rung]
        for seed in ([1] if QUICK else [1, 2]):
            for name in A1_TARGETS:
                key = f"{name}|{rung}|{seed}"
                if store.done(key):
                    continue
                D = a1_data(name, seed)
                row = run_rung(D, A1_BASE, spec, seed, steps, gn)
                store.put(key, row)
                say(f"a1 {name:12s} {rung:22s} seed {seed} final {row['final']:.2e}  [{row['time']:.0f}s]")


def study_a1b(store):
    """A1 follow-ups: (a) the mixed rungs re-run with the heavy-ball curvature taken on its own group (the first pass
    took it over all parameters, a bug); (b) an lr check for raw against whitened coordinates under Adam, since a
    coordinate change alters the parameter scale and one lr is not one setting; (c) the winner re-scored."""
    setup(fp64=True)
    steps, gn = (200, 3) if QUICK else (2000, 8)
    data = {name: a1_data(name, 0) for name in A1_TARGETS}
    for rung in ["white+mixed", "val_white+mixed"]:
        for name, D in data.items():
            key = f"{name}|{rung}|0"
            if store.get(key) and store.get(key).get("fixed_L"):
                continue
            row = run_rung(D, A1_BASE, rung, 0, steps, gn); row["fixed_L"] = True
            store.put(key, row)
            say(f"a1b {name:12s} {rung:16s} (fixed L) final {row['final']:.2e}  +GN {row.get('final_gn', float('nan')):.2e}  [{row['time']:.0f}s]")
    for rung in ["mesh", "white"]:
        bk, init, fk = A1_RUNGS[rung]
        for lr in ([2e-2] if QUICK else [1e-3, 2e-2]):
            spec = (bk, init, dict(fk, lr=lr))
            for name, D in data.items():
                key = f"{name}|{rung}+lr{lr:g}|0"
                if store.done(key):
                    continue
                row = run_rung(D, A1_BASE, spec, 0, steps, 0)
                store.put(key, row)
                say(f"a1b {name:12s} {rung}+lr{lr:g}   final {row['final']:.2e}  [{row['time']:.0f}s]")
    # (d) the periodic-head variant re-run with the head trained between its replacements (the first pass froze it)
    wname = (store.get("_winner") or {}).get("name", "mesh")
    bk, init, fk = A1_RUNGS[wname]
    for name, D in data.items():
        key = f"{name}|{wname}+periodic|0"
        if store.get(key) and store.get(key).get("trained_between"):
            continue
        row = run_rung(D, A1_BASE, (bk, init, dict(fk, head="periodic")), 0, steps, gn); row["trained_between"] = True
        store.put(key, row)
        say(f"a1b {name:12s} {wname}+periodic (trained between) final {row['final']:.2e}  trained-head {row['final_trained']:.2e}  +GN {row.get('final_gn', float('nan')):.2e}  [{row['time']:.0f}s]")
    scores = {r: gmean([store.get(f"{n}|{r}|0")["final"] for n in A1_TARGETS]) for r in A1_CANDIDATES}
    winner = min(scores, key=scores.get)
    prev = (store.get("_winner") or {}).get("name")
    store.put("_winner", dict(name=winner, scores=scores, previous=prev))
    say(f"a1b winner after the fix: {winner} (was {prev})  " + " ".join(f"{k}={v:.1e}" for k, v in scores.items()))


# ============================================================================= the winning recipe (from A1)
DEFAULT_RECIPE = ("white+mixed", (dict(coords="white"), "smooth", dict(optimizer="mixed", head="varpro", beta=1e-2)))


RECIPE_LR = 2e-2   # a1b lr check: 0.02 beat the ladder's 0.005 on the winner by 6x in geometric mean over the four targets


def recipe():
    """(name, (block kwargs, init, fit kwargs)) of the A1 winner at the a1b lr; the prior's expected winner if A1 has
    not run."""
    a1 = Store("a1")
    w = a1.get("_winner")
    if w and w["name"] in A1_RUNGS:
        bk, init, fk = A1_RUNGS[w["name"]]
        return f"{w['name']}+lr{RECIPE_LR:g}", (bk, init, dict(fk, lr=RECIPE_LR))
    say("a1 winner not available; using the default recipe", DEFAULT_RECIPE[0])
    return DEFAULT_RECIPE


def base_cfg(d):
    return dict(M=max(6, 2 * d), N1=32, K=2, N2=64)


def fit_block(D, cfg, spec, seed, steps, gn=0, d_out=1, depth=2, residual=False):
    block_kw, init, fit_kw = spec
    kw = dict(block_kw, depth=depth, residual=residual)
    net = build_block(D, cfg, kw, init, seed, d_out=d_out)
    log = qi2.fit(net, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], steps=steps, log_every=100, gn_iters=gn, seed=seed, **fit_kw)
    return log_row(log, block=kw, init=init, fit=fit_kw, cfg=cfg)


def fit_shallow(D, M, N1, spec, seed, steps, gn=0, d_out=1):
    """The checkpoint-H model: one projection layer, solved readout, directions learned by the same optimizer."""
    _, _, fit_kw = spec
    net = qi2.make_block(D["d"], M, N1, 1, 1, d_out=d_out, depth=1, seed=seed)
    log = qi2.fit(net, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], steps=steps, log_every=100, gn_iters=gn, seed=seed,
                  optimizer=fit_kw["optimizer"], head="varpro", beta=fit_kw["beta"])
    return log_row(log, M=M, N1=N1)


def fit_mlp(D, w, seed, steps, act="tanh", d_out=1, lr=3e-3, depth=2):
    """Plain MLP, standard init, Adam on everything, the solved-head diagnostic logged, final head solve."""
    net = qi2.MLP(D["d"], [w] * depth, d_out=d_out, act=act, seed=seed)
    log = qi2.fit(net, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], steps=steps, log_every=100, seed=seed, optimizer="adam",
                  head="trained", beta=0.0, lr=lr, calibrate_first=False)
    return log_row(log, w=w, act=act)


# ============================================================================= A2: scaling
A2_TARGETS = ["fast_waves", "composition", "product_peak"]
A2_D = 4
A2_SWEEP = {"N1": [8, 16, 32, 64], "N2": [32, 64, 128], "M": [4, 8, 16], "K": [1, 2, 4]}
A2_DEPTH = [(2, False), (3, False), (4, False), (3, True), (4, True)]
A2_MLP_W = [16, 32, 64, 128, 256]


def study_a2(store):
    """One knob at a time from the base config at d = 4, plus depth with and without residual streams, plus the MLP
    width ladder on the same targets; error against parameters."""
    setup(fp64=True)
    steps = 200 if QUICK else 2000
    name_r, spec = recipe()
    store.put("_recipe", name_r)
    sweep = {k: ([v[0], v[-1]] if QUICK else v) for k, v in A2_SWEEP.items()}
    for name in A2_TARGETS:
        D = tasks.analytic(name, A2_D, max(2048, 8 * (base_cfg(A2_D)["M"] * 64 + 8 * 128)), seed=0)
        if not store.done(f"{name}|oracle"):
            store.put(f"{name}|oracle", oracle_floor(D, base_cfg(A2_D)))
        for knob, vals in sweep.items():
            for v in vals:
                cfg = dict(base_cfg(A2_D)); cfg[knob] = v
                key = f"{name}|{knob}|{v}"
                if store.done(key):
                    continue
                row = fit_block(D, cfg, spec, 0, steps)
                store.put(key, row)
                say(f"a2 {name:13s} {knob}={v:<4d} final {row['final']:.2e}  params {row['counts']['params']}  [{row['time']:.0f}s]")
        for depth, res in ([(2, False), (3, True)] if QUICK else A2_DEPTH):
            key = f"{name}|depth|{depth}{'r' if res else ''}"
            if store.done(key):
                continue
            row = fit_block(D, base_cfg(A2_D), spec, 0, steps, depth=depth, residual=res)
            store.put(key, row)
            say(f"a2 {name:13s} depth={depth}{'r' if res else ' '}  final {row['final']:.2e}  params {row['counts']['params']}  [{row['time']:.0f}s]")
        for w in ([16, 64] if QUICK else A2_MLP_W):
            key = f"{name}|mlp|{w}"
            if store.done(key):
                continue
            row = fit_mlp(D, w, 0, steps)
            store.put(key, row)
            say(f"a2 {name:13s} mlp w={w:<4d} final {row['final']:.2e}  params {row['counts']['params']}  [{row['time']:.0f}s]")


# ============================================================================= B1: analytic head-to-head at d = 5
B1_TARGETS = ["gauss_bump", "fast_waves", "composition", "product_peak", "three_bumps", "random_ridges"]
B1_D = 5


def run_matched(store, exp, D, cfg, spec, seed, steps, gn):
    """block (recipe) | shallow at the same units | MLP at the same params | MLP at the same FLOPs; one key per seed."""
    key = f"{D['name']}|{seed}"
    if store.done(key):
        return store.get(key)
    row = {}
    row["block"] = fit_block(D, cfg, spec, seed, steps, gn=gn if seed == 0 else 0)
    c = row["block"]["counts"]
    row["shallow"] = fit_shallow(D, max(2, c["units"] // cfg["N1"]), cfg["N1"], spec, seed, steps, gn=gn if seed == 0 else 0)
    row["mlp_params"] = fit_mlp(D, qi2.mlp_width_for(c["params"], D["d"], "params"), seed, steps)
    row["mlp_flops"] = fit_mlp(D, qi2.mlp_width_for(c["flops"], D["d"], "flops"), seed, steps)
    store.put(key, row)
    say(f"{exp} {D['name']:13s} seed {seed}  block {row['block']['final']:.1e} (+GN {row['block'].get('final_gn', float('nan')):.1e})  "
        f"shallow {row['shallow']['final']:.1e}  mlp@p {row['mlp_params']['final']:.1e} (w={row['mlp_params']['w']})  "
        f"mlp@f {row['mlp_flops']['final']:.1e} (w={row['mlp_flops']['w']})  [{c['params']}p/{c['units']}u]")
    return row


def study_b1(store):
    setup(fp64=True)
    steps, gn = (200, 3) if QUICK else (4000, 20)          # the record's E2 budget (expI01: 4000 steps, 40 GN)
    name_r, spec = recipe()
    store.put("_recipe", name_r)
    cfg = base_cfg(B1_D)
    for name in (B1_TARGETS[:2] if QUICK else B1_TARGETS):
        for seed in ([0] if QUICK else [0, 1, 2]):
            D = tasks.analytic(name, B1_D, max(2048, 8 * (cfg["M"] * cfg["N1"] + cfg["K"] * cfg["N2"])), seed=seed)
            run_matched(store, "b1", D, cfg, spec, seed, steps, gn)


# ============================================================================= B2: structured regression (fp32 except the noise-free Lorenz map)
B2_SETS = ["friedman1", "lorenz_map", "kin8nm", "concrete"]


def b2_data(name, seed):
    if name == "friedman1":
        return tasks.friedman1(seed=seed)
    if name == "lorenz_map":
        return tasks.lorenz_map(seed=seed)
    return tasks.openml_regression(name, seed=seed)


def ridge_rmse(D):
    """Ridge regression with alpha by leave-one-out on the training set: the linear baseline."""
    from sklearn.linear_model import RidgeCV
    m = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100]).fit(D["Xtr"].cpu().numpy(), D["Ytr"].cpu().numpy())
    pred = torch.as_tensor(m.predict(D["Xte"].cpu().numpy())).to(D["Yte"].dtype).reshape(D["Yte"].shape)
    return float(((pred - D["Yte"]) ** 2).mean().sqrt())


def validated_rcond(model, D):
    """Noisy labels: the head's truncation chosen by two-fold validation under a readout-norm cap (expI01 E6 lesson)."""
    from qiblocks import pick_rcond
    with torch.no_grad():
        return pick_rcond(model.feats(D["Xtr"]), D["Ytr"])


def rmse(model, D):
    with torch.no_grad():
        return float(((model(D["Xte"]) - D["Yte"]) ** 2).mean().sqrt())


def study_b2(store):
    steps = 200 if QUICK else 2000
    name_r, spec = recipe()
    store.put("_recipe", name_r)
    for name in (["friedman1"] if QUICK else B2_SETS):
        for seed in ([0] if QUICK else [0, 1, 2]):
            key = f"{name}|{seed}"
            if store.done(key):
                continue
            setup(fp64=(name == "lorenz_map"))
            D = b2_data(name, seed)
            d, q = D["d"], D["Ytr"].shape[1]
            noisy = name != "lorenz_map"
            cfg = dict(M=2 * d, N1=16, K=4, N2=32)
            row = dict(ridge=ridge_rmse(D), n=D["n"], d=d)
            block_kw, init, fit_kw = spec
            net = build_block(D, cfg, block_kw, init, seed, d_out=q)
            rc = 1e-14
            if noisy:                                        # the truncation the VarPro head trains against, validated first
                qi2.calibrate(net, D["Xtr"]); rc = validated_rcond(net, D)
            log = qi2.fit(net, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], steps=steps, log_every=100, seed=seed, rcond=rc, **fit_kw)
            if noisy:
                rc = validated_rcond(net, D); net.solve_readout(D["Xtr"], D["Ytr"], rc)
            row["block"] = dict(rmse=rmse(net, D), rcond=rc, counts=net.counts(), time=log["time"], final_rel=log["final"])
            w = qi2.mlp_width_for(net.counts()["params"], d, "params", d_out=q)
            mlp = qi2.MLP(d, [w, w], d_out=q, seed=seed)
            log = qi2.fit(mlp, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], steps=steps, log_every=100, seed=seed, optimizer="adam",
                          head="trained", beta=0.0, lr=3e-3, calibrate_first=False)
            r_trained = rmse(mlp, D)                      # after the fit's final solve; recover the trained-head number too
            rc_m = validated_rcond(mlp, D) if noisy else 1e-14
            mlp.solve_readout(D["Xtr"], D["Ytr"], rc_m)
            row["mlp_params"] = dict(rmse=rmse(mlp, D), rmse_trained_head=log["final_trained"], rcond=rc_m, w=w, counts=mlp.counts(), time=log["time"])
            store.put(key, row)
            say(f"b2 {name:11s} seed {seed}  ridge {row['ridge']:.3f}  block {row['block']['rmse']:.3f} (rcond {rc:.0e})  "
                f"mlp {row['mlp_params']['rmse']:.3f} (trained head {log['final_trained']:.3f}, w={w})  [{row['block']['counts']['params']}p]")


# ============================================================================= B3: Fashion-MNIST (fp32, CPU)
B3_SIZES = {"small": dict(M=32, N1=8, K=32, N2=8), "large": dict(M=64, N1=8, K=64, N2=8)}
FMNIST_MEAN, FMNIST_STD = 0.2860, 0.3530


def b3_data():
    Xtr, ytr, Xte, yte = tasks.fashion_mnist()
    f = lambda X: ((X.to(torch.get_default_dtype()) / 255.0 - FMNIST_MEAN) / FMNIST_STD)
    if QUICK:
        Xtr, ytr, Xte, yte = Xtr[:5000], ytr[:5000], Xte[:2000], yte[:2000]
    return dict(Xtr=f(Xtr), Ytr=ytr, Xte=f(Xte), Yte=yte, d=784, name="fashion")


def b3_row(log, model):
    return dict(epoch=log["step"], err=log["test_trained"], err_solved=log["test_solved"], final_err=log["final_trained"],
                final_err_solved=log["final"], counts=model.counts(), time=log["time"], band=log["band"])


def study_b3(store):
    """Block against tanh and ReLU MLPs at matched parameters and logistic regression; cross-entropy, Adam,
    batch 256, 10 epochs; test error per epoch with the trained head and with a one-hot least-squares refit."""
    setup(fp64=False)
    epochs = 1 if QUICK else 10
    name_r, (block_kw, init, fit_kw) = recipe()
    store.put("_recipe", name_r)
    D = b3_data()
    if not store.done("logistic|0"):
        lin = qi2.MLP(784, [], d_out=10, seed=0)
        log = qi2.fit(lin, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], loss="ce", batch=256, epochs=epochs, optimizer="adam", head="trained",
                      beta=0.0, lr=1e-3, calibrate_first=False, seed=0)
        store.put("logistic|0", b3_row(log, lin))
        say(f"b3 logistic  err {log['final_trained']:.4f}  [{log['time']:.0f}s]")
    for size, cfg in B3_SIZES.items():
        for seed in [0]:
            key = f"{size}|block|{seed}"
            if not store.done(key):
                net = qi2.make_block(784, cfg["M"], cfg["N1"], cfg["K"], cfg["N2"], d_out=10, seed=seed,
                                     coords=block_kw.get("coords", "raw"), head_coords=block_kw.get("coords", "raw"), cap=block_kw.get("cap", 1e3))
                log = qi2.fit(net, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], loss="ce", batch=256, epochs=epochs, optimizer="adam",
                              head="trained", beta=fit_kw["beta"], lr=1e-3, seed=seed)
                store.put(key, b3_row(log, net))
                say(f"b3 {size} block   err {log['final_trained']:.4f} (solved {log['final']:.4f})  {net.counts()['params']}p  [{log['time']:.0f}s]")
            params = store.get(key)["counts"]["params"]
            for act in ["tanh", "relu"]:
                key = f"{size}|mlp_{act}|{seed}"
                if store.done(key):
                    continue
                w = qi2.mlp_width_for(params, 784, "params", d_out=10, act=act)
                mlp = qi2.MLP(784, [w, w], d_out=10, act=act, seed=seed)
                log = qi2.fit(mlp, D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], loss="ce", batch=256, epochs=epochs, optimizer="adam",
                              head="trained", beta=0.0, lr=1e-3, calibrate_first=False, seed=seed)
                store.put(key, dict(b3_row(log, mlp), w=w))
                say(f"b3 {size} mlp_{act} err {log['final_trained']:.4f} (solved {log['final']:.4f})  {mlp.counts()['params']}p w={w}  [{log['time']:.0f}s]")


# ============================================================================= B4: PINNs (fp64)
B4_PDES = ["poisson1d", "poisson2d", "burgers"]


def pinn_loss(model, P, beta):
    out, pres = model.forward_pres(P["X_col"])
    r = P["residual"](model, P["X_col"].clone())
    bc = model(P["X_bc"])[:, 0] - P["u_bc"]
    data = (r ** 2).mean() + P["bc_weight"] * (bc ** 2).mean()
    return data + beta * qi2.band_penalty(pres), data


def pinn_err(model, P):
    with torch.no_grad():
        return qi2.rel_l2(model(P["X_test"])[:, 0], P["u_test"])


def fit_pinn(model, P, steps, lr=3e-3, beta=1e-2, log_every=100, warmup=50, seed=0, op_every=None):
    """Adam on the PINN loss for every parameter. op_every (linear PDEs): the head is replaced every op_every steps
    by the operator-system solve (expF01), for whichever model, and Adam continues."""
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    log = dict(step=[], loss=[], err=[]); t0 = time.time()
    for t in range(steps):
        if op_every and t % op_every == 0:
            op_solve(model, P)
            for q in model.readout():
                if q in opt.state:
                    del opt.state[q]
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1.0, (t + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * t / steps))
        opt.zero_grad(set_to_none=True)
        total, data = pinn_loss(model, P, beta)
        total.backward(); opt.step()
        if t % log_every == 0 or t == steps - 1:
            log["step"].append(t); log["loss"].append(float(data)); log["err"].append(pinn_err(model, P))
    log["time"] = time.time() - t0
    return log


def op_solve(model, P, rcond=1e-14):
    """Linear PDE: the head solved on the stacked operator system [L phi | 0; sqrt(w) phi_bc | sqrt(w) 1] (expF01),
    feature derivatives by forward-over-reverse autodiff."""
    from torch.func import hessian, vmap
    feats_fn = lambda x: model.feats(x[None])[0]
    with torch.no_grad():
        Hs = vmap(hessian(feats_fn))(P["X_col"])                          # [n, F, d, d]
        L = -Hs.diagonal(dim1=-2, dim2=-1).sum(-1)                         # -Laplacian of every feature
        w = math.sqrt(P["bc_weight"])
        Fb = model.feats(P["X_bc"])
        A = torch.cat([torch.cat([L, torch.zeros(len(L), 1)], 1), w * torch.cat([Fb, torch.ones(len(Fb), 1)], 1)], 0)
        b = torch.cat([P["f"](P["X_col"]), w * P["u_bc"]])
        sol, rank = qi2.tsvd_solve(A, b, rcond)
        if isinstance(model, qi2.QIStack):
            last = model.layers[-1].mixer
            last.set_raw(sol[:-1].view(last.M, last.N, last.K)); last.bias.copy_(sol[-1:].view_as(last.bias))
        else:
            model.W.copy_(sol[:-1].view_as(model.W)); model.b.copy_(sol[-1:].view_as(model.b))
    return pinn_err(model, P), rank


def study_b4(store):
    setup(fp64=True)
    steps = 100 if QUICK else 6000
    name_r, (block_kw, init, fit_kw) = recipe()
    store.put("_recipe", name_r)
    for name in (["poisson1d"] if QUICK else B4_PDES):
        P = tasks.pde(name, seed=0, n_col=(512 if QUICK else 2048))
        cfg = dict(M=(2 if P["d"] == 1 else 6), N1=32, K=2, N2=64)
        key = f"{name}|0"
        if store.done(key):
            continue
        row = dict(cfg=cfg)
        op_every = 250 if P["linear"] else None
        net = qi2.make_block(P["d"], cfg["M"], cfg["N1"], cfg["K"], cfg["N2"], seed=0, coords=block_kw.get("coords", "raw"), cap=block_kw.get("cap", 1e3))
        qi2.calibrate(net, P["X_col"])
        log = fit_pinn(net, P, steps, beta=fit_kw["beta"], op_every=op_every)
        row["block"] = dict(step=log["step"], err=log["err"], loss=log["loss"], final=log["err"][-1], counts=net.counts(), time=log["time"])
        if P["linear"]:
            row["block"]["final_solved"], row["block"]["rank"] = op_solve(net, P)
        w = qi2.mlp_width_for(net.counts()["params"], P["d"], "params")
        mlp = qi2.MLP(P["d"], [w, w], seed=0)
        log = fit_pinn(mlp, P, steps, beta=0.0, op_every=op_every)
        row["mlp_params"] = dict(step=log["step"], err=log["err"], loss=log["loss"], final=log["err"][-1], counts=mlp.counts(), w=w, time=log["time"])
        if P["linear"]:
            row["mlp_params"]["final_solved"], row["mlp_params"]["rank"] = op_solve(mlp, P)
        store.put(key, row)
        say(f"b4 {name:10s} block {row['block']['final']:.2e} (solved {row['block'].get('final_solved', float('nan')):.2e})  "
            f"mlp {row['mlp_params']['final']:.2e} (solved {row['mlp_params'].get('final_solved', float('nan')):.2e}, w={w})  [{row['block']['time']:.0f}s / {row['mlp_params']['time']:.0f}s]")


# ============================================================================= main
STUDIES = dict(a0=study_a0, a1=study_a1, a1b=study_a1b, a2=study_a2, b1=study_b1, b2=study_b2, b3=study_b3, b4=study_b4)
FIG_OF = dict(a0="fig1", a1="fig2", a1b="fig2", a2="fig3", b1="fig4", b2="fig4", b3="fig5", b4="fig6")


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=list(STUDIES) + ["plot"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    QUICK = args.quick
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.study == "plot":
        for name, fn in plots.ALL.items():
            try:
                say("wrote", fn(RES))
            except Exception as e:
                say(name, "not built:", e)
        return
    store = Store(("a1" if args.study == "a1b" else args.study) + ("_quick" if QUICK else ""))
    t0 = time.time()
    STUDIES[args.study](store)
    say(f"{args.study} done in {(time.time() - t0) / 60:.1f} min -> {store.path}")
    if not QUICK and args.study in FIG_OF:
        say("wrote", plots.ALL[FIG_OF[args.study]](RES))


if __name__ == "__main__":
    main()
