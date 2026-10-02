"""Matched depth insertion probes for I02; original experiment files remain read-only.

Run from repository root:
  .venv/bin/python experiments/expI03_investigation/depth/run.py --steps 1000

Fixed I02 meshes and forward equations, fp64, one CPU thread. The exact-identity
arms insert zero residual QILayers before the original head. Calibration happens
only before insertion, preserving the depth-two function exactly. The optional
small-step arm changes Adam's step only for the inserted layers (1/N_mid).
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-depth-mpl")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "expI02_block_prior"))
import qi2
import tasks

OUT = ROOT / "results/checkpoint_I_depth_theory/investigation_20260906/depth"
ARMS = ("depth2", "random3", "random4", "identity3", "identity4", "identity3_small", "identity4_small")


def dump(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def base_model(data, seed):
    model = qi2.make_block(data["d"], M=2 * data["d"], N1=32, K=2, N2=64, seed=seed)
    qi2.calibrate(model, data["Xtr"][:4096])
    model.solve_readout(data["Xtr"], data["Ytr"])
    return model


def make_model(base, data, seed, arm):
    if arm == "depth2":
        return copy.deepcopy(base)
    depth = int(arm.split("_")[0][-1])
    if arm.startswith("random"):
        model = qi2.make_block(data["d"], M=2 * data["d"], N1=32, K=2, N2=64,
                              depth=depth, seed=seed)
        qi2.calibrate(model, data["Xtr"][:4096])
        # The first layer is identical because make_block consumes its RNG first.
        for p, q in zip(model.layers[0].parameters(), base.layers[0].parameters()):
            assert torch.equal(p, q)
        model.solve_readout(data["Xtr"], data["Ytr"])
        return model
    inserted = [qi2.QILayer(2, 2, 64, residual=True, seed=seed + 10 + i)
                for i in range(depth - 2)]
    return qi2.QIStack([copy.deepcopy(base.layers[0]), *inserted, copy.deepcopy(base.layers[-1])])


@torch.no_grad()
def diagnostics(model, data):
    pres = model.pres(data["Xtr"])
    return {
        "pres": [{"mean": p.mean(0).tolist(), "sd": p.std(0, unbiased=False).tolist(),
                  "max_abs": p.abs().max(0).values.tolist(),
                  "outside_mesh": (p.abs() > 1).double().mean(0).tolist(),
                  "outside_band": (p.abs() > .9).double().mean(0).tolist()}
                 for p in pres],
        "mixer_norms": [float(l.mixer.theta.norm()) for l in model.layers[:-1]],
        "band_penalty": float(qi2.band_penalty(pres)),
    }


def identity_checks(base, model, data):
    x = data["Xte"][:64].clone().requires_grad_(True)
    by, dy = base(x), model(x)
    bg = torch.autograd.grad(by.sum(), x)[0]
    dg = torch.autograd.grad(dy.sum(), x)[0]
    result = {"prediction_max_abs": float((by-dy).abs().max()),
              "input_gradient_max_abs": float((bg-dg).abs().max()),
              "feature_max_abs": float((base.feats(x)-model.feats(x)).abs().max()),
              "head_exactly_equal": all(torch.equal(p, q) for p, q in zip(base.readout(), model.readout()))}
    for key in ["prediction_max_abs", "input_gradient_max_abs", "feature_max_abs"]:
        assert result[key] == 0, (key, result[key])
    assert result["head_exactly_equal"]
    # Local residual branch Jacobian and a finite difference with the head fixed.
    p = model.layers[1].mixer.theta
    direction = torch.randn(p.shape, generator=torch.Generator().manual_seed(101))
    direction /= direction.norm()
    loss = ((model(x)-data["Yte"][:64, None])**2).mean()
    analytical = float((torch.autograd.grad(loss, p)[0] * direction).sum())
    saved = p.detach().clone()
    differences = []
    # A large solved-head norm amplifies fp64 cancellation, so check the usual
    # truncation/roundoff curve instead of requiring one arbitrary step size.
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        with torch.no_grad():
            p.copy_(saved + eps * direction)
            plus = float(((model(x)-data["Yte"][:64, None])**2).mean())
            p.copy_(saved - eps * direction)
            minus = float(((model(x)-data["Yte"][:64, None])**2).mean())
            p.copy_(saved)
        differences.append(dict(eps=eps, derivative=(plus-minus)/(2*eps)))
    finite_difference = min(differences, key=lambda row: abs(row["derivative"]-analytical))["derivative"]
    result.update(fd_analytical=analytical, fd_numerical=finite_difference,
                  fd_abs_error=abs(analytical-finite_difference), fd_sweep=differences,
                  head_norm=float(model.readout()[0].norm()))
    assert abs(analytical-finite_difference) < 1e-5 * max(1, abs(analytical))
    return result


def regularizer(pres, mode="sum"):
    if mode == "sum" or len(pres) < 2:
        return qi2.band_penalty(pres)
    if mode == "preserve_identity":
        # At identity insertion, every later bank sees the same z. Keep the
        # original first-bank penalty and average the repeated channel penalties.
        return qi2.band_penalty(pres[:1]) + qi2.band_penalty(pres[1:]) / (len(pres)-1)
    raise ValueError(mode)


def train(model, data, arm, steps, lr, beta, insert_lr=None, penalty_mode="sum"):
    """I02 fit's full-batch Adam+VarPro, warmup 50, cosine, head solved each pass.

    Evaluate after each logged update with the I02 QR/SVD head refit. Head
    parameters are excluded from Adam. No selection using test error.
    """
    started = time.perf_counter()
    x, y = data["Xtr"], data["Ytr"][:, None]
    params = model.nonlinear_params()
    small = arm.endswith("small")
    base_params = [p for p in model.layers[0].parameters() if p.requires_grad]
    inserted_params = [p for layer in model.layers[1:-1] for p in layer.parameters()]
    groups = [dict(params=base_params, lr=lr, base_lr=lr)] if base_params else []
    if inserted_params:
        add_lr = insert_lr if insert_lr is not None else (lr / model.layers[1].N if small else lr)
        groups.append(dict(params=inserted_params, lr=add_lr, base_lr=add_lr))
    optimizer = torch.optim.Adam(groups)
    layer_previous = [l.mixer.theta.detach().clone() for l in model.layers[:-1]]
    trajectory = []
    first_steps = []

    def evaluate(step, grad_norms=None, step_norms=None):
        with torch.no_grad():
            model.solve_readout(x, y)
            tr = qi2.rel_l2(model(x), y)
            te = qi2.rel_l2(model(data["Xte"]), data["Yte"][:, None])
            penalty = float(regularizer(model.pres(x), penalty_mode))
        trajectory.append(dict(step=step, train=tr, test=te, grad_norms=grad_norms,
                               step_norms=step_norms, optimized_band_penalty=penalty,
                               **diagnostics(model, data)))

    evaluate(0)
    for t in range(steps):
        if t > 0 and t % 250 == 0:
            model.solve_readout(x, y)
        schedule = min(1., (t+1)/50) * .5 * (1 + math.cos(math.pi*t/steps))
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * schedule
        optimizer.zero_grad(set_to_none=True)
        output, pres = model.forward_pres(x, solve=(y, 1e-14))
        data_loss = ((output-y)**2).mean()
        loss = data_loss + beta*regularizer(pres, penalty_mode)
        loss.backward()
        grads = [float(l.mixer.theta.grad.norm()) if l.mixer.theta.grad is not None else None
                 for l in model.layers[:-1]]
        optimizer.step()
        with torch.no_grad():
            delta = [float((l.mixer.theta-last).norm()) for l, last in zip(model.layers[:-1], layer_previous)]
            layer_previous = [l.mixer.theta.detach().clone() for l in model.layers[:-1]]
        if t < 20:
            first_steps.append(dict(step=t+1, data_loss=float(data_loss.detach()),
                                    grad_norms=grads, step_norms=delta,
                                    **diagnostics(model, data)))
        if t == 0 or (t+1) % 100 == 0 or t == steps-1:
            evaluate(t+1, grads, delta)
    return dict(final=trajectory[-1]["test"], train_final=trajectory[-1]["train"],
                trajectory=trajectory, first_steps=first_steps,
                seconds=time.perf_counter()-started, counts=model.counts())


def title_and_shared_legend(fig, axes, title, max_columns=4):
    """Keep a deduplicated legend above the axes, with explicit title margins."""
    entries = {}
    for ax in np.asarray(axes).reshape(-1):
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            entries.setdefault(label, handle)
    title_lines = textwrap.wrap(title, width=int(fig.get_size_inches()[0] * 11))
    title_artist = fig.suptitle("\n".join(title_lines), y=.98, fontsize=12)
    title_artist.set_in_layout(False)
    legend_top = .98 - .047 * len(title_lines)
    columns = min(max_columns, len(entries))
    rows = math.ceil(len(entries) / columns)
    legend = fig.legend(list(entries.values()), list(entries), loc="upper center",
                        bbox_to_anchor=(.5, legend_top), ncol=columns, fontsize=8,
                        frameon=False, borderaxespad=0)
    legend.set_in_layout(False)
    fig.tight_layout(rect=(0, 0, 1, legend_top - .053 * rows - .035))


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = sorted(set(row["target"] for row in rows.values()))
    fig, axes = plt.subplots(1, len(targets), figsize=(6*len(targets), 4.8), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    for ax, target in zip(axes[0], targets):
        for i, arm in enumerate(ARMS):
            matches = [r for r in rows.values() if r["target"] == target and r["arm"] == arm]
            if not matches:
                continue
            curves = np.array([[p["test"] for p in r["trajectory"]] for r in matches])
            x = [p["step"] for p in matches[0]["trajectory"]]
            ax.plot(x, np.median(curves, 0), color=colors[i], label=arm)
            if len(matches)>1:
                ax.fill_between(x, curves.min(0), curves.max(0), color=colors[i], alpha=.1)
        ax.set(yscale="log", xlabel="Adam step", ylabel="Relative test L2", title=target)
        ax.grid(alpha=.2)
    title_and_shared_legend(fig, axes, "Exact identity insertion in fixed-mesh QI blocks; median and seed range")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--targets", nargs="+", default=["fast_waves", "composition"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--lr", type=float, default=.02)
    parser.add_argument("--beta", type=float, default=.01)
    parser.add_argument("--name", default="screen")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.name}.json"
    rows = json.loads(path.read_text()) if path.exists() else {}
    dump(OUT / f"{args.name}_config.json", dict(vars(args), torch=torch.__version__, d=4,
         N1=32, N2=64, K=2, M=8, threads=1, dtype="float64", calibration="depth2 only before identity insertion"))
    for target in args.targets:
        for seed in args.seeds:
            data = tasks.analytic(target, 4, args.n_train, n_test=args.n_test, seed=seed)
            base = base_model(data, seed)
            for arm in args.arms:
                key = f"{target}|{seed}|{arm}"
                if key in rows:
                    continue
                model = make_model(base, data, seed, arm)
                checks = identity_checks(base, model, data) if arm.startswith("identity") else None
                log = train(model, data, arm, args.steps, args.lr, args.beta)
                rows[key] = dict(target=target, seed=seed, arm=arm, checks=checks, **log)
                dump(path, rows)
                torch.save(model.state_dict(), OUT / f"{args.name}_{target}_{seed}_{arm}.pt")
                plot(rows, OUT / f"{args.name}_trajectories.png")
                print(f"{key}: initial={log['trajectory'][0]['test']:.3g}, final={log['final']:.3g}, {log['seconds']:.1f}s", flush=True)
    for target in args.targets:
        for arm in args.arms:
            values = [r["final"] for r in rows.values() if r["target"] == target and r["arm"] == arm]
            if values:
                print(target, arm, "median", np.median(values), "range", min(values), max(values))


if __name__ == "__main__":
    main()
