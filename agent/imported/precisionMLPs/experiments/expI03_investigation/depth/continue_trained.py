"""Insert exact identities into the reproduced successful A2 depth-two fit.

First run run.py with the a2_matched command in this directory's report.
Then: .venv/bin/python experiments/expI03_investigation/depth/continue_trained.py
The continuation tests keep all initial predictions and the training data fixed.
"""
import argparse
import json
import torch
import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--base-lr", type=float, default=.001)
    parser.add_argument("--insert-lr", type=float, default=.0003)
    parser.add_argument("--name", default="continuation")
    parser.add_argument("--penalty-mode", choices=["sum", "preserve_identity"], default="sum")
    parser.add_argument("--no-frozen", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    data = run.tasks.analytic("fast_waves", 4, 12288, n_test=10000, seed=0)
    base = run.base_model(data, 0)
    source = run.OUT / "a2_matched_fast_waves_0_depth2.pt"
    base.load_state_dict(torch.load(source, weights_only=True))
    path = run.OUT / f"{args.name}.json"
    rows = json.loads(path.read_text()) if path.exists() else {}
    run.dump(run.OUT / f"{args.name}_config.json", dict(vars(args), source=str(source),
             n_train=12288, n_test=10000, target="fast_waves", seed=0, d=4,
             head_rcond=1e-14, beta=.01, calibration="none after restoring the trained base"))
    for arm, freeze in [("depth2", False), ("identity3_small", False),
                        ("identity4_small", False), ("identity3_small", True),
                        ("identity4_small", True)]:
        if freeze and args.no_frozen:
            continue
        key = arm + ("_frozen" if freeze else "")
        if key in rows:
            continue
        model = run.make_model(base, data, 0, arm)
        checks = run.identity_checks(base, model, data) if arm.startswith("identity") else None
        if checks is not None and args.penalty_mode == "preserve_identity":
            original, pres0 = base.forward_pres(data["Xtr"])
            inserted, pres1 = model.forward_pres(data["Xtr"])
            loss0 = ((original-data["Ytr"][:, None])**2).mean() + .01*run.regularizer(pres0, args.penalty_mode)
            loss1 = ((inserted-data["Ytr"][:, None])**2).mean() + .01*run.regularizer(pres1, args.penalty_mode)
            g0 = torch.autograd.grad(loss0, list(base.layers[0].parameters()))
            g1 = torch.autograd.grad(loss1, list(model.layers[0].parameters()))
            checks["objective_abs_difference"] = float((loss0-loss1).abs())
            checks["base_gradient_max_abs_difference"] = max(float((p-q).abs().max()) for p, q in zip(g0, g1))
            assert checks["objective_abs_difference"] < 1e-12
            assert checks["base_gradient_max_abs_difference"] < 1e-10
        if freeze:
            for p in model.layers[0].parameters():
                p.requires_grad_(False)
        log = run.train(model, data, arm, args.steps, args.base_lr, .01,
                        insert_lr=args.insert_lr, penalty_mode=args.penalty_mode)
        rows[key] = dict(arm=arm, frozen_base=freeze, checks=checks,
                         target="fast_waves", seed=0, **log)
        run.dump(path, rows)
        torch.save(model.state_dict(), run.OUT / f"{args.name}_{key}.pt")
        print(f"{key}: initial={log['trajectory'][0]['test']:.6g}, final={log['final']:.6g}, {log['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
