"""expF17 -- the PINN arm runner (SPEC 18.8). Separate from run.py so the
solved-arm runner is never touched; cells land in the same cells.jsonl with
method="pinn" (variant "base" = the trained network, variant "refit" = the
frozen-feature certificate) and tuning in the same tuning.jsonl.

Tuning mirrors the solved arms: lr swept over LR_GRID at seed 0 in the oracle
regime with the edge-walk (cap PINN_MAX_WALK = 2, Sam's cost ruling), selected on the
TRAINED score (the refit is a certificate, never a selector), frozen; the
dynamic regime inherits lr and sweeps w_mult with the same walk.

Ladder {121, 256, 529, 1024} columns, 3 seeds below 1024 and seed 0 at 1024
(Sam, 2026-09-01: the 2025 rung costs 30-70 min per run and is dropped).
Queue order: regime (oracle, then dynamic) -> width -> task -> seed, so both
plot-1 panels fill from the bottom rung up. Sharding splits the queue by (task, W) so that
a dynamic cell always finds its oracle-tuned lr in its own shard.

Usage:
    uv run --extra dev python experiments/expF17_method_scaling_suite/run_pinn.py --run
        [--shard i/n] [--threads T] [--max-cells N] [--only task=wave,C=529]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from f17 import dicts as fd, pinn, protocol as pr, store  # noqa: E402
from f17.tasks import TASKS  # noqa: E402

METHOD = "pinn"
LR_STEP = np.sqrt(10.0)  # the grid's own spacing (half a decade)
LR_BOUNDS = (1e-5, 1e-1)
W_MULT_GRID = [0.3, 1.0, 3.0]
PINN_MAX_WALK = 2  # Sam 2026-09-01: the PINN's walks are capped at 2 (not MAX_WALK)


def lr_step(value, direction):
    v = value * (LR_STEP if direction > 0 else 1.0 / LR_STEP)
    return float(v) if LR_BOUNDS[0] <= v <= LR_BOUNDS[1] else None


def _train_cell(task, n_ax, regime, seed, lr, w_mult):
    t = TASKS[task]
    m_train, m_refit, info = pinn.run_pinn(t, n_ax, regime, seed, lr,
                                           w_mult=(w_mult if regime == "dynamic" else 1.0))
    return m_train, m_refit, info


def _land(task, C, n_ax, regime, seed, lr, w_mult, m_train, m_refit, info):
    cols = (n_ax + 1) ** 2
    common = dict(task=task, method=METHOD, C=C, seed=seed, regime=regime,
                  n_ax=n_ax, cols=cols, config={"lr": lr},
                  meta=dict(method=METHOD, n_ax=n_ax, n_units=cols - 1,
                            adam_steps=pinn.ADAM_STEPS, lbfgs_cap=pinn.LBFGS_ITERS),
                  w_mult=(w_mult if regime == "dynamic" else None),
                  t_total=info["t_total"], t_adam=info["t_adam"],
                  t_lbfgs=info["t_lbfgs"], n_data=info["n_data"],
                  loss_adam=info["loss_adam"], loss_final=info["loss_final"],
                  lbfgs_evals=info["lbfgs_evals"], gamma_rms=info["gamma_rms"])
    rec = dict(common, variant="base", hist=info["hist"],
               readout_norm=info["readout_norm"], **m_train)
    store.save_cell(rec)
    store.save_cell(dict(common, variant="refit",
                         readout_norm=info["refit_readout_norm"], **m_refit))
    return rec


def tuned_lr(task, C, tuning, log):
    """Oracle-regime lr sweep at seed 0 with the edge-walk; frozen per (task, C).
    The sweep's runs are themselves the seed-0 oracle cells (landed once)."""
    tkey = f"{task}|{METHOD}|{C}|oracle"
    if tkey in tuning:
        return tuning[tkey]["chosen"]["lr"]
    n_ax = fd.n_ax_for(C)
    rows = []

    def evaluate(lr):
        m_train, m_refit, info = _train_cell(task, n_ax, "oracle", 0, lr, None)
        rows.append(dict(config={"lr": lr}, rel_l2=m_train["rel_l2"],
                         linf=m_train["linf"], refit_rel_l2=m_refit["rel_l2"],
                         t_total=info["t_total"]))
        log(f"  tune {task}|{C}|oracle lr={lr:.1e} rel_l2={m_train['rel_l2']:.2e} "
            f"refit={m_refit['rel_l2']:.2e} ({info['t_total']}s)")
        return m_train, m_refit, info

    results = {lr: evaluate(lr) for lr in pinn.LR_GRID}
    walked = 0
    while walked < PINN_MAX_WALK:
        best = min(rows, key=lambda r: r["rel_l2"])
        vals = sorted(r["config"]["lr"] for r in rows)
        v = best["config"]["lr"]
        if v not in (vals[0], vals[-1]):
            break
        nxt = lr_step(v, +1 if v == vals[-1] else -1)
        if nxt is None or any(abs(r["config"]["lr"] - nxt) < 1e-12 for r in rows):
            break
        results[nxt] = evaluate(nxt)
        walked += 1
    chosen = min(rows, key=lambda r: r["rel_l2"])["config"]
    rec = dict(task=task, method=METHOD, C=C, regime="oracle", knob=["lr"],
               grid=rows, chosen=chosen, edge_walked=walked)
    store.save_tuning(rec)
    tuning[tkey] = rec
    # the chosen sweep run IS the seed-0 oracle cell
    m_train, m_refit, info = results[chosen["lr"]]
    _land(task, C, n_ax, "oracle", 0, chosen["lr"], None, m_train, m_refit, info)
    return chosen["lr"]


def tuned_w_mult(task, C, lr, tuning, log):
    tkey = f"{task}|{METHOD}|{C}|dynamic"
    if tkey in tuning and tuning[tkey]["chosen_dict"] == {"lr": lr}:
        return tuning[tkey]["chosen"]["w_mult"]
    n_ax = fd.n_ax_for(C)
    rows = []

    def evaluate(w):
        m_train, m_refit, info = _train_cell(task, n_ax, "dynamic", 0, lr, w)
        rows.append(dict(w_mult=w, rel_l2=m_train["rel_l2"], linf=m_train["linf"],
                         refit_rel_l2=m_refit["rel_l2"], t_total=info["t_total"]))
        log(f"  tune {task}|{C}|dynamic w_mult={w:.2g} rel_l2={m_train['rel_l2']:.2e} "
            f"refit={m_refit['rel_l2']:.2e} ({info['t_total']}s)")
        return m_train, m_refit, info

    results = {w: evaluate(w) for w in W_MULT_GRID}
    for _ in range(PINN_MAX_WALK):
        best = min(rows, key=lambda r: r["rel_l2"])
        vals = sorted(r["w_mult"] for r in rows)
        if best["w_mult"] not in (vals[0], vals[-1]):
            break
        nxt = pr.knob_step(METHOD, "w_mult", best["w_mult"],
                           +1 if best["w_mult"] == vals[-1] else -1, 0)
        if nxt is None or any(r["w_mult"] == nxt for r in rows):
            break
        results[nxt] = evaluate(nxt)
    chosen = min(rows, key=lambda r: r["rel_l2"])["w_mult"]
    rec = dict(task=task, method=METHOD, C=C, regime="dynamic", knob="w_mult",
               grid=rows, chosen={"w_mult": chosen}, chosen_dict={"lr": lr})
    store.save_tuning(rec)
    tuning[tkey] = rec
    m_train, m_refit, info = results[chosen]
    _land(task, C, n_ax, "dynamic", 0, lr, chosen, m_train, m_refit, info)
    return chosen


PINN_LADDER = [128, 256, 512, 1024]   # Sam 2026-09-01: no 2025 rung for the PINN
PINN_SEEDS = {1024: [0]}               # and seed 0 only at 1024 (3 seeds below)


def build_queue():
    q = []
    tasks = [t for t in pr.TASK_ORDER
             if not TASKS[t].get("blocked") and TASKS[t]["dynamic"]]
    for regime in ("oracle", "dynamic"):
        for C in PINN_LADDER:
            for task in tasks:
                for seed in PINN_SEEDS.get(C, pr.SEEDS):
                    q.append(dict(task=task, method=METHOD, C=C, seed=seed,
                                  regime=regime, variant="base"))
    return q


def _key(e):
    return pr.cell_key(e["task"], e["method"], e["C"], e["seed"], e["regime"],
                       e["variant"])


def run_queue(shard=(0, 1), max_cells=None, only=None):
    from f17 import plots
    cells = store.load()
    tuning = store.load_tuning()
    queue = build_queue()
    if only:
        queue = [e for e in queue if all(str(e[k]) == v for k, v in only.items())]
    i, n = shard
    queue = [e for e in queue
             if (pr.TASK_ORDER.index(e["task"]) * 97 + PINN_LADDER.index(e["C"])) % n == i]
    pending = [e for e in queue if _key(e) not in cells]
    print(f"pinn queue (shard {i}/{n}): {len(queue)} cells, {len(pending)} pending",
          flush=True)

    def log(msg):
        print(msg, flush=True)

    n_done = 0
    t_start = time.time()
    for e in queue:
        if _key(e) in cells:
            continue
        task, C, seed, regime = e["task"], e["C"], e["seed"], e["regime"]
        n_ax = fd.n_ax_for(C)
        lr = tuned_lr(task, C, tuning, log)
        w_mult = tuned_w_mult(task, C, lr, tuning, log) if regime == "dynamic" else None
        cells = store.load()
        if _key(e) in cells:  # seed 0 landed by the sweep
            rec = cells[_key(e)]
        else:
            m_train, m_refit, info = _train_cell(task, n_ax, regime, seed, lr, w_mult)
            rec = _land(task, C, n_ax, regime, seed, lr, w_mult, m_train, m_refit, info)
        n_done += 1
        cells = store.load()
        refit = cells[_key(dict(e, variant="refit"))]["rel_l2"]
        print(f"[{n_done}/{len(pending)}] {rec['key']:46s} rel_l2={rec['rel_l2']:.2e} "
              f"refit={refit:.2e} lr={lr:.1e} w={w_mult} ({rec['t_total']}s)", flush=True)
        try:
            plots.replot()
        except Exception as exc:  # noqa: BLE001 -- plotting must never stop the run
            print(f"  (replot failed: {exc})", flush=True)
        if max_cells and n_done >= max_cells:
            break
    print(f"done: {n_done} new cells in {(time.time() - t_start) / 60:.1f} min",
          flush=True)


def main():
    args = sys.argv[1:]
    shard = (0, 1)
    if "--shard" in args:
        a, b = args[args.index("--shard") + 1].split("/")
        shard = (int(a), int(b))
    if "--threads" in args:
        torch.set_num_threads(int(args[args.index("--threads") + 1]))
    only = None
    for a in args:
        if a.startswith("--only"):
            kv = a.split("=", 1)[1] if "=" in a else args[args.index(a) + 1]
            only = dict(p.split("=") for p in kv.split(","))
    max_cells = None
    if "--max-cells" in args:
        max_cells = int(args[args.index("--max-cells") + 1])
    if "--run" in args:
        run_queue(shard=shard, max_cells=max_cells, only=only)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
