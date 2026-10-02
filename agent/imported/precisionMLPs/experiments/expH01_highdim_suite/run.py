"""expH01 -- the high-dimensional benchmark suite: build it and run it end to end.

This experiment builds the measuring device. It contains no adaptive model: the only
things fitted here are the even-geometry reference (directions spread evenly, centers
spaced evenly, one least-squares solve) and a random-feature control, and they exist so
that the 80 tasks, the data geometries, the test sets and the error breakdowns get
exercised at least once. Models that choose their own directions and centers come later.

Modes
-----
``--gallery``  draw the task galleries and the predicted-center-density figure (no fitting).
``--smoke``    fit the tasks listed in ``h01suite.tasks.SMOKE_IDS`` (all of d = 1 and
               d = 2, plus 3.3, 3.11, 3.12, 3.13) at the given budgets.
``--full``     fit all 80 tasks over a small budget grid.
``--plot``     redraw the smoke figure from saved data.

How much training data there is, is a knob in its own right: ``--n-train 256,1024,4096``
sets absolute training-set sizes and ``--ratios`` sets them relative to the budget
(``n_train = ratio * B``); passing both runs the union. Every fit is scored on three
fixed test sets:

``same_as_train``   fresh points drawn like the training data;
``uniform``         uniform over the whole cube;
``dense_region``    the densest part of the data only, with a margin of training data
                    all around it -- the set on which machine precision is a fair
                    question.

Usage:
    python experiments/expH01_highdim_suite/run.py --gallery
    python experiments/expH01_highdim_suite/run.py --smoke --budgets 1024,4096
    python experiments/expH01_highdim_suite/run.py --full --budgets 128,512
    python experiments/expH01_highdim_suite/run.py --plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from h01suite.baseline import EvenGeometry, RandomFeatures
from h01suite.metrics import (errors_by_data_density, error_metrics, region_errors,
                              sheet_errors)
from h01suite.tasks import SMOKE_IDS, TASKS, get_task

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH01_highdim_suite"
FIG_DIR = RESULTS_DIR / "figures"
SMOKE_PATH = RESULTS_DIR / "smoke.json"
FULL_PATH = RESULTS_DIR / "full.json"

BUDGETS = [64, 128, 256, 512, 1024, 2048, 4096]
SAMPLE_RATIOS = [1.25, 2.0, 4.0, 8.0]
SMOKE_BUDGET = 1024
SMOKE_RATIO = 8.0
TEST_SETS = ("same_as_train", "uniform", "dense_region")
CLUSTERED = ("hotspots", "stretched_hotspots")
TRAIN_SEED = 0
TEST_SEED = 10_000


def build_model(kind: str, d: int, budget: int, seed: int):
    if kind == "even_geometry":
        return EvenGeometry(d=d, budget=budget)
    if kind == "random_features":
        return RandomFeatures(d=d, budget=budget, seed=seed)
    raise KeyError(kind)


def evaluate(task, model, sets, y_true) -> dict:
    """Every number the suite records for one fitted model on one task."""
    rec: dict = {"errors": {}, "by_data_density": None, "packet": None, "jump": None,
                 "sheet": None}
    for key in TEST_SETS:
        rec["errors"][key] = error_metrics(model.predict(sets[key]), y_true[key])

    # error split by how dense the data is, where there is a density formula
    logp = task.logpdf(sets["same_as_train"])
    if logp is not None and task.density_tag in CLUSTERED:
        rec["by_data_density"] = errors_by_data_density(
            model.predict(sets["same_as_train"]), y_true["same_as_train"], logp)

    # inside/outside the burst of oscillation, and near/far from a step or slope break
    for name, mask_fn in (("packet", task.packet_mask), ("jump", task.jump_mask)):
        out = {}
        for key in TEST_SETS:
            mask = mask_fn(sets[key])
            if mask is None or mask.all() or (~mask).all():
                continue
            out[key] = region_errors(model.predict(sets[key]), y_true[key], mask, label=name)
        rec[name] = out or None

    if task.is_sheet:
        rec["sheet"] = sheet_errors(model.predict, task.F, sets)
    return rec


def run_one(task, budget: int, ratio: float, seed: int,
            kinds=("even_geometry", "random_features"),
            n_train: int | None = None) -> list[dict]:
    """Fit every model on one (task, budget, training-set size) combination.

    ``n_train`` overrides ``ratio * budget`` when given.
    """
    n_train = int(round(ratio * budget)) if n_train is None else int(n_train)
    X, y = task.train_set(n_train, seed=seed)
    sets = task.test_sets(seed=TEST_SEED)
    y_true = {k: task.F(v) for k, v in sets.items()}
    rows = []
    for kind in kinds:
        t0 = time.time()
        model = build_model(kind, task.d, budget, seed)
        model.fit(X, y)
        rec = evaluate(task, model, sets, y_true)
        geo = model.geometry()
        rows.append({"task": task.id, "name": task.name, "d": task.d,
                     "data": task.density_tag, "what_it_tests": task.what_it_tests,
                     "model": kind, "budget": budget, "ratio": ratio,
                     "n_train": n_train, "seed": seed,
                     "dense_region": task.dense_region_description(),
                     "n_features": int(len(geo["centers"])),
                     "n_directions": int(geo["n_directions"]),
                     "n_per_direction": int(geo["n_per_direction"]),
                     "rank": int(model.info["rank"]),
                     "train_mse": float(np.mean((model.predict(X) - y) ** 2)),
                     "seconds": round(time.time() - t0, 2), **rec})
    return rows


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_suite(task_ids, budgets, ratios, seeds, out_path, label, n_trains=()) -> list[dict]:
    """``ratios`` gives training-set sizes relative to the budget, ``n_trains`` absolute
    sizes; the runs are the union of the two."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    sizes = [("ratio", r) for r in ratios] + [("abs", n) for n in n_trains]
    total = len(task_ids) * len(budgets) * len(sizes) * len(seeds)
    done = 0
    for tid in task_ids:
        task = get_task(tid)
        for B in budgets:
            for kind, val in sizes:
                for seed in seeds:
                    if kind == "ratio":
                        rows.extend(run_one(task, B, val, seed))
                    else:
                        rows.extend(run_one(task, B, val / B, seed, n_train=val))
                    done += 1
                    last = rows[-2:]
                    tag = "  ".join(
                        f"{r['model']}: same_as_train={r['errors']['same_as_train']['rel_l2']:.2e} "
                        f"uniform={r['errors']['uniform']['rel_l2']:.2e} "
                        f"dense={r['errors']['dense_region']['rel_l2']:.2e}" for r in last)
                    print(f"[{done}/{total}] {tid:5s} {task.name:44s} B={B:5d} "
                          f"n_train={last[-1]['n_train']:6d} | {tag} | "
                          f"{time.time() - t0:.0f}s", flush=True)
    payload = {"label": label, "budgets": budgets, "ratios": ratios,
               "n_trains": list(n_trains), "seeds": seeds, "task_ids": list(task_ids),
               "rows": rows}
    with open(out_path, "w") as f:
        json.dump(_jsonable(payload), f)
    print(f"\nSaved {out_path}  ({time.time() - t0:.0f}s)")
    return rows


def summary_table(rows) -> str:
    ids = sorted({r["task"] for r in rows},
                 key=lambda s: (int(s.split(".")[0]), int(s.split(".")[1])))
    head = (f"{'task':6s} {'name':46s} "
            f"{'even same_as_train':>19s} {'even uniform':>13s} {'even dense':>11s} "
            f"{'random dense':>13s}")
    lines = [head]
    for tid in ids:
        ev = next((r for r in rows if r["task"] == tid and r["model"] == "even_geometry"), None)
        rd = next((r for r in rows if r["task"] == tid and r["model"] == "random_features"), None)
        if ev is None:
            continue
        lines.append(f"{tid:6s} {ev['name']:46s} "
                     f"{ev['errors']['same_as_train']['rel_l2']:19.2e} "
                     f"{ev['errors']['uniform']['rel_l2']:13.2e} "
                     f"{ev['errors']['dense_region']['rel_l2']:11.2e} "
                     f"{rd['errors']['dense_region']['rel_l2']:13.2e}")
    lines.append("(all numbers are relative L2 error)")
    return "\n".join(lines)


def render_gallery():
    import viz
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("gallery d=1 ...", flush=True)
    viz.gallery_d1(FIG_DIR / "gallery_d1.png")
    print("gallery d=2 ...", flush=True)
    viz.gallery_d2(FIG_DIR / "gallery_d2.png")
    for d in (3, 4, 5):
        print(f"gallery d={d} ...", flush=True)
        viz.gallery_high(FIG_DIR / f"gallery_d{d}.png", d)
    print("predicted center density ...", flush=True)
    viz.predicted_center_density_figure(FIG_DIR / "predicted_center_density.png")
    print(f"figures written to {FIG_DIR} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gallery", action="store_true", help="draw the gallery figures")
    ap.add_argument("--smoke", action="store_true", help="fit the first-pass task list")
    ap.add_argument("--full", action="store_true", help="fit all 80 tasks")
    ap.add_argument("--plot", action="store_true", help="redraw from saved data")
    ap.add_argument("--budgets", type=str, default=None, help="comma-separated budgets B")
    ap.add_argument("--ratios", type=str, default=None,
                    help="comma-separated n_train/B values")
    ap.add_argument("--n-train", type=str, default=None,
                    help="comma-separated absolute training-set sizes (added to the ratio runs)")
    ap.add_argument("--seeds", type=str, default=str(TRAIN_SEED))
    args = ap.parse_args()

    if not any([args.gallery, args.smoke, args.full, args.plot]):
        ap.print_help()
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    n_trains = [int(v) for v in args.n_train.split(",")] if args.n_train else []
    if args.gallery:
        render_gallery()

    rows = None
    if args.smoke:
        budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else [SMOKE_BUDGET]
        ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else [SMOKE_RATIO]
        if n_trains and args.ratios is None:
            ratios = []
        rows = run_suite(SMOKE_IDS, budgets, ratios, seeds, SMOKE_PATH, "smoke", n_trains)
        print("\n" + summary_table([r for r in rows if r["budget"] == budgets[-1]]))
    if args.full:
        budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else [128, 512]
        ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else [4.0]
        if n_trains and args.ratios is None:
            ratios = []
        full = run_suite([t.id for t in TASKS], budgets, ratios, seeds, FULL_PATH, "full",
                         n_trains)
        print("\n" + summary_table([r for r in full if r["budget"] == budgets[-1]]))
    if args.plot and rows is None:
        if not SMOKE_PATH.exists():
            print(f"No data at {SMOKE_PATH}; run --smoke first.")
            return
        with open(SMOKE_PATH) as f:
            rows = json.load(f)["rows"]

    if rows is not None:
        import viz
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        budget = max(r["budget"] for r in rows)
        out = viz.smoke_baseline(FIG_DIR / "smoke_baseline.png",
                                 [r for r in rows if r["budget"] == budget], budget=budget)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
