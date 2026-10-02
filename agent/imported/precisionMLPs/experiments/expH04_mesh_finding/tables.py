"""Markdown tables for the expH04 writeup, from the saved ladder data.

Usage:
    uv run --extra dev python experiments/expH04_mesh_finding/tables.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_H_highdim" / "expH04_mesh_finding"


def load():
    rows = []
    for p in sorted(glob.glob(str(RESULTS_DIR / "ladder_*.json"))):
        with open(p) as f:
            rows += json.load(f)["rows"]
    return rows


def _key(s):
    return (int(s.split(".")[0]), int(s.split(".")[1]))


def _short(name):
    return name.split("-", 1)[1].replace("-", " ")


def fmt(x):
    if x is None or not np.isfinite(x):
        return "--"
    m, e = f"{x:.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def get(rows, tid, rung, B, key):
    r = next((r for r in rows if r["task"] == tid and r["rung"] == rung and r["budget"] == B),
             None)
    return None if r is None else r["errors"][key]["rel_l2"]


def table(rows, tasks, rungs, B, key, labels):
    head = "| task | " + " | ".join(labels[r] for r in rungs) + " |"
    sep = "|---|" + "---|" * len(rungs)
    out = [head, sep]
    for tid in tasks:
        name = next((r["name"] for r in rows if r["task"] == tid), None)
        if name is None:
            continue
        cells = [fmt(get(rows, tid, rg, B, key)) for rg in rungs]
        out.append(f"| {tid} {_short(name)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


LABELS = {"even": "even", "data_p1": "data $p$", "oracle_r1": "true slope",
          "surr_r1": "est. slope", "surr_r2": "est. curvature", "residual": "residual",
          "freq": "est. frequency", "active_oracle": "active (true)", "active": "active (est.)",
          "active_x3": "active, iterated", "dir_surr": "angles (est.)", "both_surr": "angles+centers"}


def main():
    rows = load()
    out = []
    d1 = ["1.1", "1.7", "1.8", "1.11", "1.12", "1.13", "1.14", "1.15", "1.16"]
    r1 = ["even", "data_p1", "oracle_r1", "surr_r1", "surr_r2", "residual", "freq"]
    out.append("### 1-D\n\nRelative $L_2$ on the dense region at $B=128$ (where the meshes differ most):\n")
    out.append(table(rows, d1, r1, 128, "dense_region", LABELS))
    out.append("\nThe same at $B=1024$ (everything resolved; the price of adaptation):\n")
    out.append(table(rows, d1, r1, 1024, "dense_region", LABELS))
    out.append("\nUniform-cube test at $B=1024$:\n")
    out.append(table(rows, d1, r1, 1024, "uniform", LABELS))
    d2 = ["2.1", "2.3", "2.7", "2.8", "2.11", "2.12", "2.13", "2.14", "2.15", "2.16"]
    r2 = ["even", "data_p1", "surr_r1", "residual", "freq", "dir_surr", "active"]
    out.append("\n### 2-D\n\nDense region at $B=1024$:\n")
    out.append(table(rows, d2, r2, 1024, "dense_region", LABELS))
    out.append("\nDense region at $B=4096$:\n")
    out.append(table(rows, d2, r2, 4096, "dense_region", LABELS))
    out.append("\nUniform-cube test at $B=4096$:\n")
    out.append(table(rows, d2, r2, 4096, "uniform", LABELS))
    d3 = ["3.5", "3.7", "3.11", "3.12", "3.13", "3.16", "5.5", "5.16"]
    r3 = ["even", "data_p1", "surr_r1", "freq", "active_oracle", "active", "active_x3"]
    out.append("\n### $d=3$ and $d=5$ at $B=4096$\n\nDense region:\n")
    out.append(table(rows, d3, r3, 4096, "dense_region", LABELS))
    out.append("\nUniform-cube test:\n")
    out.append(table(rows, d3, r3, 4096, "uniform", LABELS))
    # split sweep
    sp = RESULTS_DIR / "split_d3.json"
    if sp.exists():
        with open(sp) as f:
            srows = json.load(f)["rows"]
        tasks = sorted({r["task"] for r in srows}, key=_key)
        npers = sorted({r["n_per"] for r in srows})
        out.append("\n### $d=3$ split sweep (even mesh, $B=4096$), dense region\n")
        out.append("| task | " + " | ".join(f"{n} per dir" for n in npers) + " |")
        out.append("|---|" + "---|" * len(npers))
        for tid in tasks:
            name = _short(next(r["name"] for r in srows if r["task"] == tid))
            cells = [fmt(next((r["errors"]["dense_region"]["rel_l2"] for r in srows
                               if r["task"] == tid and r["n_per"] == n), None)) for n in npers]
            out.append(f"| {tid} {name} | " + " | ".join(cells) + " |")
    kp = RESULTS_DIR / "known_answer_2d.json"
    if kp.exists():
        with open(kp) as f:
            krows = json.load(f)["rows"]
        out.append("\n### Known-answer ridge in 2-D (uniform test)\n")
        rungs = [("even", None), ("dir_est", 1 / 3), ("active_true", None), ("active_est", None),
                 ("active_iter", None)]
        names = ["even angles", r"angles from $A(\theta)^{1/3}$", "active (true)",
                 "active (est.)", "active, iterated"]
        Bs = sorted({r["budget"] for r in krows})
        out.append("| $B$ | " + " | ".join(names) + " |")
        out.append("|---|" + "---|" * len(names))
        for B in Bs:
            cells = []
            for rg, al in rungs:
                r = next((r for r in krows if r["budget"] == B and r["rung"] == rg
                          and (al is None or abs((r["alpha"] or 0) - al) < 1e-9)), None)
                cells.append(fmt(None if r is None else r["errors"]["rel_l2"]))
            out.append(f"| {B} | " + " | ".join(cells) + " |")
    text = "\n".join(out)
    print(text)
    with open(RESULTS_DIR / "tables.md", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
