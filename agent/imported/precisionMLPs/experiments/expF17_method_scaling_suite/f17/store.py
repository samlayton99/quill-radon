"""Crash-safe JSONL store: one line per landed cell, atomic append, keyed by
(task, method, C, seed, regime, variant). Reload -> skip landed cells (13.13)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .protocol import cell_key

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF17_method_scaling_suite"
CELLS_PATH = RESULTS_DIR / "cells.jsonl"
TUNING_PATH = RESULTS_DIR / "tuning.jsonl"


def _append(path, rec):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, sort_keys=True) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def load(path=CELLS_PATH):
    """-> {key: record}; a later line for the same key wins (allows re-runs)."""
    out = {}
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail line from a crash; the cell re-runs
                out[rec["key"]] = rec
    return out


def save_cell(rec):
    rec["key"] = cell_key(rec["task"], rec["method"], rec["C"], rec["seed"],
                          rec["regime"], rec["variant"])
    _append(CELLS_PATH, rec)
    return rec["key"]


def save_tuning(rec):
    rec["key"] = f"{rec['task']}|{rec['method']}|{rec['C']}|{rec['regime']}"
    _append(TUNING_PATH, rec)


def load_tuning():
    return load(TUNING_PATH)
