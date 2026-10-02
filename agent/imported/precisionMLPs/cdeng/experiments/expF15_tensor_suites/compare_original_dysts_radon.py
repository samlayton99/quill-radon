"""Key matched-budget Radon comparison for the original five expF14 systems.

Tests the largest genuinely compressed output rank (d-1) against a dense 1-D
Radon readout at the same coefficient budget.  This is an oracle representation
ceiling, matching compare_extra_dysts_radon.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "cdeng" / "results" / "checkpoint_F_applications" / "expF15_tensor_suites"

spec = importlib.util.spec_from_file_location("cmp", HERE / "compare_extra_dysts_radon.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

DYSTS = ROOT / "experiments" / "expF14_dysts_chaos"
sys.path.insert(0, str(DYSTS))
import reference  # noqa: E402
import systems  # noqa: E402

# Focused oracle sweep: the broad extra-system sweep consistently selected
# these basins.  This avoids thirty large decompositions per system.
cmp.LAMBDAS = [0.16, 0.25, 0.30]
cmp.COLLARS = [1.0, 1.6]
cmp.RCONDS = [1e-13]


def main():
    tensor = json.loads((OUT / "data.json").read_text())["dysts"]
    out = {"scope": "d-1 tensor rank versus dense 1-D Radon at matched coefficients; oracle representation ceiling",
           "systems": {}}
    for name in systems.SYSTEM_ORDER:
        S = systems.System(name)
        T = S.horizon(3.0)
        ts = np.linspace(0.0, T, 6001)
        Y, nfev = reference.rk_trajectory(S, T, ts, 1e-13, 1e-14)
        s = 2.0 * ts / T - 1.0
        trec = tensor[name]
        tc = next(c for c in trec["cells"] if c["rank"] == S.d - 1)
        width = max(4, tc["parameters"] // S.d - 4)
        radon = cmp.best_radon(s, Y, width)
        rec = {"d": S.d, "tensor_rank": S.d - 1, "budget": tc["parameters"],
               "tensor_relative_l2": tc["relative_l2"], "radon": radon,
               "reference_nfev": nfev}
        out["systems"][name] = rec
        print(f"{name:10s} P={tc['parameters']} tensor={tc['relative_l2']:.3e} "
              f"Radon={radon['relative_l2']:.3e}", flush=True)
    path = OUT / "original_dysts_tensor_vs_radon.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
