"""expF17 dysts arm: 8 chaotic ODE systems, whole-trajectory (expF14 framing).

Horizon lambda_max * T = 3 Lyapunov times, reference = expF14's mpmath odefun
at 30 digits on a 6001-point grid (cached; MacArthur is the acknowledged
exception -- its C^0 field breaks the Taylor method, so it carries a
convergence-verified DOP853 reference at ~5e-14, recorded as reference-capped).
Systems and their FD/complex-step-verified F, J come from expF14/systems.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
_F14 = REPO_ROOT / "experiments" / "expF14_dysts_chaos"
if str(_F14) not in sys.path:
    sys.path.insert(0, str(_F14))

import systems as f14_systems  # noqa: E402
import reference as f14_reference  # noqa: E402

LYAP_TIMES = 3.0
N_EVAL = 6001
SYSTEMS_1D = list(f14_systems.SYSTEM_ORDER)

_CACHE = {}


def dysts_task(name):
    """-> dict(key, system, T, ts, Yref, family). Reference pulled lazily
    (cache is warm for all 8 at this horizon)."""
    if name not in _CACHE:
        sys_obj = f14_systems.System(name)
        T = sys_obj.horizon(LYAP_TIMES)
        ts, Yref = f14_reference.reference(sys_obj, T, N_EVAL, verbose=False)
        _CACHE[name] = dict(key=f"dysts_{name}", system=sys_obj, T=float(T),
                            ts=np.asarray(ts), Yref=np.asarray(Yref),
                            family="dysts", title=f"{name} (d={sys_obj.d})")
    return _CACHE[name]


def verify_dysts_references(verbose=True):
    """Purity gate for the 1-D arm: every reference loads, is finite, starts at
    the exact IC, and spans the recorded horizon."""
    for name in SYSTEMS_1D:
        t = dysts_task(name)
        assert np.all(np.isfinite(t["Yref"])), name
        assert np.abs(t["Yref"][0] - t["system"].ic).max() < 1e-12, name
        assert len(t["ts"]) == N_EVAL and abs(t["ts"][-1] - t["T"]) < 1e-12, name
        if verbose:
            print(f"  verified dysts reference {name} (d={t['system'].d}, "
                  f"T={t['T']:.3f})")
    if verbose:
        print("dysts reference gate PASS (8 systems)")
