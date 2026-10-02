"""Tests for expF14 -- dysts chaotic ODEs via frozen QI geometry + collocation.

These are the verification tests for the experiment's machinery, in the repo's
usual order: (1) the re-implemented vector fields match the benchmark library,
(2) the analytic Jacobian matches an exact derivative, (3) the assembled
collocation Jacobian matches the residual it claims to differentiate, (4) the
frozen dictionary reaches machine precision on a known smooth target, and
(5) the extended-precision reference is better than the fp64 solution it is
used to certify.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / "experiments" / "expF14_dysts_chaos"


def _load(name):
    """Load by explicit path: several experiments define same-named modules."""
    spec = importlib.util.spec_from_file_location(f"expF14_{name}",
                                                  EXP_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    spec.loader.exec_module(mod)
    return mod


dysts = pytest.importorskip("dysts.flows")
systems = _load("systems")
core = _load("core")
reference = _load("reference")


@pytest.mark.parametrize("name", systems.SYSTEM_ORDER)
def test_rhs_matches_dysts(name):
    """Our vectorised RHS reproduces dysts' own, to rounding."""
    S = systems.System(name)
    _, rel = systems.verify_rhs(S, n=128)
    assert rel < 1e-14, f"{name}: RHS differs from dysts by {rel:.2e} (relative)"


@pytest.mark.parametrize("name", systems.SYSTEM_ORDER)
def test_jacobian_matches_complex_step(name):
    """Analytic dF/du against the complex-step derivative (no cancellation)."""
    S = systems.System(name)
    _, rel = systems.verify_jacobian(S, n=128)
    assert rel < 1e-13, f"{name}: Jacobian differs by {rel:.2e} (relative)"


def test_collocation_jacobian_matches_residual():
    """The assembled block Jacobian differentiates the residual it is built from."""
    S = systems.System("Lorenz")
    T = S.horizon(1.0)
    centers, gamma = core.geometry(16)
    p = core.n_params(centers)
    s = np.linspace(-1, 1, 200)
    D0 = core.dict_rows(s, centers, gamma, 0)
    D1 = core.dict_rows(s, centers, gamma, 1)
    rng = np.random.default_rng(0)
    A = 0.02 * rng.standard_normal((S.d, p))
    sigma = np.array([8.0, 9.0, 24.0])
    _, J = core._assemble(S, A, D0, D1, sigma, T)
    h, worst = 1e-7, 0.0
    for _ in range(10):
        c, m = int(rng.integers(S.d)), int(rng.integers(p))
        Rp, _ = core._assemble(S, _bump(A, c, m, h), D0, D1, sigma, T, need_jac=False)
        Rm, _ = core._assemble(S, _bump(A, c, m, -h), D0, D1, sigma, T, need_jac=False)
        fd = (Rp - Rm).T.ravel() / (2 * h)
        worst = max(worst, np.max(np.abs(fd - J[:, c * p + m]))
                    / max(np.max(np.abs(fd)), 1e-300))
    assert worst < 1e-6, f"collocation Jacobian off by {worst:.2e}"


def _bump(A, c, m, h):
    B = A.copy()
    B[c, m] += h
    return B


def test_dictionary_reaches_machine_precision_on_a_known_target():
    """The frozen geometry itself is not the limit: plain lstsq on a smooth
    target hits the repo's fp64 floor. Anything worse downstream is the solve."""
    centers, gamma = core.geometry(256)
    s = np.linspace(-1, 1, 4001)
    y = np.sin(3.0 * s) * np.exp(-0.3 * s)
    D0 = core.dict_rows(s, centers, gamma, 0)
    a = np.linalg.lstsq(D0, y, rcond=core.RCOND)[0]
    rel = np.linalg.norm(D0 @ a - y) / np.linalg.norm(y)
    assert rel < 1e-13, f"frozen dictionary only reached {rel:.2e}"


def test_reference_is_better_than_what_it_certifies():
    """A fp64 Runge-Kutta reference is NOT good enough on a chaotic IVP: the
    mpmath reference must be, and its two-precision self-check proves it."""
    S = systems.System("Lorenz")
    T = S.horizon(3.0)
    sc = reference.selfcheck(S, T, n_eval=501, dps_lo=25, dps_hi=40)
    assert sc["rel_l2"] < 1e-20, f"mpmath reference self-check {sc['rel_l2']:.2e}"
    cc = reference.crosscheck(S, T, n_eval=501, tols=((1e-13, 1e-14),))
    assert cc["rtol1e-13"]["rel_l2"] > 1e-14, (
        "DOP853 at rtol=1e-13 is unexpectedly good; re-check the claim that a "
        "fp64 reference cannot certify these numbers")


@pytest.mark.slow
def test_lorenz_solve_reaches_the_floor():
    """End-to-end: Lorenz over three Lyapunov times, warm start ~1e-7 -> ~1e-13."""
    S = systems.System("Lorenz")
    T = S.horizon(3.0)
    ts, Yref = reference.reference(S, T, 2001, verbose=False)
    cell = core.solve_cell(S, T, 256, warm_rtol=1e-8, warm_atol=1e-11)
    rel, _, _ = core.errors(core.model_trajectory(cell, ts), Yref)
    warm, _, _ = core.errors(core.warm_trajectory(cell, ts), Yref)
    assert warm > 1e-9, "warm start is suspiciously good; it should be the cheap solve"
    assert rel < 1e-11, f"Lorenz solve only reached {rel:.2e}"
