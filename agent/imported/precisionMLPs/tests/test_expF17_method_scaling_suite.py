"""expF17 verification battery.

What must hold before any benchmark number is produced:
  1. oracle purity            every closed-form oracle passes its FD-residual gate
  2. exact column identity    all families report (n_ax+1)^2 actual columns (+3 PHS)
  3. derivative rows          every family x every needed (ax,ay) matches central FD
  4. seed jitter              seeded, reproducible, perturbs geometry not counts
  5. QI floor anchor          radon/tensor oracle fits reproduce the step-0 numbers
  6. spectral sanity          smooth-task fit reaches near machine precision
  7. lstsq behaviour          min-norm on rank-deficient systems (gelsd semantics)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF17_method_scaling_suite"))

from f17 import dicts as fd, solve as sv
from f17.tasks import TASKS, verify_oracles

RNG = np.random.default_rng(3)


def test_oracle_purity_gate():
    verify_oracles(verbose=False)


def test_exact_column_identity():
    for C in (128, 256, 512, 1024, 2048):
        n_ax = fd.n_ax_for(C)
        want = (n_ax + 1) ** 2
        rng = np.random.default_rng(0)
        halo = max(1, min(round(n_ax / 8), 8))
        built = {
            "qi_radon": fd.build_dictionary("qi_radon", n_ax, {"halo": halo}, rng),
            "qi_tensor": fd.build_dictionary("qi_tensor", n_ax, {"halo": halo}, rng),
            "elm": fd.build_dictionary("elm", n_ax, {"R": 2.0}, rng),
            "spectral": fd.build_dictionary("spectral", n_ax, {}, rng),
            "bwler": fd.build_dictionary("bwler", n_ax, {}, rng),
            "rbf_imq": fd.build_dictionary("rbf_imq", n_ax, {"epsh": 1.0}, rng),
        }
        for m, d in built.items():
            assert d.cols == want, (C, m, d.cols, want)
        assert fd.build_dictionary("rbf_phs", n_ax, {}, rng).cols == want + 3


@pytest.mark.parametrize("method,config", [
    ("qi_radon", {"halo": 2}), ("qi_tensor", {"halo": 2}), ("elm", {"R": 2.0}),
    ("rbf_imq", {"epsh": 1.0}), ("rbf_phs", {}), ("spectral", {}), ("bwler", {}),
])
def test_derivative_rows_match_fd(method, config):
    n_ax = 12
    rng = np.random.default_rng(11)
    d = fd.build_dictionary(method, n_ax, config, rng,
                            fourier_x=(method == "spectral"))
    v = np.random.default_rng(5).standard_normal(d.cols)
    P = np.random.default_rng(7).uniform(-0.9, 0.9, (40, 2))

    def u(Q):
        return d.rows(Q, [((0, 0), 1.0)]) @ v

    for (ax, ay), h in [((1, 0), 1e-6), ((0, 1), 1e-6), ((2, 0), 1e-4), ((0, 2), 1e-4)]:
        got = d.rows(P, [((ax, ay), 1.0)]) @ v
        E = np.zeros((1, 2)); E[0, 0 if ax > 0 else 1] = h  # step on the differentiated axis
        if (ax + ay) == 1:
            ref = (u(P + E) - u(P - E)) / (2 * h)
        else:
            ref = (u(P + E) - 2 * u(P) + u(P - E)) / h ** 2
        scale = max(1.0, np.abs(ref).max())
        assert np.abs(got - ref).max() / scale < 2e-5, (method, (ax, ay))


def test_callable_coefficients_and_term_sums():
    rng = np.random.default_rng(0)
    d = fd.build_dictionary("qi_radon", 10, {"halo": 1}, rng)
    P = np.random.default_rng(1).uniform(-1, 1, (30, 2))
    c = lambda Q: Q[:, 0] ** 2 + 1.0
    A = d.rows(P, [((1, 0), c), ((0, 0), 2.0)])
    B = (c(P)[:, None] * d.rows(P, [((1, 0), 1.0)]) + 2.0 * d.rows(P, [((0, 0), 1.0)]))
    assert np.abs(A - B).max() < 1e-12


def test_seed_jitter_reproducible_and_effective():
    cfg = {"halo": 2}
    d0a = fd.build_dictionary("qi_radon", 15, cfg, np.random.default_rng(1000))
    d0b = fd.build_dictionary("qi_radon", 15, cfg, np.random.default_rng(1000))
    d1 = fd.build_dictionary("qi_radon", 15, cfg, np.random.default_rng(1001))
    assert np.array_equal(d0a.b, d0b.b) and not np.array_equal(d0a.b, d1.b)
    assert d0a.cols == d1.cols
    t0 = fd.build_dictionary("qi_tensor", 15, cfg, np.random.default_rng(1000))
    t1 = fd.build_dictionary("qi_tensor", 15, cfg, np.random.default_rng(1001))
    assert t0.cols == t1.cols and not np.allclose(t0.cx, t1.cx)


def test_qi_floor_anchor_step0():
    """Reproduce the step-0 oracle-fit floor (geom_setup.py protocol, zero phase):
    convection_c40 at n_ax=32, halo 4, 3x data, rcond 1e-13, zero phase gave
    4.64e-5 (radon) and 4.24e-7 (tensor) -- geom_setup.json. Gate at 3x (the
    data draw differs by rng stream position)."""
    prob = TASKS["convection_c40"]
    rng = np.random.default_rng(0)
    Pte = rng.uniform(-1, 1, (20000, 2))
    n_ax = 32
    C = (n_ax + 1) ** 2
    Ptr = rng.uniform(-1, 1, (3 * C, 2))
    ytr, yte = prob["exact"](Ptr), prob["exact"](Pte)
    for method, anchor in [("qi_radon", 4.64e-5), ("qi_tensor", 4.24e-7)]:
        d = fd.build_dictionary(method, n_ax, {"halo": 4}, np.random.default_rng(0))
        if method == "qi_radon":
            d = fd.radon_dict(n_ax, 4, seed_phase=(0.0, 0.0))
        else:
            d = fd.TensorDict(n_ax, 4, shift=(0.0, 0.0))
        A = d.rows(Ptr, [((0, 0), 1.0)])
        v, *_ = np.linalg.lstsq(A, ytr, rcond=1e-13)
        err = np.linalg.norm(d.rows(Pte, [((0, 0), 1.0)]) @ v - yte) / np.linalg.norm(yte)
        assert err < 3 * anchor, (method, err, anchor)


def test_oracle_footprint_is_the_closed_square():
    """SPEC 18.9: an open-square draw leaves the halo columns unpinned on the
    boundary lines the score grid contains (measured 7-10x at the floor, with
    the dynamic arm landing BELOW the oracle). The protocol oracle must match a
    fit on the full score grid to within 2x and carry no edge layer."""
    prob = TASKS["darcy_man"]
    n_ax = 31
    d = fd.build_dictionary("qi_radon", n_ax, {"halo": 5}, np.random.default_rng(0))
    a, info = sv.oracle_fit(prob, d, 0)
    assert info["n_data"] == 4 * d.cols + 640
    m = sv.score(prob, d, a)
    Pg, ug, _ = sv.eval_data(prob)
    a_full, _ = sv.lstsq(d.rows(Pg, [((0, 0), 1.0)]), ug)
    m_full = sv.score(prob, d, a_full)
    assert m["rel_l2"] < 2 * m_full["rel_l2"], (m["rel_l2"], m_full["rel_l2"])
    err = np.abs(d.rows(Pg, [((0, 0), 1.0)]) @ a - ug)
    dist = 1 - np.max(np.abs(Pg), axis=1)
    edge = np.sqrt(np.mean(err[dist < 0.02] ** 2))
    inner = np.sqrt(np.mean(err[dist >= 0.1] ** 2))
    assert edge < 5 * inner, (edge, inner)


def test_spectral_hits_floor_on_wave():
    """wave needs ~32 Chebyshev modes in eta (5 periods); at n_ax=44 the measured
    fit is 3.2e-10 (n_ax=22 is genuinely under-resolved at 2e-1 -- the cliff)."""
    prob = TASKS["wave"]
    n_ax = 44  # C = 2025, resolved regime
    d = fd.build_dictionary("spectral", n_ax, {}, np.random.default_rng(0))
    rng = np.random.default_rng(2)
    Ptr = rng.uniform(-1, 1, (4 * d.cols, 2))
    ytr = prob["exact"](Ptr)
    A = d.rows(Ptr, [((0, 0), 1.0)])
    v, *_ = np.linalg.lstsq(A, ytr, rcond=1e-15)
    Pte = rng.uniform(-1, 1, (20000, 2))
    err = (np.linalg.norm(d.rows(Pte, [((0, 0), 1.0)]) @ v - prob["exact"](Pte))
           / np.linalg.norm(prob["exact"](Pte)))
    assert err < 1e-8, err


def test_bwler_same_span_as_chebyshev():
    """Value-space and coefficient-space rows must span the same space: the fit
    of a smooth target should agree to a modest factor at moderate n."""
    prob = TASKS["reaction"]
    n_ax = 15
    rng = np.random.default_rng(4)
    errs = {}
    for method in ("spectral", "bwler"):
        d = fd.build_dictionary(method, n_ax, {}, np.random.default_rng(0))
        Ptr = np.random.default_rng(9).uniform(-1, 1, (4 * d.cols, 2))
        A = d.rows(Ptr, [((0, 0), 1.0)])
        v, *_ = np.linalg.lstsq(A, prob["exact"](Ptr), rcond=1e-15)
        Pte = np.random.default_rng(10).uniform(-1, 1, (10000, 2))
        errs[method] = (np.linalg.norm(d.rows(Pte, [((0, 0), 1.0)]) @ v
                                       - prob["exact"](Pte))
                        / np.linalg.norm(prob["exact"](Pte)))
    ratio = max(errs.values()) / min(errs.values())
    assert ratio < 50, errs


def test_lstsq_min_norm_on_rank_deficient():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((60, 10))
    A[:, 5:] = A[:, :5]  # rank 5
    y = A @ np.ones(10)
    v, res, rank, sv = np.linalg.lstsq(A, y, rcond=1e-15)
    assert rank == 5
    assert np.abs(A @ v - y).max() < 1e-10


# ---------------------------------------------------------------------------
# the 1-D (dysts) arm
# ---------------------------------------------------------------------------

from f17 import dicts1d as f1  # noqa: E402
from f17 import solve1d as s1  # noqa: E402


def test_1d_column_identity():
    rng = np.random.default_rng(0)
    for W in (48, 96, 256):
        for method, cfg in [("qi_grid", {"halo": 4}), ("elm", {"R": 2.0}),
                            ("spectral", {}), ("bwler", {}),
                            ("rbf_imq", {"epsh": 1.0}), ("rbf_phs", {})]:
            d = f1.build_dictionary_1d(method, W, cfg, np.random.default_rng(0))
            assert d.cols == W, (method, W, d.cols)


@pytest.mark.parametrize("method,config", [
    ("qi_grid", {"halo": 3}), ("elm", {"R": 2.0}), ("spectral", {}),
    ("bwler", {}), ("rbf_imq", {"epsh": 1.0}), ("rbf_phs", {}),
])
def test_1d_derivative_rows_match_fd(method, config):
    d = f1.build_dictionary_1d(method, 32, config, np.random.default_rng(3))
    v = np.random.default_rng(5).standard_normal(d.cols)
    s = np.random.default_rng(7).uniform(-0.9, 0.9, 50)
    h = 1e-6
    got = d.rows(s, 1) @ v
    ref = ((d.rows(s + h, 0) - d.rows(s - h, 0)) @ v) / (2 * h)
    scale = max(1.0, np.abs(ref).max())
    assert np.abs(got - ref).max() / scale < 2e-5, method


def test_1d_qi_floor_anchor_lorenz():
    """Measured in-build: qi_grid oracle fit of the Lorenz mpmath reference at
    W=256, halo 8, seed 0 gives 1.07e-13. Gate at 10x."""
    from f17.tasks1d import dysts_task
    from f17.dicts import dict_rng
    t = dysts_task("Lorenz")
    d = f1.build_dictionary_1d("qi_grid", 256, {"halo": 8}, dict_rng("qi_grid", 256, 0))
    A, sigma, info = s1.oracle_fit_1d(t, d, 0)
    m = s1.score_1d(t, d, A, sigma)
    assert m["rel_l2"] < 1.1e-12, m["rel_l2"]


def test_1d_spectral_dynamic_floor_lorenz():
    """Measured in-build: the Chebyshev collocation solve of Lorenz at W=384
    reaches 1.7e-14 under Chebyshev-density collocation (uniform-density data
    at this degree is the Platte-Trefethen-Kuijlaars unstable regime and gave
    9.3e0 -- the sampling rule is load-bearing). Gate at 100x."""
    from f17.tasks1d import dysts_task
    t = dysts_task("Lorenz")
    d, A, sigma, info = s1.dynamic_solve_1d(t, "spectral", 384, {}, 0)
    m = s1.score_1d(t, d, A, sigma)
    assert not info["diverged"]
    assert m["rel_l2"] < 2e-12, m["rel_l2"]


def test_sampling_density_rule():
    """spectral/bwler draw arcsine-density data; everyone else uniform."""
    rng = np.random.default_rng(0)
    idx_u = s1._sample_nodes(6001, 1000, np.random.default_rng(1), False)
    idx_c = s1._sample_nodes(6001, 1000, np.random.default_rng(1), True)
    assert len(idx_u) == len(idx_c) == 1000
    assert len(np.unique(idx_c)) == 1000
    # arcsine density concentrates at the ends: raw draw puts 41% in the outer
    # 10% bands vs 20% uniform; grid dedup + uniform top-up dilutes to ~1.7x
    outer = lambda idx: np.mean((idx < 600) | (idx > 5400))
    assert outer(idx_c) > 1.4 * outer(idx_u)


def test_translation_end_to_end():
    """The coordinate maps, folded into the operator coefficients, verified
    through OUR row assembly (not just expF13's FD check): fit a spectral
    dictionary to the exact solution, then the assembled interior operator rows
    applied to that fit must reproduce the forcing to ~ the fit's own accuracy
    (amplified by the operator's derivative scale)."""
    from f17 import dicts as fdm, solve as svm
    # poisson_man is excluded: its oracle's log singularities sit just inside
    # the holes, so a spectral fit converges only algebraically and the n^4
    # derivative amplification swamps the residual; its map is scale-invariant
    # (unit Laplacian) and FD-verified in expF13. The tasks below carry the
    # actual coefficient folding (2u_eta + (c/pi)u_xi; 4/-16; the a-terms).
    for key, nax, tol in [("convection_c40", 44, 1e-4), ("wave", 44, 1e-3),
                          ("darcy_man", 31, 1e-5)]:
        t = svm.TASKS[key]
        d = fdm.build_dictionary("spectral", nax, {}, np.random.default_rng(0),
                                 fourier_x=t["periodic_fourier"])
        rng = np.random.default_rng(1)
        P = svm.interior_points(t, 4 * d.cols, rng, method="spectral")
        a, _ = np.linalg.lstsq(d.rows(P, [((0, 0), 1.0)]), t["exact"](P),
                               rcond=1e-15)[:2]
        Q = svm.interior_points(t, 500, np.random.default_rng(2), method="spectral")
        lin = None
        for idx, c in t["lin_terms"]:
            cc = fdm._coeff_col(c, Q)
            rows = d.rows(Q, [(idx, 1.0)])
            lin = (0.0 if lin is None else lin) + (cc * rows if np.ndim(cc) else cc * rows)
        vals = {i: d.rows(Q, [(i, 1.0)]) @ a for i in t["nl"]["fields"]}
        r = lin @ a + t["nl"]["res"](vals, Q)
        f = t["forcing"]
        fv = f(Q) if callable(f) else np.full(len(Q), float(f))
        scale = max(1.0, np.abs(lin @ a).max())
        rel = np.abs(r - fv).max() / scale
        assert rel < tol, (key, rel)


# --- the PINN arm (SPEC 18.8) --------------------------------------------------

def test_pinn_closed_form_derivatives_match_autograd():
    import torch
    from f17 import pinn
    params = pinn.init_params(6, 0)
    P = (torch.rand(40, 2, dtype=torch.float64) * 2 - 1).requires_grad_(True)
    D = pinn.net_derivs(params, P, [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2)])
    g = torch.autograd.grad(D[(0, 0)].sum(), P, create_graph=True)[0]
    gxx = torch.autograd.grad(g[:, 0].sum(), P, create_graph=True)[0][:, 0]
    gyy = torch.autograd.grad(g[:, 1].sum(), P, create_graph=True)[0][:, 1]
    for got, want in ((D[(1, 0)], g[:, 0]), (D[(0, 1)], g[:, 1]),
                      (D[(2, 0)], gxx), (D[(0, 2)], gyy)):
        assert float((got - want).abs().max()) < 1e-13


def test_pinn_is_the_elm_architecture_and_footprint():
    """Column identity (n_ax+1)^2, the RidgeDict readback reproduces the
    network, and the data footprints equal the solved arms' (oracle: 4C + the
    closed square's perimeter; dynamic: 4C interior + build_bcs' point sets)."""
    import torch
    from f17 import pinn
    t = TASKS["darcy_man"]
    n_ax = 10
    params = pinn.init_params(n_ax, 1)
    d, a = pinn.as_dict(params)
    assert d.cols == (n_ax + 1) ** 2
    P = np.random.default_rng(0).uniform(-1, 1, (30, 2))
    u_net = pinn.net_derivs(params, torch.as_tensor(P), [(0, 0)])[(0, 0)].detach().numpy()
    assert np.max(np.abs(d.rows(P, [((0, 0), 1.0)]) @ a - u_net)) < 1e-14
    od = pinn.oracle_data(t, n_ax, 0)
    assert od["n_data"] == 4 * d.cols + 640
    dd = pinn.dynamic_data(t, n_ax, 0)
    assert dd["n_data"] == 4 * d.cols
    bcs = sv.build_bcs(t, d, dd["n_data"])
    assert [len(g) for (_, g, _) in bcs] == [len(g) for (_, _, _, g) in dd["blocks"]]
    # seeds perturb the init, reproducibly
    p2 = pinn.init_params(n_ax, 1)
    assert all(torch.equal(x, y) for x, y in zip(params, p2))
    assert not torch.equal(params[0], pinn.init_params(n_ax, 2)[0])


def test_pinn_short_budget_trains_and_refit_never_worse():
    """A reduced-budget run on a smooth task: the loss falls by orders of
    magnitude, the trained score is finite, and the frozen-feature refit (the
    certificate) is at least as good as the trained readout in both regimes."""
    from f17 import pinn
    t = TASKS["darcy_man"]
    for regime in ("oracle", "dynamic"):
        m_tr, m_rf, info = pinn.run_pinn(t, 8, regime, 0, 1e-3, w_mult=1.0,
                                         adam_steps=800, lbfgs_iters=200)
        assert info["hist"][0][1] / info["loss_final"] > 30
        assert np.isfinite(m_tr["rel_l2"]) and m_tr["rel_l2"] < 1.0
        assert m_rf["rel_l2"] <= 1.05 * m_tr["rel_l2"]
        assert info["loss_final"] <= info["loss_adam"]
