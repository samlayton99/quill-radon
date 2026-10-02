"""expF18: the drifting Taylor-Green oracle satisfies NS, derivative rows match finite
differences, the block rule holds on the cube, the oracle fit converges with width, and a
small Gauss-Newton solve descends from the Stokes step toward the oracle."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "expF18_ns_spacetime"))

import problem as pb  # noqa: E402
import ridge3d as r3  # noqa: E402


def test_drifting_taylor_green_is_an_exact_ns_solution():
    mom, div, curl = pb.verify(verbose=False)
    assert mom < 1e-7 and div < 1e-9
    assert curl > 0.1                       # the advection is not a gradient: Newton has work to do


def test_derivative_rows_match_finite_differences():
    d = r3.SpaceTimeDict(12, 8, seed=2)
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.8, 0.8, (40, 3))
    c = rng.normal(size=d.cols)
    f = lambda Q: d.rows(Q, [(r3.VAL, 1.0)]) @ c  # noqa: E731
    for alpha, ax, o in [(r3.DX, 0, 1), (r3.DY, 1, 1), (r3.DT, 2, 1), (r3.DXX, 0, 2), (r3.DYY, 1, 2)]:
        h = 1e-5 if o == 1 else 1e-4
        fd = pb._fd(f, X, ax, o, h)
        an = d.rows(X, [(alpha, 1.0)]) @ c
        assert np.abs(fd - an).max() < (1e-8 if o == 1 else 1e-5) * np.abs(an).max()
    # a variable coefficient enters as a row scaling and the bias column only at order 0
    coeff = rng.normal(size=len(X))
    A = d.rows(X, [(r3.DX, coeff), (r3.VAL, 2.0)])
    B = coeff[:, None] * d.rows(X, [(r3.DX, 1.0)]) + 2.0 * d.rows(X, [(r3.VAL, 1.0)])
    assert np.allclose(A, B)


def test_blocks_are_even_qi_grids_on_the_cube():
    d = r3.SpaceTimeDict(8, 16)
    for b in d.geom.blocks:
        assert abs(b.gamma * b.h - 0.25) < 1e-14
        assert b.T == pytest.approx(1.25 * np.abs(b.v).sum())     # support function of the cube
        assert np.allclose(np.diff(b.offsets), b.h)
    assert d.cols == 8 * 16 + 1


def test_oracle_fit_converges_with_width():
    errs = []
    for W in (256, 1024):
        d = r3.SpaceTimeDict(W // 16, 16)
        pts = r3.point_sets(3 * d.cols, seed=0)
        model, info = r3.oracle_fit(d, pts)
        errs.append(r3.quick_score(model)["rel_l2_v"])
    assert errs[1] < 1e-2 * errs[0] and errs[1] < 1e-5


def test_gauss_newton_descends_from_stokes_toward_the_oracle():
    W = 512
    d = r3.SpaceTimeDict(W // 16, 16)
    pts = r3.point_sets(3 * d.cols, seed=0)
    asm = r3.Assembly(d, pts)
    a, hist = r3.gauss_newton(asm, max_iter=6, verbose=False,
                              eval_fn=lambda a: r3.quick_score(r3.Model(d, a)))
    assert hist[0]["rel_l2_v"] > 0.05                           # the Stokes step alone is wrong (drift)
    assert hist[-1]["rel_l2_v"] < 5e-3                          # Newton recovers the flow
    assert hist[-1]["res_after"] < 0.05 * hist[0]["res_before"]
    ora, _ = r3.oracle_fit(d, pts)
    assert hist[-1]["rel_l2_v"] < 30 * r3.quick_score(ora)["rel_l2_v"]
    mom, div = r3.Model(d, a).ns_residual(r3.interior_points(500, np.random.default_rng(1)))
    assert np.abs(div).max() < 0.1 and np.abs(mom).max() < 0.5


def test_augmented_qr_solve_matches_the_svd_solve():
    from h06.core import solve_augmented
    rng = np.random.default_rng(5)
    A = rng.normal(size=(300, 40)) @ np.diag(np.logspace(0, -12, 40))   # gapless, ill-conditioned
    y = rng.normal(size=300)
    ref = solve_augmented(A, y, rcond=1e-10)
    buf = np.empty((300, 41), order="F")
    buf[:, :40], buf[:, 40] = A, y
    x, rank = r3.solve_stacked([buf], rcond=1e-10)
    assert rank == ref.rank
    assert np.linalg.norm(x - ref.coef) < 1e-9 * np.linalg.norm(ref.coef)
