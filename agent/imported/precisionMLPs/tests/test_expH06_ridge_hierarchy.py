"""expH06: block geometry, nested directions, atom finding, Gauss-Newton polish, grower."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "expH06_ridge_hierarchy"))

from h06.core import (Geometry, make_block, nested_directions, fit_geometry, rel_l2, ball, origin,  # noqa: E402
                      solve_augmented, LAMBDA)
from h06.targets import get_target                                                                # noqa: E402
from h06.atoms import projection_pursuit, varpro_polish, tangent_basis, block_columns             # noqa: E402
from h06.grow import Grower                                                                       # noqa: E402


@pytest.fixture(scope="module")
def data3():
    d, r = 3, 0.3
    Z = ball(6000, d, r, np.random.default_rng(0))
    Zt = ball(3000, d, 0.27, np.random.default_rng(1))
    return d, Z, Zt, origin(d)


def test_block_is_an_even_qi_grid(data3):
    d, Z, _, _ = data3
    b = make_block(np.array([1.0, 0.0, 0.0]), Z, 16)
    assert abs(b.gamma * b.h - LAMBDA) < 1e-14
    t = b.offsets
    assert np.allclose(np.diff(t), b.h)
    assert t[0] == pytest.approx(-b.T + 0.5 * b.h) and t[-1] == pytest.approx(b.T - 0.5 * b.h)
    assert b.T == pytest.approx(1.25 * np.max(np.abs(Z[:, 0])))


def test_nested_directions_are_prefixes_and_spread():
    V = nested_directions(3, 64)
    V32 = nested_directions(3, 32)
    assert np.allclose(V[:32], V32)
    G = np.abs(V @ V.T)
    np.fill_diagonal(G, 0)
    assert G[:16, :16].max() < G[:64, :64].max() < 0.99


def test_exact_directions_reach_the_floor(data3):
    d, Z, Zt, x0 = data3
    tgt = get_target("ridge2", d)
    g = Geometry([make_block(u, Z, 48, "atom") for u in tgt.U])
    fit = fit_geometry(g, Z, tgt(x0 + Z))
    assert rel_l2(fit.predict(g, Zt), tgt(x0 + Zt)) < 1e-11


def test_gauss_newton_jacobian_matches_finite_differences(data3):
    d, Z, _, x0 = data3
    Zs = Z[:1500]
    tgt = get_target("ridge1", d)
    y = tgt(x0 + Zs)
    v0 = tgt.U[0] + 0.05 * np.array([0.3, -0.2, 0.1])
    g = Geometry([make_block(v0, Zs, 24, "atom")])
    A = g.augmented(Zs)
    fit = solve_augmented(A, y)
    b = g.blocks[0]
    U = tangent_basis(b.v)
    _, sech2 = block_columns(b, Zs)
    dA_ddelta0 = b.gamma * sech2 * (Zs @ U[:, 0])[:, None]
    eps = 1e-6
    v1 = b.v + eps * U[:, 0]
    b1 = make_block(v1, Zs, 24, "atom")
    b1.T = b.T                                   # hold the band fixed for the derivative check
    th1, _ = block_columns(b1, Zs)
    th0, _ = block_columns(b, Zs)
    fd = (th1 - th0) / eps
    assert np.max(np.abs(fd - dA_ddelta0)) < 1e-4 * np.max(np.abs(dA_ddelta0))


def test_projection_pursuit_plus_polish_recovers_a_ridge(data3):
    d, Z, Zt, x0 = data3
    tgt = get_target("ridge1", d)
    y = tgt(x0 + Z)
    v, sc, sp = projection_pursuit(Z, y, n_off=32)
    assert sp < 1e-8
    assert abs(abs(float(v @ tgt.U[0])) - 1.0) < 1e-10


def test_joint_polish_recovers_four_ridges(data3):
    d, Z, Zt, x0 = data3
    tgt = get_target("ridge4", d)
    y = tgt(x0 + Z)
    found, res = [], y.copy()
    for _ in range(4):
        v, _, _ = projection_pursuit(Z, res, n_off=32)
        found.append(v)
        g = Geometry([make_block(u, Z, 48, "atom") for u in found])
        fit = fit_geometry(g, Z, y)
        res = y - g.augmented(Z) @ fit.coef
    g, hist = varpro_polish(g, Z, y, which=[0, 1, 2, 3], iters=20)
    fit = fit_geometry(g, Z, y)
    assert rel_l2(fit.predict(g, Zt), tgt(x0 + Zt)) < 1e-10
    for b in g.blocks:
        assert max(abs(float(b.v @ u)) for u in tgt.U) > 1 - 1e-10


def test_grower_finds_two_ridges_cheaply(data3):
    d, Z, Zt, x0 = data3
    tgt = get_target("ridge2", d)
    y = tgt(x0 + Z)
    gr = Grower(d, Z[500:], y[500:], Z[:500], y[:500], budget=256, verbose=False)
    geom, fit, hist = gr.run()
    assert geom.describe()["n_atoms"] >= 2
    assert rel_l2(fit.predict(geom, Zt), tgt(x0 + Zt)) < 1e-10
    assert geom.units <= 256
