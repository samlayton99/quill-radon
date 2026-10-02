"""Tests for the checkpoint-H high-dimensional benchmark suite (expH01).

These check the measuring device, not a model: the direction basis, every target's
analytic gradient against finite differences, the 80-task list and the tasks that are
meant to share a function, the common scaling, every data geometry, the three test sets,
the predicted center density, and two reference fits that pin the even-geometry model to
the accuracy floors it is known to reach.

Run with ``-s`` to see the numbers.
"""

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Append (never insert-0): the experiment directory holds generically named modules
# (run/viz) shared with sibling experiments, so keep it off sys.path[0]; run.py is
# loaded by explicit path below.
sys.path.append(str(REPO_ROOT / "experiments" / "expH01_highdim_suite"))

from scipy.stats import qmc

from h01suite import baseline as BL
from h01suite import densities as D
from h01suite import metrics as M
from h01suite import targets as TG
from h01suite.basis import dct_basis, l1_scales, u, z_coord, z_of
from h01suite.tasks import TASKS, SMOKE_IDS, get_task, packet_anchor_margins, tasks_for_dim

DIMS = (1, 2, 3, 4, 5)


def _load_expH01(name, relpath):
    """Load an expH01 module by explicit path under a unique name."""
    path = REPO_ROOT / "experiments" / "expH01_highdim_suite" / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ===========================================================================
# directions and coordinates
# ===========================================================================

def test_dct_basis_orthogonal():
    """Q_d must be orthogonal to 1e-14 in every dimension the suite uses."""
    for d in DIMS:
        Q = dct_basis(d)
        err = float(np.abs(Q.T @ Q - np.eye(d)).max())
        print(f"d={d}  ||Q^T Q - I||_max = {err:.2e}")
        assert err < 1e-14
    assert np.allclose(dct_basis(1), [[1.0]])


def test_z_coordinate_range_on_cube_corners():
    """z_k runs over [-1,1] on the cube, with both ends attained at corners."""
    for d in DIMS:
        C = np.array(list(itertools.product([-1.0, 1.0], repeat=d)))
        for k in range(1, d + 1):
            z = z_coord(C, u(d, k))
            assert z.min() >= -1.0 - 1e-14 and z.max() <= 1.0 + 1e-14
            assert abs(z.min() + 1.0) < 1e-14 and abs(z.max() - 1.0) < 1e-14
        # z_of computes all d of them at once and must agree
        assert np.allclose(z_of(C, d), (C @ dct_basis(d)) / l1_scales(d))
    print("z_k in [-1,1] with both ends attained, for d = 1..5")


# ===========================================================================
# the target functions
# ===========================================================================

def _all_families(d):
    """One instance of every family in dimension ``d``."""
    return [TG.MultiscaleBumps(d), TG.WideBump(d),
            TG.RadialOscillation(d, 1.0), TG.RadialOscillation(d, 6.0),
            TG.Composition(d), TG.Polynomial(d),
            TG.RadialRunge(d, 4.0), TG.RadialRunge(d, 12.0),
            TG.ProductPeak(d),
            TG.SpatialPacket(d, TG.aligned_packet_anchor(d), "at hotspot"),
            TG.SpatialPacket(d, TG.antialigned_packet_anchor(d), "away"),
            TG.SphereJump(d), TG.WavyJump(d),
            TG.KinkRing(d), TG.OneSidedKink(d), TG.Piecewise(d)]


FAMILY_NAMES = [f.label() for f in _all_families(3)]


@pytest.mark.parametrize("which", range(len(FAMILY_NAMES)), ids=FAMILY_NAMES)
def test_every_target_gradient_matches_finite_differences(which):
    """Every family's analytic gradient must match central differences in every d.

    Points within 0.05 of a step or slope break are excluded: there the analytic
    gradient is the gradient of the smooth part and finite differences straddle the
    surface, so the two are not supposed to agree.
    """
    h = 1e-6
    worst = 0.0
    for d in DIMS:
        f = _all_families(d)[which]
        X = np.random.default_rng(4 + d).uniform(-0.92, 0.92, size=(400, d))
        g = f.grad(X)
        fd = np.zeros_like(g)
        for j in range(d):
            e = np.zeros(d)
            e[j] = h
            fd[:, j] = (f.value(X + e) - f.value(X - e)) / (2 * h)
        mask = f.interface_mask(X)
        keep = np.ones(len(X), bool) if mask is None else ~mask
        assert keep.sum() > 50
        scale = max(1e-12, float(np.abs(g[keep]).max()))
        rel = float(np.abs(fd[keep] - g[keep]).max() / scale)
        worst = max(worst, rel)
    print(f"{FAMILY_NAMES[which]:38s} worst relative gradient error over d=1..5: {worst:.2e}")
    assert worst < 1e-6


def test_step_families_are_flagged_and_kinks_are_not():
    """Only the two step families are marked as having no usable gradient."""
    d = 3
    flags = {f.family: f.differentiable for f in _all_families(d)}
    assert flags["sphere_jump"] is False and flags["wavy_jump"] is False
    assert flags["kink_ring"] is True and flags["piecewise"] is True
    assert flags["one_sided_kink"] is True and flags["radial_oscillation"] is True
    print("step families flagged, kink families not:", flags)


def test_step_surfaces_are_where_the_value_actually_jumps():
    """The marked interface really is where the function jumps, for both step families."""
    d, rng = 3, np.random.default_rng(0)
    for f, probe in ((TG.SphereJump(d), "sphere"), (TG.WavyJump(d), "wavy")):
        X = rng.uniform(-0.9, 0.9, size=(200000, d))
        v = f.value(X)
        m = f.interface_mask(X)
        # inside the band the values must span both levels; away from it, locally smooth
        assert m.sum() > 100
        near = v[m]
        assert near.max() - near.min() > 0.5, probe
        print(f"{probe}: {m.sum()} of {len(X)} points within 0.05 of the step, "
              f"value spread there {near.max() - near.min():.3f}")


def test_no_target_is_a_sum_of_one_dimensional_pieces():
    """A sum of profiles along fixed directions has a zero mixed second difference in
    the basis it is written in. The targets must not: they are genuinely multivariate.

    The check is a mixed second difference in the normalized coordinates ``z_1, z_2``,
    which is exactly zero, at any step size, for any ``sum_k g_k(z_k)`` and for any sum
    of profiles along the ``u_k``. The step is a coarse 0.05 rather than a small one so
    that the four corners of the stencil can straddle a curved interface: the one-sided
    kink is separable on each side of its sphere and only the sphere itself couples the
    coordinates.
    """
    d, h = 3, 0.05
    Q, s = dct_basis(d), l1_scales(d)
    e1 = h * s[0] * Q[:, 0]
    e2 = h * s[1] * Q[:, 1]
    rng = np.random.default_rng(2)
    X = rng.uniform(-0.85, 0.85, size=(4000, d))

    def mixed_difference(fn):
        return (fn(X + e1 + e2) - fn(X + e1 - e2) - fn(X - e1 + e2) + fn(X - e1 - e2)) / (4 * h * h)

    additive = lambda A: np.sin(np.pi * z_of(A, d)[:, 0]) + 0.5 * z_of(A, d)[:, 1] ** 3
    control = float(np.abs(mixed_difference(additive)).max())
    print(f"{'(control) a genuine sum of profiles':38s} largest mixed second difference "
          f"= {control:.3e}")
    assert control < 1e-9

    for f in _all_families(d):
        size = float(np.abs(mixed_difference(f.value)).max())
        print(f"{f.label():38s} largest mixed second difference d2F/dz1 dz2 = {size:.3e}")
        assert size > 1e-3, f"{f.label()} behaves like a sum of one-dimensional pieces"


def test_polynomial_reduces_correctly_in_one_dimension():
    """The cyclic index makes the 1-D polynomial exactly z^3 - z^4."""
    f = TG.Polynomial(1)
    x = np.linspace(-1, 1, 501)[:, None]
    assert np.allclose(f.value(x), x[:, 0] ** 3 - x[:, 0] ** 4)
    print("d=1 polynomial = z^3 - z^4 exactly")


def test_anchor_of_the_away_packet_is_far_from_every_cluster():
    """The away-from-hotspots burst must sit well clear of all three clusters.

    Distances are Euclidean in ``z``, divided by the cluster's standard deviation (an
    upper bound on its standard deviation in ``z``). The target is 2 widths. It holds
    with room to spare for d >= 2; in d = 1 the cube leaves nowhere to put the burst
    that is both 2 widths from the +0.45 cluster and still inside the domain, and the
    anchor at 0.85 lands at 1.82 widths.
    """
    for d in DIMS:
        m = packet_anchor_margins(d)
        print(f"d={d}  away-packet anchor at {TG.antialigned_packet_anchor(d)} sits "
              + ", ".join(f"{v:.2f} widths from mu_{k}" for k, v in m.items()))
        floor = 1.8 if d == 1 else 2.0
        assert min(m.values()) >= floor


# ===========================================================================
# the task list
# ===========================================================================

def test_task_list_shape_and_uniqueness():
    assert len(TASKS) == 80
    for d in DIMS:
        assert len(tasks_for_dim(d)) == 16
        assert [t.id for t in tasks_for_dim(d)] == [f"{d}.{i}" for i in range(1, 17)]
    assert len({t.id for t in TASKS}) == 80
    assert len({t.name for t in TASKS}) == 80
    assert all(t.what_it_tests.strip() for t in TASKS)
    assert len(SMOKE_IDS) == 36 and all(get_task(i) for i in SMOKE_IDS)
    print(f"80 tasks, 16 per dimension, ids and names unique; "
          f"{len(SMOKE_IDS)} in the first-pass list")


def _same_function_pairs():
    pairs = [(f"{d}.3", f"{d}.11") for d in DIMS]          # same waves, different data
    pairs += [(f"{d}.15", f"{d}.16") for d in (2, 3, 4, 5)]  # same sheet, with/without noise
    return pairs


@pytest.mark.parametrize("a,b", _same_function_pairs(),
                         ids=[f"{a}_and_{b}" for a, b in _same_function_pairs()])
def test_tasks_that_share_a_function_agree_bit_for_bit(a, b):
    """Only the data changes across these pairs, so the scaled target must be identical."""
    ta, tb = get_task(a), get_task(b)
    assert ta.d == tb.d and ta.density_tag != tb.density_tag
    assert ta.target is tb.target
    X = np.random.default_rng(11).uniform(-1.0, 1.0, size=(4000, ta.d))
    fa, fb = ta.F(X), tb.F(X)
    print(f"{a} vs {b}: identical at {int((fa == fb).sum())}/{len(X)} points "
          f"(data {ta.density_tag} vs {tb.density_tag})")
    assert np.array_equal(fa, fb)
    assert ta.mean_uniform == tb.mean_uniform and ta.sd_uniform == tb.sd_uniform


def test_the_two_bursts_are_the_same_family_in_different_places():
    """Tasks 12 and 13 differ only in where the burst of oscillation sits."""
    for d in DIMS:
        t12, t13 = get_task(f"{d}.12"), get_task(f"{d}.13")
        assert t12.target.family == t13.target.family == "spatial_packet"
        assert t12.target.tau == t13.target.tau and t12.target.omega == t13.target.omega
        assert not np.allclose(t12.target.a, t13.target.a)
    print("tasks 12 and 13 share the burst's width and frequency and differ only in place")


def test_one_dimensional_substitutions():
    """d = 1 replaces the four rows that need a second dimension or a sphere."""
    assert get_task("1.9").target.family == "piecewise"
    assert get_task("1.10").target.family == "one_sided_kink"
    assert get_task("1.15").target.family == "wavy_jump"      # the 1-D step at z_1 = .78
    assert get_task("1.16").target.family == "radial_runge"
    assert np.allclose(get_task("1.16").target.a, TG.aligned_packet_anchor(1))
    for d in (2, 3, 4, 5):
        assert get_task(f"{d}.9").target.family == "wavy_jump"
        assert get_task(f"{d}.10").target.family == "kink_ring"
        assert get_task(f"{d}.15").density_tag == "curved_sheet"
        assert get_task(f"{d}.16").density_tag == "curved_sheet_noisy"
    print("d=1 substitutions in place: 9 piecewise, 10 one-sided kink, "
          "15 sparse-region step, 16 spike on the cluster")


# ===========================================================================
# the common scaling
# ===========================================================================

def test_scaling_holds_on_an_independent_uniform_sample():
    """F must have mean ~0 and standard deviation ~1 on a uniform sample that is
    independent of the reference set used to compute the scaling.

    The check sample is a scrambled Sobol set (2^21 points, a different seed): plain
    independent draws cannot resolve a 1e-3 mean at any affordable size.
    """
    worst_mu = worst_sd = 0.0
    for d in DIMS:
        X = 2.0 * qmc.Sobol(d=d, scramble=True, seed=13579).random_base2(m=21) - 1.0
        for t in tasks_for_dim(d):
            v = t.F(X)
            mu, sd = abs(float(v.mean())), abs(float(v.std()) - 1.0)
            assert mu < 1e-3, f"task {t.id}: |mean| = {mu:.2e}"
            assert sd < 1e-2, f"task {t.id}: |sd - 1| = {sd:.2e}"
            worst_mu, worst_sd = max(worst_mu, mu), max(worst_sd, sd)
    print(f"worst |mean| = {worst_mu:.2e} (tolerance 1e-3), "
          f"worst |sd - 1| = {worst_sd:.2e} (tolerance 1e-2)")


# ===========================================================================
# data geometries
# ===========================================================================

def test_every_data_geometry_stays_inside_the_cube():
    for t in TASKS:
        X = t.sample(4000, seed=3)
        assert X.shape == (len(X), t.d)
        assert np.all(np.abs(X) <= 1.0), f"task {t.id} left the cube ({np.abs(X).max()})"
    print("all 80 task data geometries stay strictly inside [-1,1]^d")


def test_hotspot_mixture_fractions():
    """Per-cluster rejection keeps the realized fractions at the nominal weights
    (.20 uniform, .40, .25, .15)."""
    for d in (1, 3, 5):
        X, lab = D.Hotspots(d).sample_with_labels(200_000, seed=1)
        frac = np.array([float((lab == i).mean()) for i in range(4)])
        print(f"d={d}  fractions {np.round(frac, 4)}  (nominal .20 .40 .25 .15)")
        assert np.allclose(frac, [0.20, 0.40, 0.25, 0.15], atol=5e-3)
        assert np.all(np.abs(X) <= 1.0)


def test_hotspot_widths_and_densest_cluster():
    """The three clusters have widths .22, .28, .25, and the first one is the densest."""
    for d in (1, 3):
        h = D.Hotspots(d)
        got = [float(np.sqrt(c[0, 0])) for c in h.covs]
        assert np.allclose(got, [0.22, 0.28, 0.25])
        assert h.densest_cluster() == 0
    print("hotspot widths .22/.28/.25; the cluster at +0.45 is the densest")


def test_stretched_hotspot_covariance_recovered():
    """Sigma = Q diag(.25^2, .083^2, .15^2, ...) Q^T must come back out of the sample
    (a 3:1 stretch between u_1 and u_2)."""
    d = 3
    X, lab = D.StretchedHotspots(d).sample_with_labels(300_000, seed=2)
    C = np.cov(X[lab == 1].T)
    got = np.diag(dct_basis(d).T @ C @ dct_basis(d))
    want = np.array([0.25 ** 2, 0.083 ** 2, 0.15 ** 2])
    print(f"widths along u_1,u_2,u_3: {np.round(np.sqrt(got), 4)} "
          f"vs target {np.round(np.sqrt(want), 4)}")
    assert np.all(np.abs(got - want) / want < 0.10)
    assert np.sqrt(got[0] / got[1]) > 2.5           # the stretch really is about 3:1


def test_flat_sheet_perpendicular_coordinates():
    """Clean flat sheets have exactly zero perpendicular coordinates; the thickened
    variant has standard deviation .015."""
    for d in (2, 3, 4, 5):
        clean = D.FlatSheet(d=d, noisy=False)
        Xc = clean.sample(20_000, seed=4)
        assert float(np.abs(clean.perpendicular_coords(Xc)).max()) < 1e-14
        noisy = D.FlatSheet(d=d, noisy=True)
        sd = noisy.perpendicular_coords(noisy.sample(60_000, seed=4)).std(axis=0)
        print(f"d={d}  clean perpendicular coords ~ 0, thickened sd {np.round(sd, 5)} "
              f"(target {D.SHEET_NOISE_SD})")
        assert np.all(np.abs(sd - D.SHEET_NOISE_SD) < 1e-3)


def test_curved_sheet_noise_is_perpendicular_and_everything_stays_inside():
    """The thickening must be orthogonal to the analytic tangent, and the sheet itself
    must fit inside the cube with room to spare."""
    for d in (2, 3, 4, 5):
        cs = D.CurvedSheet(d=d, noisy=True)
        n = 20000
        p = cs.params(n, np.random.default_rng(6))
        X0 = cs.embed(p)
        Xn = cs.sample(n, seed=6)
        J = cs.tangent_jacobian(p)
        resid = float(np.abs(np.einsum("nik,ni->nk", J, Xn - X0)).max())
        print(f"d={d}  curved sheet max|x| clean {np.abs(X0).max():.4f}, "
              f"thickened {np.abs(Xn).max():.4f}, "
              f"max |tangent . displacement| = {resid:.2e}, redrawn {cs.n_redrawn}")
        assert float(np.abs(X0).max()) < 1.0
        assert resid < 1e-10
        assert np.all(np.abs(Xn) <= 1.0)


def test_fixed_distance_sets_sit_at_the_requested_distance():
    fs = D.FlatSheet(d=3, noisy=False)
    for r in D.SHEET_DISTANCES:
        X = fs.offset_points(4000, r, seed=8)
        assert np.allclose(fs.distance_to_sheet(X), r, atol=1e-12)
        assert np.all(np.abs(X) <= 1.0)
    print(f"flat-sheet offset sets exact at distances {D.SHEET_DISTANCES}")


def test_dense_region_sets_have_training_data_all_around_them():
    """The ``dense_region`` test set must sit strictly inside the densest part of the
    data, with training data on every side, for every family."""
    # even grid / uniform: shrunken cube, and the training data covers the whole cube.
    for tid in ("1.5", "2.1", "4.6"):
        t = get_task(tid)
        C = t.test_sets()["dense_region"]
        assert np.abs(C).max() <= D.DENSE_CUBE + 1e-12
        X = t.sample(20000, seed=3)
        assert np.abs(X).max() > D.DENSE_CUBE + 0.05
    # hotspots: within one standard deviation of the densest cluster, with most of the
    # training data outside that ball.
    for tid in ("1.12", "2.12", "3.14", "5.13"):
        t = get_task(tid)
        dens = t.density
        i = dens.densest_cluster()
        C = t.test_sets()["dense_region"]
        L = np.linalg.cholesky(dens.covs[i])
        sd_units = np.sqrt(np.sum(np.linalg.solve(L, (C - dens.means[i]).T) ** 2, axis=0))
        assert sd_units.max() <= D.DENSE_SD + 1e-12
        X = t.sample(20000, seed=3)
        sd_X = np.sqrt(np.sum(np.linalg.solve(L, (X - dens.means[i]).T) ** 2, axis=0))
        outside = float(np.mean(sd_X > D.DENSE_SD))
        assert outside > 0.5
        print(f"{tid}: dense region = {t.dense_region_description()} | furthest test point "
              f"{sd_units.max():.3f} widths out, {100 * outside:.0f}% of the training data "
              f"further out still")
    # d = 1: the densest cluster is the one at +0.45 with width 0.22, so [0.23, 0.67].
    C1 = get_task("1.12").test_sets()["dense_region"]
    assert C1.min() >= 0.45 - 0.22 - 1e-12 and C1.max() <= 0.45 + 0.22 + 1e-12
    # sheets: exactly on the sheet, parameters in the inner 80%.
    for tid in ("2.15", "3.16", "5.16"):
        t = get_task(tid)
        C = t.test_sets()["dense_region"]
        dist = t.distance_to_sheet(C[:400])
        assert dist.max() < 2e-2, (tid, dist.max())     # curved: grid-search upper bound
        y1 = C @ u(t.d, 1)
        y1_full = t.density.sheet_points(20000, seed=5) @ u(t.d, 1)
        assert np.abs(y1).max() < 0.95 * np.abs(y1_full).max()
        print(f"{tid}: dense region = {t.dense_region_description()} | furthest test point "
              f"{dist.max():.1e} from the sheet")


def test_test_set_names_and_sizes():
    s2 = get_task("2.1").test_sets()
    assert set(s2) == {"same_as_train", "uniform", "dense_region"}
    assert all(len(s2[k]) == 20000 for k in s2)
    s5 = get_task("5.16").test_sets()
    assert len(s5["same_as_train"]) == 40000 and len(s5["on_sheet"]) == 40000
    assert all(f"distance_{r:g}" in s5 for r in D.SHEET_DISTANCES)
    assert get_task("5.16").is_sheet and not get_task("2.1").is_sheet
    print("test sets: same_as_train / uniform / dense_region, 20000 points for d<=2 and "
          "40000 for d>=3, plus on_sheet and four fixed distances for the sheet tasks")


# ===========================================================================
# region masks
# ===========================================================================

def test_burst_and_step_masks():
    t = get_task("1.12")                       # burst centered on z = 0.45, width .18
    x = np.linspace(-1, 1, 20001)[:, None]
    m = t.packet_mask(x)
    lo, hi = float(x[m].min()), float(x[m].max())
    print(f"1.12 burst region = [{lo:.3f}, {hi:.3f}] (expected 0.45 +/- 2*0.18)")
    assert abs(lo - (0.45 - 0.36)) < 1e-3 and abs(hi - (0.45 + 0.36)) < 1e-3
    assert t.jump_mask(x) is None

    tj = get_task("1.15")                       # step at z_1 = 0.78
    mj = tj.jump_mask(x)
    assert abs(float(x[mj].min()) - 0.73) < 1e-3 and abs(float(x[mj].max()) - 0.83) < 1e-3
    assert tj.packet_mask(x) is None
    print("1.15 step band = [0.73, 0.83]")


# ===========================================================================
# the predicted center density
# ===========================================================================

def test_predicted_density_integrates_to_the_number_of_centers_and_is_flat_when_it_should_be():
    """The prediction integrates to the number of centers, and is flat both for a target
    whose slope is exactly constant and for one whose slope energy is constant on
    average, in both cases under uniform data."""
    v = np.array([1.0])
    X = np.random.default_rng(0).uniform(-1.0, 1.0, size=(200_000, 1))
    cases = {"constant slope (F = z)": lambda A: np.ones_like(A),
             "constant slope energy (F = sin(10 pi z))":
                 lambda A: np.pi * 10.0 * np.cos(np.pi * 10.0 * A)}
    for label, grad in cases.items():
        res = M.predicted_center_density(v, grad, X, n_centers=64.0, r=1)
        assert abs(res["integral"] - 64.0) < 1e-9
        inner = np.abs(res["t"]) < 0.80         # away from the ends of the data
        rho = res["density"][inner]
        ripple = float((rho.max() - rho.min()) / rho.mean())
        print(f"{label:44s} integral {res['integral']:.6f} (target 64), "
              f"interior ripple {100 * ripple:.2f}%")
        assert ripple < 0.05


def test_predicted_density_is_undefined_for_the_step_targets():
    for tid in ("1.8", "2.9"):
        t = get_task(tid)
        assert t.differentiable is False
        res = M.predicted_center_density(u(t.d, 1), t.grad_F, t.sample(1000, seed=0),
                                         n_centers=32.0, differentiable=t.differentiable)
        assert res["defined"] is False
        print(f"{tid} ({t.name}): prediction undefined -- {res['reason']}")


def test_predicted_density_follows_the_burst_and_the_data():
    """Moving the burst away from the clusters must move the prediction's peak."""
    peaks = {}
    for tid in ("1.12", "1.13"):
        t = get_task(tid)
        res = M.predicted_center_density(u(1, 1), t.grad_F, t.sample(120_000, seed=5),
                                         n_centers=64.0)
        peaks[tid] = float(res["t"][int(np.argmax(res["density"]))])
    print(f"peak of the predicted density: {peaks} "
          f"(burst at 0.45 in 1.12, at 0.85 in 1.13; the densest cluster is at 0.45)")
    assert abs(peaks["1.12"] - 0.45) < 0.10
    assert peaks["1.13"] > peaks["1.12"] + 0.10


def test_predicted_density_uses_the_gradient_of_the_scaled_target():
    """grad_F must be the gradient of F, not of the raw target."""
    t = get_task("2.5")
    X = np.random.default_rng(3).uniform(-0.9, 0.9, size=(200, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        fd = (t.F(X + e) - t.F(X - e)) / (2 * h)
        assert np.abs(fd - t.grad_F(X)[:, j]).max() < 1e-6 * max(1.0, np.abs(fd).max())
    print("grad_F matches finite differences of the scaled target F")


# ===========================================================================
# the reference models
# ===========================================================================

def test_even_geometry_shape():
    m = BL.EvenGeometry(d=3, budget=512)
    geo = m.geometry()
    assert geo["n_per_direction"] == 8 and geo["n_directions"] == 64
    assert len(geo["centers"]) == 512 and geo["directions"].shape == (512, 3)
    assert np.allclose(np.linalg.norm(geo["unique_directions"], axis=1), 1.0)
    for v in geo["unique_directions"]:
        sel = np.all(np.isclose(geo["directions"], v), axis=1)
        T = BL.EDGE_MARGIN * float(np.abs(v).sum())
        assert geo["centers"][sel].min() > -T and geo["centers"][sel].max() < T
        assert geo["centers"][sel].min() < -float(np.abs(v).sum())   # the margin is used
        assert np.allclose(geo["gammas"][sel], BL.LAMBDA / (2 * T / 8))
    print("even geometry d=3 B=512: 64 directions x 8 centers, width lambda/h per direction")


def test_reference_reaches_the_accuracy_floor_on_a_smooth_target():
    """Task 1.1 (three smooth bumps) at B=128 with 8B training points must reach the
    known one-dimensional fp64 floor. If this fails the grid, margin or width wiring is
    wrong -- do not loosen the bound."""
    t = get_task("1.1")
    X, y = t.train_set(8 * 128, seed=0)
    sets = t.test_sets()
    even = BL.EvenGeometry(d=1, budget=128).fit(X, y)
    rand = BL.RandomFeatures(d=1, budget=128, seed=0).fit(X, y)
    e = {k: M.error_metrics(even.predict(sets[k]), t.F(sets[k])) for k in
         ("same_as_train", "uniform", "dense_region")}
    r = {k: M.error_metrics(rand.predict(sets[k]), t.F(sets[k])) for k in
         ("same_as_train", "uniform")}
    print("1.1 B=128 even   relative L2: " + "  ".join(f"{k} {v['rel_l2']:.2e}"
                                                       for k, v in e.items()))
    print("1.1 B=128 random relative L2: " + "  ".join(f"{k} {v['rel_l2']:.2e}"
                                                       for k, v in r.items()))
    assert e["same_as_train"]["rel_l2"] < 1e-11 and e["uniform"]["rel_l2"] < 1e-11
    for k in ("same_as_train", "uniform"):
        assert r[k]["rel_l2"] > 1e3 * e[k]["rel_l2"]


def test_reference_in_two_dimensions():
    """Task 2.1 (three smooth bumps in 2-D) at B=1024: relative L2 below 1e-7, and
    random features three or more orders worse."""
    t = get_task("2.1")
    X, y = t.train_set(8 * 1024, seed=0)
    sets = t.test_sets()
    even = BL.EvenGeometry(d=2, budget=1024).fit(X, y)
    rand = BL.RandomFeatures(d=2, budget=1024, seed=0).fit(X, y)
    e = {k: M.error_metrics(even.predict(sets[k]), t.F(sets[k])) for k in
         ("same_as_train", "uniform", "dense_region")}
    r = {k: M.error_metrics(rand.predict(sets[k]), t.F(sets[k])) for k in
         ("same_as_train", "uniform")}
    print("2.1 B=1024 even   relative L2: " + "  ".join(f"{k} {v['rel_l2']:.2e}"
                                                        for k, v in e.items()))
    print("2.1 B=1024 random relative L2: " + "  ".join(f"{k} {v['rel_l2']:.2e}"
                                                        for k, v in r.items()))
    assert e["same_as_train"]["rel_l2"] < 1e-7 and e["uniform"]["rel_l2"] < 1e-7
    for k in ("same_as_train", "uniform"):
        assert r[k]["rel_l2"] > 1e3 * e[k]["rel_l2"]


def test_concentric_waves_reach_the_floor():
    """The concentric waves are ``cos(pi omega rho)``: cosine is even, so this is a smooth
    function of rho^2 with no cone point at its center. Task 1.2 (omega = 1) must therefore
    reach machine precision at B = 128 like every other smooth 1-D target, and 2.2 must be
    near it at B = 1024. (An earlier ``sin`` version had a cone point and stalled at 1e-3.)
    """
    t = get_task("1.2")
    X, y = t.train_set(8 * 128, seed=0)
    sets = t.test_sets()
    m = BL.EvenGeometry(d=1, budget=128).fit(X, y)
    e1 = M.error_metrics(m.predict(sets["uniform"]), t.F(sets["uniform"]))["rel_l2"]
    t2 = get_task("2.2")
    X, y = t2.train_set(8 * 1024, seed=0)
    sets = t2.test_sets()
    m2 = BL.EvenGeometry(d=2, budget=1024).fit(X, y)
    e2 = M.error_metrics(m2.predict(sets["uniform"]), t2.F(sets["uniform"]))["rel_l2"]
    print(f"1.2 (slow concentric waves) B=128 relative L2 on uniform points: {e1:.2e}; "
          f"2.2 at B=1024: {e2:.2e}")
    assert e1 < 1e-11
    assert e2 < 1e-9


# ===========================================================================
# the experiment driver
# ===========================================================================

def test_run_one_records_everything_the_suite_asks_for():
    h01 = _load_expH01("h01_run", "run.py")
    rows = h01.run_one(get_task("1.12"), budget=64, ratio=4.0, seed=0)
    assert len(rows) == 2
    assert {r["model"] for r in rows} == {"even_geometry", "random_features"}
    abs_rows = h01.run_one(get_task("1.12"), budget=64, ratio=4.0, seed=0, n_train=300)
    assert abs_rows[0]["n_train"] == 300                  # the absolute data-size knob
    rec = rows[0]
    for key in ("same_as_train", "uniform", "dense_region"):
        assert set(rec["errors"][key]) >= {"mse", "rel_l2", "max_abs"}
    assert "dense_region" in rec and "what_it_tests" in rec
    assert rec["by_data_density"] is not None and len(rec["by_data_density"]) == 10
    assert rec["packet"] is not None and "packet_inside" in rec["packet"]["same_as_train"]
    assert "angular" not in rec
    print(f"1.12 B=64: relative L2 on points like the training data "
          f"{rec['errors']['same_as_train']['rel_l2']:.2e}, inside the burst "
          f"{rec['packet']['same_as_train']['packet_inside']['rel_l2']:.2e}, outside "
          f"{rec['packet']['same_as_train']['packet_outside']['rel_l2']:.2e}, sparsest "
          f"bin {rec['by_data_density'][0]['rel_l2']:.2e}")

    sheet = h01.run_one(get_task("2.16"), budget=64, ratio=4.0, seed=0)[0]
    assert sheet["sheet"] is not None and "on_sheet" in sheet["sheet"]
    assert all(f"distance_{r:g}" in sheet["sheet"] for r in D.SHEET_DISTANCES)
    print("2.16 sheet errors: " + ", ".join(f"{k}={v['rel_l2']:.2e}"
                                            for k, v in sheet["sheet"].items()))
