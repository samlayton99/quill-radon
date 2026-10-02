"""Independent analytical checks for quarantined encoding diagnostics."""
from pathlib import Path
import sys
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_hard_encoding_sweep import (analytical_target_coefficients, halo_growth,
    RecurrenceAffineDiskProfileOperator, ideal_jets, audit_encoding, ORDERS)


@pytest.mark.parametrize('frequency', [5.3, 2*np.pi, 8*np.pi])
def test_bessel_coefficients_match_independent_normalized_disk_quadrature(frequency):
    features = RecurrenceAffineDiskProfileOperator([[0, 1], [0, 1]], 32, centers=33)
    coefficients = analytical_target_coefficients(features, frequency)
    nodes, weights = leggauss(72)
    radius = np.sqrt((nodes+1)/2)
    angle = 2*np.pi*np.arange(192)/192
    reference = (radius[:, None, None] * np.stack((np.cos(angle), np.sin(angle)), axis=1)[None]).reshape(-1, 2)
    points = reference/features.scale + features.midpoint
    quadrature = np.broadcast_to((weights/2/192)[:, None], (72, 192)).ravel()
    target = np.sin(frequency*points[:, 0])*np.sin(frequency*points[:, 1])
    indices = np.unique(np.r_[0, 1, 2, np.flatnonzero(abs(coefficients)>1e-6)[::7], features.size-1])
    basis = features.selected_ideal_columns(points, indices, ((0, 0),))[(0, 0)]
    expected = basis.T @ (quadrature*target)
    np.testing.assert_allclose(coefficients[indices], expected, atol=8e-15, rtol=8e-13)


def test_halo_chebyshev_identity_and_fixed_degree_limit():
    for degree in (16, 32, 64):
        for centers in (257, 513, 1025):
            result = halo_growth(degree, centers)
            np.testing.assert_allclose(result['chebyshev_u_at_outer_center'],
                                       result['independently_evaluated_u'], rtol=2e-14)
    result = halo_growth(32, 10**12)
    assert abs(result['chebyshev_u_at_outer_center']/33-1) < .001
    assert abs(result['acosh_outer_center']/result['leading_acosh']-1) < 1e-5


def test_diagnostic_does_not_fit_or_change_given_coefficients(monkeypatch):
    import scipy.linalg
    import solver.route2 as route2
    def forbidden(*args, **kwargs):
        pytest.fail('A fixed-coefficient audit must not fit or solve a PDE')
    monkeypatch.setattr(np.linalg, 'lstsq', forbidden)
    monkeypatch.setattr(scipy.linalg, 'lstsq', forbidden)
    monkeypatch.setattr(route2, 'solve_route2', forbidden)
    features = RecurrenceAffineDiskProfileOperator([[0, 1], [0, 1]], 4, centers=33)
    coefficients = np.arange(features.size)*.01
    before = coefficients.copy()
    points = np.random.default_rng(971).uniform(0, 1, (7, 2))
    ideal = ideal_jets(features, coefficients, points, ORDERS)
    result = audit_encoding(features, coefficients, points, points, ideal[(0, 0)], ideal, 8*np.pi)
    np.testing.assert_array_equal(coefficients, before)
    assert np.isfinite(result['ordinary_pde_rms'])
    assert result['readout_l1'] > 0
