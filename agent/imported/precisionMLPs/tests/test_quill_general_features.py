"""Checks of PDE-independent constructed features and physical derivatives."""
import sys
from pathlib import Path
from math import comb

import numpy as np
import pytest
from scipy.special import eval_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'experiments/expF19_radon_direct_pde'))
from solver.general_features import ConstructedFeatures


def _column(features, index):
    return np.flatnonzero(np.all(features.multiindices == index, axis=1))[0]


@pytest.mark.parametrize('backend', ['quill', 'polynomial'])
def test_constant_and_physical_derivatives(backend):
    features = ConstructedFeatures([[2., 6.], [-3., -2.]], 4,
                                   backend=backend, max_derivative=4)
    points = np.array([[2.1, -2.8], [3.6, -2.5], [5.1, -2.1]])
    # P2((x-4)/2)*P1(2y+5); independent elementary expressions.
    z = (points[:, 0]-4)/2
    w = 2*points[:, 1]+5
    col = _column(features, (2, 1))
    np.testing.assert_allclose(features.evaluate(points)[:, col],
                               .5*(3*z*z-1)*w, atol=1e-10)
    np.testing.assert_allclose(features.evaluate(points, (1, 1))[:, col],
                               3*z, atol=1e-9)
    np.testing.assert_allclose(features.evaluate(points, (2, 1))[:, col],
                               np.full(len(z), 1.5), atol=1e-8)
    np.testing.assert_allclose(features.evaluate(points)[:, 0], 1., atol=0.)
    np.testing.assert_allclose(features.evaluate(points, (0, 4))[:, 0], 0., atol=0.)
    # P4 fourth derivative is 105 in normalized coordinates, scaled by 1/2^4.
    fourth = features.evaluate(points, (4, 0))[:, _column(features, (4, 0))]
    np.testing.assert_allclose(fourth, 105/16, atol=1e-7)


@pytest.mark.parametrize('dimension,degree', [(1, 16), (2, 8), (3, 5), (4, 3)])
def test_independent_multivariate_values(dimension, degree):
    bounds = np.column_stack([np.arange(dimension)*.3-1,
                              np.arange(dimension)*.3+1])
    features = ConstructedFeatures(bounds, degree)
    assert features.size == comb(dimension+degree, degree)
    z = np.random.default_rng(197).uniform(-1, 1, (43, dimension))
    points = z+(bounds[:, 0]+bounds[:, 1])/2
    expected = np.ones((len(z), features.size))
    for axis in range(dimension):
        expected *= eval_legendre(features.multiindices[:, axis][None, :],
                                  z[:, axis, None])
    np.testing.assert_allclose(features.evaluate(points), expected, atol=2e-8)
    assert features.metrics['construction_uses_target_data'] is False
    assert features.metrics['tanh_count'] == dimension*len(features.encoding.centers)
    assert features.encoding.halo_per_side == int(np.ceil(np.sqrt(
        features.metrics['interior_centers_per_axis'])))


def test_analytic_derivatives_match_finite_differences_of_actual_network():
    features = ConstructedFeatures([[-2., 3.]], 8, max_derivative=4)
    x = np.linspace(-1.8, 2.8, 31)[:, None]
    step = 2e-5
    for derivative in range(1, 5):
        fd = (features.evaluate(x+step, (derivative-1,))-
              features.evaluate(x-step, (derivative-1,)))/(2*step)
        actual = features.evaluate(x, (derivative,))
        scale = np.maximum(1., np.max(np.abs(actual), axis=0))
        assert np.max(np.abs(fd-actual)/scale) < 2e-7
    # The evaluations really use the tanh bank and do not silently substitute
    # exact polynomial values after calibration.
    z = ((x-features.midpoint)*features.scale)[:, 0]
    raw = features.encoding.evaluate(z)
    np.testing.assert_allclose(features.evaluate(x), raw, atol=1e-14)


def test_derivative_calibration_and_rejection_of_underresolved_explicit_bank():
    features = ConstructedFeatures([[-1., 1.]], 5, max_derivative=4)
    assert len(features.metrics['encoding_derivative_errors']) == 4
    assert max(row['max_scaled_error'] for row in
               features.metrics['normalized_derivative_checks']) <= 1e-8
    with pytest.raises(ValueError, match='did not meet'):
        ConstructedFeatures([[-1., 1.]], 5, centers=9, lam=.2, max_derivative=4)


def test_zero_degree_input_checks_and_empty_queries():
    features = ConstructedFeatures([[0., 1.], [0., 2.]], 0, max_derivative=4)
    assert features.metrics['tanh_count'] == 0
    np.testing.assert_equal(features.evaluate([[.3, 1.]]), [[1.]])
    np.testing.assert_equal(features.evaluate([[.3, 1.]], (5, 0)), [[0.]])
    assert features.evaluate(np.empty((0, 2))).shape == (0, 1)
    with pytest.raises(ValueError, match='bounds'):
        ConstructedFeatures([[1., 0.]], 3)
    with pytest.raises(ValueError, match='degree'):
        ConstructedFeatures([[0., 1.]], 1.5)
    with pytest.raises(ValueError, match='max_derivative'):
        ConstructedFeatures([[0., 1.]], 2, max_derivative=-1)
    with pytest.raises(ValueError, match='derivative'):
        features.evaluate([[.3, 1.]], (1.0, 0.0))
    with pytest.raises(ValueError, match='points'):
        features.evaluate([[.3, np.nan]])


def test_precision_controls_and_anchored_actual_tanh_evaluation():
    # High-degree halo readouts are large; subtracting their saturated
    # constants before summing materially affects accurate evaluation.
    features = ConstructedFeatures([[-1.,1.]], 16, centers=257, lam=.2,
                                   encoding_tolerance=1e-13,
                                   evaluation='anchored')
    assert features.metrics['derivative_check_tolerance'] == 1e-13
    x = np.array([-1.,-.991,-.421,.037,.713,.999,1.])
    expected = eval_legendre(np.arange(17)[None,:], x[:,None])
    np.testing.assert_allclose(features.evaluate(x[:,None]),expected,
                               rtol=0,atol=4e-14)
    # Verify against high-precision evaluation of the ACTUAL encoded network,
    # not just its intended polynomial. Coefficients remain their stored FP64
    # values, so this also checks the precise anchored bias convention.
    import mpmath as mp
    with mp.workdps(45):
        b = features.encoding
        for point in [-.421,.713]:
            reference = []
            for j in [3,8,16]:
                value = mp.mpf((-1)**j)+mp.fsum(
                    mp.mpf(w)*(mp.tanh(mp.mpf(b.gamma)*(mp.mpf(point)-mp.mpf(c)))-
                               mp.tanh(mp.mpf(b.gamma)*(-1-mp.mpf(c))))
                    for c,w in zip(b.centers,b.weights[:,j]))
                reference.append(float(value))
            actual = features.evaluate([[point]])[0,[3,8,16]]
            np.testing.assert_allclose(actual,reference,rtol=0,atol=4e-14)
    with pytest.raises(ValueError,match='did not meet'):
        ConstructedFeatures([[-1.,1.]],8,centers=129,lam=.2,
                            encoding_tolerance=1e-18,evaluation='anchored')
    for tolerance in [0.,-1.,np.inf,np.nan,True]:
        with pytest.raises(ValueError,match='encoding_tolerance'):
            ConstructedFeatures([[-1.,1.]],2,encoding_tolerance=tolerance)
    with pytest.raises(ValueError,match='evaluation'):
        ConstructedFeatures([[-1.,1.]],2,evaluation='polynomial_shortcut')
