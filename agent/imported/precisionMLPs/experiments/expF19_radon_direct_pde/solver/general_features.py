"""PDE-independent, explicitly constructed multivariate neural features.

Every feature is a product of one-dimensional Legendre polynomials, with total
degree bounded by ``degree``.  The ``quill`` backend evaluates actual corrected
tanh encodings of those polynomials, including their analytic derivatives;
``polynomial`` is a matched exact-basis control. No target values are fitted.

Products are explicit product gates. The tanh count therefore does not measure
the number of product operations or independent coefficients.
"""
from __future__ import annotations

from functools import lru_cache
from math import ceil, sqrt
import time

import numpy as np
from numpy.polynomial import legendre, polynomial
from scipy.special import expit

from quill_boundary import encode


def _compositions(total, dimension):
    if dimension == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for tail in _compositions(total - first, dimension - 1):
                yield (first,) + tail


@lru_cache(maxsize=32)
def _tanh_derivative_polynomial(order):
    """For order >=1 return q with tanh^(order)(z)=sech²(z) q(tanh(z))."""
    if order < 1:
        raise ValueError("Derivative polynomial requires positive order")
    q = np.ones(1)
    for _ in range(order - 1):
        q = polynomial.polysub(
            polynomial.polymul([1., 0., -1.], polynomial.polyder(q)),
            polynomial.polymul([0., 2.], q))
    q.setflags(write=False)
    return q


def _evaluate_encoding(encoding, x, order):
    """Evaluate tanh derivatives without subtracting nearly equal numbers."""
    x = np.asarray(x, dtype=float).reshape(-1)
    result = np.empty((len(x), encoding.weights.shape[1]))
    # Bound the largest intermediate rather than caching arrays of query points.
    block = max(1, min(2048, 1_000_000 // len(encoding.centers)))
    for start in range(0, len(x), block):
        z = encoding.gamma * (x[start:start + block, None] - encoding.centers)
        t = np.tanh(z)
        if order == 0:
            if getattr(encoding, 'evaluation_mode', 'standard') == 'anchored':
                # Evaluate the same prescribed anchored tanh expansion. Halo
                # constants cancel analytically before arithmetic, rather
                # than adding large saturated terms and a cancelling bias.
                z0 = encoding.gamma*(encoding.anchor_x-encoding.centers)
                diff = t-np.tanh(z0)
                both_positive = (z >= 0) & (z0 >= 0)
                both_negative = (z <= 0) & (z0 <= 0)
                positive = 2*(expit(-2*z0)-expit(-2*z))
                negative = 2*(expit(2*z)-expit(2*z0))
                diff = np.where(both_positive,positive,
                                np.where(both_negative,negative,diff))
                result[start:start + block] = diff @ encoding.weights + encoding.anchor_value
            else:
                result[start:start + block] = t @ encoding.weights + encoding.bias
        else:
            e = np.exp(-2 * np.abs(z))
            sech2 = 4 * e / (1 + e)**2
            phi = encoding.gamma**order * sech2 * polynomial.polyval(
                t, _tanh_derivative_polynomial(order))
            result[start:start + block] = phi @ encoding.weights
    return result


def _exact_legendre(x, degree, order):
    if order == 0:
        return legendre.legvander(x, degree)
    coefficients = legendre.legder(np.eye(degree + 1), m=order, axis=0)
    return legendre.legval(x, coefficients).T


def _validate_bank(bank, degree, max_derivative):
    # Independent endpoint-concentrated and nonuniform interior checks. These
    # are not the contour quadrature or the bank's construction centers.
    x = np.unique(np.r_[np.cos(np.pi*np.arange(258)/257),
                        2*np.mod(np.arange(1, 132)*np.sqrt(2), 1)-1])
    details = []
    for order in range(max_derivative + 1):
        exact = _exact_legendre(x, degree, order)
        error = np.max(np.abs(_evaluate_encoding(bank, x, order) - exact), axis=0)
        scale = np.maximum(1., np.max(np.abs(exact), axis=0))
        details.append(dict(order=order, max_absolute_error=float(error.max()),
                            max_scaled_error=float(np.max(error/scale)),
                            per_mode_absolute_error=error.tolist()))
    return details


@lru_cache(maxsize=24)
def _construct_bank(degree, centers, lam, max_derivative,
                    tolerance=1e-8,evaluation='standard'):
    if centers is None:
        first = max(33, 4*degree + 1)
        center_choices = [first, 2*(first-1)+1, 4*(first-1)+1,
                          8*(first-1)+1, 16*(first-1)+1]
        if tolerance < 1e-8:
            center_choices = sorted(set(center_choices+[65,129,257,513,1025,2049]))
    else:
        center_choices = [centers]
    lambdas = [lam] if lam is not None else ([.12,.15,.18,.20,.22,.25,.30]
                                            if tolerance < 1e-8 else [.15,.20,.25,.30])
    attempts = []
    # Requested derivatives are checked column by column, so a large
    # high-degree derivative cannot hide inaccurate low-degree columns.
    selected = None
    for n in center_choices:
        candidates = []
        for bandwidth in lambdas:
            bank = encode(lambda z: legendre.legvander(z, degree), n - 1,
                          lam=bandwidth, halo=ceil(sqrt(n)))
            bank.evaluation_mode = evaluation
            bank.anchor_x = -1.
            bank.anchor_value = (-1.)**np.arange(degree+1)
            checks = _validate_bank(bank, degree, max_derivative)
            score = max(row['max_scaled_error'] for row in checks)
            attempts.append(dict(interior_centers=n, lam=float(bandwidth),
                                 maximum_scaled_error=score))
            candidates.append((score, bank, checks))
        score, bank, checks = min(candidates, key=lambda item: item[0])
        if score <= tolerance:
            selected = bank, checks
            break
    if selected is None:
        best = min(attempts, key=lambda row: row['maximum_scaled_error'])
        raise ValueError("QUILL basis derivative encoding did not meet the "
                         f"independent {tolerance:g} tolerance; best={best}")
    bank, checks = selected
    # Sharing the normalized bank across coordinates/constructions is safe:
    # callers cannot accidentally mutate its coefficient arrays.
    for value in vars(bank).values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return bank, checks, tuple(attempts), tolerance


class ConstructedFeatures:
    """Total-degree polynomial product features on a physical bounding box.

    ``bounds`` is a finite array with shape (dimension, 2), with lower and upper
    coordinates. ``evaluate`` takes a Q-by-dimension point array and returns a
    Q-by-size matrix. A derivative is a dimension-length tuple of nonnegative
    integers, in PHYSICAL coordinates. The bounding box defines the coordinate
    scaling, not a restriction to rectangular PDE domains.

    The encoding sweep sees only known basis functions, never the PDE/target.
    The independent check covers orders zero through ``max_derivative`` on the
    normalized interval. This can be inferred from the equation's differential
    order. Higher derivatives are evaluable but not certified by that check.
    ``encoding_tolerance`` bounds the sampled per-mode normalized error, not
    the physical PDE error. Physical coordinate scaling and coefficient
    magnitudes can amplify it. ``evaluation='anchored'`` evaluates the same
    constructed tanh differences before summation to avoid cancellation of
    saturated halo constants; derivatives remain actual tanh derivatives.
    """
    def __init__(self, bounds, degree, backend='quill', centers=None, lam=None,
                 max_derivative=2, encoding_tolerance=1e-8,
                 evaluation='standard'):
        started = time.perf_counter()
        self.bounds = np.asarray(bounds, dtype=float).copy()
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2 or
                len(self.bounds) == 0 or not np.all(np.isfinite(self.bounds)) or
                np.any(self.bounds[:, 1] <= self.bounds[:, 0])):
            raise ValueError("bounds must be finite increasing pairs with shape (d,2)")
        if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        if backend not in ('quill', 'polynomial'):
            raise ValueError("backend must be 'quill' or 'polynomial'")
        if centers is not None and (isinstance(centers, (bool, np.bool_)) or
                not isinstance(centers, (int, np.integer)) or centers < 3):
            raise ValueError("centers must count at least three interior centers")
        if lam is not None and (not np.isfinite(lam) or lam <= 0):
            raise ValueError("lam must be positive and finite")
        if (isinstance(max_derivative, (bool, np.bool_)) or
                not isinstance(max_derivative, (int, np.integer)) or max_derivative < 0):
            raise ValueError("max_derivative must be a nonnegative integer")
        if (isinstance(encoding_tolerance,(bool,np.bool_)) or
                not np.isfinite(encoding_tolerance) or encoding_tolerance <= 0):
            raise ValueError("encoding_tolerance must be positive and finite")
        if evaluation not in ('standard','anchored'):
            raise ValueError("evaluation must be 'standard' or 'anchored'")
        self.dimension = len(self.bounds)
        self.degree = int(degree)
        self.backend = backend
        self.multiindices = np.array([index for total in range(degree + 1)
                                      for index in _compositions(total, self.dimension)], dtype=int)
        self.size = len(self.multiindices)
        self.scale = 2 / (self.bounds[:, 1] - self.bounds[:, 0])
        self.midpoint = .5 * (self.bounds[:, 1] + self.bounds[:, 0])
        self.encoding = None
        self.metrics = dict(backend=backend, dimension=self.dimension, degree=self.degree,
                            coefficient_count=self.size, tanh_count=0,
                            basis='total-degree Legendre products',
                            max_derivative_calibrated=int(max_derivative),
                            evaluation=evaluation,
                            product_gates=True, construction_uses_target_data=False)
        if backend == 'quill' and degree > 0:
            self.encoding, checks, attempts, tolerance = _construct_bank(
                degree, centers, lam, int(max_derivative),
                float(encoding_tolerance),evaluation)
            self.metrics.update(
                tanh_count=self.dimension*len(self.encoding.centers),
                interior_centers_per_axis=self.encoding.n_cells + 1,
                halo_per_side=self.encoding.halo_per_side,
                lam=self.encoding.lam, normalized_gamma=self.encoding.gamma,
                physical_gamma=(self.scale*self.encoding.gamma).tolist(),
                normalized_derivative_checks=checks,
                derivative_check_tolerance=tolerance,
                encoding_value_error=checks[0]['max_absolute_error'],
                encoding_derivative_errors=[r['max_absolute_error'] for r in checks[1:]],
                encoding_sweep=list(attempts),
                encoding_check_scope=f'normalized 1D basis derivatives through order {max_derivative}')
        else:
            self.metrics.update(encoding_value_error=0., encoding_derivative_errors=[0.]*max_derivative)
        self.metrics['setup_seconds'] = time.perf_counter() - started

    def evaluate(self, points, derivative=None):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[:, None] if self.dimension == 1 else points[None, :]
        if points.ndim != 2 or points.shape[1] != self.dimension or not np.all(np.isfinite(points)):
            raise ValueError("points must be a finite array with shape (Q, dimension)")
        if derivative is None:
            derivative = (0,)*self.dimension
        raw_order = np.asarray(derivative)
        if (raw_order.shape != (self.dimension,) or
                not np.issubdtype(raw_order.dtype, np.integer) or np.any(raw_order < 0)):
            raise ValueError("derivative must be a dimension-length nonnegative integer tuple")
        derivative = tuple(int(value) for value in raw_order)
        z = (points - self.midpoint)*self.scale
        result = np.ones((len(points), self.size))
        for axis, order in enumerate(derivative):
            if self.encoding is None:
                values = _exact_legendre(z[:, axis], self.degree, order)
            else:
                values = _evaluate_encoding(self.encoding, z[:, axis], order)
            result *= values[:, self.multiindices[:, axis]] * self.scale[axis]**order
        return result
