"""Explicit readout coordinates for a genuinely flat tanh ridge network.

On the unit disk the Logan--Shepp polynomials U_n(v_nk dot x)/sqrt(pi)
form an orthonormal basis. A finite angular reproducing formula expresses
them on one common direction grid. Corrected QUILL then maps each known
univariate U_n profile to tanh readouts. No PDE solution enters construction,
and the quill backend evaluates only actual tanh features and derivatives.

This is a restricted linear readout parameterization w=E a, not independent
optimization of every neuron readout. There are no product gates.
"""
from __future__ import annotations

from math import ceil, sqrt, factorial
import time
import numpy as np
from numpy.polynomial import polynomial
from scipy.special import eval_gegenbauer

from quill_boundary import encode
from solver.general_features import _evaluate_encoding, _tanh_derivative_polynomial


def chebyshev_u(z, degree):
    """Second-kind Chebyshev values, including complex arguments."""
    z = np.asarray(z)
    out = np.empty(z.shape + (degree+1,), dtype=np.result_type(z, float))
    out[..., 0] = 1.
    if degree:
        out[..., 1] = 2*z
    for n in range(2, degree+1):
        out[..., n] = 2*z*out[..., n-1]-out[..., n-2]
    return out


def _u_coefficients(degree):
    coefficients = [np.array([1.])]
    if degree:
        coefficients.append(np.array([0., 2.]))
    for n in range(2, degree+1):
        coefficients.append(polynomial.polysub(
            polynomial.polymul([0., 2.], coefficients[-1]), coefficients[-2]))
    return coefficients


class RidgePINNFeatures:
    """Known basis functions encoded inside b+sum w*tanh(gamma*(omega.x-c)).

    Domain is the closed physical disk of given radius and midpoint. The
    p(p+3)/2+1 readout coordinates correspond to degree/index pairs (n,k).
    M>=p+1 suffices for exact angular polynomial reproduction. Smaller M is
    allowed ONLY as an explicitly underresolved experimental control.
    Every quill residual jet is computed from the actual tanh bank.
    """
    def __init__(self, degree, directions=None, centers=129, lam=.2,
                 radius=1., midpoint=(0., 0.), backend='quill',
                 encoding_tolerance=1e-9, allow_angular_underresolution=False):
        began = time.perf_counter()
        if isinstance(degree, bool) or int(degree) != degree or degree < 0:
            raise ValueError('degree must be a nonnegative integer')
        if directions is None:
            directions = int(degree)+1
        if int(directions) != directions or directions < 1:
            raise ValueError('directions must be a positive integer')
        if int(centers) != centers or centers < 3:
            raise ValueError('centers must count at least 3 interior centers')
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('radius must be positive')
        if backend not in ('quill', 'polynomial'):
            raise ValueError('backend must be quill or polynomial')
        if directions <= degree and not allow_angular_underresolution:
            raise ValueError('M>=degree+1 is required; opt in for negative control')
        self.degree, self.direction_count = int(degree), int(directions)
        self.dimension, self.radius, self.backend = 2, float(radius), backend
        self.midpoint = np.asarray(midpoint, dtype=float)
        if self.midpoint.shape != (2,) or not np.all(np.isfinite(self.midpoint)):
            raise ValueError('midpoint must be a finite 2-vector')
        self.bounds = np.column_stack([self.midpoint-self.radius,
                                       self.midpoint+self.radius])
        self.mode_pairs = [(n, k) for n in range(self.degree+1) for k in range(n+1)]
        # Do not call these multiindices: generic parity preconditioning would
        # incorrectly treat the angular index k as a coordinate exponent.
        self.size = len(self.mode_pairs)
        self.angles = np.pi*np.arange(self.direction_count)/self.direction_count
        self.directions = np.column_stack([np.cos(self.angles), np.sin(self.angles)])
        self.angular_maps = []
        for n in range(self.degree+1):
            reference = np.pi*np.arange(n+1)/(n+1)
            values = chebyshev_u(np.cos(self.angles[:, None]-reference[None, :]), n)[..., n]
            self.angular_maps.append(values/(self.direction_count*np.sqrt(np.pi)))
        self.encoding = None
        self.exact_coefficients = _u_coefficients(self.degree)
        checks = []
        if backend == 'quill':
            self.encoding = encode(lambda z: chebyshev_u(z, self.degree),
                                   int(centers)-1, lam=lam, halo=ceil(sqrt(centers)))
            self.encoding.evaluation_mode = 'anchored'
            self.encoding.anchor_x = -1.
            self.encoding.anchor_value = (-1.)**np.arange(self.degree+1)*(np.arange(self.degree+1)+1)
            z = np.unique(np.r_[np.cos(np.pi*np.arange(258)/257),
                                2*np.mod(np.arange(1, 131)*np.sqrt(2), 1)-1])
            for order in range(3):
                truth = self._exact_profiles(z, order)
                error = np.max(np.abs(_evaluate_encoding(self.encoding,z,order)-truth),axis=0)
                scale = np.maximum(1., np.max(np.abs(truth),axis=0))
                checks.append(dict(order=order, maximum_absolute=float(error.max()),
                                   maximum_scaled=float(np.max(error/scale))))
            if max(r['maximum_scaled'] for r in checks) > encoding_tolerance:
                raise ValueError(f'Known-profile derivative encoding failed {encoding_tolerance:g}: {checks}')
        self.metrics = dict(architecture='one hidden layer tanh ridge MLP',
            backend=backend, product_gates=False, degree=self.degree,
            coefficient_count=self.size, directions=self.direction_count,
            interior_centers_per_direction=int(centers), halo_per_side=ceil(sqrt(centers)),
            tanh_count=0 if self.encoding is None else self.direction_count*len(self.encoding.centers),
            lam=float(lam), radius=self.radius, readout_parameterization='explicit E @ a',
            construction_uses_target_data=False,
            angular_reproduction_guaranteed=directions>degree,
            normalized_profile_checks=checks, setup_seconds=time.perf_counter()-began)

    def _exact_profiles(self, z, order):
        # d^r C_n^(1)=2^r r! C_(n-r)^(1+r). Evaluating expanded
        # monomials instead loses many digits for moderate n near |z|=1.
        return np.column_stack([np.zeros_like(z) if n<order else
            (2**order*factorial(order))*eval_gegenbauer(n-order,1+order,z)
            for n in range(self.degree+1)])

    def _points_order(self, points, derivative):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError('points must be finite Q-by-2 array')
        z = (points-self.midpoint)/self.radius
        if np.any(np.linalg.norm(z,axis=1) > 1+1e-12):
            raise ValueError('The encoded domain is the closed disk; point outside')
        if derivative is None:
            derivative = (0,0)
        raw = np.asarray(derivative)
        if raw.shape != (2,) or not np.issubdtype(raw.dtype,np.integer) or np.any(raw<0):
            raise ValueError('derivative must be a nonnegative integer pair')
        return z, tuple(int(v) for v in raw)

    def evaluate(self, points, derivative=None):
        z, derivative = self._points_order(points, derivative)
        order = sum(derivative)
        projection = z@self.directions.T
        if self.encoding is None:
            profiles = self._exact_profiles(projection.ravel(),order)
        else:
            profiles = _evaluate_encoding(self.encoding,projection.ravel(),order)
        profiles = profiles.reshape(len(z),self.direction_count,self.degree+1)
        chain = np.prod(self.directions**np.array(derivative)[None,:],axis=1)/self.radius**order
        result = np.empty((len(z),self.size))
        start = 0
        for n, angular in enumerate(self.angular_maps):
            result[:,start:start+n+1] = (profiles[:,:,n]*chain)@angular
            start += n+1
        # The constant is an exact output bias, with no hidden neurons.
        result[:,0] = 1/np.sqrt(np.pi) if order == 0 else 0.
        return result

    def compile(self, coefficients):
        """Return standard dense-layer arrays; no products or polynomial gates."""
        if self.encoding is None:
            raise ValueError('Only the quill backend is a tanh MLP')
        coefficients = np.asarray(coefficients,float)
        if coefficients.ndim == 1:
            coefficients = coefficients[:,None]
        if coefficients.shape[0] != self.size:
            raise ValueError('Expected one coefficient row per constructed feature')
        q = coefficients.shape[1]
        profile_coeffs = np.zeros((self.direction_count,self.degree+1,q))
        start = 1
        for n in range(1,self.degree+1):
            profile_coeffs[:,n,:] = self.angular_maps[n]@coefficients[start:start+n+1]
            start += n+1
        weights = np.einsum('jn,mnf->mjf',self.encoding.weights,profile_coeffs)
        bias = coefficients[0]/np.sqrt(np.pi)+np.einsum('n,mnf->f',self.encoding.bias,profile_coeffs)
        first_weights = np.repeat(self.encoding.gamma*self.directions/self.radius,
                                  len(self.encoding.centers),axis=0)
        first_bias = (-self.encoding.gamma*self.encoding.centers[None,:]
                      -self.encoding.gamma*(self.directions@self.midpoint/self.radius)[:,None])
        return dict(first_weights=first_weights, first_bias=first_bias.ravel(),
                    output_weights=weights.reshape(-1,q), output_bias=bias,
                    profile_coefficients=profile_coeffs)

    def evaluate_flat(self, points, coefficients, derivative=None, block=256):
        """Audit independently compiled standard tanh MLP and exact derivatives."""
        _, derivative = self._points_order(points,derivative)
        order = sum(derivative)
        net = self.compile(coefficients)
        out = np.empty((len(points),net['output_weights'].shape[1]))
        for start in range(0,len(points),block):
            z = np.asarray(points)[start:start+block]@net['first_weights'].T+net['first_bias']
            t = np.tanh(z)
            if order == 0:
                values = t
            else:
                e = np.exp(-2*np.abs(z))
                values = (4*e/(1+e)**2)*polynomial.polyval(t,_tanh_derivative_polynomial(order))
                values *= np.prod(net['first_weights']**np.array(derivative)[None,:],axis=1)
            out[start:start+block] = values@net['output_weights']
            if order == 0:
                out[start:start+block] += net['output_bias']
        return out
