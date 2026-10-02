"""Target-specific boundary-corrected sine coefficients from theorem_for_sam A--B.

This is a detached reference, not an initializer or a fit. The older Toeplitz QI
implementation uses a different halo construction and is not substituted here.
"""

import math

import mpmath as mp
import numpy as np


def sine_coefficients(n=512, dps=60, quadrature_degree=8):
    with mp.workdps(dps):
        h, lam, delta = mp.mpf(2)/n, mp.mpf(1)/4, mp.mpf(1)/4
        gamma = lam/h
        radius = math.ceil(math.sqrt(n))
        m = (radius + 1)//2
        d = mp.pi/(2*gamma)
        sigma = 2*mp.floor(delta/(4*d))*d
        if d > delta/8 or (radius + mp.mpf('.5'))*h > delta:
            raise ValueError("Reference width does not satisfy the note's local representation conditions")
        centers = [-1+j*h for j in range(-radius, n+radius+1)]
        left = centers[0]-h/2
        right = centers[-1]+h/2
        f = lambda z: mp.sqrt(2)*mp.sin(2*mp.pi*z)
        density = lambda z: (f(z+1j*d)-f(z-1j*d))/(2j*d)
        base = [mp.re(h*density(x)/2) for x in centers]
        ts = [mp.exp(2*lam*(i-mp.mpf('.5'))) for i in range(1, m+1)]
        denominator = [mp.mpf(1)]
        for t in ts:
            denominator = [denominator[0]] + [denominator[k]+t*denominator[k-1]
                           for k in range(1, len(denominator))] + [t*denominator[-1]]
        corrections = []
        for endpoint, sign in [(left, 1), (right, -1)]:
            moments = [mp.mpf(0)]
            for ell in range(1, m+1):
                mu = mp.quad(lambda y: mp.re(f(endpoint+1j*y)*mp.exp(sign*2j*ell*gamma*y)),
                             [0, d], maxdegree=quadrature_degree)/d
                # Conjugate symmetry reduces B.4--B.5 to a real integrand.
                nu = 2*(-1)**ell * mp.quad(
                    lambda y: mp.im(density(endpoint+1j*y)*mp.exp(sign*2j*ell*gamma*y))
                    /(1+mp.exp(2*mp.pi*y/h)), [0, sigma], maxdegree=quadrature_degree)
                moments.append(mu-nu)
            numerator = [sum(denominator[j]*moments[k-j] for j in range(k+1)) for k in range(m+1)]
            corrections.append([mp.fsum(coefficient*(-1/t)**k for k, coefficient in enumerate(numerator))
                / mp.fprod(1-other/t for j, other in enumerate(ts) if j != i)
                for i, t in enumerate(ts)])
        weights = base.copy()
        for i in range(m):
            weights[i] += corrections[0][i]/2
            weights[-i-1] -= corrections[1][i]/2
        bias = f(-1)-sum(w*mp.tanh(gamma*(-1-x)) for w, x in zip(weights, centers))
        baseline_bias = f(-1)-sum(w*mp.tanh(gamma*(-1-x)) for w, x in zip(base, centers))
        return {"centers": np.array(centers, float), "c": np.array([bias]+weights, float),
                "baseline_c": np.array([baseline_bias]+base, float),
                "gamma": np.full(len(centers), float(gamma)), "h": float(h),
                "radius": radius, "dps": dps, "quadrature_degree": quadrature_degree,
                "correction_coefficients": np.array(corrections, float)}
