"""Putting every target on a common scale.

Every target is centered and scaled once,

    F(x) = (Ftilde(x) - mean_uniform) / sd_uniform,

using a fixed set of points spread uniformly over the whole cube -- the same set for
every task of a given ``d``, whatever that task's own data geometry is, sheet tasks
included. That is what stops a clustered data geometry from silently changing the scale
of the loss, and it is why two tasks that share a target function agree bit for bit.

The reference set is a scrambled Sobol sequence with a frozen seed, ``2^20`` points per
dimension, cached in process (two dimensions at a time; tasks are built in dimension
order, so the cache never thrashes). Two deliberate choices:

* Sobol rather than independent uniform draws: independent points estimate a
  unit-variance mean only to ``sd/sqrt(n)``, which at any affordable ``n`` is larger
  than the tolerance the scaling has to hold.
* ``2^20`` rather than ``2^16``: the fast oscillatory targets in ``d = 5`` run through
  many periods across the cube, and at ``2^16`` their mean swings by several times the
  claimed tolerance between independent scrambles. At ``2^20`` every task agrees with an
  independent ``2^21`` scramble well inside the tolerance the tests assert, and scaling
  all 80 tasks costs a few seconds.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.stats import qmc

__all__ = ["REFERENCE_N", "REFERENCE_SEED", "reference_points", "reference_moments",
           "normalize_callable"]

REFERENCE_N = 1 << 20
REFERENCE_SEED = 20260828


@functools.lru_cache(maxsize=2)
def reference_points(d: int, n: int = REFERENCE_N, seed: int = REFERENCE_SEED) -> np.ndarray:
    """The frozen uniform reference set for dimension ``d`` (scrambled Sobol)."""
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.log2(n))
    if 2 ** m != n:
        raise ValueError("reference set size must be a power of two (Sobol balance)")
    X = sampler.random_base2(m=m)
    X.setflags(write=False)
    return np.ascontiguousarray(2.0 * X - 1.0)


def reference_moments(raw, d: int) -> tuple[float, float]:
    """``(mean, standard deviation)`` of a raw target on the frozen reference set."""
    vals = np.asarray(raw(reference_points(d)), dtype=np.float64)
    mu = float(vals.mean())
    sd = float(vals.std())
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0.0:
        raise ValueError(f"degenerate scaling: mean={mu}, sd={sd}")
    return mu, sd


def normalize_callable(raw, d: int):
    """Return ``(F, mean, sd)`` where ``F(X) = (raw(X) - mean)/sd``."""
    mu, sd = reference_moments(raw, d)

    def F(X):
        return (np.asarray(raw(X), dtype=np.float64) - mu) / sd

    return F, mu, sd
