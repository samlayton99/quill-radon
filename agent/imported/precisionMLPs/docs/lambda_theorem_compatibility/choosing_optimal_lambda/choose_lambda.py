"""Select lambda from an activation and precision, optionally a target summary.

Standard library only. Inputs use angular frequencies (radians per input unit).
This is a practical selector. The tanh representation theorem also requires
admissible geometry and width; a representative frequency is not an alias certificate.
"""
from __future__ import annotations

import math
import sys


def log_khat(activation: str, xi: float) -> float:
    """log(Khat(xi)/Khat(0)), evaluated without Fourier-tail underflow."""
    if activation == "gaussian":
        return -xi * xi / 4
    if activation == "gelu":
        return math.log1p(xi * xi) - xi * xi / 2
    if activation != "tanh":
        raise ValueError("activation must be 'tanh', 'gelu', or 'gaussian'")
    a = abs(xi) * math.pi / 2
    if a == 0:
        return 0.0
    return math.log(2 * a) - a - math.log(-math.expm1(-2 * a))


def log_alias_score(activation: str, lam: float, theta: float | None = None) -> float:
    """Basic score if theta is omitted; the practical first-pair score otherwise."""
    if theta is None:
        return log_khat(activation, 2 * math.pi / lam)
    if not 0 < theta < math.pi:
        raise ValueError("The representative grid frequency h*omega_scale must be in (0, pi)")
    r = {"tanh": 1, "gelu": 2, "gaussian": 0}[activation]
    minus, plus = 2 * math.pi - theta, 2 * math.pi + theta
    terms = [r * math.log(theta / t) + log_khat(activation, t / lam)
             for t in (minus, plus)]
    high, low = max(terms), min(terms)
    numerator = high + math.log1p(math.exp(low - high))
    denominator = log_khat(activation, theta / lam)
    return numerator - denominator



def choose_lambda(activation: str, *, spacing: float,
                  e_tol: float = sys.float_info.epsilon,
                  omega_scale: float | None = None,
                  interval: tuple[float, float] = (0.03, 1.5)) -> dict:
    """Solve the stated scalar budget equation; return lambda and gamma=lambda/h.

    No frequency estimate: use the basic kernel ratio. With omega_scale: keep
    the first alias pair. omega_scale is a rough angular frequency in radians
    per input unit, about 2*pi divided by a typical wavelength. e_tol is an
    effective tolerance absorbing amplitude and constant factors; its default
    is binary64 machine epsilon. Width N on [-1,1] gives spacing=2/N.
    Search bounds are explicit; 'upper_search_limit' is not a located root.
    """
    if activation not in ("tanh", "gelu", "gaussian"):
        raise ValueError("activation must be 'tanh', 'gelu', or 'gaussian'")
    lo, hi = interval
    if not (0 < lo < hi and 0 < e_tol < 1 and spacing > 0):
        raise ValueError("Invalid search interval, e_tol or spacing")
    if not all(math.isfinite(x) for x in (lo, hi, e_tol, spacing)):
        raise ValueError("Inputs must be finite")
    theta = None
    if omega_scale is not None:
        if not (omega_scale > 0 and math.isfinite(omega_scale)):
            raise ValueError("omega_scale must be positive and finite")
        theta = spacing * omega_scale
        if not 0 < theta < math.pi:
            raise ValueError("The representative grid frequency h*omega_scale must be in (0, pi)")
    # GELU's transform has a small central rise. Keep the first alias on its
    # decreasing tail, where both basic and refined scores are monotone.
    if activation == "gelu" and hi > 2 * math.pi - (theta or 0):
        raise ValueError("For GELU, the interval must keep (2*pi-theta)/lambda >= 1")
    rule = "basic" if theta is None else "refined"
    budget = math.log(e_tol)
    def score(lam):
        return log_alias_score(activation, lam, theta)
    if score(lo) >= budget:
        return {"lambda": None, "gamma": None, "status": "no_feasible_point", "rule": rule}
    if score(hi) < budget:
        return {"lambda": hi, "gamma": hi / spacing,
                "status": "upper_search_limit", "rule": rule}
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if score(mid) < budget:
            lo = mid
        else:
            hi = mid
    return {"lambda": lo, "gamma": lo / spacing, "status": "threshold", "rule": rule}


if __name__ == "__main__":
    import json
    print(json.dumps({
        "basic": choose_lambda("tanh", spacing=2/128),
        "sine_mixture": choose_lambda("tanh", spacing=2/128,
                                       omega_scale=30*math.pi/7),
        "gelu_sine_mixture": choose_lambda("gelu", spacing=2/128,
                                            omega_scale=30*math.pi/7),
    }, indent=2))
