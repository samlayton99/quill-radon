"""expF14 -- extended-precision reference trajectories for the dysts systems.

A chaotic IVP amplifies any error by exp(lambda t), so a fp64 reference is not
good enough to certify a fp64 solution: DOP853 at rtol=atol=1e-13/1e-14 is
already only ~6e-13 relative on Lorenz over three Lyapunov times (measured in
`crosscheck`).  The reference here is mpmath's Taylor-method `odefun` at 30
decimal digits, rounded to fp64 afterwards -- exactly the repo's mpmath
convention for the QI construction: an offline extended-precision
precomputation producing fp64 constants.

Parameters and the initial condition are passed to mpmath as the *exact fp64
values*, so the reference solves the same IVP the fp64 solver is given.

Trajectories are cached (uncompressed .npz) keyed by (system, T, n_eval, dps).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = (REPO_ROOT / "results" / "checkpoint_F_applications"
             / "expF14_dysts_chaos" / "ref_cache")

DPS = 30

# mpmath's odefun is a Taylor method and assumes the field is analytic. That
# fails on MacArthur, whose growth rate is a min over resources: the field is
# only C^0 and the window crosses 9 kinks. Measured, at lambda_max*T = 3:
# DOP853@1e-13, DOP853@1e-14 and Radau@1e-12 agree with each other to 8e-14,
# while mpmath disagrees with ALL of them by 8.4e-5 and with itself (25 vs 40
# digits) by 3.4e-7. The extended-precision route is simply wrong there, so
# that system gets a convergence-verified fp64 reference instead.
RK_REFERENCE = {"MacArthur": dict(rtol=1e-13, atol=1e-15, method="DOP853")}
ODE_TOL_EXP = 25       # mpmath odefun tol = 10^-25
DEGREE = 12


def _mp_rhs(sys):
    """Return a scalar-list rhs f(t, y) in mpmath for `sys`."""
    import mpmath as mp
    def _mp(v):
        arr = np.asarray(v)
        if arr.ndim == 0:
            return mp.mpf(float(arr))
        return np.vectorize(lambda z: mp.mpf(float(z)), otypes=[object])(arr)

    p = {k: _mp(v) for k, v in sys.params.items()}
    name = sys.name
    if name == "Lorenz":
        def f(t, y):
            x, yy, z = y
            return [p["sigma"] * (yy - x), p["rho"] * x - x * z - yy,
                    x * yy - p["beta"] * z]
    elif name == "Rossler":
        def f(t, y):
            x, yy, z = y
            return [-yy - z, x + p["a"] * yy, p["b"] + z * x - p["c"] * z]
    elif name == "Thomas":
        def f(t, y):
            x, yy, z = y
            a, b = p["a"], p["b"]
            return [-a * x + b * mp.sin(yy), -a * yy + b * mp.sin(z),
                    -a * z + b * mp.sin(x)]
    elif name == "Halvorsen":
        def f(t, y):
            x, yy, z = y
            a, b = p["a"], p["b"]
            return [-a * x - b * yy - b * z - yy ** 2,
                    -a * yy - b * z - b * x - z ** 2,
                    -a * z - b * x - b * yy - x ** 2]
    elif name == "Lorenz96":
        d = sys.d
        FF = p["f"]

        def f(t, y):
            return [(y[(i + 1) % d] - y[(i - 2) % d]) * y[(i - 1) % d] - y[i] + FF
                    for i in range(d)]
    elif name == "InteriorSquirmer":
        a, g, tau = p["a"], p["g"], p["tau"]
        nmodes = len(a)

        def f(t, y):
            r, th, tt = y
            phase = mp.mpf(1) / 2 + mp.tanh(tau * 20 * mp.sin(2 * mp.pi * tt / tau)) / 2
            vr = mp.mpf(0)
            vt = mp.mpf(0)
            for i in range(nmodes):
                nn = mp.mpf(i + 1)
                A = a[i] * phase
                G = g[i] * (1 - phase)
                sn, cs = mp.sin(th * nn), mp.cos(th * nn)
                rn = r ** nn
                vr += (G * cs + A * sn) * (nn * rn * (r ** 2 - 1) / r)
                vt += (2 * r + (r ** 2 - 1) * nn / r) * (A * cs - G * sn) * rn
            return [vr, vt / r, mp.mpf(1)]

    elif name == "DoublePendulum":
        g, l1, l2, m1, m2 = (p["g"], p["l1"], p["l2"], p["m1"], p["m2"])

        def f(t, y):
            th1, th2, p1, p2 = y
            cd, sd = mp.cos(th1 - th2), mp.sin(th1 - th2)
            den = l1 * l2 * (m1 + m2 * sd ** 2)
            th1d = (l2 * p1 - l1 * p2 * cd) / (l1 * den)
            th2d = ((m1 + m2) * l1 * p2 - m2 * l2 * p1 * cd) / (m2 * l2 * den)
            h1 = p1 * p2 * sd / den
            h2 = (m2 * l2 * p1 ** 2) / (2 * l1 * den ** 2)
            h2 += m2 * p2 * l2 * l1 * th2d / den / 2
            h2 *= mp.sin(2 * (th1 - th2))
            return [th1d, th2d,
                    -(m1 + m2) * g * l1 * mp.sin(th1) - h1 + h2,
                    -m2 * g * l2 * mp.sin(th2) + h1 - h2]

    elif name == "MacArthur":
        kk, cc, ss, rr0, dd, mm = (p["k"], p["c"], p["s"], p["r"], p["d"], p["m"])
        ns = kk.shape[0]

        def f(t, y):
            nn, R = list(y[:ns]), list(y[ns:])
            mu = [min(rr0 * R[j] / (kk[j][i] + R[j]) for j in range(ns))
                  for i in range(ns)]
            out = [nn[i] * (mu[i] - mm) for i in range(ns)]
            for a in range(ns):
                acc = dd * (ss[a] - R[a])
                for i in range(ns):
                    acc -= cc[a][i] * mu[i] * nn[i]
                out.append(acc)
            return out

    else:
        raise ValueError(name)
    return f


def reference(sys, T, n_eval, dps=DPS, use_cache=True, verbose=True):
    """Dense reference trajectory: (ts, Y) with ts uniform on [0, T], Y (n_eval, d)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if sys.name in RK_REFERENCE and dps == DPS:
        return _rk_reference(sys, T, n_eval, verbose=verbose)
    key = f"{sys.name}_T{T:.10g}_n{n_eval}_dps{dps}.npz"
    path = CACHE_DIR / key
    if use_cache and path.exists():
        z = np.load(path)
        return z["ts"], z["Y"]

    import mpmath as mp
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        f = _mp_rhs(sys)
        ic = [mp.mpf(float(v)) for v in sys.ic]
        t0 = time.time()
        sol = mp.odefun(f, mp.mpf(0), ic, tol=mp.mpf(10) ** (-ODE_TOL_EXP),
                        degree=DEGREE)
        sol(mp.mpf(T))                     # drive the integrator to the end once
        ts = np.linspace(0.0, T, n_eval)
        Y = np.empty((n_eval, sys.d))
        for i, t in enumerate(ts):
            Y[i] = [float(v) for v in sol(mp.mpf(t))]
        if verbose:
            print(f"  mpmath reference {sys.name} T={T:.4f} dps={dps}: "
                  f"{time.time() - t0:.1f}s", flush=True)
    finally:
        mp.mp.dps = old
    np.savez(path, ts=ts, Y=Y)
    return ts, Y


def _rk_reference(sys, T, n_eval, verbose=True):
    """Convergence-verified fp64 reference for a non-analytic field."""
    cfg = RK_REFERENCE[sys.name]
    path = CACHE_DIR / f"{sys.name}_T{T:.10g}_n{n_eval}_rk.npz"
    if path.exists():
        z = np.load(path)
        return z["ts"], z["Y"]
    ts = np.linspace(0.0, T, n_eval)
    Y, _ = rk_trajectory(sys, T, ts, cfg["rtol"], cfg["atol"], cfg["method"])
    if verbose:
        print(f"  RK reference {sys.name} T={T:.4f} ({cfg['method']} "
              f"rtol={cfg['rtol']:.0e}); mpmath is not usable on this field",
              flush=True)
    np.savez(path, ts=ts, Y=Y)
    return ts, Y


def reference_uncertainty(sys, T, n_eval=2001):
    """How well can we even measure error here? Spread over independent methods.

    For an analytic field this compares mpmath at two working precisions. For a
    field in RK_REFERENCE it compares three independent fp64 integrator families
    (explicit RK, implicit RK, multistep) against the chosen reference -- the
    only honest way to bound a reference that cannot be taken to extra digits.
    """
    ts, Y = reference(sys, T, n_eval, verbose=False)
    if sys.name in RK_REFERENCE:
        out = {}
        for tag, (m, rt, at) in {"DOP853@1e-14": ("DOP853", 1e-14, 1e-16),
                                 "Radau@1e-12": ("Radau", 1e-12, 1e-14),
                                 "LSODA@1e-12": ("LSODA", 1e-12, 1e-14)}.items():
            Z, _ = rk_trajectory(sys, T, ts, rt, at, m)
            out[tag] = float(np.linalg.norm(Z - Y) / np.linalg.norm(Y))
        return out
    return {"mpmath 25 vs 40 digits": selfcheck(sys, T, min(n_eval, 501))["rel_l2"]}


def rk_trajectory(sys, T, ts, rtol, atol, method="DOP853"):
    """fp64 Runge-Kutta trajectory sampled at `ts` (used as warm start / baseline)."""
    from scipy.integrate import solve_ivp

    def g(t, y):
        return sys.F(y[None, :])[0]

    s = solve_ivp(g, [0.0, T], sys.ic, rtol=rtol, atol=atol, method=method,
                  dense_output=True)
    if not s.success:
        raise RuntimeError(f"{sys.name}: solve_ivp failed -- {s.message}")
    return s.sol(ts).T, int(s.nfev)


def crosscheck(sys, T, n_eval=2001, tols=((1e-13, 1e-14), (1e-10, 1e-12))):
    """How good is a *fp64* Runge-Kutta reference? Compare DOP853 to mpmath."""
    ts, Y = reference(sys, T, n_eval, verbose=False)
    out = {}
    for rtol, atol in tols:
        Z, nfev = rk_trajectory(sys, T, ts, rtol, atol)
        out[f"rtol{rtol:g}"] = dict(
            rel_l2=float(np.linalg.norm(Z - Y) / np.linalg.norm(Y)),
            linf=float(np.max(np.abs(Z - Y))), nfev=nfev)
    return out


def selfcheck(sys, T, n_eval=2001, dps_lo=25, dps_hi=40):
    """Reference self-consistency: the same integration at two working precisions."""
    _, Ylo = reference(sys, T, n_eval, dps=dps_lo, verbose=False)
    _, Yhi = reference(sys, T, n_eval, dps=dps_hi, verbose=False)
    return dict(rel_l2=float(np.linalg.norm(Ylo - Yhi) / np.linalg.norm(Yhi)),
                linf=float(np.max(np.abs(Ylo - Yhi))))
