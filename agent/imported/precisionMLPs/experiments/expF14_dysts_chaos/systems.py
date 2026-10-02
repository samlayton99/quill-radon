"""expF14 -- the five dysts chaotic ODE systems, with analytic RHS and Jacobian.

Canonical parameters, initial conditions, dominant period and estimated maximal
Lyapunov exponent are read from `dysts` (Gilpin, NeurIPS 2021 D&B).  The
right-hand sides are re-implemented here in vectorised numpy because the
Gauss-Newton solve needs (a) evaluation at n collocation points at once and
(b) an analytic Jacobian dF/du at those points.

Both re-implementations are verified:
  * `verify_rhs`      -- our F(U) against dysts' own rhs, elementwise;
  * `verify_jacobian` -- our J(U) against the complex-step derivative
                         Im F(U + i h e_c)/h, which is exact to rounding
                         (no subtractive cancellation, unlike finite differences).

Everything is fp64 except the complex-step probe.
"""

from __future__ import annotations

import numpy as np

SYSTEM_ORDER = ["Lorenz", "Rossler", "Thomas", "Halvorsen", "Lorenz96",
                "InteriorSquirmer", "DoublePendulum", "MacArthur"]

# Systems whose Jacobian is obtained by complex step rather than by hand. The
# complex-step derivative Im F(u + i h e_k)/h is exact to rounding for analytic
# F, so this is not an approximation -- it just moves the algebra from us to the
# machine. Verified against central differences in `verify_jacobian_fd`.
CSTEP_JAC = {"InteriorSquirmer", "DoublePendulum", "MacArthur"}


# ---------------------------------------------------------------------------
# right-hand sides:  F(U) with U of shape (n, d) -> (n, d)
# Jacobians:         J(U) with U of shape (n, d) -> (n, d, d), J[:, c, k] = dF_c/du_k
# Both are written so they also accept complex input (for the complex-step test).
# ---------------------------------------------------------------------------

def _lorenz_rhs(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    return np.stack([p["sigma"] * (y - x),
                     p["rho"] * x - x * z - y,
                     x * y - p["beta"] * z], axis=1)


def _lorenz_jac(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    n = len(U)
    J = np.zeros((n, 3, 3))
    J[:, 0, 0] = -p["sigma"]; J[:, 0, 1] = p["sigma"]
    J[:, 1, 0] = p["rho"] - z; J[:, 1, 1] = -1.0; J[:, 1, 2] = -x
    J[:, 2, 0] = y; J[:, 2, 1] = x; J[:, 2, 2] = -p["beta"]
    return J


def _rossler_rhs(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    return np.stack([-y - z,
                     x + p["a"] * y,
                     p["b"] + z * x - p["c"] * z], axis=1)


def _rossler_jac(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    n = len(U)
    J = np.zeros((n, 3, 3))
    J[:, 0, 1] = -1.0; J[:, 0, 2] = -1.0
    J[:, 1, 0] = 1.0; J[:, 1, 1] = p["a"]
    J[:, 2, 0] = z; J[:, 2, 2] = x - p["c"]
    return J


def _thomas_rhs(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    a, b = p["a"], p["b"]
    return np.stack([-a * x + b * np.sin(y),
                     -a * y + b * np.sin(z),
                     -a * z + b * np.sin(x)], axis=1)


def _thomas_jac(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    a, b = p["a"], p["b"]
    n = len(U)
    J = np.zeros((n, 3, 3))
    J[:, 0, 0] = -a; J[:, 0, 1] = b * np.cos(y)
    J[:, 1, 1] = -a; J[:, 1, 2] = b * np.cos(z)
    J[:, 2, 2] = -a; J[:, 2, 0] = b * np.cos(x)
    return J


def _halvorsen_rhs(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    a, b = p["a"], p["b"]
    return np.stack([-a * x - b * y - b * z - y ** 2,
                     -a * y - b * z - b * x - z ** 2,
                     -a * z - b * x - b * y - x ** 2], axis=1)


def _halvorsen_jac(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    a, b = p["a"], p["b"]
    n = len(U)
    J = np.zeros((n, 3, 3))
    J[:, 0, 0] = -a; J[:, 0, 1] = -b - 2 * y; J[:, 0, 2] = -b
    J[:, 1, 0] = -b; J[:, 1, 1] = -a; J[:, 1, 2] = -b - 2 * z
    J[:, 2, 0] = -b - 2 * x; J[:, 2, 1] = -b; J[:, 2, 2] = -a
    return J


def _l96_rhs(U, p):
    """Cyclic Lorenz-96: xdot_i = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F."""
    Xp1 = np.roll(U, -1, axis=1)
    Xm1 = np.roll(U, 1, axis=1)
    Xm2 = np.roll(U, 2, axis=1)
    return (Xp1 - Xm2) * Xm1 - U + p["f"]


def _l96_jac(U, p):
    n, d = U.shape
    J = np.zeros((n, d, d))
    idx = np.arange(d)
    ip1, im1, im2 = (idx + 1) % d, (idx - 1) % d, (idx - 2) % d
    # accumulate (indices can collide for very small d)
    np.add.at(J, (slice(None), idx, ip1), U[:, im1])
    np.add.at(J, (slice(None), idx, im2), -U[:, im1])
    np.add.at(J, (slice(None), idx, im1), U[:, ip1] - U[:, im2])
    np.add.at(J, (slice(None), idx, idx), -1.0)
    return J


# ---------------------------------------------------------------------------
# the three added systems (Sam's second batch), each breaking a different
# assumption: a near-square-wave drive, a conservative Hamiltonian flow, and a
# vector field that is only C^0.
# ---------------------------------------------------------------------------

def _squirmer_rhs(U, p):
    """Interior squirmer in polar coords (r, th, tt); tt is a clock, d(tt)/dt = 1.

    The system is non-autonomous dressed as autonomous: the mode amplitudes are
    gated by protocol(tt) = 0.5 + 0.5 tanh(tau * 20 * sin(2 pi tt / tau)), a
    near-square wave (tau*20 = 60 at the default tau = 3).
    """
    r, th, tt = U[:, 0], U[:, 1], U[:, 2]
    a, g, tau = p["a"], p["g"], p["tau"]
    phase = 0.5 + 0.5 * np.tanh(tau * 20.0 * np.sin(2 * np.pi * tt / tau))
    A = a[None, :] * phase[:, None]
    G = g[None, :] * (1.0 - phase)[:, None]
    nv = np.arange(1, a.shape[0] + 1)[None, :]
    rc, thc = r[:, None], th[:, None]
    sinv, cosv = np.sin(thc * nv), np.cos(thc * nv)
    rn = rc ** nv
    vrn = (G * cosv + A * sinv) * (nv * rn * (rc ** 2 - 1.0) / rc)
    vth = (2 * rc + (rc ** 2 - 1.0) * nv / rc) * (A * cosv - G * sinv) * rn
    return np.stack([vrn.sum(axis=1), vth.sum(axis=1) / r,
                     np.ones_like(r)], axis=1)


def _pendulum_rhs(U, p):
    """Double pendulum in (th1, th2, p1, p2). Hamiltonian: no attractor."""
    th1, th2, p1, p2 = U[:, 0], U[:, 1], U[:, 2], U[:, 3]
    g, l1, l2, m1, m2 = p["g"], p["l1"], p["l2"], p["m1"], p["m2"]
    cd, sd = np.cos(th1 - th2), np.sin(th1 - th2)
    denom = l1 * l2 * (m1 + m2 * sd ** 2)
    th1dot = (l2 * p1 - l1 * p2 * cd) / (l1 * denom)
    th2dot = ((m1 + m2) * l1 * p2 - m2 * l2 * p1 * cd) / (m2 * l2 * denom)
    h1 = p1 * p2 * sd / denom
    h2 = (m2 * l2 * p1 ** 2) / (2 * l1 * denom ** 2)
    h2 = h2 + 0.5 * m2 * p2 * l2 * l1 * th2dot / denom
    h2 = h2 * np.sin(2 * (th1 - th2))
    p1dot = -(m1 + m2) * g * l1 * np.sin(th1) - h1 + h2
    p2dot = -m2 * g * l2 * np.sin(th2) + h1 - h2
    return np.stack([th1dot, th2dot, p1dot, p2dot], axis=1)


def _macarthur_rhs(U, p):
    """MacArthur consumer-resource: 5 species, 5 resources, Liebig's law.

    mu_i = min_j r * R_j / (k[j,i] + R_j) -- the growth rate is the MINIMUM over
    resources, so the vector field is only C^0: it has kinks wherever the
    limiting resource changes. The argmin is taken on the real part so that a
    complex-step probe differentiates the currently active branch, which is the
    correct one-sided derivative.
    """
    ns = p["k"].shape[0]
    nn, rr = U[:, :ns], U[:, ns:]
    kT = p["k"].T[None, :, :]                    # kT[.., i, j] = k[j, i]
    u = p["r"] * rr[:, None, :] / (kT + rr[:, None, :])
    idx = np.argmin(np.real(u), axis=2)[..., None]
    mu = np.take_along_axis(u, idx, axis=2)[..., 0]
    nndot = nn * (mu - p["m"])
    rrdot = p["d"] * (p["s"][None, :] - rr) - (mu * nn) @ p["c"].T
    return np.concatenate([nndot, rrdot], axis=1)


def complex_step_jac(rhs, U, params, h=1e-30):
    """J[:, c, k] = dF_c/du_k by complex step -- exact to rounding, no cancellation."""
    n, d = U.shape
    J = np.empty((n, d, d))
    for k in range(d):
        Uc = U.astype(np.complex128)
        Uc[:, k] += 1j * h
        J[:, :, k] = np.imag(rhs(Uc, params)) / h
    return J


_IMPL = {
    "Lorenz": (_lorenz_rhs, _lorenz_jac),
    "Rossler": (_rossler_rhs, _rossler_jac),
    "Thomas": (_thomas_rhs, _thomas_jac),
    "Halvorsen": (_halvorsen_rhs, _halvorsen_jac),
    "Lorenz96": (_l96_rhs, _l96_jac),
    "InteriorSquirmer": (_squirmer_rhs, None),
    "DoublePendulum": (_pendulum_rhs, None),
    "MacArthur": (_macarthur_rhs, None),
}


# ---------------------------------------------------------------------------
# system objects
# ---------------------------------------------------------------------------

class System:
    def __init__(self, name):
        import dysts.flows as flows
        m = getattr(flows, name)()
        self.name = name
        self.params = {k: (np.asarray(v, dtype=np.float64) if np.ndim(v)
                           else float(v)) for k, v in m.params.items()}
        self.ic = np.atleast_1d(np.asarray(m.ic, dtype=np.float64)).copy()
        self.d = int(self.ic.size)
        self.period = float(m.period)
        self.lyapunov = float(m.maximum_lyapunov_estimated)
        self._dysts = m
        self._rhs, self._jac = _IMPL[name]

    def F(self, U):
        """(n, d) -> (n, d)."""
        return self._rhs(np.atleast_2d(U), self.params)

    def J(self, U):
        """(n, d) -> (n, d, d), J[:, c, k] = dF_c/du_k."""
        U = np.atleast_2d(U)
        if self._jac is None:
            return complex_step_jac(self._rhs, U, self.params)
        return self._jac(U, self.params)

    def horizon(self, lyap_times):
        """Window length T such that lambda_max * T = lyap_times."""
        return float(lyap_times) / self.lyapunov

    def __repr__(self):
        return (f"<{self.name} d={self.d} lambda={self.lyapunov:.4f} "
                f"period={self.period:.4f}>")


def load_all():
    return {n: System(n) for n in SYSTEM_ORDER}


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def sample_states(sys, n=64):
    """States taken from a short trajectory of the system itself.

    Random perturbations of the initial condition are NOT usable here: the
    squirmer needs 0 < r < 1 and divides by r, and MacArthur's resources must
    stay positive. Verifying on the states the solver actually visits is both
    safer and more relevant.
    """
    from scipy.integrate import solve_ivp
    T = sys.horizon(3.0)
    sol = solve_ivp(lambda t, y: sys.F(y[None, :])[0], [0.0, T], sys.ic,
                    rtol=1e-10, atol=1e-12, method="DOP853", dense_output=True)
    return sol.sol(np.linspace(0.0, T, n)).T


def verify_rhs(sys, n=64):
    """Max |our F - dysts rhs| over on-trajectory states, absolute and relative."""
    U = sample_states(sys, n)
    ours = sys.F(U)
    theirs = np.empty_like(ours)
    for i in range(n):
        theirs[i] = np.asarray(sys._dysts.rhs(U[i], 0.0), dtype=np.float64).ravel()
    absd = np.max(np.abs(ours - theirs))
    return absd, absd / max(np.max(np.abs(theirs)), 1e-300)


def verify_jacobian(sys, n=64, h=1e-30):
    """Analytic dF/du against the complex-step derivative.

    Vacuous for the systems in CSTEP_JAC (their J *is* complex step); those are
    gated by `verify_jacobian_fd` instead.
    """
    if sys.name in CSTEP_JAC:
        return 0.0, 0.0
    U = sample_states(sys, n)
    Ja = sys.J(U)
    Jc = complex_step_jac(sys._rhs, U, sys.params, h)
    absd = np.max(np.abs(Ja - Jc))
    return absd, absd / max(np.max(np.abs(Jc)), 1e-300)


def verify_jacobian_fd(sys, n=32, rel_h=1e-6):
    """dF/du against central differences on-trajectory.

    Reported as the 95th percentile of the per-entry relative error, not the
    max: MacArthur's field is only C^0, so a difference stencil that straddles
    a kink is wrong about the derivative there and says nothing about our
    Jacobian. The max is returned alongside so the outliers stay visible.
    """
    U = sample_states(sys, n)
    Ja = sys.J(U)
    scale = np.maximum(np.abs(U), 1.0)
    Jf = np.empty_like(Ja)
    for k in range(sys.d):
        step = rel_h * scale[:, k]
        Up, Um = U.copy(), U.copy()
        Up[:, k] += step
        Um[:, k] -= step
        Jf[:, :, k] = (sys.F(Up) - sys.F(Um)) / (2 * step)[:, None]
    denom = np.maximum(np.abs(Jf), np.max(np.abs(Jf)) * 1e-8)
    rel = np.abs(Ja - Jf) / denom
    return float(np.percentile(rel, 95)), float(np.max(rel))


def verify_all(verbose=True):
    """Returns name -> (rhs_rel, jac_rel, jac_fd_p95). Raises if anything is off."""
    out = {}
    for name in SYSTEM_ORDER:
        s = System(name)
        _, r_rel = verify_rhs(s)
        _, j_rel = verify_jacobian(s)
        fd_p95, fd_max = verify_jacobian_fd(s)
        how = "complex-step" if name in CSTEP_JAC else "analytic"
        if verbose:
            print(f"  {name:17s} d={s.d:2d}  rhs vs dysts: {r_rel:.2e}   "
                  f"jac({how}) vs analytic: {j_rel:.2e}   "
                  f"vs central-diff p95: {fd_p95:.2e} (max {fd_max:.1e})")
        assert r_rel < 1e-13, f"{name}: rhs mismatch {r_rel:.3e}"
        assert j_rel < 1e-12, f"{name}: jacobian mismatch {j_rel:.3e}"
        assert fd_p95 < 1e-5, f"{name}: jacobian vs FD p95 {fd_p95:.3e}"
        out[name] = (r_rel, j_rel, fd_p95)
    return out


if __name__ == "__main__":
    print("verifying RHS and Jacobian for the five dysts systems")
    verify_all()
    print("\nsystem metadata:")
    for n in SYSTEM_ORDER:
        s = System(n)
        for lt in (3,):
            print(f"  {s!r}  T(lambdaT={lt})={s.horizon(lt):.3f} "
                  f"= {s.horizon(lt)/s.period:.2f} periods  ic={np.round(s.ic,4).tolist()}")
