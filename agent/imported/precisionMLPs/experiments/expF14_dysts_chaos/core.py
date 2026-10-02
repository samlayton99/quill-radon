"""expF14 -- frozen QI geometry + collocation Gauss-Newton for autonomous ODE systems.

Exactly the expF01/expF02 recipe, lifted from a scalar unknown to a coupled
d-component system (the expF09 `solve_system` idea in one coordinate).

Model.  Time is rescaled to s = 2t/T - 1 on [-1, 1], so d/dt = (2/T) d/ds.  Each
component of the (scale-normalised) state gets the SAME frozen 1-D QI
dictionary,

    u~_c(s) = sum_m A[c, m] tanh(gamma (s - s_m)) + sum_{k<=3} A[c, W+k] s^k,

with uniform centres s_m = -1 + m h (h = 2/N) extended by a halo of
R = max(70, ceil(0.4 N)) nodes on each side, and one shared bandwidth
gamma = lambda / h.  Nothing about the geometry is learned; only A is solved for.

Normalisation.  u_c = sigma_c u~_c with sigma_c a per-component scale taken from
a cheap fp64 Runge-Kutta pre-solve (NOT from the reference), so every residual
block is O(1).  This is a deployable step, not oracle information.

Residual and Jacobian.  With D0, D1 the value / d-ds dictionary rows at the
collocation points,

    R[:, c] = (2/T) D1 A[c]  -  F_c(sigma * (D0 A^T)) / sigma_c ,
    dR[:, c] / dA[c'] = (2/T) delta_{cc'} D1  -  (sigma_c'/sigma_c) diag(dF_c/du_c') D0 ,

plus one initial-condition row per component, D0(s=-1) A[c] = u0_c / sigma_c,
weighted sqrt(n_col) as in expF02.  Damped Gauss-Newton with backtracking; each
step is one min-norm lstsq of the stacked system.
"""

from __future__ import annotations

import numpy as np

RCOND = 1e-13
LAM = 0.25
MONO = [0, 1, 2, 3]
COLLOC_PER_NEURON = 4
MAX_NEWTON = 30


# ---------------------------------------------------------------------------
# geometry + dictionary
# ---------------------------------------------------------------------------

def geometry(N, lam=LAM):
    """Uniform QI centres on [-1, 1] plus halo, and the shared bandwidth."""
    h = 2.0 / N
    R = max(70, int(np.ceil(0.4 * N)))
    centers = -1.0 + h * np.arange(-R, N + R + 1)
    return centers, lam / h


def random_centers(N, lam=LAM, seed=0):
    """Ablation geometry: same count/span/bandwidth, centres drawn uniformly."""
    centers, gamma = geometry(N, lam)
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(centers.min(), centers.max(), centers.size)), gamma


def chebyshev_centers(N, lam=LAM):
    """Ablation geometry: Chebyshev-clustered centres over the same span."""
    centers, gamma = geometry(N, lam)
    W = centers.size
    lo, hi = centers.min(), centers.max()
    k = np.arange(W)
    c = np.cos(np.pi * (2 * k + 1) / (2 * W))
    return np.sort(lo + (hi - lo) * (1 - c) / 2), gamma


def _psi(order, t):
    """d^order/dz^order tanh(z), as a polynomial in t = tanh(z)."""
    if order == 0:
        return t
    if order == 1:
        return 1.0 - t * t
    if order == 2:
        return -2.0 * t * (1.0 - t * t)
    raise ValueError(order)


def _ffact(d, o):
    out = 1
    for k in range(o):
        out *= (d - k)
    return out


def dict_rows(s, centers, gamma, order):
    """[d^order/ds^order Phi | d^order/ds^order poly] at the points `s`."""
    s = np.asarray(s, dtype=np.float64)
    t = np.tanh(gamma * (s[:, None] - centers[None, :]))
    A = (gamma ** order) * _psi(order, t)
    P = np.zeros((s.size, len(MONO)))
    for k, dpow in enumerate(MONO):
        if order <= dpow:
            P[:, k] = _ffact(dpow, order) * s ** (dpow - order)
    return np.hstack([A, P])


def n_params(centers):
    return centers.size + len(MONO)


def evaluate(A, centers, gamma, s):
    """Model state (scaled) at points `s`: (len(s), d)."""
    return dict_rows(s, centers, gamma, 0) @ A.T


# ---------------------------------------------------------------------------
# the stacked nonlinear system
# ---------------------------------------------------------------------------

def _assemble(sys, A, D0, D1, sigma, T, need_jac=True):
    """Residual (n_col, d) and, if asked, the dense Jacobian (d*n_col, d*p)."""
    d, p = A.shape
    n = D0.shape[0]
    Us = D0 @ A.T                      # (n, d) scaled state
    U = Us * sigma[None, :]
    Fv = sys.F(U)
    dU = (2.0 / T) * (D1 @ A.T)        # (n, d) scaled d/dt
    R = dU - Fv / sigma[None, :]
    if not need_jac:
        return R, None
    Jf = sys.J(U)                      # (n, d, d)
    J = np.zeros((d * n, d * p))
    for c in range(d):
        rs = slice(c * n, (c + 1) * n)
        J[rs, c * p:(c + 1) * p] = (2.0 / T) * D1
        for k in range(d):
            coef = -(sigma[k] / sigma[c]) * Jf[:, c, k]
            if np.any(coef):
                J[rs, k * p:(k + 1) * p] += coef[:, None] * D0
    return R, J


def _ic_block(sys, centers, gamma, sigma, p, w):
    """(rows, rhs) for the d initial-condition rows, already weighted."""
    d = sys.d
    row = dict_rows(np.array([-1.0]), centers, gamma, 0)[0][:p]   # (p,)
    B = np.zeros((d, d * p))
    g = np.empty(d)
    for c in range(d):
        B[c, c * p:(c + 1) * p] = w * row
        g[c] = w * sys.ic[c] / sigma[c]
    return B, g


def _stacked(sys, A, D0, D1, sigma, T, B, g, scale):
    R, _ = _assemble(sys, A, D0, D1, sigma, T, need_jac=False)
    a = A.ravel()
    r = np.concatenate([R.T.ravel() / scale, B @ a - g])
    return r, float(np.linalg.norm(r)), float(np.max(np.abs(R)))


def gauss_newton(sys, A0, D0, D1, sigma, T, B, g, rcond=RCOND, max_it=MAX_NEWTON,
                 verbose=False):
    """Damped Gauss-Newton on the readout. Returns (A, history)."""
    A = A0.copy()
    d, p = A.shape
    hist = []
    no_progress = 0
    diverged = False
    for it in range(max_it):
        R, J = _assemble(sys, A, D0, D1, sigma, T)
        # A non-finite Jacobian means the iterate left the domain where the
        # vector field is defined (the squirmer divides by r, so a cold start at
        # r == 0 lands here). That is a divergence, not a crash: record it.
        if not np.all(np.isfinite(J)):
            diverged = True
            break
        scale = max(np.abs(J).max(), 1e-300)
        Jst = np.vstack([J / scale, B])
        r, rn, rmax = _stacked(sys, A, D0, D1, sigma, T, B, g, scale)
        hist.append(rmax)
        if not np.isfinite(rn):
            diverged = True
            break
        try:
            step = np.linalg.lstsq(Jst, -r, rcond=rcond)[0].reshape(d, p)
        except np.linalg.LinAlgError:
            # LAPACK gelsd occasionally fails to converge; gelsy is a different
            # algorithm (complete orthogonal factorisation) and usually succeeds.
            from scipy.linalg import lstsq as _slstsq
            try:
                step = _slstsq(Jst, -r, cond=rcond, lapack_driver="gelsy")[0]
                step = step.reshape(d, p)
            except Exception:
                diverged = True
                break
        alpha = 1.0
        for _ in range(10):
            _, new_n, _ = _stacked(sys, A + alpha * step, D0, D1, sigma, T, B, g, scale)
            if new_n <= rn * (1.0 - 1e-4 * alpha) or new_n < 1e-14:
                break
            alpha *= 0.5
        A = A + alpha * step
        if verbose:
            print(f"    it {it:2d}  |res|inf={rmax:.3e}  alpha={alpha:.3g}", flush=True)
        small = alpha * np.linalg.norm(step) < 1e-14 * max(1.0, np.linalg.norm(A))
        # A step that fails to halve the residual has not necessarily stopped
        # helping the *solution*: the first such step is still typically worth
        # ~5x in eval error. Require two in a row before declaring convergence.
        no_progress = (no_progress + 1) if (len(hist) >= 2
                                            and hist[-1] > 0.5 * hist[-2]) else 0
        if small or no_progress >= 2:
            break
    return A, hist, diverged


# ---------------------------------------------------------------------------
# the full per-cell solve
# ---------------------------------------------------------------------------

def solve_cell(sys, T, N, lam=LAM, rcond=RCOND, geom="uniform", seed=0,
               warm=True, warm_rtol=1e-6, warm_atol=1e-9, use_poly=True,
               max_it=MAX_NEWTON, verbose=False, init=None, w_mult=1.0):
    """One (system, horizon, width) cell. Returns a dict with the model + diagnostics."""
    from reference import rk_trajectory

    if geom == "uniform":
        centers, gamma = geometry(N, lam)
    elif geom == "random":
        centers, gamma = random_centers(N, lam, seed)
    elif geom == "chebyshev":
        centers, gamma = chebyshev_centers(N, lam)
    else:
        raise ValueError(geom)
    W = centers.size
    p = n_params(centers)
    n_col = COLLOC_PER_NEURON * W
    s_col = np.linspace(-1.0, 1.0, n_col)
    t_col = T * (s_col + 1.0) / 2.0

    D0 = dict_rows(s_col, centers, gamma, 0)
    D1 = dict_rows(s_col, centers, gamma, 1)
    if not use_poly:
        D0 = D0[:, :W]
        D1 = D1[:, :W]
        p = W

    # cheap fp64 RK pre-solve: sets the component scales AND the warm start
    Yrk, nfev = rk_trajectory(sys, T, t_col, warm_rtol, warm_atol)
    sigma = np.maximum(np.sqrt(np.mean(Yrk ** 2, axis=0)), 1e-12)

    w_ic = w_mult * np.sqrt(n_col)
    B, g = _ic_block(sys, centers, gamma, sigma, p, w_ic)

    # --- Newton initialisation ladder (expF03 part 1: the dominant knob) -----
    if init is None:
        init = "warm" if warm else "cold"
    ic_row = dict_rows(np.array([-1.0]), centers, gamma, 0)[:, :p]

    def _fit(Y):
        """lstsq fit of a trajectory Y (physical units) with the IC row enforced."""
        M = np.vstack([D0, w_ic * ic_row])
        rhs = np.vstack([Y / sigma[None, :], w_ic * (sys.ic / sigma)[None, :]])
        return np.linalg.lstsq(M, rhs, rcond=rcond)[0].T

    sub = None
    if init == "cold":
        A0 = np.zeros((sys.d, p))
    elif init == "warm":
        A0 = _fit(Yrk)
    elif init == "bcfit":
        A0 = np.linalg.lstsq(w_ic * ic_row,
                             w_ic * (sys.ic / sigma)[None, :], rcond=rcond)[0].T
    elif init == "cascade":
        # converge at N//4, then refit the full-width dictionary to that solution
        sub = solve_cell(sys, T, max(16, N // 4), lam=lam, rcond=rcond, geom=geom,
                         seed=seed, warm_rtol=warm_rtol, warm_atol=warm_atol,
                         use_poly=use_poly, max_it=max_it, init="warm",
                         w_mult=w_mult)
        A0 = _fit(model_trajectory(sub, t_col))
    else:
        raise ValueError(init)

    A, hist, diverged = gauss_newton(sys, A0, D0, D1, sigma, T, B, g, rcond=rcond,
                                     max_it=max_it, verbose=verbose)
    cell = dict(A=A, A0=A0, centers=centers, gamma=gamma, sigma=sigma, T=T,
                W=W, p=p, n_col=n_col, iters=len(hist), hist=hist,
                rk_nfev=nfev, lam=lam, rcond=rcond, geom=geom, use_poly=use_poly,
                init=init, w_mult=w_mult, diverged=diverged)
    if sub is not None:
        cell["cascade_iters"] = sub["iters"]
        cell["cascade_W"] = sub["W"]
    try:
        cell["fresh"] = fresh_residual(sys, cell)
    except (ValueError, FloatingPointError):
        cell["fresh"] = dict(pde=float("inf"), ic=float("inf"),
                             stacked=float("inf"))
    return cell


def fresh_residual(cell_sys, cell):
    """expF03 part 2's deployable signal: the STACKED residual (PDE rows AND the
    initial-condition row) at collocation points the solve never saw.

    The condition row is not optional. Every autonomous system here has fixed
    points, and u == fixed point is an exact solution of the ODE with an
    identically zero PDE residual -- only the IC row rules it out. expF03 hit
    exactly this trap on the logistic IVP, where a diverged solve returned an
    exact-but-wrong branch whose PDE residual was literally zero.
    """
    n = cell["n_col"]
    s_f = np.linspace(-1.0, 1.0, n + 1)[:-1] + 1.0 / n     # staggered, unseen
    D0f = dict_rows(s_f, cell["centers"], cell["gamma"], 0)[:, :cell["p"]]
    D1f = dict_rows(s_f, cell["centers"], cell["gamma"], 1)[:, :cell["p"]]
    R, _ = _assemble(cell_sys, cell["A"], D0f, D1f, cell["sigma"], cell["T"],
                     need_jac=False)
    ic_row = dict_rows(np.array([-1.0]), cell["centers"], cell["gamma"], 0)[:, :cell["p"]]
    ic_err = np.abs((ic_row @ cell["A"].T)[0] * cell["sigma"] - cell_sys.ic)
    return dict(pde=float(np.max(np.abs(R))),
                ic=float(np.max(ic_err)),
                stacked=float(max(np.max(np.abs(R)), np.max(ic_err))))


def model_trajectory(cell, ts):
    """Physical-state trajectory of a solved cell at times `ts`."""
    s = 2.0 * np.asarray(ts, dtype=np.float64) / cell["T"] - 1.0
    Us = dict_rows(s, cell["centers"], cell["gamma"], 0)[:, :cell["p"]] @ cell["A"].T
    return Us * cell["sigma"][None, :]


def warm_trajectory(cell, ts):
    s = 2.0 * np.asarray(ts, dtype=np.float64) / cell["T"] - 1.0
    Us = dict_rows(s, cell["centers"], cell["gamma"], 0)[:, :cell["p"]] @ cell["A0"].T
    return Us * cell["sigma"][None, :]


def errors(Yhat, Yref):
    """(aggregate rel L2, max per-component rel L2, absolute Linf)."""
    E = Yhat - Yref
    rel = float(np.linalg.norm(E) / np.linalg.norm(Yref))
    per = np.linalg.norm(E, axis=0) / np.maximum(np.linalg.norm(Yref, axis=0), 1e-300)
    return rel, float(np.max(per)), float(np.max(np.abs(E)))


def interpolation_floor(sys, T, N, ts, Yref, lam=LAM, rcond=RCOND, use_poly=True):
    """Best rel L2 the frozen dictionary can reach by fitting the reference directly.

    Separates 'the basis cannot resolve this trajectory' (resolution) from
    'the collocation solve cannot find it' (conditioning / chaos).
    """
    centers, gamma = geometry(N, lam)
    s = 2.0 * np.asarray(ts) / T - 1.0
    D0 = dict_rows(s, centers, gamma, 0)
    if not use_poly:
        D0 = D0[:, :centers.size]
    sigma = np.maximum(np.sqrt(np.mean(Yref ** 2, axis=0)), 1e-12)
    A = np.linalg.lstsq(D0, Yref / sigma[None, :], rcond=rcond)[0]
    return errors((D0 @ A) * sigma[None, :], Yref)
