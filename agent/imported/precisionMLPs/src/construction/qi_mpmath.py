"""QI construction for single-hidden-layer tanh MLPs.

Constructs a single-hidden-layer tanh MLP that approximates a target function
on [-1, 1]. Two precision regimes are supported (see construct_qi docstring
and papers/practical_implementation.tex):

    precision="fp64"   : fast (~10ms), L_inf ~ 1e-12 (default).
    precision="mpmath" : slow (~55s cold, ~0.25s cached), L_inf ~ 3e-15.

Both paths produce valid fp64 coefficients for fp64 PyTorch models.

Index conventions (two distinct families -- see also docs/explanation.md 5.1):
    n = center/neuron index, n in [-halo, N+halo]  (W = N + 2*halo + 1 neurons).
    k = cardinal stencil index, k in [-Kc, Kc]     (2*Kc+1 coefficients, default 321).

Algorithm:
    1. Grid (centers): x_n = -1 + n*h, h = 2/N, n in [-halo, N+halo].
    2. Set gamma = lambda_star / h (inner-layer bandwidth, grows as O(N)).
    3. Toeplitz solve for cardinal coefficients c_k, k in [-Kc, Kc]:
         sum_j c_j * h * Kd((r-j)*h) = h * delta_{r,0},  r,j in [-Kc, Kc],
       where Kd(x) = gamma * sech^2(gamma * x).
    4. Convolve with target derivative (stencil index k in [-Kc, Kc]):
         a_n = sum_k c_k * g'(x_{n-k})
       (vectorized sliding-window matmul in fp64; Kahan in the mpmath path).
    5. Compute bias from boundary: c0 = g(-1) - sum_n a_n * tanh(gamma*(-1 - x_n)).

The resulting MLP: q(x) = c0 + sum_n a_n * tanh(gamma * (x - x_n)).

Caching: Cardinal coefficients c_j depend only on (lambda_star, Kc, N,
precision, mp_dps) and are target-independent. They are cached on disk at
DEFAULT_CACHE_DIR (override via QI_CACHE_DIR env var or cache_dir kwarg).

Adapted from continuous-mlps/src/construction/explicit_quasi_interpolant.py.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# Default on-disk cache for cardinal coefficients (target-independent).
DEFAULT_CACHE_DIR = Path(
    os.environ.get("QI_CACHE_DIR",
                   str(Path(__file__).resolve().parents[2] / "results" / "qi_cache"))
)


def default_halo(N: int, *, lambda_star: float = 0.30,
                 ratio: float = 0.4, safety: float = 35.0) -> int:
    """Recommended halo size for width N at a given lambda.

    Two constraints drive the choice:
      1. Scaling with N (matches continuous-mlps): halo >= ratio * N.
      2. Exponential halo-tail decay: exp(-2 * lambda * halo) must be below
         the target precision floor, so halo >= safety / (2 * lambda).

    For lambda=0.30, safety=35 gives halo >= 59 (floor below 1e-15).
    For lambda=0.25 (mpmath path), halo >= 70 (floor below 1e-15).
    """
    min_halo = int(np.ceil(safety / (2.0 * lambda_star)))
    return max(min_halo, int(ratio * N))


@dataclass(frozen=True)
class QIResult:
    """Immutable result of QI construction.

    This is a pure data object -- it never references a model.
    Use initialize.py to copy values into a QIMlp.

    Fields:
        N:                  Number of interior grid intervals.
        h:                  Grid spacing 2/N.
        gamma:              Bandwidth parameter lambda_star / h.
        lambda_val:         Dimensionless bandwidth gamma * h = lambda_star.
        centers:            Grid positions including halo, shape [N + 2*halo + 1].
        interior_centers:   Grid positions without halo, shape [N + 1].
        a_coeffs:           Outer weights (QI coefficients), shape [N + 2*halo + 1].
        interior_a_coeffs:  Outer weights without halo, shape [N + 1].
        c0:                 Bias constant from boundary condition.
        toeplitz_residual:  Residual norm of the Toeplitz solve (for diagnostics).
        halo:               Number of ghost nodes on each side.
        Kc:                 Toeplitz stencil half-width.
    """
    N: int
    h: float
    gamma: float
    lambda_val: float
    centers: np.ndarray
    interior_centers: np.ndarray
    a_coeffs: np.ndarray
    interior_a_coeffs: np.ndarray
    c0: float
    toeplitz_residual: float
    halo: int
    Kc: int


# ---------------------------------------------------------------------------
# Fast fp64 path (no mpmath dependency for the Toeplitz solve)
# ---------------------------------------------------------------------------

def _build_toeplitz_c_f64(*, h, gamma, Kc):
    """Solve Toeplitz system for cardinal coefficients in fp64.

    Same math as the mpmath version but uses numpy.linalg.solve.
    """
    n = 2 * Kc + 1
    idx = np.arange(-Kc, Kc + 1, dtype=np.int64)

    d = np.arange(-2 * Kc, 2 * Kc + 1, dtype=np.int64)
    x = d.astype(np.float64) * h
    z = gamma * x
    t = np.tanh(z)
    kd = gamma * (1.0 - t * t)
    dvals = h * kd

    T = np.empty((n, n), dtype=np.float64)
    for i, r in enumerate(idx):
        diffs = r - idx
        T[i, :] = dvals[diffs + 2 * Kc]

    b = np.zeros(n, dtype=np.float64)
    b[Kc] = h
    c = np.linalg.solve(T, b)

    rvec = T @ c - b
    info = {"toeplitz_resid_inf": float(np.max(np.abs(rvec)))}
    return c.tolist(), info


def _build_a_f64(*, gprime, h, gamma, N, halo, Kc, c_f64, sample_dtype=np.float64):
    """Compute outer weights a_n via a vectorized convolution in fp64.

        a_n = sum_{k=-Kc..Kc} c_k * g'(x_{n-k})   for n in [-halo, N+halo]

    This is a single sliding-window BLAS matmul rather than a Python loop with
    compensated summation. The deviation from the Kahan-summed result is below
    the fp64 path's intrinsic cancellation floor (~1e-12; |c_j| reach O(300)
    with alternating signs), so this is ~2000x faster at no usable precision
    cost. The mpmath path (used for machine-eps baselines) keeps Kahan.
    """
    c_arr = np.asarray(c_f64, dtype=np.float64)

    # Sample g' on the extended grid j in [-(halo+Kc), N+halo+Kc].
    jmin = -(halo + Kc)
    jmax = N + halo + Kc
    y = np.array(
        [_sample_scalar(gprime, -1.0 + j * h, sample_dtype) for j in range(jmin, jmax + 1)],
        dtype=np.float64,
    )

    # a_n = sum_k c_k y_{n-k}. Sliding windows of width 2Kc+1 over y give
    # exactly one row per output node n in [-halo, N+halo]; dotting each window
    # with the reversed coefficients realizes the correlation a = windows @ c[::-1].
    windows = sliding_window_view(y, 2 * Kc + 1)
    a = windows @ c_arr[::-1]

    n_idx = list(range(-halo, N + halo + 1))
    return n_idx, a


def _compute_c0_f64(*, g, x0, x_centers, a_f64, gamma, sample_dtype=np.float64):
    """Compute bias c0 in fp64 so that q(x0) = g(x0).

    Uses math.fsum (correctly-rounded summation) for the boundary sum, matching
    the compensated summation used elsewhere in the construction.
    """
    gx0 = _sample_scalar(g, x0, sample_dtype)
    x_centers_f64 = np.asarray(x_centers, dtype=np.float64)
    a_arr = np.asarray(a_f64, dtype=np.float64)
    terms = a_arr * np.tanh(gamma * (float(x0) - x_centers_f64))
    return float(gx0 - math.fsum(terms))


# ---------------------------------------------------------------------------
# mpmath high-precision path
# ---------------------------------------------------------------------------

def _sech2_mp(z, mp):
    """sech^2(z) in mpmath."""
    t = mp.cosh(z)
    return 1 / (t * t)


def _Kd_mp(x, gamma, mp):
    """Kd(x) = gamma * sech^2(gamma * x) in mpmath."""
    return mp.mpf(gamma) * _sech2_mp(mp.mpf(gamma) * x, mp)


def _mp_kahan_dot(c_list, y_list, mp):
    """Kahan compensated dot product in mpmath arithmetic."""
    s = mp.mpf("0")
    comp = mp.mpf("0")
    for ci, yi in zip(c_list, y_list):
        prod = ci * yi
        yk = prod - comp
        t = s + yk
        comp = (t - s) - yk
        s = t
    return s


def _sample_scalar(func, x, sample_dtype=np.float64):
    """Sample func at x with quantized input/output."""
    x_q = sample_dtype(float(x))
    y = func(float(x_q))
    y_q = sample_dtype(y)
    return float(y_q)


def _build_toeplitz_c_mpmath(*, h, gamma, Kc, mp_dps):
    """Solve Toeplitz system for cardinal coefficients c_k in mpmath.

    T[r, j] = h * Kd((r-j)*h), solve T @ c = e_center for c.
    Returns (c_list_mp, info_dict).
    """
    import mpmath as mp
    mp.mp.dps = int(mp_dps)

    n = 2 * Kc + 1
    idx = list(range(-Kc, Kc + 1))

    h_mp = mp.mpf(h)

    # Precompute Toeplitz values for all differences d in [-2Kc, 2Kc]
    dvals = {}
    for d in range(-2 * Kc, 2 * Kc + 1):
        x = mp.mpf(d) * h_mp
        dvals[d] = h_mp * _Kd_mp(x, gamma, mp)

    # Build dense Toeplitz matrix
    T = mp.matrix(n, n)
    for i, r in enumerate(idx):
        for j, jj in enumerate(idx):
            T[i, j] = dvals[r - jj]

    # RHS: unit vector at center
    b = mp.matrix(n, 1)
    for i in range(n):
        b[i] = mp.mpf("0")
    b[Kc] = mp.mpf(h)

    # Solve
    c = mp.lu_solve(T, b)
    c_list = [c[i] for i in range(n)]

    # Residual
    rvec = T * c - b
    resid_inf = max(abs(rvec[i]) for i in range(n))

    info = {
        "toeplitz_resid_inf": float(resid_inf),
    }
    return c_list, info


def _build_a_mpmath_kahan(*, gprime, h, gamma, N, halo, Kc, c_mp, mp_dps, sample_dtype=np.float64):
    """Compute outer weights a_n via convolution with Kahan summation in mpmath.

    a_n = sum_{k=-Kc..Kc} c_k * g'(x_{n-k})
    for n in [-halo, N+halo].
    """
    import mpmath as mp
    mp.mp.dps = int(mp_dps)

    h_mp = mp.mpf(h)

    # Sample g' on extended grid
    jmin = -(halo + Kc)
    jmax = N + halo + Kc
    y = {}
    for j in range(jmin, jmax + 1):
        xj = mp.mpf(-1) + mp.mpf(j) * h_mp
        y_samp = _sample_scalar(gprime, xj, sample_dtype)
        y[j] = mp.mpf(y_samp)

    k_list = list(range(-Kc, Kc + 1))
    n_idx = list(range(-halo, N + halo + 1))

    a_mp = []
    for n in n_idx:
        seg = [y[n - k] for k in k_list]
        a_n = _mp_kahan_dot(c_mp, seg, mp)
        a_mp.append(a_n)

    return n_idx, a_mp


def _compute_c0_mpmath(*, g, x0, x_centers, a_mp, gamma, mp_dps, sample_dtype=np.float64):
    """Compute bias c0 in mpmath so that q(x0) = g(x0)."""
    import mpmath as mp
    mp.mp.dps = int(mp_dps)

    gx0 = mp.mpf(_sample_scalar(g, x0, sample_dtype))
    s = mp.mpf("0")
    x0_mp = mp.mpf(x0)
    gamma_mp = mp.mpf(gamma)

    for an, xn in zip(a_mp, x_centers):
        xn_mp = mp.mpf(float(xn))
        s += an * mp.tanh(gamma_mp * (x0_mp - xn_mp))

    return gx0 - s


# ---------------------------------------------------------------------------
# Cardinal-coefficient cache (target-independent)
# ---------------------------------------------------------------------------
#
# The cardinal coefficients c_j depend only on (lambda_star, Kc, N, precision,
# mp_dps). They do NOT depend on the target function. Since mpmath computation
# takes ~55s per call, caching them once per (lambda, Kc, N, precision) pays
# off enormously when running experiments across many targets/seeds.
#
# Convolution with g'(x_k) and bias c0 are cheap and never cached.

def _cache_key(*, lambda_star: float, Kc: int, N: int,
               precision: str, mp_dps: int) -> str:
    payload = f"{precision}|lam={lambda_star:.12f}|Kc={Kc}|N={N}|dps={mp_dps}"
    h = hashlib.sha1(payload.encode()).hexdigest()[:12]
    ext = "txt" if precision == "mpmath" else "npz"
    return f"c_{precision}_lam{lambda_star:.4f}_Kc{Kc}_N{N}_dps{mp_dps}_{h}.{ext}"


def _load_cached_c_f64(path: Path) -> Optional[tuple[np.ndarray, dict]]:
    """Load fp64 cardinal coefficients from npz cache."""
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return data["c"].astype(np.float64), {
            "toeplitz_resid_inf": float(data["resid"]),
            "cached": True,
        }
    except Exception:
        return None


def _save_cached_c_f64(path: Path, c: np.ndarray, resid: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, c=np.asarray(c, dtype=np.float64), resid=float(resid))


def _load_cached_c_mpmath(path: Path, mp_dps: int):
    """Load mpmath cardinal coefficients from text cache (full precision)."""
    if not path.exists():
        return None
    try:
        import mpmath as mp
        mp.mp.dps = int(mp_dps)
        with open(path) as f:
            header = f.readline().strip()
            resid = float(header.split("=")[1])
            c_list = [mp.mpf(line.strip()) for line in f if line.strip()]
        return c_list, {"toeplitz_resid_inf": resid, "cached": True}
    except Exception:
        return None


def _save_cached_c_mpmath(path: Path, c_mp: list, resid: float, mp_dps: int) -> None:
    """Serialize mpmath coefficients as strings with full precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import mpmath as mp
    with mp.workdps(int(mp_dps)):
        with open(path, "w") as f:
            f.write(f"resid={resid:.18e}\n")
            for v in c_mp:
                f.write(mp.nstr(v, int(mp_dps)) + "\n")


def construct_qi(
    target_fn: Callable[[float], float],
    target_deriv: Callable[[float], float],
    N: int,
    *,
    precision: Literal["fp64", "mpmath"] = "fp64",
    lambda_star: Optional[float] = None,
    Kc: int = 160,
    halo: Optional[int] = None,
    mp_dps: int = 30,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> QIResult:
    """Construct a QI MLP approximation from a target function.

    Two precision regimes are supported (see papers/practical_implementation.tex
    for the full nuance):

    - ``precision="fp64"`` (default, fast, ~10ms per call):
          Uses numpy.linalg.solve for the Toeplitz system and Python-level
          Kahan summation for the convolution. Achieves L_inf ~ 1e-12 on
          smooth targets. Limited by cancellation error in the convolution
          (c_j has magnitudes ~300 with alternating signs).
          Defaults for this path: lambda_star=0.30.

    - ``precision="mpmath"`` (slow, ~55s per call, reaches machine epsilon):
          Uses mpmath arbitrary-precision for Toeplitz solve AND convolution.
          Achieves L_inf ~ 2-6e-15 (true fp64 machine epsilon).
          Required because at the smaller lambda needed for machine-eps
          precision, the Toeplitz matrix becomes fp64-ill-conditioned.
          Defaults for this path: lambda_star=0.25.

    The cardinal coefficients c_j (the expensive part) are cached to disk
    keyed by (lambda_star, Kc, N, precision, mp_dps). Since c_j does NOT
    depend on the target function, one construction amortizes across all
    targets and seeds for a given (N, lambda) configuration.

    Args:
        target_fn:    Target function g(x). Callable on floats.
        target_deriv: Derivative g'(x). Callable on floats.
        N:            Number of interior grid intervals.
        precision:    "fp64" or "mpmath" (see docstring above).
        lambda_star:  Dimensionless bandwidth gamma*h. If None, uses the
                      default for the chosen precision (0.30 for fp64, 0.25
                      for mpmath).
        Kc:           Toeplitz stencil half-width (default 160, matches
                      continuous-mlps).
        halo:         Ghost nodes on each side. If None, uses default_halo(N)
                      = max(ceil(35/(2*lambda)), int(0.4*N)).
        mp_dps:       mpmath decimal places when precision="mpmath" (default
                      30 is enough; larger values don't change the result).
        cache_dir:    Directory for c_j cache. Defaults to results/qi_cache.
        use_cache:    If True (default), load and save c_j to disk.

    Returns:
        QIResult with all construction data (fp64 coefficients either way).
    """
    # Apply precision-specific defaults
    if lambda_star is None:
        lambda_star = 0.25 if precision == "mpmath" else 0.30
    if halo is None:
        halo = default_halo(N, lambda_star=lambda_star)

    h = 2.0 / N
    gamma = lambda_star / h

    # ---- Step 1: cardinal coefficients c_j (cacheable) ----
    cache_dir_p = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_path = cache_dir_p / _cache_key(
        lambda_star=lambda_star, Kc=Kc, N=N,
        precision=precision, mp_dps=mp_dps,
    )
    toe_info: dict = {}
    c_f64_array: Optional[np.ndarray] = None
    c_mp_list: Optional[list] = None

    if use_cache:
        if precision == "mpmath":
            hit_mp = _load_cached_c_mpmath(cache_path, mp_dps)
            if hit_mp is not None:
                c_mp_list, toe_info = hit_mp
                c_f64_array = np.array([float(v) for v in c_mp_list],
                                        dtype=np.float64)
        else:
            hit_f = _load_cached_c_f64(cache_path)
            if hit_f is not None:
                c_f64_array, toe_info = hit_f

    if c_f64_array is None:
        if precision == "mpmath":
            c_mp_list, toe_info = _build_toeplitz_c_mpmath(
                h=h, gamma=gamma, Kc=Kc, mp_dps=mp_dps,
            )
            c_f64_array = np.array([float(v) for v in c_mp_list],
                                    dtype=np.float64)
            toe_info["cached"] = False
            if use_cache:
                _save_cached_c_mpmath(
                    cache_path, c_mp_list,
                    toe_info.get("toeplitz_resid_inf", 0.0), mp_dps,
                )
        else:
            c_list, toe_info = _build_toeplitz_c_f64(h=h, gamma=gamma, Kc=Kc)
            c_f64_array = np.array(c_list, dtype=np.float64)
            toe_info["cached"] = False
            if use_cache:
                _save_cached_c_f64(
                    cache_path, c_f64_array,
                    toe_info.get("toeplitz_resid_inf", 0.0),
                )

    # ---- Step 2: convolve with target derivative (cheap, per-target) ----
    if precision == "mpmath":
        assert c_mp_list is not None
        n_idx, a_mp = _build_a_mpmath_kahan(
            gprime=target_deriv, h=h, gamma=gamma,
            N=N, halo=halo, Kc=Kc, c_mp=c_mp_list, mp_dps=mp_dps,
        )
        n_idx_np = np.array(n_idx, dtype=np.int64)
        centers = -1.0 + n_idx_np.astype(np.float64) * h
        c0_mp = _compute_c0_mpmath(
            g=target_fn, x0=-1.0, x_centers=centers,
            a_mp=a_mp, gamma=gamma, mp_dps=mp_dps,
        )
        a_coeffs = np.array([float(v) for v in a_mp], dtype=np.float64)
        c0 = float(c0_mp)
    else:
        n_idx, a_f64 = _build_a_f64(
            gprime=target_deriv, h=h, gamma=gamma,
            N=N, halo=halo, Kc=Kc, c_f64=c_f64_array,
        )
        n_idx_np = np.array(n_idx, dtype=np.int64)
        centers = -1.0 + n_idx_np.astype(np.float64) * h
        c0 = _compute_c0_f64(
            g=target_fn, x0=-1.0, x_centers=centers,
            a_f64=a_f64, gamma=gamma,
        )
        a_coeffs = np.array(a_f64, dtype=np.float64)

    # Extract interior (non-halo) portion
    interior_mask = (n_idx_np >= 0) & (n_idx_np <= N)
    interior_centers = centers[interior_mask]
    interior_a_coeffs = a_coeffs[interior_mask]

    return QIResult(
        N=N,
        h=h,
        gamma=gamma,
        lambda_val=lambda_star,
        centers=centers,
        interior_centers=interior_centers,
        a_coeffs=a_coeffs,
        interior_a_coeffs=interior_a_coeffs,
        c0=c0,
        toeplitz_residual=toe_info.get("toeplitz_resid_inf", 0.0),
        halo=halo,
        Kc=Kc,
    )


def evaluate_qi(qi_result: QIResult, x: np.ndarray, kahan: bool = False) -> np.ndarray:
    """Evaluate the QI interpolant at points x.

    q(x) = c0 + sum_n a_n * tanh(gamma * (x - x_n))

    Args:
        qi_result: Construction output.
        x: Evaluation points, shape [n_points].
        kahan: If True, use Kahan compensated summation (slower but more
            accurate; reduces evaluation roundoff floor by ~1-2 orders of magnitude).

    Returns:
        q(x), shape [n_points].
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    z = qi_result.gamma * (x[:, None] - qi_result.centers[None, :])
    terms = qi_result.a_coeffs[None, :] * np.tanh(z)

    if kahan:
        # Kahan compensated summation along axis=1
        total = np.zeros(x.shape[0], dtype=np.float64)
        comp = np.zeros(x.shape[0], dtype=np.float64)
        for k in range(terms.shape[1]):
            y = terms[:, k] - comp
            t = total + y
            comp = (t - total) - y
            total = t
        return qi_result.c0 + total

    return qi_result.c0 + np.sum(terms, axis=1)
