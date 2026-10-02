"""p-bit tanh model and least-squares solve with no higher-precision leakage.

The computation is specified in experiments/expC11_true_precision_law/SPEC.md and implemented
once in pbit_algo.h (model, tanh) and lapack_gelss.h (reference LAPACK 3.12.1 DGELSS, ported
statement by statement). It is compiled three ways:

- ``emul``: binary64 carrier; every +, -, *, /, sqrt correctly rounded in a binary format with a
  p-bit significand, exponent range [emin, emax] and gradual underflow;
- ``f64``, ``f32``: the same code on native binary64 / binary32 hardware.

A format is ``(p, emin, emax)``. ``FP64`` and ``FP32`` are the IEEE formats; ``panel_format(p)``
is the format of the precision sweep. Python does no model arithmetic: it prepares the data and
the stored constants (each rounded once to the format) and reads results back.
"""
from __future__ import annotations

import ctypes
from fractions import Fraction
import hashlib
from math import ceil, factorial, floor
from pathlib import Path
import subprocess

import mpmath
import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE / "_build"

FP64 = (53, -1022, 1023)
FP32 = (24, -126, 127)
# The sweep's formats keep binary64's exponent field but pull its range in by 64 binades at each
# end, so every value of every emulated format (subnormals included) is a normal binary64 number.
PANEL_EMIN, PANEL_EMAX = -958, 959
NATIVE = {"f64": FP64, "f32": FP32}
_NATIVE_TYPE = {"f64": 1, "f32": 2}

FLAGS = ["-O2", "-std=c11", "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared"]
_LIBS: dict[str, ctypes.CDLL] = {}
_d = ctypes.POINTER(ctypes.c_double)
_i = ctypes.POINTER(ctypes.c_int)
_z = ctypes.c_size_t
_c = ctypes.c_int


def panel_format(p: int) -> tuple[int, int, int]:
    return (p, PANEL_EMIN, PANEL_EMAX)


def _lib(backend: str) -> ctypes.CDLL:
    if backend in _LIBS:
        return _LIBS[backend]
    if backend != "emul" and backend not in NATIVE:
        raise ValueError(f"Unknown backend {backend!r}")
    source = HERE / ("pbit_emul.c" if backend == "emul" else "pbit_native.c")
    flags = list(FLAGS)
    if backend != "emul":
        # Native builds also avoid auto-vectorization, so each loop runs the scalar code as written.
        flags += [f"-DPBIT_TYPE={_NATIVE_TYPE[backend]}", "-fno-vectorize", "-fno-slp-vectorize"]
    text = b"".join((HERE / f).read_bytes() for f in (source.name, "pbit_algo.h", "lapack_gelss.h"))
    tag = hashlib.sha256(text + " ".join(flags).encode()).hexdigest()[:16]
    BUILD.mkdir(exist_ok=True)
    so = BUILD / f"pbit_{backend}_{tag}.so"
    if not so.exists():
        tmp = so.with_suffix(".tmp.so")
        subprocess.run(["cc", *flags, "-I", str(HERE), str(source), "-o", str(tmp), "-lm"], check=True)
        tmp.replace(so)
    lib = ctypes.CDLL(str(so))
    lib.pbit_run.argtypes = [_d, _z, _d, _d, _z, _d, ctypes.c_double, _z, _d, ctypes.c_double,
                             _z, _d, _d, _d, _i, _d]
    lib.pbit_tanh.argtypes = [_d, _z, _d, _d]
    lib.pbit_eval.argtypes = [_d, _z, _d, ctypes.c_double, _z, _d, ctypes.c_double, _d, _d]
    lib.pbit_features.argtypes = [_d, _z, _d, ctypes.c_double, _z, _d, ctypes.c_double, _d]
    lib.pbit_op.argtypes = [_c, _c, _c, _c, _z, _d, _d, _d]
    lib.pbit_gelss.argtypes = [_d, _c, _c, _d, _d, _c, _d, _d, _i, _d]
    if backend == "emul":
        lib.pbit_round.argtypes = [_c, _c, _c, _z, _d, _d]
        lib.pbit_counters.argtypes = [ctypes.POINTER(ctypes.c_long)]
    _LIBS[backend] = lib
    return lib


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_d)


def _arr(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


# ---------------------------------------------------------------- exact constants

def round_fraction(value, p: int, emin: int | None = None) -> float:
    """Round an exact rational to the format: p significant bits, nearest, ties to even; below
    2^emin (when given) the quantum is fixed at 2^(emin - p + 1) (gradual underflow)."""
    value = Fraction(value)
    if value == 0:
        return 0.0
    sign, a = (-1 if value < 0 else 1), abs(value)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if emin is not None and e < emin:
        e = emin
    scaled = a * Fraction(2) ** (p - 1 - e)
    m, rem = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * rem
    if twice > scaled.denominator or (twice == scaled.denominator and m % 2):
        m += 1
    out = sign * Fraction(m) * Fraction(2) ** (e - p + 1)
    f = float(out)
    assert Fraction(f) == out
    return f


def _round_real(x: mpmath.mpf, p: int, guard: int = 1500) -> float:
    """Round a real known to 2000 bits; refuse if it is within 2^-guard of a rounding midpoint."""
    sign, man, exp, _ = mpmath.mpf(x)._mpf_
    fr = (-1) ** sign * Fraction(man) * Fraction(2) ** exp
    lo = round_fraction(fr * (1 - Fraction(1, 2 ** guard)), p)
    hi = round_fraction(fr * (1 + Fraction(1, 2 ** guard)), p)
    assert lo == hi, "constant too close to a rounding midpoint"
    return lo


def constants(fmt: tuple[int, int, int]) -> dict:
    """The stored constants of the spec for format (p, emin, emax), each a format value.

    The LAPACK entries reproduce what reference LAPACK computes for a Fortran real kind with
    DIGITS = p, MINEXPONENT = emin + 1, MAXEXPONENT = emax + 1 (DLAMCH, LA_CONSTANTS, DNRM2), and
    EPS**(-0.125) of DBDSQR as a correctly rounded power.
    """
    p, emin, emax = fmt
    with mpmath.workprec(2000):
        ln2 = mpmath.log(2)
        kb = (p + 4).bit_length()
        ln2hi = _round_real(ln2, max(p - kb, 1))
        ln2lo = _round_real(ln2 - mpmath.mpf(ln2hi), p)
        invln2 = _round_real(1 / ln2, p)
        sat = _round_real((p + 3) * ln2 / 2, p)
        epspow = _round_real(mpmath.mpf(2) ** (mpmath.mpf(p) / 8), p)
    d = 1
    while Fraction(35, 100) ** d / factorial(d + 1) > Fraction(1, 2 ** (p + 1)):
        d += 1
    coeffs = [round_fraction(Fraction(1, factorial(j + 1)), p) for j in range(d)]
    two = Fraction(2)
    tiny = two ** emin
    huge = (2 - two ** (1 - p)) * two ** emax
    eps = two ** -p
    small = round_fraction(1 / huge, p, emin)
    sfmin = tiny
    if small >= tiny:
        sfmin = round_fraction(Fraction(small) * Fraction(round_fraction(1 + eps, p, emin)), p, emin)
    safmin_e = max(emin, -emax)
    return {
        "format": fmt, "INVLN2": invln2, "LN2HI": ln2hi, "LN2LO": ln2lo, "SAT": sat,
        "coeffs": coeffs,
        "eps": float(eps), "prec": float(two ** (1 - p)), "sfmin": float(sfmin), "huge": float(huge),
        "safmin": float(two ** safmin_e), "safmax": float(two ** -safmin_e),
        "tsml": float(two ** ceil(emin * 0.5)),
        "tbig": float(two ** floor((emax + 1 - p + 1) * 0.5)),
        "ssml": float(two ** -floor((emin + 1 - p) * 0.5)),
        "sbig": float(two ** -ceil((emax + 1 + p - 1) * 0.5)),
        "epspow": epspow, "hndrth": round_fraction(Fraction(1, 100), p, emin),
    }


def _pack(c: dict) -> np.ndarray:
    p, emin, emax = c["format"]
    return _arr([p, emin, emax, c["INVLN2"], c["LN2HI"], c["LN2LO"], c["SAT"],
                 c["eps"], c["prec"], c["sfmin"], c["huge"], c["safmin"], c["safmax"],
                 c["tsml"], c["tbig"], c["ssml"], c["sbig"], c["epspow"], c["hndrth"],
                 len(c["coeffs"]), *c["coeffs"]])


# ---------------------------------------------------------------- entry points

def _check_backend(fmt, backend):
    fmt = tuple(fmt)
    if backend != "emul" and fmt != NATIVE[backend]:
        raise ValueError(f"backend {backend} runs only in format {NATIVE[backend]}")
    return fmt


def counters() -> dict:
    """Emulator event counters since the last read (non-finite, non-format inputs, overflows,
    results in the subnormal range)."""
    buf = (ctypes.c_long * 4)()
    _lib("emul").pbit_counters(buf)
    return {"nonfinite": buf[0], "not_format_input": buf[1], "overflow": buf[2],
            "subnormal": buf[3]}


def round_p(x, fmt) -> np.ndarray:
    """Round binary64 values to the format (nearest, ties to even)."""
    x = _arr(x)
    out = np.empty_like(x)
    if _lib("emul").pbit_round(*fmt, x.size, _ptr(x), _ptr(out)):
        raise ValueError(f"unsupported format {fmt}")
    return out


def op(fmt, name: str, a, b=None, backend: str = "emul") -> np.ndarray:
    """One rounded primitive, elementwise (for tests)."""
    fmt = _check_backend(fmt, backend)
    code = {"add": 0, "sub": 1, "mul": 2, "div": 3, "sqrt": 4}[name]
    a = _arr(a)
    b = _arr(a if b is None else b)
    out = np.empty_like(a)
    if _lib(backend).pbit_op(*fmt, code, a.size, _ptr(a), _ptr(b), _ptr(out)):
        raise ValueError("bad format or op")
    return out


def tanh(z, fmt, backend: str = "emul") -> np.ndarray:
    """tanh_p of the spec, elementwise; z must already be format values."""
    fmt = _check_backend(fmt, backend)
    z = _arr(z)
    out = np.empty_like(z)
    if backend == "emul":
        counters()
    if _lib(backend).pbit_tanh(_ptr(_pack(constants(fmt))), z.size, _ptr(z), _ptr(out)):
        raise RuntimeError("pbit_tanh failed")
    if backend == "emul":
        c = counters()
        assert c["nonfinite"] == 0 and c["not_format_input"] == 0, c
    return out


def _geometry_inputs(N, halo, lam, fmt):
    W = N + 2 * halo + 1
    jr = round_p(np.arange(-halo, N + halo + 1, dtype=np.float64), fmt)
    return W, jr, float(round_p([N], fmt)[0]), float(round_p([lam], fmt)[0])


def features(x, N: int, halo: int, lam: float, fmt, backend: str = "emul") -> np.ndarray:
    """The M x W feature matrix phi_ij = tanh_p(gamma (x_i - c_j)) at the format."""
    fmt = _check_backend(fmt, backend)
    xr = round_p(x, fmt)
    W, jr, Nr, lamr = _geometry_inputs(N, halo, lam, fmt)
    phi = np.empty((xr.size, W))
    if _lib(backend).pbit_features(_ptr(_pack(constants(fmt))), xr.size, _ptr(xr), Nr, W,
                                   _ptr(jr), lamr, _ptr(phi)):
        raise RuntimeError("pbit_features failed")
    return phi


def evaluate(weights, x_eval, N: int, halo: int, lam: float, fmt, backend: str = "emul") -> np.ndarray:
    """The model output at the format for given weights (bias last)."""
    fmt = _check_backend(fmt, backend)
    xer = round_p(x_eval, fmt)
    W, jr, Nr, lamr = _geometry_inputs(N, halo, lam, fmt)
    w = _arr(weights)
    assert w.size == W + 1 and np.array_equal(round_p(w, fmt), w)
    fit = np.empty(xer.size)
    if _lib(backend).pbit_eval(_ptr(_pack(constants(fmt))), xer.size, _ptr(xer), Nr, W, _ptr(jr),
                               lamr, _ptr(w), _ptr(fit)):
        raise RuntimeError("pbit_eval failed")
    return fit


def gelss(a, b, fmt, rconds, backend: str = "emul") -> dict:
    """The ported DGELSS on a given matrix and right-hand side (both already format values)."""
    fmt = _check_backend(fmt, backend)
    a = np.asarray(a, dtype=np.float64)
    m, n = a.shape
    af = _arr(a.T.ravel())  # column-major
    b = _arr(b)
    rc = _arr(rconds)
    assert np.array_equal(round_p(af, fmt), af) and np.array_equal(round_p(b, fmt), b)
    assert np.array_equal(round_p(rc, fmt), rc)
    x, s = np.empty((rc.size, n)), np.empty(n)
    ranks = np.zeros(rc.size, dtype=np.intc)
    if backend == "emul":
        counters()
    status = _lib(backend).pbit_gelss(_ptr(_pack(constants(fmt))), m, n, _ptr(af), _ptr(b), rc.size,
                                      _ptr(rc), _ptr(x), ranks.ctypes.data_as(_i), _ptr(s))
    return {"status": status, "x": x, "sigma": s, "rank": ranks.tolist(),
            "events": counters() if backend == "emul" else None}


def run(x, y, x_eval, N: int, halo: int, lam: float, fmt, backend: str = "emul",
        rconds=None, strict: bool = True) -> dict:
    """Round the data to the format, then run the model and the DGELSS solve in its arithmetic.

    x, y, x_eval are binary64 data; lam is the offline bandwidth. ``rconds`` are DGELSS cutoffs
    (default [2^(1-p)]), each applied to the one decomposition. Returns the weights (bias last),
    outputs on x_eval, singular values (descending) and ranks, per cutoff in the ``*_all`` keys
    and for the first cutoff in the plain keys.
    """
    fmt = _check_backend(fmt, backend)
    p = fmt[0]
    xr, yr, xer = round_p(x, fmt), round_p(y, fmt), round_p(x_eval, fmt)
    W, jr, Nr, lamr = _geometry_inputs(N, halo, lam, fmt)
    n = W + 1
    rc = round_p([2.0 ** (1 - p)] if rconds is None else rconds, fmt)
    weights, sigma = np.empty((rc.size, n)), np.empty(n)
    fit = np.empty((rc.size, xer.size))
    ranks = np.zeros(rc.size, dtype=np.intc)
    lib = _lib(backend)
    if backend == "emul":
        counters()
    status = lib.pbit_run(_ptr(_pack(constants(fmt))), xr.size, _ptr(xr), _ptr(yr), xer.size,
                          _ptr(xer), Nr, W, _ptr(jr), lamr, rc.size, _ptr(rc), _ptr(weights),
                          _ptr(fit), ranks.ctypes.data_as(_i), _ptr(sigma))
    events = counters() if backend == "emul" else None
    if status:
        raise RuntimeError(f"pbit_run failed with status {status}")
    if events and (events["not_format_input"] or (strict and (events["nonfinite"] or events["overflow"]))):
        raise RuntimeError(f"left the format: {events}")
    return {"format": fmt, "backend": backend, "weights": weights[0], "fit": fit[0],
            "rank": int(ranks[0]), "weights_all": weights, "fit_all": fit,
            "rank_all": ranks.tolist(), "rconds": rc, "sigma": sigma, "lambda_p": lamr,
            "events": events}
