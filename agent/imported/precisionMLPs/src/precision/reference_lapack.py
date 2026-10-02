"""Build and call netlib reference LAPACK 3.12.1 DGELSS / SGELSS, the anchors for the p-bit port.

The library is compiled from the unmodified netlib Fortran (src/precision/_build/lapack-3.12.1,
fetched from https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz) with
gfortran and ``-ffp-contract=off`` (no fused multiply-add), using only the files DGELSS and SGELSS
reach. One file is changed: ILAENV returns block size 1, the unblocked configuration the port
implements (blocking only reorders the same operations).
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import re
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "_build" / "lapack-3.12.1"
TARBALL_SHA256 = "2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422"
FFLAGS = ["-O2", "-ffp-contract=off", "-fPIC"]
_LIB = None
_SEARCH = ["SRC", "BLAS/SRC", "INSTALL"]
_EXT = [".f", ".f90", ".F90", ".F"]


def _find(symbol: str) -> Path | None:
    for d in _SEARCH:
        for ext in _EXT:
            path = SOURCE / d / f"{symbol}{ext}"
            if path.exists():
                return path
    return None


def _patched_ilaenv(build: Path) -> Path:
    text = (SOURCE / "SRC" / "ilaenv.f").read_text()
    anchor = "      GO TO ( 10, 10, 10, 80, 90, 100, 110, 120,"
    assert text.count(anchor) == 1
    patch = ("*     expC11: unblocked configuration (block size 1 for every routine).\n"
             "      IF( ISPEC.EQ.1 ) THEN\n"
             "         ILAENV = 1\n"
             "         RETURN\n"
             "      END IF\n")
    out = build / "ilaenv.f"
    out.write_text(text.replace(anchor, patch + anchor))
    return out


def build() -> Path:
    """Compile the DGELSS/SGELSS closure into a shared library and return its path."""
    tarball = SOURCE.parent / "lapack-3.12.1.tar.gz"
    if not SOURCE.exists():
        raise FileNotFoundError(f"reference LAPACK source missing: {SOURCE}")
    if tarball.exists():
        assert hashlib.sha256(tarball.read_bytes()).hexdigest() == TARBALL_SHA256
    tag = hashlib.sha256((" ".join(FFLAGS) + Path(__file__).read_text()).encode()).hexdigest()[:12]
    build_dir = SOURCE.parent / f"reflapack_{tag}"
    lib = build_dir / "libreflapack.dylib"
    if lib.exists():
        return lib
    build_dir.mkdir(exist_ok=True)
    fc = ["gfortran", *FFLAGS, "-J", str(build_dir), "-c"]
    for module in ("la_constants.f90", "la_xisnan.F90"):
        subprocess.run([*fc, str(SOURCE / "SRC" / module), "-o", str(build_dir / (module + ".o"))], check=True)
    objects = {"la_constants": build_dir / "la_constants.f90.o", "la_xisnan": build_dir / "la_xisnan.F90.o"}
    queue, seen = ["dgelss", "sgelss"], set()
    while queue:
        sym = queue.pop()
        if sym in seen:
            continue
        seen.add(sym)
        src = _patched_ilaenv(build_dir) if sym == "ilaenv" else _find(sym)
        if src is None:
            continue  # a runtime symbol (libgfortran / libm), resolved at link time
        obj = build_dir / f"{sym}.o"
        subprocess.run([*fc, str(src), "-o", str(obj)], check=True)
        objects[sym] = obj
        undefined = subprocess.run(["nm", "-u", str(obj)], check=True, capture_output=True, text=True).stdout
        for name in re.findall(r"^_([a-z][a-z0-9_]*)_$", undefined, flags=re.M):
            if name not in seen:
                queue.append(name)
    subprocess.run(["gfortran", "-shared", "-o", str(lib), *map(str, objects.values())], check=True)
    return lib


def _load():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(build()))
    return _LIB


def gelss(a: np.ndarray, b: np.ndarray, rcond: float, dtype) -> dict:
    """Reference xGELSS on A (m x n) and b (m): returns x, singular values, rank, info."""
    lib = _load()
    fn = lib.dgelss_ if dtype == np.float64 else lib.sgelss_
    ct = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    a = np.asfortranarray(a, dtype=dtype).copy(order="F")
    m, n = a.shape
    bb = np.zeros(max(m, n), dtype=dtype)
    bb[:m] = b
    s = np.zeros(min(m, n), dtype=dtype)
    i = ctypes.c_int
    rank, info = i(0), i(0)
    work_query = np.zeros(1, dtype=dtype)
    args = lambda work, lwork: (ctypes.byref(i(m)), ctypes.byref(i(n)), ctypes.byref(i(1)),
                                a.ctypes.data_as(ctypes.POINTER(ct)), ctypes.byref(i(m)),
                                bb.ctypes.data_as(ctypes.POINTER(ct)), ctypes.byref(i(max(m, n))),
                                s.ctypes.data_as(ctypes.POINTER(ct)), ctypes.byref(ct(rcond)),
                                ctypes.byref(rank), work.ctypes.data_as(ctypes.POINTER(ct)),
                                ctypes.byref(i(lwork)), ctypes.byref(info))
    fn(*args(work_query, -1))
    lwork = int(work_query[0])
    work = np.zeros(max(lwork, 1), dtype=dtype)
    a = np.asfortranarray(a)
    fn(*args(work, lwork))
    return {"x": bb[:n].astype(np.float64), "sigma": s.astype(np.float64), "rank": rank.value,
            "info": info.value}


def machine_constants() -> dict:
    """What gfortran computes for the LAPACK machine parameters, in both kinds (for checking the
    port's constants). Compiles and runs a tiny Fortran program."""
    prog = HERE / "_build" / "machine_constants.f90"
    prog.write_text("""program mc
  use la_constants, only: dsafmin, dsafmax, ssafmin, ssafmax
  implicit none
  double precision :: dlamch, deps
  real :: slamch, seps
  external dlamch, slamch
  deps = dlamch('Epsilon')
  seps = slamch('Epsilon')
  write(*, '(a, 20(1x, es25.17e3))') 'D', dlamch('E'), dlamch('P'), dlamch('S'), dlamch('O'), &
    dsafmin, dsafmax, &
    2d0**ceiling((minexponent(0d0)-1)*0.5d0), 2d0**floor((maxexponent(0d0)-digits(0d0)+1)*0.5d0), &
    2d0**(-floor((minexponent(0d0)-digits(0d0))*0.5d0)), &
    2d0**(-ceiling((maxexponent(0d0)+digits(0d0)-1)*0.5d0)), deps**(-0.125d0), 0.01d0, &
    sqrt(dsafmin), sqrt(dsafmax/2)
  write(*, '(a, 20(1x, es25.17e3))') 'S', dble(slamch('E')), dble(slamch('P')), dble(slamch('S')), &
    dble(slamch('O')), dble(ssafmin), dble(ssafmax), &
    dble(2e0**ceiling((minexponent(0e0)-1)*0.5e0)), dble(2e0**floor((maxexponent(0e0)-digits(0e0)+1)*0.5e0)), &
    dble(2e0**(-floor((minexponent(0e0)-digits(0e0))*0.5e0))), &
    dble(2e0**(-ceiling((maxexponent(0e0)+digits(0e0)-1)*0.5e0))), dble(seps**(-0.125e0)), dble(0.01e0), &
    dble(sqrt(ssafmin)), dble(sqrt(ssafmax/2))
end program
""")
    lib_path = build()
    exe = prog.with_suffix("")
    subprocess.run(["gfortran", *FFLAGS, "-I", str(lib_path.parent), str(prog), str(lib_path),
                    "-o", str(exe), f"-Wl,-rpath,{lib_path.parent}"], check=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    keys = ["eps", "prec", "sfmin", "huge", "safmin", "safmax", "tsml", "tbig", "ssml", "sbig",
            "epspow", "hndrth", "rtmin", "rtmax"]
    result = {}
    for line in out.splitlines():
        tag, *vals = line.split()
        result["fp64" if tag == "D" else "fp32"] = dict(zip(keys, map(float, vals)))
    return result
