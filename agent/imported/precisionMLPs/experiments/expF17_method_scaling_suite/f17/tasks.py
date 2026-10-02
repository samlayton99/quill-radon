"""expF17 task registry: 8 tasks, machine-precision oracles, one coordinate frame.

Every task is posed in scaled P = (xi, eta) in [-1,1]^2 (SPEC 13.3); the physical
map is folded into the operator coefficients. xi = space, eta = time (or y).
The five BWLer tasks + poisson_man are imported from expF13/problems.py (their
definitions are FD-verified there, incl. the Cole-Hopf Burgers oracle); darcy_man
imports expF08's FD-verified manufactured control; darcy_orig loads the FNO
darcy_421 test set if a local copy exists (reference-capped, oracle arm only).

Task dict fields (superset of the expF13 format):
  key, family, title            family drives the plot colour cue
  exact(P) | None               the oracle on scaled coords
  lin_terms, nl, bc_blocks      dynamic-arm operator (expF13/expF02 format)
  forcing                       rhs (scalar or callable)
  mask(P) | None                domain mask (poisson_man holes)
  periodic_fourier: bool        spectral arm uses Fourier on the xi axis
                                (True ONLY for convection, where the solution is
                                analytically periodic; reaction is only value-
                                periodic with a C^1 seam, and Chebyshev converges
                                geometrically on the closed square, so it stays
                                Chebyshev x Chebyshev)
  eval_grid()                   -> (P_eval [n,2], u_true [n])
  dynamic: bool                 False = oracle arm only (darcy_orig: coefficient
                                field is piecewise constant, strong-form
                                collocation with smooth dictionaries undefined)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
_F13 = REPO_ROOT / "experiments" / "expF13_bwler_suite"
_F08 = REPO_ROOT / "experiments" / "expF08_darcy_sweep"
for _p in (str(_F13), str(_F08)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import problems as f13  # noqa: E402  (expF13 problem suite, FD-verified)
import darcy_problems as f08  # noqa: E402  (expF08 manufactured Darcy, FD-verified)

PI = np.pi

# candidate locations for the FNO darcy_421 test set (first hit wins)
DARCY_NPZ_CANDIDATES = [
    REPO_ROOT / "data" / "darcy_test_421_jax.npz",
    REPO_ROOT.parent.parent / "continuous-mlps" / "data" / "fno_datasets_jax" / "darcy_test_421_jax.npz",
    Path("/scr/cdeng/continuous-mlps/data/fno_datasets_jax/darcy_test_421_jax.npz"),
]
DARCY_INSTANCE = 0  # which test instance darcy_orig uses (recorded, fixed)


def _uniform_grid(n=200):
    g = np.linspace(-1.0, 1.0, n)
    GX, GE = np.meshgrid(g, g)
    return np.stack([GX.ravel(), GE.ravel()], axis=1)


def _from_f13(key, family, periodic_fourier=False, mask=None, eval_n=200):
    prob = f13.PROBLEMS[key]
    exact = prob["exact"]

    def eval_grid():
        P = _uniform_grid(eval_n)
        if mask is not None:
            P = P[mask(P)]
        return P, exact(P)

    return dict(
        key=key, family=family, title=prob["title"], exact=exact,
        lin_terms=prob["lin_terms"], nl=prob["nl"], bc_blocks=prob["bc_blocks"],
        forcing=prob["forcing"], mask=mask, periodic_fourier=periodic_fourier,
        eval_grid=eval_grid, dynamic=True, order=prob["order"],
    )


def _darcy_man():
    # -a lap u - grad a . grad u = f on scaled [-1,1]^2, u* = 0 on the boundary
    exact = f08.control_u
    lin_terms = [((2, 0), lambda P: -f08.control_a(P)),
                 ((0, 2), lambda P: -f08.control_a(P)),
                 ((1, 0), lambda P: -f08.control_ax(P)),
                 ((0, 1), lambda P: -f08.control_ay(P))]

    def eval_grid():
        P = _uniform_grid(200)
        return P, exact(P)

    return dict(
        key="darcy_man", family="darcy",
        title="Darcy control  $-\\nabla\\!\\cdot(a\\nabla u)=f$, smooth $a$ (manufactured)",
        exact=exact, lin_terms=lin_terms,
        nl=dict(fields=[], res=lambda v, p: 0.0, jac=lambda v, p: {}),
        bc_blocks=[dict(where="square", terms=[((0, 0), 1.0)], value=0.0)],
        forcing=f08.control_forcing, mask=None, periodic_fourier=False,
        eval_grid=eval_grid, dynamic=True, order=2,
    )


def darcy_orig_path():
    for cand in DARCY_NPZ_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _darcy_orig():
    path = darcy_orig_path()

    def eval_grid():
        if path is None:
            raise RuntimeError("darcy_orig: no local darcy_test_421_jax.npz "
                               f"(tried {[str(c) for c in DARCY_NPZ_CANDIDATES]})")
        raw = np.load(path)
        # accept either (u, a) arrays or FNO-style keys
        ukey = "u" if "u" in raw else ("y" if "y" in raw else "sol")
        u = np.asarray(raw[ukey], dtype=np.float64)
        if u.ndim == 3:
            u = u[DARCY_INSTANCE]
        n0, n1 = u.shape
        sub = slice(0, None, 2)  # 421 -> 211 per axis
        gx = np.linspace(-1.0, 1.0, n0)[sub]
        gy = np.linspace(-1.0, 1.0, n1)[sub]
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        P = np.stack([GX.ravel(), GY.ravel()], axis=1)
        return P, u[sub, :][:, sub].ravel()

    return dict(
        key="darcy_orig", family="darcy",
        title="Darcy (FNO darcy_421, reference-capped)",
        exact=None, lin_terms=None, nl=None, bc_blocks=None, forcing=None,
        mask=None, periodic_fourier=False, eval_grid=eval_grid,
        dynamic=False, order=2, blocked=(path is None),
    )


def build_tasks():
    tasks = {}
    for key, fam, per in [("convection_c40", "transport", True),
                          ("convection_c80", "transport", True),
                          ("reaction", "reaction", False),
                          ("wave", "wave", False),
                          ("burgers", "burgers", False)]:
        tasks[key] = _from_f13(key, fam, periodic_fourier=per)
    tasks["poisson_man"] = _from_f13("poisson_man", "steady",
                                     mask=f13.in_poisson_domain, eval_n=241)
    tasks["darcy_man"] = _darcy_man()
    tasks["darcy_orig"] = _darcy_orig()
    return tasks


TASKS = build_tasks()
TASK_ORDER = ["convection_c40", "convection_c80", "reaction", "wave",
              "burgers", "poisson_man", "darcy_man", "darcy_orig"]


def verify_oracles(verbose=True):
    """The oracle purity gate: run before any benchmark number is produced.

    expF13.verify_all() FD-checks every closed-form residual, the Cole-Hopf
    IC/BC, and both reference files; expF08.verify_control() FD-checks the
    manufactured-Darcy derivatives and forcing identity."""
    f13.verify_all(verbose=verbose)
    f08.verify_control()
    if verbose:
        print("  verified darcy_man (expF08 control)")
    for key in TASK_ORDER:
        t = TASKS[key]
        if t["exact"] is not None:
            P, u = t["eval_grid"]()
            assert np.all(np.isfinite(u)) and len(P) == len(u)
    if verbose:
        blocked = [k for k in TASK_ORDER if TASKS[k].get("blocked")]
        print(f"oracle gate PASS ({len(TASK_ORDER)} tasks; blocked: {blocked or 'none'})")
