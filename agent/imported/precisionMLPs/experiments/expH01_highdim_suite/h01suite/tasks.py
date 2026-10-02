"""The 80 tasks: 16 in each of ``d = 1..5``.

One ``Task`` is one approximation problem: a target function, a way of drawing the
training points, the centering and scaling that puts the target on a common scale, the
three test sets, the region masks the localized targets need, and a one-line statement
of what the task is there to test.

The 16 tasks per dimension are the same list in every dimension, so a result can be
read across ``d`` without wondering whether the function changed:

     1  even grid          bumps at three very different widths
     2  even grid          slow concentric waves
     3  even grid          fast concentric waves
     4  even grid          polynomial coupling neighbouring coordinates
     5  uniform            a function of a function
     6  uniform            broad radial spike
     7  uniform            narrow radial spike
     8  uniform            step across a sphere
     9  uniform            step across a curved surface  (d = 1: two formulas glued)
    10  uniform            slope flip across a sphere    (d = 1: one-sided kink)
    11  hotspots           fast concentric waves, same function as 3
    12  hotspots           burst of oscillation centered on the densest cluster
    13  hotspots           the same burst, placed away from every cluster
    14  hotspots           product peak
    15  curved sheet       a function of a function       (d = 1: step in the sparse tail)
    16  curved sheet noisy  the same function as 15       (d = 1: narrow spike at the cluster)

Deliberate comparisons built into the list:

* 2 against 3 -- identical geometry, six times the oscillation.
* 3 against 11 -- literally the same function object, only the data changes.
* 12 against 13 -- the same burst of oscillation, once where the data is and once where
  it is not.
* 6 against 7 -- the same shape, one sharp and one broad.
* 15 against 16 -- the same function object on the same sheet, with and without a thin
  layer of noise perpendicular to the sheet.

Tasks that are meant to be the same function share the *same target object*, so their
centered-and-scaled versions agree bit for bit and the only difference is the data.

``d = 1`` substitutions. There is no curved sheet inside a line and a sphere in one
dimension is two points, so four rows are replaced: 9 becomes the two-formula piecewise
target, 10 becomes the one-sided kink (so that both kinds of kink appear somewhere in
the suite), 15 becomes a step out in the sparse tail of the hotspot density, and 16
becomes a narrow spike sitting exactly on the densest cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from h01suite import targets as TG
from h01suite.densities import Density, make_density
from h01suite.normalize import normalize_callable

__all__ = ["Task", "TASKS", "get_task", "task_ids", "tasks_for_dim", "SMOKE_IDS",
           "packet_anchor_margins"]

# The tasks to run before paying for the full campaign: everything in d = 1 and d = 2,
# plus the four d = 3 tasks that carry the comparisons the suite is built around.
SMOKE_IDS = ([f"1.{i}" for i in range(1, 17)] + [f"2.{i}" for i in range(1, 17)]
             + ["3.3", "3.11", "3.12", "3.13"])


def packet_anchor_margins(d: int) -> dict[str, float]:
    """How far the away-from-hotspots burst sits from each cluster, in cluster widths.

    Distances are Euclidean in ``z``; each is divided by that cluster's standard
    deviation. The standard deviation in ``z`` is at most the one quoted in ``x`` (the
    normalized coordinates divide by ``||u_k||_1 >= 1``), so these ratios are lower
    bounds on the true separation in standard deviations.
    """
    a = TG.antialigned_packet_anchor(d)
    means = TG.hotspot_means_z(d)
    sds = {"plus": 0.22, "minus": 0.28, "perp": 0.25}
    return {k: float(np.linalg.norm(a - means[k]) / sds[k]) for k in means}


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """One benchmark problem: target + data geometry + test sets + metadata."""

    id: str
    name: str
    d: int
    density_tag: str
    target: TG.Target
    what_it_tests: str
    density: Density = field(init=False)
    F: Callable[[np.ndarray], np.ndarray] = field(init=False)
    mean_uniform: float = field(init=False)
    sd_uniform: float = field(init=False)

    def __post_init__(self):
        self.density = make_density(self.density_tag, self.d)
        self.F, self.mean_uniform, self.sd_uniform = normalize_callable(self.target, self.d)

    # -- data ------------------------------------------------------------
    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw ``n`` training inputs from this task's data geometry."""
        return self.density.sample(n, seed=seed)

    def train_set(self, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        X = self.sample(n, seed=seed)
        return X, self.F(X)

    def test_sets(self, seed: int = 10_000) -> dict[str, np.ndarray]:
        """``same_as_train``, ``uniform``, ``dense_region``, plus ``on_sheet`` and the
        fixed-distance sets for the sheet geometries."""
        return self.density.test_sets(seed=seed)

    def dense_region_description(self) -> str:
        """Plain-English definition of this task's ``dense_region`` test set."""
        return self.density.dense_region_description()

    def logpdf(self, X: np.ndarray):
        return self.density.logpdf(X)

    # -- the function -----------------------------------------------------
    @property
    def differentiable(self) -> bool:
        """False for the two step families, whose gradient is undefined on the step."""
        return self.target.differentiable

    def grad_F(self, X: np.ndarray) -> np.ndarray:
        """Gradient of the centered-and-scaled target, shape ``[n, d]``.

        For the step families this is the gradient of the smooth part; the points where
        that is wrong are exactly the ones ``interface_mask`` marks.
        """
        return self.target.grad(X) / self.sd_uniform

    def directional_derivative(self, X: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``dF/dv`` at each point, for a direction ``v`` (used as given, not normalized)."""
        return self.grad_F(X) @ np.asarray(v, dtype=np.float64)

    # -- regions ----------------------------------------------------------
    def packet_mask(self, X: np.ndarray) -> np.ndarray | None:
        """Points inside the burst of oscillation, or ``None`` for targets without one."""
        return self.target.packet_mask(X)

    def jump_mask(self, X: np.ndarray) -> np.ndarray | None:
        """Points within 0.05 of a step or slope break, or ``None`` if there is none."""
        return self.target.interface_mask(X)

    # -- sheets -----------------------------------------------------------
    @property
    def is_sheet(self) -> bool:
        return self.density.is_sheet

    def distance_to_sheet(self, X: np.ndarray):
        """Distance from the data sheet (exact when flat, a grid bound when curved)."""
        if not self.is_sheet:
            return None
        return self.density.distance_to_sheet(X)

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "d": self.d,
                "data": self.density_tag, "target": self.target.label(),
                "what_it_tests": self.what_it_tests,
                "mean_uniform": self.mean_uniform, "sd_uniform": self.sd_uniform,
                "differentiable": self.differentiable,
                "has_packet": self.target.packet_mask(np.zeros((1, self.d))) is not None,
                "has_jump": self.target.interface_mask(np.zeros((1, self.d))) is not None,
                "is_sheet": self.is_sheet,
                "dense_region": self.dense_region_description()}


# ---------------------------------------------------------------------------
# building the list
# ---------------------------------------------------------------------------

def _targets_for(d: int) -> dict[str, TG.Target]:
    """One instance of every target this dimension needs, so that tasks meant to share
    a function share the object."""
    out = {
        "bumps": TG.MultiscaleBumps(d),
        "waves_slow": TG.RadialOscillation(d, 1.0),
        "waves_fast": TG.RadialOscillation(d, 6.0),
        "polynomial": TG.Polynomial(d),
        "composition": TG.Composition(d),
        "runge_broad": TG.RadialRunge(d, 4.0),
        "runge_narrow": TG.RadialRunge(d, 12.0),
        "sphere_jump": TG.SphereJump(d),
        "wavy_jump": TG.WavyJump(d),
        "kink_ring": TG.KinkRing(d),
        "one_sided_kink": TG.OneSidedKink(d),
        "piecewise": TG.Piecewise(d),
        "product_peak": TG.ProductPeak(d),
        "packet_at_hotspot": TG.SpatialPacket(d, TG.aligned_packet_anchor(d), "at hotspot"),
        "packet_away": TG.SpatialPacket(d, TG.antialigned_packet_anchor(d),
                                        "away from hotspots"),
    }
    if d == 1:
        out["runge_at_hotspot"] = TG.RadialRunge(
            d, 12.0, anchor_point=TG.aligned_packet_anchor(d), anchor_name="hotspot")
    return out


def _rows(d: int) -> list[tuple[str, str, str, str]]:
    """``(id, name, data geometry, target key, what it tests)`` for one dimension."""
    p = f"{d}"
    rows = [
        (f"{p}.1", f"d{d}-even-grid-multiscale-bumps", "even_grid", "bumps",
         "Three smooth bumps whose widths differ by 5x: does one global resolution "
         "serve all of them?"),
        (f"{p}.2", f"d{d}-even-grid-radial-oscillation-freq1", "even_grid", "waves_slow",
         "The easiest genuinely multivariate target: one slow concentric wave."),
        (f"{p}.3", f"d{d}-even-grid-radial-oscillation-freq6", "even_grid", "waves_fast",
         "Same shape as task 2 at six times the oscillation; the cost of resolution alone."),
        (f"{p}.4", f"d{d}-even-grid-polynomial", "even_grid", "polynomial",
         "A low-degree polynomial that couples neighbouring coordinates, so no single "
         "direction explains it."),
        (f"{p}.5", f"d{d}-uniform-data-composition", "uniform", "composition",
         "A function of a function: two coordinates enter multiplicatively inside an "
         "exponential."),
        (f"{p}.6", f"d{d}-uniform-data-radial-runge-a4", "uniform", "runge_broad",
         "Smooth everywhere, with a mild peak; the easy end of the sharpness scale."),
        (f"{p}.7", f"d{d}-uniform-data-radial-runge-a12", "uniform", "runge_narrow",
         "The same shape three times sharper: how fast does accuracy fall with peak width?"),
        (f"{p}.8", f"d{d}-uniform-data-sphere-jump", "uniform", "sphere_jump",
         "A genuine discontinuity, and the surface it lives on is curved."),
        None,   # slot 9, filled below
        None,   # slot 10
        (f"{p}.11", f"d{d}-hotspot-data-radial-oscillation-freq6", "hotspots", "waves_fast",
         "Exactly the function of task 3; only the data changes, from even coverage to "
         "three tight clusters."),
        (f"{p}.12", f"d{d}-hotspot-data-packet-at-hotspot", "hotspots", "packet_at_hotspot",
         "A short burst of oscillation sitting exactly where most of the data is."),
        (f"{p}.13", f"d{d}-hotspot-data-packet-away-from-hotspots", "hotspots", "packet_away",
         "The same burst placed away from every cluster: the hard part of the function "
         "is where the data is thin."),
        (f"{p}.14", f"d{d}-hotspot-data-product-peak", "hotspots", "product_peak",
         "A hard smooth function that is not built from any simple structure, on "
         "clustered data."),
        None,   # slot 15
        None,   # slot 16
    ]
    if d == 1:
        rows[8] = (f"{p}.9", f"d{d}-uniform-data-piecewise", "uniform", "piecewise",
                   "Two different formulas glued together: the value matches across the "
                   "seam, the slope does not.")
        rows[9] = (f"{p}.10", f"d{d}-uniform-data-one-sided-kink", "uniform", "one_sided_kink",
                   "Flat on one side, quadratic on the other: value and slope are "
                   "continuous, curvature is not.")
        rows[14] = (f"{p}.15", f"d{d}-hotspot-data-jump-in-sparse-region", "hotspots",
                    "wavy_jump",
                    "A step placed out in the thin tail of the data, where there is "
                    "little to locate it with.")
        rows[15] = (f"{p}.16", f"d{d}-hotspot-data-radial-runge-a12-at-hotspot", "hotspots",
                    "runge_at_hotspot",
                    "A narrow spike sitting exactly on the densest cluster: sharpness and "
                    "data both concentrated in the same place.")
    else:
        rows[8] = (f"{p}.9", f"d{d}-uniform-data-wavy-jump", "uniform", "wavy_jump",
                   "A step across a curved surface, so no single direction describes "
                   "where it is.")
        rows[9] = (f"{p}.10", f"d{d}-uniform-data-kink-ring", "uniform", "kink_ring",
                   "The slope flips sign across a sphere: a milder failure than a step.")
        rows[14] = (f"{p}.15", f"d{d}-curved-sheet-composition", "curved_sheet",
                    "composition",
                    "The data lies exactly on a bent sheet: does accuracy follow the "
                    "sheet's dimension rather than the ambient one?")
        rows[15] = (f"{p}.16", f"d{d}-curved-sheet-noisy-composition", "curved_sheet_noisy",
                    "composition",
                    "Same function, same sheet, plus a thin layer of noise perpendicular "
                    "to it: how fast does the extra direction start to matter?")
    return rows


def _build_tasks() -> list[Task]:
    out: list[Task] = []
    for d in range(1, 6):
        targets = _targets_for(d)
        for tid, name, tag, key, what in _rows(d):
            out.append(Task(tid, name, d, tag, targets[key], what))
    return out


TASKS: list[Task] = _build_tasks()
_BY_ID = {t.id: t for t in TASKS}
_BY_NAME = {t.name: t for t in TASKS}


def get_task(key: str) -> Task:
    """Look a task up by id (``"3.12"``) or by name."""
    if key in _BY_ID:
        return _BY_ID[key]
    if key in _BY_NAME:
        return _BY_NAME[key]
    raise KeyError(f"unknown task {key!r}")


def task_ids() -> list[str]:
    return [t.id for t in TASKS]


def tasks_for_dim(d: int) -> list[Task]:
    return [t for t in TASKS if t.d == d]
