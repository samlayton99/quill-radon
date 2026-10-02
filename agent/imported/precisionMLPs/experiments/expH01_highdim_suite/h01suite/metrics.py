"""What gets measured for every fit.

Nothing is reduced to a single test error. For every run the suite records:

* errors on each of the three test sets (``same_as_train``, ``uniform``, ``dense_region``)
  as mean squared error, relative ``L_2`` and largest absolute error;
* for the clustered data geometries, error split by how dense the data is at each test
  point (ten equal-size bins, sparsest first);
* for the burst-of-oscillation targets, error inside versus outside the burst; for the
  step and kink targets, error near versus far from the surface;
* for the sheet geometries, error on the sheet and at fixed distances from it;
* the predicted center density along a direction.

The predicted center density
----------------------------
How densely approximation theory says centers should be placed along a direction ``v``,
given the data density and how fast the function changes along ``v``:

    predicted(t) proportional to [ p_v(t) * D_v(t) ] ^ (1 / (2r + 1)),

where ``t = v . x`` is the position along the direction, ``p_v(t)`` is the density of
the projected training points, and

    D_v(t) = E[ |dF/dv (X)|^2  |  v . X = t ]

is the average squared slope along ``v`` among the points that project to ``t``. It is
estimated by binning the projected sample and averaging ``|grad F . v|^2`` inside each
bin. ``r`` is the order the approximation error is controlled at (``r = 1`` for tanh
centers, the default). The result is scaled so that it integrates to the number of
centers being placed, which makes it directly comparable with a model's actual centers.

``D_v`` is used through a Gaussian-smoothed local mean rather than pointwise. For an
oscillating function the slope vanishes twice per period, and a pointwise version would
ask for zero centers at every one of those points; the classical statement is about the
local envelope. With the default smoothing, a function whose slope energy is constant
along ``v`` under uniform data gives a flat prediction, as it must. Set ``smooth_bw=0``
for the raw pointwise version.

The prediction is undefined for the two step families -- their gradient does not exist
on the step, which is exactly where all the difficulty is -- and the function says so
rather than returning a number.
"""

from __future__ import annotations

import numpy as np

__all__ = ["error_metrics", "errors_by_data_density", "region_errors",
           "sheet_errors", "projected_density", "predicted_center_density"]


# ---------------------------------------------------------------------------
# basic errors
# ---------------------------------------------------------------------------

def error_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Mean squared error, relative ``L_2`` and largest absolute error."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    if pred.size == 0:
        return {"n": 0, "mse": float("nan"), "rel_l2": float("nan"), "max_abs": float("nan")}
    r = pred - true
    denom = float(np.linalg.norm(true))
    return {"n": int(pred.size),
            "mse": float(np.mean(r * r)),
            "rel_l2": float(np.linalg.norm(r) / denom) if denom > 0 else float("nan"),
            "max_abs": float(np.max(np.abs(r)))}


def errors_by_data_density(pred, true, logp, n_bins: int = 10) -> list[dict]:
    """Error in ten equal-size bins ordered by how dense the data is there.

    Bin 0 holds the tenth of the test points that sit in the sparsest part of the data,
    bin 9 the tenth in the densest part. Only the ranking of ``logp`` is used, so an
    unnormalized log density is fine.
    """
    logp = np.asarray(logp, dtype=np.float64).ravel()
    order = np.argsort(logp)
    out = []
    edges = np.linspace(0, len(order), n_bins + 1).astype(int)
    for b in range(n_bins):
        idx = order[edges[b]:edges[b + 1]]
        m = error_metrics(np.asarray(pred)[idx], np.asarray(true)[idx])
        m["bin"] = b
        m["logp_low"] = float(logp[idx].min()) if idx.size else float("nan")
        m["logp_high"] = float(logp[idx].max()) if idx.size else float("nan")
        out.append(m)
    return out


def region_errors(pred, true, mask, label: str = "region") -> dict:
    """Error inside and outside a region."""
    mask = np.asarray(mask, dtype=bool).ravel()
    return {f"{label}_inside": error_metrics(np.asarray(pred)[mask], np.asarray(true)[mask]),
            f"{label}_outside": error_metrics(np.asarray(pred)[~mask], np.asarray(true)[~mask])}


def sheet_errors(model_predict, F, sets: dict) -> dict:
    """Error on the data sheet and at each fixed distance from it."""
    out = {}
    for key, X in sets.items():
        if key != "on_sheet" and not key.startswith("distance_"):
            continue
        if len(X) == 0:
            continue
        out[key] = error_metrics(model_predict(X), F(X))
    return out


# ---------------------------------------------------------------------------
# the predicted center density
# ---------------------------------------------------------------------------

def _gaussian_smooth(f: np.ndarray, dt: float, bw: float) -> np.ndarray:
    """Reflect-padded Gaussian smoothing of a signal on an evenly spaced grid."""
    if bw <= 0.0:
        return f
    half = max(1, int(np.ceil(4.0 * bw / dt)))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * dt / bw) ** 2)
    k /= k.sum()
    padded = np.concatenate([f[half:0:-1], f, f[-2:-half - 2:-1]])
    if len(padded) != len(f) + 2 * half:           # short signals: edge-pad instead
        padded = np.pad(f, half, mode="edge")
    return np.convolve(padded, k, mode="valid")


def projected_density(t_samples: np.ndarray, grid: np.ndarray,
                      bw: float | None = None) -> np.ndarray:
    """The density of the projected training points, by histogram plus smoothing."""
    t_samples = np.asarray(t_samples, dtype=np.float64).ravel()
    grid = np.asarray(grid, dtype=np.float64)
    dt = float(grid[1] - grid[0])
    edges = np.concatenate([grid - 0.5 * dt, [grid[-1] + 0.5 * dt]])
    counts, _ = np.histogram(t_samples, bins=edges)
    p = counts.astype(np.float64) / (len(t_samples) * dt)
    if bw is None:
        bw = 4.0 * dt
    return _gaussian_smooth(p, dt, bw)


def _slope_energy(t: np.ndarray, dv: np.ndarray, grid: np.ndarray,
                  dt: float, bw: float) -> np.ndarray:
    """Average ``|dF/dv|^2`` among the points that project near each grid position.

    The sum of ``|dF/dv|^2`` and the count of points are binned separately, both are
    smoothed with the same Gaussian, and only then divided. Smoothing the ratio instead
    would drag the zeros of the empty cells beyond the ends of the data inwards and make
    the estimate sag near the edges. Positions with essentially no data get zero, which
    is harmless: the data density there is zero as well.
    """
    edges = np.concatenate([grid - 0.5 * dt, [grid[-1] + 0.5 * dt]])
    total, _ = np.histogram(t, bins=edges, weights=dv * dv)
    count, _ = np.histogram(t, bins=edges)
    num = _gaussian_smooth(total, dt, bw)
    den = _gaussian_smooth(count.astype(np.float64), dt, bw)
    floor = 1e-8 * float(den.max()) if den.max() > 0 else 1.0
    return np.where(den > floor, num / np.maximum(den, floor), 0.0)


def predicted_center_density(v: np.ndarray, grad_F, X_sample: np.ndarray,
                             n_centers: float, r: int = 1,
                             grid: np.ndarray | None = None, n_grid: int = 401,
                             smooth_bw: float | None = None,
                             density_bw: float | None = None,
                             margin: float = 1.25,
                             differentiable: bool = True) -> dict:
    """How densely centers should be placed along direction ``v`` (see module docstring).

    Args:
        v:              the direction, used as given (not normalized).
        grad_F:         callable ``X -> [n, d]`` gradient of the target, or ``None``.
        X_sample:       training points, used both for the projected density and for the
                        average squared slope.
        n_centers:      how many centers are being placed along ``v``; the returned
                        curve integrates to exactly this.
        margin:         the grid runs over ``|t| <= margin * ||v||_1``, a little past the
                        range the projected data can reach, matching the reference model.
        r:              the derivative order the theory controls (1 for tanh centers).
        differentiable: pass ``False`` for a target with a step in it; the function then
                        reports that the prediction is undefined.

    Returns a dict with ``t``, ``p`` (projected data density), ``slope_energy``
    (smoothed ``D_v``), ``density`` (the prediction), and ``integral``.
    """
    if grad_F is None or not differentiable:
        return {"t": None, "p": None, "slope_energy": None, "density": None,
                "integral": float("nan"), "defined": False,
                "reason": "the target has a step in it, so its slope is undefined there"}
    v = np.asarray(v, dtype=np.float64)
    X_sample = np.asarray(X_sample, dtype=np.float64)
    if grid is None:
        T = margin * float(np.abs(v).sum())
        grid = np.linspace(-T, T, n_grid)
    dt = float(grid[1] - grid[0])
    if smooth_bw is None:
        smooth_bw = 0.08 * (grid[-1] - grid[0])

    t = X_sample @ v
    dv = np.asarray(grad_F(X_sample), dtype=np.float64) @ v
    p = projected_density(t, grid, bw=density_bw)
    energy = _slope_energy(t, dv, grid, dt, smooth_bw)

    w = np.power(np.maximum(p, 0.0) * np.maximum(energy, 0.0), 1.0 / (2 * r + 1))
    total = float(np.trapezoid(w, grid)) if hasattr(np, "trapezoid") else float(np.trapz(w, grid))
    density = np.zeros_like(w) if total <= 0 else n_centers * w / total
    integral = (float(np.trapezoid(density, grid)) if hasattr(np, "trapezoid")
                else float(np.trapz(density, grid)))
    return {"t": grid, "p": p, "slope_energy": energy, "density": density,
            "integral": integral, "defined": True, "r": r, "smooth_bw": float(smooth_bw)}
