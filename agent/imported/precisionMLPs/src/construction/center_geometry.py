"""1D center-placement strategies for the geometry-comparison experiment.

All four regimes place W centers over a common span [lo, hi] and (in the
experiment) share the same gamma = lambda/h, so only the *placement* of the
tanh centers differs. This isolates the effect of center geometry on the
least-squares readout's accuracy.

  uniform   -- equispaced (the QI grid that reaches machine precision)
  random    -- uniform random over the span
  clustered -- two-stage: a few uniform meta-centers, then Gaussian draws around
               them (pseudo-clustered, but with overlap so there are no large gaps)
  trained   -- centers extracted from a trained tanh net: x0 = -b/w (the point
               where neuron k's argument is zero). The training lives in the
               experiment; this module only does the extraction.

All functions return a sorted float64 array of length W.
"""

from __future__ import annotations

import numpy as np


def uniform_centers(W: int, span: tuple[float, float]) -> np.ndarray:
    lo, hi = span
    return np.linspace(lo, hi, W, dtype=np.float64)


def random_centers(W: int, span: tuple[float, float], seed: int = 0) -> np.ndarray:
    lo, hi = span
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(lo, hi, size=W)).astype(np.float64)


def clustered_centers(
    W: int,
    span: tuple[float, float],
    seed: int = 0,
    n_meta: int | None = None,
) -> np.ndarray:
    """Two-stage pseudo-clustered placement with no large gaps.

    A few meta-centers are placed uniformly across the span; the W centers are
    then drawn from Gaussians around the meta-centers (assigned round-robin for
    even cluster sizes). The Gaussian width is comparable to the meta spacing so
    clusters overlap and there are no super-large gaps. Centers are clipped to
    the span.
    """
    lo, hi = span
    length = hi - lo
    rng = np.random.default_rng(seed)
    if n_meta is None:
        n_meta = max(2, round(W / 8))
    metas = np.linspace(lo, hi, n_meta)
    sigma = 0.75 * length / n_meta
    assign = np.arange(W) % n_meta            # round-robin -> even cluster sizes
    centers = metas[assign] + rng.normal(0.0, sigma, size=W)
    centers = np.clip(centers, lo, hi)
    return np.sort(centers).astype(np.float64)


def _cluster_offsets(n: int, ratio: float, s: float) -> np.ndarray:
    """Offsets of one cluster of n points, centered at 0.

    Gaps grow geometrically away from the cluster center: the m-th of the n-1
    gaps is s * ratio**(min(m, n-2-m) + 1), so the innermost gaps are smallest
    (points clump toward the center) and ratio=1 gives a plain uniform run.
    For n=7, ratio=0.5, s=1 this returns
    [-0.875, -0.375, -0.125, 0, 0.125, 0.375, 0.875].
    """
    if n < 2:
        return np.zeros(max(n, 1), dtype=np.float64)
    m = np.arange(n - 1)
    levels = np.minimum(m, (n - 2) - m) + 1
    gaps = s * ratio ** levels
    pos = np.concatenate([[0.0], np.cumsum(gaps)])
    return pos - 0.5 * (pos[0] + pos[-1])     # center symmetrically


def regular_clustered_centers(
    W: int,
    span: tuple[float, float],
    cluster_size: int = 8,
    ratio: float = 0.75,
) -> np.ndarray:
    """Regular grid broken into evenly-spaced clusters of `cluster_size` points.

    Each cluster of `cluster_size` points is compressed toward its center by the
    geometric `ratio` (see _cluster_offsets); clusters are tiled with center
    spacing cluster_size * s, where s = span_length / W is the regular spacing.
    ratio=1 reproduces the uniform grid; smaller ratio clumps each cluster
    tighter and opens voids between clusters. Deterministic. Returns exactly W
    sorted centers within the span.
    """
    lo, hi = span
    length = hi - lo
    mid = 0.5 * (lo + hi)
    n = int(cluster_size)
    s = length / W
    offsets = _cluster_offsets(n, ratio, s)
    cluster_spacing = n * s

    # Tile enough clusters to comfortably cover the span, centered on its mid.
    n_clusters = int(np.ceil(length / cluster_spacing)) + 4
    k0 = n_clusters // 2
    cluster_centers = mid + (np.arange(n_clusters) - k0) * cluster_spacing
    pts = np.sort((cluster_centers[:, None] + offsets[None, :]).ravel())
    pts = pts[(pts >= lo - 1e-12) & (pts <= hi + 1e-12)]

    # Trim symmetrically (drop outermost extras) to exactly W.
    extra = pts.size - W
    if extra > 0:
        left = extra // 2
        pts = pts[left: pts.size - (extra - left)]
    elif extra < 0:                            # rare: shrink spacing and retry once
        return regular_clustered_centers(
            W, (lo, hi), cluster_size, ratio) if False else _pad_to_W(pts, W, lo, hi)
    return np.clip(pts, lo, hi).astype(np.float64)


def _pad_to_W(pts: np.ndarray, W: int, lo: float, hi: float) -> np.ndarray:
    """Fallback: pad a short center list up to W by inserting uniform fillers."""
    extra = W - pts.size
    fillers = np.linspace(lo, hi, extra + 2)[1:-1]
    return np.sort(np.concatenate([pts, fillers])).astype(np.float64)


def trained_centers(
    w: np.ndarray,
    b: np.ndarray,
    span: tuple[float, float],
    eps: float = 1e-8,
) -> np.ndarray:
    """Extract tanh-neuron centers x0 = -b/w from a trained inner layer.

    w, b are the per-neuron input weight and bias of tanh(w*x + b). Neurons with
    |w| < eps (effectively dead / horizontal) have an undefined center; they are
    placed at a span endpoint deterministically by the sign of their argument.
    All centers are clipped to the span. Returns a sorted array of length len(w).
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    lo, hi = span
    safe = np.abs(w) >= eps
    centers = np.empty_like(w)
    centers[safe] = -b[safe] / w[safe]
    # Dead neurons: park at the endpoint matched to the bias sign (arbitrary but
    # deterministic), then clip everything into the span.
    centers[~safe] = np.where(b[~safe] >= 0, lo, hi)
    centers = np.clip(centers, lo, hi)
    return np.sort(centers).astype(np.float64)
