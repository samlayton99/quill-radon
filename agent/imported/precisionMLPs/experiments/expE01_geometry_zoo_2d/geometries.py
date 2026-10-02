"""Six 2D ridge geometries, unified to a common interface.

Every geometry returns (dirs, offsets): dirs[M,2] unit normals and offsets[M],
so neuron m is the ridge tanh(gamma*(dirs_m . x - offsets_m)) -- a line at
signed distance offsets_m from the origin in direction dirs_m. This is the
(direction, offset) = Radon-line view shared by all candidates.

  hex              -- (1) hex-packed points, radial tangent ridge per point.
  mode_radial      -- (2) uncapped exact-N mode radial grid (sqrt-radial gaps,
                          DP-allocated angular counts, staggered rings), tangents.
  sixn_rings       -- (4) exactly 6k points per ring, sqrt-radial gaps, all rings
                          phase-aligned to the same axes (symmetry-first), tangents.
  random_ridges    -- (5) random disk points; 3 ridges each at 0/60/120 deg with a
                          random per-point base orientation (decoupled directions).
  radon_tensor     -- (6a) uniform tensor grid in Radon (theta,t): J equispaced
                          directions x M uniform signed offsets.
  radon_interlaced -- (6b) same but the offset grid is shifted by h_t/2 on
                          alternate directions (Natterer efficient/hex sampling).

Point-based geometries (1,2,4) use the tangent rule dir = c/||c||, offset = ||c||.
Modules 2 and 4 are ported from sandbox.ipynb (trimmed to return points).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.hex_geometry import hex_pack


# --------------------------------------------------------------------------
# point -> tangent ridge
# --------------------------------------------------------------------------

def _tangent_ridges(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dir = c/||c|| (radial normal), offset = ||c|| (tangent to circle at c).
    The r=0 point (if any) gets an arbitrary direction (1,0), offset 0."""
    points = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(points, axis=1)
    dirs = np.zeros_like(points)
    interior = r > 1e-12
    dirs[interior] = points[interior] / r[interior, None]
    dirs[~interior] = np.array([1.0, 0.0])
    return dirs, r.copy()


# --------------------------------------------------------------------------
# shared ring helpers (ported from sandbox modules 2-4)
# --------------------------------------------------------------------------

def _ring_radii_from_gaps(num_rings, outer_radius, radial_gap_power=0.5):
    ring_numbers = np.arange(1, num_rings + 1, dtype=float)
    radial_gaps = ring_numbers ** radial_gap_power
    radial_gaps *= outer_radius / radial_gaps.sum()
    return np.cumsum(radial_gaps), radial_gaps


def _mode_divisors(mode_lattice_count, min_count, max_count):
    return np.array(
        [n for n in range(max(1, min_count), max_count + 1) if mode_lattice_count % n == 0],
        dtype=float,
    )


def _small_prime_leftover(n, primes=(2, 3, 5)):
    leftover = int(n)
    for prime in primes:
        while leftover > 1 and leftover % prime == 0:
            leftover //= prime
    return leftover


def _mode_count_penalty(n, mode_counts):
    if len(mode_counts) == 0:
        return 0.0
    n_float = float(n)
    divisor_distance = np.min(np.abs(mode_counts - n_float) / np.maximum(mode_counts, n_float))
    leftover = _small_prime_leftover(n)
    smoothness_penalty = 0.0 if leftover == 1 else (np.log(leftover) / np.log(max(n, 2))) ** 2
    return 0.7 * divisor_distance ** 2 + 0.3 * smoothness_penalty


def _choose_num_rings(N, outer_radius, radial_gap_power=0.5, angular_gap_ratio=0.75, min_ring_count=6):
    if N <= min_ring_count:
        return 1
    max_rings = max(1, N // min_ring_count)
    best_num_rings, best_score = 1, np.inf
    for num_rings in range(1, max_rings + 1):
        radii, radial_gaps = _ring_radii_from_gaps(num_rings, outer_radius, radial_gap_power)
        ideal_counts = 2 * np.pi * radii / (angular_gap_ratio * radial_gaps)
        score = abs(np.log(ideal_counts.sum() / N))
        if score < best_score:
            best_score, best_num_rings = score, num_rings
    return best_num_rings


def _allocate_counts_exact(N, ideal_counts, min_ring_count, mode_weight, outer_slack_weight, mode_lattice_count):
    num_rings = len(ideal_counts)
    if N < num_rings * min_ring_count:
        raise ValueError("N too small for the selected number of rings")
    mode_counts = _mode_divisors(mode_lattice_count, min_ring_count, N)
    costs_by_ring = []
    for ring_idx, ideal_count in enumerate(ideal_counts):
        is_outer = ring_idx == num_rings - 1
        spacing_weight = outer_slack_weight if is_outer else 1.0
        mode_ring_weight = 0.0 if is_outer else mode_weight
        ring_costs = np.full(N + 1, np.inf)
        for count in range(min_ring_count, N + 1):
            spacing_error = ((count - ideal_count) / max(ideal_count, 1.0)) ** 2
            ring_costs[count] = spacing_weight * spacing_error + mode_ring_weight * _mode_count_penalty(count, mode_counts)
        costs_by_ring.append(ring_costs)

    dp = np.full((num_rings + 1, N + 1), np.inf)
    prev = np.full((num_rings + 1, N + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for ring_idx in range(num_rings):
        rings_left = num_rings - ring_idx - 1
        for used in range(N + 1):
            if not np.isfinite(dp[ring_idx, used]):
                continue
            max_count = N - used - rings_left * min_ring_count
            for count in range(min_ring_count, max_count + 1):
                cand = dp[ring_idx, used] + costs_by_ring[ring_idx][count]
                if cand < dp[ring_idx + 1, used + count]:
                    dp[ring_idx + 1, used + count] = cand
                    prev[ring_idx + 1, used + count] = count
    if not np.isfinite(dp[num_rings, N]):
        raise RuntimeError("could not allocate ring counts exactly")
    counts, used = [], N
    for ring_idx in range(num_rings, 0, -1):
        count = prev[ring_idx, used]
        counts.append(int(count))
        used -= count
    counts.reverse()
    return np.array(counts, dtype=int)


def _candidate_offsets(count, mode_lattice_count, rotation_candidates):
    period = 2 * np.pi / count
    if mode_lattice_count % count == 0:
        lattice_offsets = np.arange(mode_lattice_count) * (2 * np.pi / mode_lattice_count)
        return np.unique(np.round(np.mod(lattice_offsets, period), decimals=14))
    return np.linspace(0.0, period, rotation_candidates, endpoint=False)


def _choose_ring_offset(radius, count, placed, mode_lattice_count, rotation_candidates):
    base_angles = 2 * np.pi * np.arange(count) / count
    best_offset, best_min_dist = 0.0, -np.inf
    for offset in _candidate_offsets(count, mode_lattice_count, rotation_candidates):
        cand = radius * np.column_stack([np.cos(base_angles + offset), np.sin(base_angles + offset)])
        min_dist = np.inf if placed.size == 0 else np.linalg.norm(
            cand[:, None, :] - placed[None, :, :], axis=2).min()
        if min_dist > best_min_dist:
            best_min_dist, best_offset = min_dist, float(offset)
    return best_offset


def _ring_points(radii, counts, stagger, mode_lattice_count=360, rotation_candidates=180):
    placed = np.empty((0, 2))
    out = []
    for radius, count in zip(radii, counts):
        offset = (_choose_ring_offset(radius, int(count), placed, mode_lattice_count, rotation_candidates)
                  if stagger else 0.0)
        angles = 2 * np.pi * np.arange(count) / count + offset
        pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        out.append(pts)
        placed = np.vstack([placed, pts])
    return np.vstack(out)


# --------------------------------------------------------------------------
# the six geometries
# --------------------------------------------------------------------------

def build_hex(N, radius, seed=0):
    centers, _, _ = hex_pack(N, radius)
    return _tangent_ridges(centers)


def build_mode_radial(N, radius, seed=0):
    num_rings = _choose_num_rings(N, radius, min_ring_count=min(6, N))
    radii, radial_gaps = _ring_radii_from_gaps(num_rings, radius)
    ideal_counts = 2 * np.pi * radii / (0.75 * radial_gaps)
    counts = _allocate_counts_exact(N, ideal_counts, min_ring_count=min(6, N),
                                    mode_weight=0.15, outer_slack_weight=0.15,
                                    mode_lattice_count=360)
    return _tangent_ridges(_ring_points(radii, counts, stagger=True))


def _sixn_counts(N):
    counts, remaining, k = [], int(N), 1
    while remaining > 0:
        counts.append(min(6 * k, remaining))
        remaining -= counts[-1]
        k += 1
    return np.array(counts, dtype=int)


def build_sixn_rings(N, radius, seed=0):
    counts = _sixn_counts(N)
    radii, _ = _ring_radii_from_gaps(len(counts), radius)
    return _tangent_ridges(_ring_points(radii, counts, stagger=False))   # phase-aligned


def build_random_ridges(N, radius, seed=0):
    rng = np.random.default_rng(seed)
    n_pts = int(np.ceil(N / 3))
    r = radius * np.sqrt(rng.random(n_pts))
    ang = 2 * np.pi * rng.random(n_pts)
    pts = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
    base = rng.random(n_pts) * np.pi
    dirs, offs = [], []
    for p, b in zip(pts, base):
        for k in range(3):
            a = b + k * np.pi / 3.0                 # 0, 60, 120 deg
            w = np.array([np.cos(a), np.sin(a)])
            dirs.append(w); offs.append(float(w @ p))
    return np.array(dirs)[:N], np.array(offs)[:N]


def _jm(N, radius):
    """Pick J directions x M offsets over [-radius, radius] so that the INTERIOR
    [-1,1] offset resolution (= M/radius) matches the direction count J (a square
    interior grid), and J*M ~= N. Gives interior linear resolution ~0.63*sqrt(N),
    matching the tangent geometries at R=1.6, with the rest of the offsets forming
    the node-count halo per direction."""
    J = max(int(round(np.sqrt(N / radius))), 2)
    M = max(int(round(N / J)), 3)
    return J, M


def build_radon_tensor(N, radius, seed=0):
    J, M = _jm(N, radius)
    T = radius
    h_step = 2.0 * T / M
    thetas = np.arange(J) * np.pi / J       # endpoint rule: theta=0 included (x-axis)
    t1d = -T + (np.arange(M) + 0.5) * h_step                  # symmetric, cell-centered
    dirs, offs = [], []
    for th in thetas:
        w = np.array([np.cos(th), np.sin(th)])
        for t in t1d:
            dirs.append(w); offs.append(t)
    return np.array(dirs), np.array(offs)


def build_radon_interlaced(N, radius, seed=0):
    J, M = _jm(N, radius)
    T = radius
    h_step = 2.0 * T / M
    thetas = np.arange(J) * np.pi / J
    dirs, offs = [], []
    for j, th in enumerate(thetas):
        w = np.array([np.cos(th), np.sin(th)])
        shift = 0.5 * h_step if (j % 2 == 1) else 0.0        # half-step on alternate angles
        t1d = -T + (np.arange(M) + 0.5) * h_step + shift
        for t in t1d:
            dirs.append(w); offs.append(t)
    return np.array(dirs), np.array(offs)


GEOMETRIES = {
    "hex": build_hex,
    "mode_radial": build_mode_radial,
    "sixn_rings": build_sixn_rings,
    "random_ridges": build_random_ridges,
    "radon_tensor": build_radon_tensor,
    "radon_interlaced": build_radon_interlaced,
}
