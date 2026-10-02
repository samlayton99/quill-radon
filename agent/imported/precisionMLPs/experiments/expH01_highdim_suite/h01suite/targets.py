"""The twelve function families the suite approximates.

None of these is a sum of one-dimensional profiles along fixed directions. That is
the point: a target built as ``sum_i g_i(v_i . x)`` would bake the model's own way of
representing functions into the benchmark, and every question about "did the method
find the right directions" would already be answered by the construction. The families
here are genuinely multivariate -- radial, product, composed, or split by a curved
surface -- and are written in the normalized coordinates ``z(x)`` of ``basis.py``.

Distances to a point are measured with the *scaled* radial distance

    rho_a(x) = ||z(x) - a||_2 / sqrt(d),

so that a corner of the cube sits at distance about 1 in every dimension and a width
like ``sigma = 0.25`` means the same thing in ``d = 1`` and ``d = 5``.

Three anchor points (given in ``z`` coordinates and truncated to ``d``) are reused
across the families so that features land in comparable places as ``d`` grows:

    anchor 1 = ( .30, -.20,  .25, -.15,  .10)
    anchor 2 = (-.40,  .35, -.10,  .20, -.25)
    anchor 3 = ( .15,  .10, -.30,  .05,  .30)

The families
------------
``multiscale_bumps``    three Gaussian bumps of very different widths at the three anchors
``wide_bump``           one broad Gaussian bump; the smooth background under the
                        packet, jump and kink families
``radial_oscillation``  ``cos(pi omega rho)`` -- concentric waves, no preferred direction
                        (cosine, not sine: cosine is even, so this is a smooth function
                        of rho^2 with no cone point at the center)
``composition``         ``exp(sin(pi z_1) cos(pi z_2))`` -- a function of a function
``polynomial``          ``sum_k (z_k^2 z_{k+1} - z_k z_{k+1}^3)``, cyclic in ``k``
``radial_runge``        ``1/(1 + alpha^2 rho^2)`` -- smooth, but with a nearby complex pole
``product_peak``        ``prod_k 1/(1 + a_k^2 (z_k - b_k)^2)`` -- the standard Genz peak
``spatial_packet``      wide bump plus a short burst of oscillation in a small ball
``sphere_jump``         wide bump plus a step across a sphere
``wavy_jump``           wide bump plus a step across a curved surface
``kink_ring``           ``|rho - 0.4|`` -- continuous, slope flips across a sphere
``one_sided_kink``      ``max(0, rho_0^2 - 0.25)`` -- value and slope continuous, curvature is not
``piecewise``           one formula inside a ball, another outside; continuous, slope jumps

Every family provides ``value(X)``, an analytic ``grad(X)`` of shape ``[n, d]`` in ``x``
coordinates, and a flag ``differentiable``. The two step families set
``differentiable = False``: their ``grad`` returns the gradient of the smooth part
(correct everywhere except on the step surface) and ``interface_mask(X)`` marks points
near that surface. The predicted-center-density calculation refuses to run on them.
"""

from __future__ import annotations

import numpy as np

from h01suite.basis import dct_basis, grad_z_to_grad_x, l1_scales, z_of

__all__ = ["Target", "ANCHORS", "anchor", "MultiscaleBumps", "WideBump",
           "RadialOscillation", "Composition", "Polynomial", "RadialRunge",
           "ProductPeak", "SpatialPacket", "SphereJump", "WavyJump", "KinkRing",
           "OneSidedKink", "Piecewise", "PACKET_TAU", "PACKET_OMEGA",
           "JUMP_BAND", "aligned_packet_anchor", "antialigned_packet_anchor",
           "hotspot_means_z"]

ANCHORS = {
    1: (0.30, -0.20, 0.25, -0.15, 0.10),
    2: (-0.40, 0.35, -0.10, 0.20, -0.25),
    3: (0.15, 0.10, -0.30, 0.05, 0.30),
}

PACKET_TAU = 0.18       # width of the burst of oscillation in spatial_packet
PACKET_OMEGA = 10.0     # its frequency
JUMP_BAND = 0.05        # half-width of the "near the step" band used by the metrics


def anchor(which: int, d: int) -> np.ndarray:
    """Anchor point ``which`` (1, 2 or 3) truncated to ``d`` coordinates."""
    return np.array(ANCHORS[which][:d], dtype=np.float64)


def aligned_packet_anchor(d: int) -> np.ndarray:
    """``mu_+`` in ``z`` coordinates: ``(0.45, 0, ..., 0)``.

    The dominant hotspot mean is ``mu_+ = 0.45 * (1,...,1)`` in ``x``. Since ``u_1`` is
    proportional to ``(1,...,1)`` and every other ``u_k`` is orthogonal to it,
    ``z_1(mu_+) = 0.45`` and ``z_k(mu_+) = 0`` for ``k > 1``.
    """
    a = np.zeros(d, dtype=np.float64)
    a[0] = 0.45
    return a


def antialigned_packet_anchor(d: int) -> np.ndarray:
    """A point far from all three hotspot means (see ``hotspot_means_z``)."""
    if d == 1:
        return np.array([0.85])
    a = np.zeros(d, dtype=np.float64)
    a[0], a[1] = 0.40, -0.55
    return a


def hotspot_means_z(d: int) -> dict[str, np.ndarray]:
    """The three hotspot means of the ``hotspots`` density, in ``z`` coordinates.

    ``mu_+`` and ``mu_-`` sit at ``z_1 = +-0.45`` with every other coordinate zero;
    ``mu_perp = 0.35 u_2/||u_2||_inf`` has only ``z_2`` nonzero. In ``d = 1`` the three
    means are ``0.45``, ``-0.45`` and ``0``.
    """
    if d == 1:
        return {"plus": np.array([0.45]), "minus": np.array([-0.45]),
                "perp": np.array([0.0])}
    Q, s = dct_basis(d), l1_scales(d)
    u2 = Q[:, 1]
    mu_perp = 0.35 * u2 / np.abs(u2).max()
    zp = np.zeros(d); zp[0] = 0.45
    zm = np.zeros(d); zm[0] = -0.45
    return {"plus": zp, "minus": -zp, "perp": (mu_perp @ Q) / s}


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------

class Target:
    """A function on the cube, written in ``z``, with an analytic gradient in ``x``."""

    family: str = "target"
    differentiable: bool = True

    def __init__(self, d: int):
        self.d = int(d)

    # -- to implement in subclasses --------------------------------------
    def value_z(self, Z: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def grad_value_z(self, Z: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """Gradient with respect to ``z``, shape ``[n, d]``."""
        raise NotImplementedError

    # -- shared ----------------------------------------------------------
    def value(self, X: np.ndarray) -> np.ndarray:
        return self.value_z(z_of(X, self.d))

    def grad(self, X: np.ndarray) -> np.ndarray:
        return grad_z_to_grad_x(self.grad_value_z(z_of(X, self.d)), self.d)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.value(X)

    def interface_mask(self, X: np.ndarray) -> np.ndarray | None:
        """Points within ``JUMP_BAND`` of a step surface, or ``None`` if there is none."""
        return None

    def packet_mask(self, X: np.ndarray) -> np.ndarray | None:
        """Points inside the burst of oscillation, or ``None`` if there is none."""
        return None

    def label(self) -> str:
        return self.family

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.label()}(d={self.d})"


# ---------------------------------------------------------------------------
# radial helpers
# ---------------------------------------------------------------------------

def _radial(Z: np.ndarray, a: np.ndarray, d: int):
    """``(rho, diff)`` with ``rho = ||z-a||/sqrt(d)`` and ``diff = z-a``."""
    diff = Z - a[None, :]
    rho = np.sqrt(np.einsum("nk,nk->n", diff, diff) / d)
    return rho, diff


def _drho(rho: np.ndarray, diff: np.ndarray, d: int) -> np.ndarray:
    """``d rho / d z = (z-a)/(d rho)``; zero at the anchor itself, where every family used with it is smooth (cosine carriers, even envelopes)."""
    safe = np.where(rho > 0, rho, 1.0)
    out = diff / (d * safe[:, None])
    return np.where(rho[:, None] > 0, out, 0.0)


# ---------------------------------------------------------------------------
# the families
# ---------------------------------------------------------------------------

class MultiscaleBumps(Target):
    """Three Gaussian bumps whose widths differ by 5x, at the three anchors.

        B(x) = sum_k A_k exp(-rho_{a_k}(x)^2 / (2 sigma_k^2)),
        (A, sigma) = (1.0, .50), (0.7, .25), (0.5, .10).
    """

    family = "multiscale_bumps"
    AMPS = (1.0, 0.7, 0.5)
    SIGMAS = (0.50, 0.25, 0.10)

    def __init__(self, d: int):
        super().__init__(d)
        self.anchors = [anchor(k, d) for k in (1, 2, 3)]

    def value_z(self, Z):
        out = np.zeros(len(Z), dtype=np.float64)
        for A, sig, a in zip(self.AMPS, self.SIGMAS, self.anchors):
            rho, _ = _radial(Z, a, self.d)
            out += A * np.exp(-rho * rho / (2.0 * sig * sig))
        return out

    def grad_value_z(self, Z):
        g = np.zeros_like(Z)
        for A, sig, a in zip(self.AMPS, self.SIGMAS, self.anchors):
            rho, diff = _radial(Z, a, self.d)
            e = A * np.exp(-rho * rho / (2.0 * sig * sig))
            g += -(e / (self.d * sig * sig))[:, None] * diff
        return g


class WideBump(Target):
    """One broad Gaussian bump, ``exp(-rho_{a_1}^2 / 0.5)``.

    Used on its own and as the smooth background that the packet, step and kink
    families sit on, so those tasks are not dominated by a constant.
    """

    family = "wide_bump"
    WIDTH = 0.5

    def __init__(self, d: int):
        super().__init__(d)
        self.a = anchor(1, d)

    def value_z(self, Z):
        rho, _ = _radial(Z, self.a, self.d)
        return np.exp(-rho * rho / self.WIDTH)

    def grad_value_z(self, Z):
        rho, diff = _radial(Z, self.a, self.d)
        e = np.exp(-rho * rho / self.WIDTH)
        return -(2.0 * e / (self.d * self.WIDTH))[:, None] * diff


class RadialOscillation(Target):
    """Concentric waves around anchor 1: ``cos(pi omega rho)``. No preferred direction.

    Cosine rather than sine on purpose: ``rho`` is a distance, and ``sin(pi omega rho)``
    would have a cone point at the anchor. ``cos`` is even, so ``cos(pi omega rho)`` is an
    analytic function of ``rho^2`` and therefore smooth everywhere.
    """

    family = "radial_oscillation"

    def __init__(self, d: int, omega: float):
        super().__init__(d)
        self.omega = float(omega)
        self.a = anchor(1, d)

    def value_z(self, Z):
        rho, _ = _radial(Z, self.a, self.d)
        return np.cos(np.pi * self.omega * rho)

    def grad_value_z(self, Z):
        rho, diff = _radial(Z, self.a, self.d)
        c = -np.pi * self.omega * np.sin(np.pi * self.omega * rho)
        return c[:, None] * _drho(rho, diff, self.d)

    def label(self):
        return f"radial_oscillation(omega={self.omega:g})"


class Composition(Target):
    """``exp(sin(pi z_1) cos(pi z_2))``; in ``d = 1``, ``exp(sin(pi z_1))``.

    Not a sum of anything: the two coordinates enter multiplicatively inside an
    exponential, so no finite sum of one-dimensional profiles reproduces it.
    """

    family = "composition"

    def value_z(self, Z):
        s = np.sin(np.pi * Z[:, 0])
        if self.d == 1:
            return np.exp(s)
        return np.exp(s * np.cos(np.pi * Z[:, 1]))

    def grad_value_z(self, Z):
        g = np.zeros_like(Z)
        s, cs = np.sin(np.pi * Z[:, 0]), np.cos(np.pi * Z[:, 0])
        if self.d == 1:
            g[:, 0] = np.exp(s) * np.pi * cs
            return g
        c2, s2 = np.cos(np.pi * Z[:, 1]), np.sin(np.pi * Z[:, 1])
        val = np.exp(s * c2)
        g[:, 0] = val * np.pi * cs * c2
        g[:, 1] = -val * np.pi * s * s2
        return g


class Polynomial(Target):
    """``sum_{k=1..d} (z_k^2 z_{k+1} - z_k z_{k+1}^3)`` with ``z_{d+1} = z_1``.

    Every term couples two coordinates, so the whole thing is a genuine polynomial in
    several variables rather than a sum of single-variable pieces. In ``d = 1`` the
    cyclic index collapses it to ``z_1^3 - z_1^4``.
    """

    family = "polynomial"

    def value_z(self, Z):
        nxt = np.roll(Z, -1, axis=1)
        return np.sum(Z * Z * nxt - Z * nxt ** 3, axis=1)

    def grad_value_z(self, Z):
        nxt = np.roll(Z, -1, axis=1)
        d_first = 2.0 * Z * nxt - nxt ** 3              # d/d z_k of term k
        d_second = Z * Z - 3.0 * Z * nxt ** 2           # d/d z_{k+1} of term k
        return d_first + np.roll(d_second, 1, axis=1)


class RadialRunge(Target):
    """``1/(1 + alpha^2 rho^2)`` -- smooth, but with a pole a distance ``1/alpha`` away
    in the complex plane, so large ``alpha`` is a sharp radial spike."""

    family = "radial_runge"

    def __init__(self, d: int, alpha: float, anchor_point: np.ndarray | None = None,
                 anchor_name: str = "anchor1"):
        super().__init__(d)
        self.alpha = float(alpha)
        self.a = anchor(1, d) if anchor_point is None else np.asarray(anchor_point, float)
        self.anchor_name = anchor_name

    def value_z(self, Z):
        rho, _ = _radial(Z, self.a, self.d)
        return 1.0 / (1.0 + (self.alpha ** 2) * rho * rho)

    def grad_value_z(self, Z):
        rho, diff = _radial(Z, self.a, self.d)
        den = 1.0 + (self.alpha ** 2) * rho * rho
        return (-2.0 * self.alpha ** 2 / (self.d * den * den))[:, None] * diff

    def label(self):
        return f"radial_runge(alpha={self.alpha:g},at={self.anchor_name})"


class ProductPeak(Target):
    """The standard Genz product peak ``prod_k 1/(1 + a_k^2 (z_k - b_k)^2)``.

    A product, not a sum: it is sharply peaked in every coordinate at once, and the
    peak widths ``1/a_k`` differ by a factor of four across coordinates.
    """

    family = "product_peak"
    A = (3.0, 5.0, 8.0, 12.0, 6.0)
    B = (0.35, -0.25, 0.10, -0.40, 0.25)

    def __init__(self, d: int):
        super().__init__(d)
        self.a = np.array(self.A[:d], dtype=np.float64)
        self.b = np.array(self.B[:d], dtype=np.float64)

    def _factors(self, Z):
        s = Z - self.b[None, :]
        den = 1.0 + (self.a[None, :] ** 2) * s * s
        return s, den

    def value_z(self, Z):
        _, den = self._factors(Z)
        return np.prod(1.0 / den, axis=1)

    def grad_value_z(self, Z):
        s, den = self._factors(Z)
        val = np.prod(1.0 / den, axis=1)
        return val[:, None] * (-2.0 * (self.a[None, :] ** 2) * s / den)


class SpatialPacket(Target):
    """Wide bump plus a short burst of oscillation confined to a small ball.

        W(x) = wide_bump(x) + 0.8 exp(-rho_a^2/tau^2) cos(pi omega rho_a),
        tau = 0.18, omega = 10.

    Cosine carrier so the burst is smooth at its own center (see RadialOscillation).

    Two versions are used: the burst centered on the dominant hotspot (``at hotspot``)
    and the same burst placed away from every hotspot (``away from hotspots``). Same
    function, different place: the only thing that changes is whether the hard part of
    the function is where the data is.
    """

    family = "spatial_packet"

    def __init__(self, d: int, packet_anchor: np.ndarray, where: str,
                 tau: float = PACKET_TAU, omega: float = PACKET_OMEGA):
        super().__init__(d)
        self.background = WideBump(d)
        self.a = np.asarray(packet_anchor, dtype=np.float64)
        self.where = where
        self.tau, self.omega = float(tau), float(omega)

    def value_z(self, Z):
        rho, _ = _radial(Z, self.a, self.d)
        env = np.exp(-rho * rho / (self.tau ** 2))
        return self.background.value_z(Z) + 0.8 * env * np.cos(np.pi * self.omega * rho)

    def grad_value_z(self, Z):
        rho, diff = _radial(Z, self.a, self.d)
        env = np.exp(-rho * rho / (self.tau ** 2))
        arg = np.pi * self.omega * rho
        # d/dz of env*cos(arg): the envelope part keeps a factor rho that cancels the
        # 1/rho in d rho/dz, so only the carrier term needs the safe division.
        term_env = (-2.0 * env * np.cos(arg) / (self.d * self.tau ** 2))[:, None] * diff
        term_carrier = (-env * np.pi * self.omega * np.sin(arg))[:, None] * _drho(rho, diff, self.d)
        return self.background.grad_value_z(Z) + 0.8 * (term_env + term_carrier)

    def packet_mask(self, X):
        rho, _ = _radial(z_of(X, self.d), self.a, self.d)
        return rho <= 2.0 * self.tau

    def label(self):
        return f"spatial_packet({self.where})"


class SphereJump(Target):
    """Wide bump with a step of height 0.8 across the sphere ``rho_{a_2} = 0.35``."""

    family = "sphere_jump"
    differentiable = False
    RADIUS = 0.35

    def __init__(self, d: int):
        super().__init__(d)
        self.background = WideBump(d)
        self.a = anchor(2, d)

    def value_z(self, Z):
        rho, _ = _radial(Z, self.a, self.d)
        return self.background.value_z(Z) + 0.8 * (rho < self.RADIUS)

    def grad_value_z(self, Z):
        """The gradient of the smooth part -- correct everywhere off the sphere."""
        return self.background.grad_value_z(Z)

    def interface_mask(self, X):
        rho, _ = _radial(z_of(X, self.d), self.a, self.d)
        return np.abs(rho - self.RADIUS) <= JUMP_BAND


class WavyJump(Target):
    """Wide bump with a step of height 0.8 across a curved surface.

    For ``d >= 2`` the surface is ``z_2 = 0.3 sin(pi z_1) + 0.1`` -- not a plane, so no
    single direction describes it. In ``d = 1`` there is no room to curve and the step
    is at ``z_1 = 0.78``, out in the sparse tail of the hotspot density.
    """

    family = "wavy_jump"
    differentiable = False
    ONE_D_LOCATION = 0.78

    def __init__(self, d: int):
        super().__init__(d)
        self.background = WideBump(d)

    def _signed(self, Z: np.ndarray) -> np.ndarray:
        """Signed distance-like coordinate: positive on the raised side."""
        if self.d == 1:
            return Z[:, 0] - self.ONE_D_LOCATION
        return Z[:, 1] - 0.3 * np.sin(np.pi * Z[:, 0]) - 0.1

    def value_z(self, Z):
        return self.background.value_z(Z) + 0.8 * (self._signed(Z) > 0.0)

    def grad_value_z(self, Z):
        return self.background.grad_value_z(Z)

    def interface_mask(self, X):
        return np.abs(self._signed(z_of(X, self.d))) <= JUMP_BAND

    def label(self):
        return "wavy_jump" if self.d >= 2 else "wavy_jump(1d step at z1=0.78)"


class KinkRing(Target):
    """``|rho_{a_1} - 0.4|``: continuous, with the slope flipping sign across a sphere.

    A milder failure than a step -- the value is continuous but the first derivative
    jumps, and it does so on a curved surface rather than a plane.
    """

    family = "kink_ring"
    RADIUS = 0.4

    def __init__(self, d: int):
        super().__init__(d)
        self.a = anchor(1, d)

    def value_z(self, Z):
        rho, _ = _radial(Z, self.a, self.d)
        return np.abs(rho - self.RADIUS)

    def grad_value_z(self, Z):
        rho, diff = _radial(Z, self.a, self.d)
        return np.sign(rho - self.RADIUS)[:, None] * _drho(rho, diff, self.d)

    def interface_mask(self, X):
        rho, _ = _radial(z_of(X, self.d), self.a, self.d)
        return np.abs(rho - self.RADIUS) <= JUMP_BAND


class OneSidedKink(Target):
    """``max(0, rho_0^2 - 0.25)`` with ``rho_0`` the scaled distance to the origin.

    Flat inside the ball, quadratic outside. Value and first derivative are continuous
    across the sphere; the second derivative is not.
    """

    family = "one_sided_kink"
    THRESHOLD = 0.25

    def value_z(self, Z):
        r2 = np.einsum("nk,nk->n", Z, Z) / self.d
        return np.maximum(0.0, r2 - self.THRESHOLD)

    def grad_value_z(self, Z):
        r2 = np.einsum("nk,nk->n", Z, Z) / self.d
        active = (r2 > self.THRESHOLD).astype(np.float64)
        return (2.0 * active / self.d)[:, None] * Z


class Piecewise(Target):
    """Two different formulas, glued across the sphere ``rho_{a_1} = 0.4``:

        sin(2 pi rho)                      for rho < 0.4,
        sin(0.8 pi) + 3 (rho - 0.4)^2      otherwise.

    Continuous by construction; the slope jumps from ``2 pi cos(0.8 pi)`` to ``0``.
    """

    family = "piecewise"
    RADIUS = 0.4

    def __init__(self, d: int):
        super().__init__(d)
        self.a = anchor(1, d)

    def value_z(self, Z):
        rho, _ = _radial(Z, self.a, self.d)
        inner = np.sin(2.0 * np.pi * rho)
        outer = np.sin(0.8 * np.pi) + 3.0 * (rho - self.RADIUS) ** 2
        return np.where(rho < self.RADIUS, inner, outer)

    def grad_value_z(self, Z):
        rho, diff = _radial(Z, self.a, self.d)
        dinner = 2.0 * np.pi * np.cos(2.0 * np.pi * rho)
        douter = 6.0 * (rho - self.RADIUS)
        return np.where(rho < self.RADIUS, dinner, douter)[:, None] * _drho(rho, diff, self.d)

    def interface_mask(self, X):
        rho, _ = _radial(z_of(X, self.d), self.a, self.d)
        return np.abs(rho - self.RADIUS) <= JUMP_BAND
