"""Standalone weak/entropy residual diagnostic, not a universal shock solver.

A moving tanh front is a deliberately restricted constructed-neuron family.
Only flux, entropy, initial states, and integration cells enter the fits. Exact
Rankine--Hugoniot speeds are evaluation references, never fitting residuals.
The same flux-balance functions are used for quadratic and cubic fluxes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class ScalarLaw:
    name: str
    flux: object
    entropy: object
    entropy_flux: object


@dataclass(frozen=True)
class TanhFront:
    left: float
    right: float
    epsilon: float
    position: float
    speed: float

    def __call__(self, x, t):
        y = (np.asarray(x) - self.position - self.speed * np.asarray(t)) / self.epsilon
        return (self.left + self.right) / 2 + (self.right - self.left) / 2 * np.tanh(y)

    def spatial_breaks(self, t):
        center = self.position + self.speed * t
        return [center + a * self.epsilon for a in (-12., 0., 12.)]

    def temporal_breaks(self, x):
        if abs(self.speed) < 1e-14:
            return []
        return [(x - self.position + a * self.epsilon) / self.speed for a in (-12., 0., 12.)]


@dataclass(frozen=True)
class RampFront:
    """Piecewise-linear front with unknown propagation and spreading rates.

    This is a restricted ReLU-type family, not a solved PDE formula. At any
    fixed time its clipped ramp is exactly a difference of two ReLUs.
    """
    epsilon: float
    position: float
    speed: float
    spreading: float

    def __call__(self, x, t):
        width = self.epsilon + self.spreading * np.asarray(t)
        return np.clip(.5 + (np.asarray(x)-self.position-self.speed*np.asarray(t))/width, 0., 1.)

    def spatial_breaks(self, t):
        center = self.position + self.speed*t
        width = self.epsilon + self.spreading*t
        return [center-width/2, center+width/2]

    def temporal_breaks(self, x):
        edges = []
        for sign in (-1., 1.):
            speed = self.speed + sign*self.spreading/2
            if abs(speed) > 1e-14:
                edges.append((x-self.position-sign*self.epsilon/2)/speed)
        return edges


GX, GW = leggauss(48)


def integral(fn, a, b, breaks=()):
    """Quadrature split at representation transitions, not reference features."""
    edges = [a] + sorted({float(z) for z in breaks if a < z < b}) + [b]
    value = 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        points = (lo + hi) / 2 + (hi - lo) / 2 * GX
        value += (hi - lo) / 2 * np.dot(GW, fn(points))
    return float(value)


def control_volume_balance(field, density, flux, cell):
    """Integral of d_t density(u) + d_x flux(u) on a spacetime rectangle."""
    xl, xr, t0, t1 = cell
    top = integral(lambda x: density(field(x, t1)), xl, xr, field.spatial_breaks(t1))
    bottom = integral(lambda x: density(field(x, t0)), xl, xr, field.spatial_breaks(t0))
    right = integral(lambda t: flux(field(xr, t)), t0, t1, field.temporal_breaks(xr))
    left = integral(lambda t: flux(field(xl, t)), t0, t1, field.temporal_breaks(xl))
    return top - bottom + right - left


def damped_gauss_newton(residual, start, lower=None, upper=None):
    """Small-parameter diagnostic; no claim to be the general LSMR engine."""
    theta = np.asarray(start, dtype=float)
    size = len(theta)
    lower = np.full(size, -np.inf) if lower is None else np.asarray(lower)
    upper = np.full(size, np.inf) if upper is None else np.asarray(upper)
    damping = 1e-6
    trace = []
    for iteration in range(30):
        r = residual(theta)
        trace.append({"iteration": iteration, "parameters": theta.tolist(),
                      "residual_l2": float(np.linalg.norm(r))})
        if np.linalg.norm(r) < 2e-12:
            break
        jac = np.column_stack([(residual(theta + h * np.eye(size)[j]) -
                                residual(theta - h * np.eye(size)[j])) / (2*h)
                               for j in range(size) for h in [1e-5]])
        accepted = False
        for _ in range(15):
            delta = np.linalg.solve(jac.T @ jac + damping * np.eye(size), -jac.T @ r)
            candidate = np.clip(theta + delta, lower, upper)
            if np.linalg.norm(residual(candidate)) < np.linalg.norm(r):
                theta = candidate
                damping = max(damping / 5, 1e-14)
                accepted = True
                break
            damping *= 10
        if not accepted:
            break
    return theta, trace


def fit_expanding_front(law, enforce_entropy):
    """Same weak residual + optional inequality selects among two branches."""
    epsilon = 1e-5
    cells = [(-.12, .16, .04, .51), (.02, .37, .11, .83),
             (.14, .43, .20, .77), (-.31, .12, .05, .62),
             (.25, .64, .21, .93), (-.6, .6, .17, .71)]
    entropy_cells = cells + [(-2., 2., .11, .73)]

    def residual(theta):
        field = RampFront(epsilon, *theta)
        balances = [control_volume_balance(field, lambda u: u, law.flux, c) for c in cells]
        initial_mass = integral(lambda x: field(x, 0.), -2., 2., field.spatial_breaks(0.))-2.
        if enforce_entropy:
            balances += [max(control_volume_balance(field, law.entropy, law.entropy_flux, c), 0.) for c in entropy_cells]
        return np.asarray(balances+[initial_mass])

    theta, trace = damped_gauss_newton(residual, [.0, .5, .001],
                                     [-.2, .1, 0.], [.2, .9, 2.])
    field = RampFront(epsilon, *theta)
    validation_cells = [(-.073, .213, .057, .533), (.093, .583, .213, .977),
                        (-.4, .62, .032, .822), (-2., 2., .137, .719)]
    entropy = [control_volume_balance(field, law.entropy, law.entropy_flux, c) for c in validation_cells]
    conservation = [control_volume_balance(field, lambda u: u, law.flux, c) for c in validation_cells]
    x = np.linspace(-.5, 1.5, 40001)
    # Exact self-similar reference is used only here after optimization.
    t = .8
    reference = np.clip(x/t, 0., 1.)
    l1_error = float(np.trapezoid(np.abs(field(x, t)-reference), x))
    return {"entropy_enforced": enforce_entropy, "epsilon": epsilon,
            "position": float(theta[0]), "speed": float(theta[1]), "spreading_rate": float(theta[2]),
            "fit_residual_l2": float(np.linalg.norm(residual(theta))),
            "independent_weak_balance_l2": float(np.linalg.norm(conservation)),
            "maximum_independent_entropy_violation": float(max(max(entropy), 0.)),
            "heldout_l1_error_t0_8": l1_error,
            "trace": trace}


def fit_front(law, left, right, epsilon=1e-3):
    # The prescribed initial step is at x=0. Its mass on [-2,2] is known data.
    initial_mass = 2 * (left + right)
    cells = [(-2., 2., t0, t1) for t0, t1 in ((0., .2), (.2, .47), (.47, .8))]

    def residual(theta):
        field = TanhFront(left, right, epsilon, theta[0], theta[1])
        balances = [control_volume_balance(field, lambda u: u, law.flux, c) for c in cells]
        initial = integral(lambda x: field(x, 0.), -2., 2., field.spatial_breaks(0.)) - initial_mass
        return np.asarray(balances + [initial])

    parameters, trace = damped_gauss_newton(residual, [.3, -.15])
    field = TanhFront(left, right, epsilon, *parameters)
    expected_speed = (law.flux(right) - law.flux(left)) / (right - left)
    enclosing = (-2., 2., .11, .73)
    entropy_rate = control_volume_balance(field, law.entropy, law.entropy_flux, enclosing) / (.73-.11)
    expected_entropy_rate = law.entropy_flux(right) - law.entropy_flux(left) - expected_speed * (law.entropy(right)-law.entropy(left))
    return {
        "law": law.name, "left": left, "right": right,
        "epsilon": epsilon, "position": float(parameters[0]), "speed": float(parameters[1]),
        "reference_speed_only_for_evaluation": float(expected_speed),
        "speed_absolute_error": float(abs(parameters[1]-expected_speed)),
        "fit_residual_l2": float(np.linalg.norm(residual(parameters))),
        "independent_entropy_production_rate": float(entropy_rate),
        "entropy_violation": float(max(entropy_rate, 0.)),
        "reference_entropy_rate_only_for_evaluation": float(expected_entropy_rate),
        "trace": trace,
    }


def width_diagnostic(law):
    # Independent rectangles do not all enclose the entire front. Their traces
    # see different fractions of the transition and test weak consistency.
    cells = [(-.12, .16, .04, .51), (.02, .37, .11, .83),
             (.14, .43, .20, .77), (-.31, .12, .05, .62),
             (.25, .64, .21, .93), (-.6, .6, .17, .71)]
    rows = []
    for epsilon in (.1, .03, .01, .003, .001, .0003):
        field = TanhFront(1., 0., epsilon, 0., .5)
        weak = np.asarray([control_volume_balance(field, lambda u: u, law.flux, c) for c in cells])
        # For Burgers at the RH speed: R=(tanh y sech^2 y)/(4 epsilon).
        strong_squared = integral(lambda y: (np.tanh(y) / np.cosh(y)**2)**2, -24., 24., [-12., 0., 12.]) / (16*epsilon)
        rows.append({"epsilon": epsilon, "weak_cell_balance_l2": float(np.linalg.norm(weak)),
                     "weak_cell_balance_max": float(np.max(np.abs(weak))),
                     "strong_residual_l2_space": float(np.sqrt(strong_squared)),
                     "strong_residual_squared_exact": float(1/(60*epsilon)),
                     "strong_integral_relative_error": float(abs(strong_squared*60*epsilon-1))})
    return rows


def main():
    quadratic = ScalarLaw("quadratic flux", lambda u: u*u/2, lambda u: u*u/2, lambda u: u*u*u/3)
    cubic = ScalarLaw("cubic flux", lambda u: u*u*u/3, lambda u: u*u/2, lambda u: u**4/4)
    fits = [fit_front(law, left, right) for law in (quadratic, cubic)
            for left, right in ((1., 0.), (0., 1.))]
    width = width_diagnostic(quadratic)
    expansion = [fit_expanding_front(quadratic, enabled) for enabled in (False, True)]
    for row in fits:
        assert row["speed_absolute_error"] < 1e-9
        assert abs(row["position"]) < 1e-9
        assert abs(row["independent_entropy_production_rate"] - row["reference_entropy_rate_only_for_evaluation"]) < 1e-9
        assert row["independent_entropy_production_rate"] * (row["left"] - row["right"]) < 0
    assert max(row["strong_integral_relative_error"] for row in width) < 1e-10
    assert width[-1]["weak_cell_balance_l2"] < width[0]["weak_cell_balance_l2"] / 100
    assert expansion[0]["heldout_l1_error_t0_8"] > .1
    assert expansion[1]["heldout_l1_error_t0_8"] < 1e-4
    output = {
        "scope": "Restricted moving tanh-front family; generic scalar flux and entropy callbacks; not automatic shock/rarefaction discovery.",
        "fit_input": "Initial step mass and spacetime conservation balances only. No PDE solution labels or analytic speed enters residuals.",
        "optimization": "Damped Gauss-Newton in two/three front parameters, finite-difference Jacobian, tiny dense linear solve; separate diagnostic, not general residual engine.",
        "quadrature": "48-point Gauss-Legendre on subintervals split at the current represented transition; no reference solution breakpoints.",
        "interpretation": "Conservation determines front speed but permits an expansion shock. Adding an unknown spreading rate and entropy inequalities selects the rarefaction in a restricted ramp family. This is not automatic representation discovery.",
        "fits": fits, "width_diagnostic": width, "expansion_branch_selection": expansion,
    }
    root = Path(__file__).resolve().parents[2]
    out = root / "results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual"
    out.mkdir(parents=True, exist_ok=True)
    (out / "shock_diagnostics.json").write_text(json.dumps(output, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.linspace(-.3, 1.1, 3000)
    axes[0].plot(x, np.clip(x/.8, 0., 1.), color="black", lw=3, alpha=.55,
                 label="Physical rarefaction (evaluation only)")
    for row, color, label in zip(expansion, ("#c64b48", "#2862bd"),
                                 ("Conservation only", "Conservation + entropy")):
        field = RampFront(row["epsilon"], row["position"], row["speed"], row["spreading_rate"])
        axes[0].plot(x, field(x, .8), color=color, lw=1.8, label=label)
    axes[0].set(xlabel="x", ylabel="u(x, 0.8)", title="Same family and optimizer; added entropy inequality")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].loglog([row["epsilon"] for row in width], [row["strong_residual_l2_space"] for row in width],
                   "o-", color="#c64b48", label="Strong residual L2")
    axes[1].loglog([row["epsilon"] for row in width], [row["weak_cell_balance_l2"] for row in width],
                   "o-", color="#2862bd", label="Weak cell-balance norm")
    axes[1].invert_xaxis()
    axes[1].set(xlabel="Transition width ε (sharper toward right)", ylabel="Residual norm (different definitions)",
                title="Correct shock sharpens: strong loss worsens")
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.savefig(out / "shock_diagnostics.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"fits": [{k: row[k] for k in ("law", "left", "right", "speed", "speed_absolute_error", "fit_residual_l2", "independent_entropy_production_rate")} for row in fits],
                      "width_diagnostic": width, "expansion_branch_selection": [{k: v for k, v in row.items() if k != "trace"} for row in expansion]}, indent=2))


if __name__ == "__main__":
    main()
