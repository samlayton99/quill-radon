"""Recompute the revised rule and redraw the existing least-squares observations.

This script fits no networks. Original source_data/anchor_rule_*.json files are
read-only inputs. Predictions use effective tolerance 2**(1-p), the original
representative angular frequency, and a continuous root on [0.03, 1.5].
The practical score keeps the first pair and the exact central denominator;
amplitude and anchoring constants are absorbed into the effective tolerance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
OUT = ROOT / "figures"
ACTIVATIONS = ("tanh", "gelu", "swish", "gaussian")
ORDER = {"tanh": 1, "gelu": 2, "swish": 2, "gaussian": 0}
SEARCH = (0.03, 1.5)
ROOT_XTOL = 1e-13
TWO_PI = 2 * np.pi
GREEN, RED = "#21864B", "#C63839"
DISPLAY_ACTIVATIONS = ("tanh", "gelu")
DISPLAY_TARGETS = ("mix_2_6_10", "expsin")
DISPLAY_WIDTHS = (32, 64, 128, 256, 512)
# At p=16, the GELU refined thresholds exceed the observed lambda interval.
DISPLAY_PRECISIONS = (24, 32, 40, 48, 53)


def log_khat(activation, xi):
    """Log of the normalized transform magnitude Khat/Khat(0), without tail underflow."""
    xi = np.abs(np.asarray(xi, dtype=float))
    if activation == "gaussian":
        return -xi**2 / 4
    if activation == "gelu":
        return np.log1p(xi**2) - xi**2 / 2
    a = xi * (np.pi / 2 if activation == "tanh" else np.pi)
    nonzero = a > 0
    safe_a = np.where(nonzero, a, 1.0)
    log_one_minus = np.log(-np.expm1(-2 * safe_a))
    if activation == "tanh":
        value = np.log(2 * safe_a) - safe_a - log_one_minus
    elif activation == "swish":
        value = (2 * np.log(safe_a) - safe_a + np.log(2)
                 + np.log1p(np.exp(-2 * safe_a)) - 2 * log_one_minus)
    else:
        raise ValueError(activation)
    return np.where(nonzero, value, 0.0)


def log_refined_ratio(activation, lam, theta):
    """Log of the practical first-pair score displayed in the section."""
    if not 0 < theta < np.pi:
        raise ValueError("The representative grid frequency must lie in (0, pi).")
    r = ORDER[activation]
    first_minus = (r * np.log(theta / (TWO_PI - theta))
                   + log_khat(activation, (TWO_PI - theta) / lam))
    first_plus = (r * np.log(theta / (TWO_PI + theta))
                  + log_khat(activation, (TWO_PI + theta) / lam))
    denominator = log_khat(activation, theta / lam)
    return np.logaddexp(first_minus, first_plus) - denominator


def largest_feasible(log_score, eps):
    """Continuous threshold, with search-limit status and a monotonicity check."""
    lo, hi = SEARCH
    grid = np.geomspace(lo, hi, 1001)
    values = np.asarray(log_score(grid), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite alias score in the search interval.")
    # A numerical check over the stated interval guards these four built-ins;
    # it is not advertised as a monotonicity proof for arbitrary new kernels.
    if np.any(np.diff(values) < -1e-10):
        raise ValueError("The score is not numerically monotone on the search interval.")
    budget = np.log(eps)
    if values[0] > budget:
        return {"lambda": None, "status": "no_feasible_point"}
    if values[-1] <= budget:
        return {"lambda": hi, "status": "upper_search_limit"}
    root = brentq(lambda lam: float(log_score(lam) - budget), lo, hi,
                  xtol=ROOT_XTOL, rtol=1e-14)
    return {"lambda": float(root), "status": "root",
            "log_equation_residual": float(log_score(root) - budget)}


def make_predictions(original):
    inputs = {
        name: {"omega_scale": float(spec["omega_M"]),
               "source_convention": spec["how"]}
        for name, spec in original["spectral"].items()
    }
    prediction = {
        "version": "effective_tolerance_first_pair_continuous_root",
        "precision": "epsilon_p=2**(1-p), spacing above one; not unit roundoff",
        "basic_formula": "abs(Khat(2*pi/lambda)/Khat(0)) <= epsilon_p",
        "refined_formula": "(t_minus+t_plus)/Khat(theta/lambda) <= e_tol",
        "terms": {"t_minus": "(theta/(2*pi-theta))**r * Khat((2*pi-theta)/lambda)",
                  "t_plus": "(theta/(2*pi+theta))**r * Khat((2*pi+theta)/lambda)",
                  "theta": "2*omega_scale/N", "e_tol": "epsilon_p=2**(1-p)"},
        "normalization": "Target amplitude and the anchoring constant are absorbed into e_tol; no spectral mass is used. Later-pair correction is omitted.",
        "search": {"interval": list(SEARCH), "method": "scipy.optimize.brentq",
                   "xtol": ROOT_XTOL, "rtol": 1e-14,
                   "numerical_monotonicity_check": "1001 logarithmically spaced points",
                   "upper_search_limit_is_threshold": False},
        "scope": "Practical prediction range; the tanh representation theorem additionally requires lambda<=1 and admissible width. The representative frequency is a rough target summary, instantiated here with expC08's original weighted frequency. General kernels are predictors, not consequences of the tanh theorem.",
        "observations": "Original frozen-grid least-squares fits; no network was rerun.",
        "spectral": inputs, "width": {}, "precision": {},
    }
    for sweep in ("width", "precision"):
        for key in original[sweep]:
            activation, target, parameter = key.split("|")
            n, p = (int(parameter), 53) if sweep == "width" else (128, int(parameter))
            spec = inputs[target]
            eps = 2.0 ** (1-p)
            theta = 2 * spec["omega_scale"] / n
            basic = largest_feasible(lambda lam: log_khat(activation, TWO_PI/lam), eps)
            if 0 < theta < np.pi:
                refined = largest_feasible(
                    lambda lam: log_refined_ratio(activation, lam, theta), eps)
            else:
                refined = {"lambda": None, "status": "frequency_outside_domain"}
            prediction[sweep][key] = {"N": n, "p": p, "epsilon": eps,
                                      "theta_scale": theta,
                                      "basic": basic, "refined": refined}
    return prediction


def curve(rows, sweep, act, fn, n, p):
    points = sorted((r for r in rows[sweep] if r["act"] == act and r["fn"] == fn
                     and r["N"] == n and r["p"] == p), key=lambda r: r["lam"])
    return np.array([r["lam"] for r in points]), np.array([r["rel_l2"] for r in points])


def error_at(x, y, lam, smoothed=False):
    z = np.log10(np.maximum(y, 1e-300))
    if smoothed:
        z = np.array([np.median(z[max(0, i-2):min(len(z), i+3)]) for i in range(len(z))])
    return float(10**np.interp(math.log(lam), np.log(x), z))


def score(x, y, lam):
    z = np.log10(np.maximum(y, 1e-300))
    smoothed = np.array([np.median(z[max(0, i-2):min(len(z), i+3)]) for i in range(len(z))])
    return float(10**(np.interp(math.log(lam), np.log(x), smoothed) - np.min(smoothed)))


def draw_line(ax, prediction, color, style, **kwargs):
    # A capped point is a search boundary, not a located error transition.
    if prediction["status"] == "root":
        ax.axvline(prediction["lambda"], color=color, ls=style, **kwargs)


def save_figure(fig, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}")
    plt.close(fig)


def draw_figures(prediction, rows):
    """Match expC08's viridis, log-log axes, complete curves and light grid.

    Separate equal-sized halves, each with its own labels and legend. Rule
    markers use interpolated observations only for their vertical coordinates.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "figure.dpi": 160, "savefig.dpi": 210,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(16, 8.3))
    displayed = []
    cmap = plt.get_cmap("viridis")
    for block, sweep in enumerate(("width", "precision")):
        parameters = DISPLAY_WIDTHS if sweep == "width" else DISPLAY_PRECISIONS
        colors = [cmap(i/(len(parameters)-1)) for i in range(len(parameters))]
        # The same margins inside each half keep the divider exactly at x=1/2.
        gs = fig.add_gridspec(2, 2, left=.058+.5*block, right=.481+.5*block,
                             bottom=.145, top=.835, wspace=.055, hspace=.08)
        for row, target in enumerate(DISPLAY_TARGETS):
            for col, act in enumerate(DISPLAY_ACTIVATIONS):
                ax = fig.add_subplot(gs[row, col])
                traces = {"basic": [], "refined": []}
                for value, color in zip(parameters, colors):
                    n, p = (value, 53) if sweep == "width" else (128, value)
                    x, y = curve(rows, sweep, act, target, n, p)
                    if len(x) != 40:
                        raise ValueError("Expected an original complete 40-point curve.")
                    ax.loglog(x, y, color=color, lw=1.3, zorder=2)
                    key = f"{act}|{target}|{value}"
                    pr = prediction[sweep][key]
                    for arm in ("basic", "refined"):
                        if pr[arm]["status"] != "root":
                            raise ValueError(f"Unlocated prediction: {sweep}, {key}, {arm}")
                        lam = pr[arm]["lambda"]
                        measured = error_at(x, y, lam)
                        traces[arm].append((lam, measured))
                        displayed.append({"sweep": sweep, "act": act, "fn": target,
                                          "N": n, "p": p, "rule": arm,
                                          "predicted_lambda": lam,
                                          "measured_error_at_prediction": measured})
                for arm, color, marker, style in [
                    ("basic", GREEN, "o", "--"), ("refined", RED, "D", "-")]:
                    points = np.asarray(traces[arm])
                    if sweep == "width" and arm == "basic":
                        ax.axvline(points[0, 0], color=color, lw=1.3, ls=style, zorder=3)
                    else:
                        ax.plot(points[:, 0], points[:, 1], color=color,
                                lw=1.3, ls=style, zorder=4)
                    ax.scatter(points[:, 0], points[:, 1], c=colors,
                               marker=marker, s=28, edgecolors=color,
                               linewidths=1.25, zorder=6 if arm == "refined" else 5)
                ax.set_xlim(.03, 1.5)
                ax.set_ylim(1e-16, 1e1)
                ax.set_yticks([10.**k for k in range(-16, 1, 2)])
                ax.grid(True, which="both", alpha=.25, lw=.65)
                ax.tick_params(axis="both", labelsize=9, length=3, width=.7)
                if col == 0:
                    name = "Sine mixture (2, 6, 10)" if row == 0 else r"$\exp(\sin(3\pi x))$"
                    ax.set_ylabel(name + "\n" + r"relative $L^2$ error", fontsize=10)
                else:
                    ax.tick_params(axis="y", labelleft=False)
                if row == 1:
                    ax.set_xlabel(r"$\lambda$", labelpad=3)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                    title = "GELU" if act == "gelu" else act
                    ax.set_title(f"{title}  ($r={ORDER[act]}$)", pad=8)
        center = .25+.5*block
        header = (r"Width sweep: $\varepsilon=2^{-52}$" if sweep == "width"
                  else r"Precision sweep: $N=128$, $\varepsilon_p=2^{1-p}$")
        fig.text(center, .965, header, ha="center", va="top", fontsize=13)
        symbol = "N" if sweep == "width" else "p"
        handles = [Line2D([0], [0], color=c, lw=1.6,
                          label=rf"${symbol}={value}$")
                   for value, c in zip(parameters, colors)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(center, .931),
                   ncol=len(parameters), frameon=False, fontsize=9,
                   handlelength=1.8, columnspacing=1.1, handletextpad=.4)
    fig.add_artist(Line2D([.5, .5], [.115, .975], transform=fig.transFigure,
                         color=".35", lw=1.0, zorder=0))
    rule_handles = [Line2D([0], [0], color=GREEN, lw=1.5, ls="--", marker="o",
                          mfc="white", label="Basic rule"),
                    Line2D([0], [0], color=RED, lw=1.5, marker="D",
                           mfc="white", label="Refined rule")]
    fig.legend(handles=rule_handles, loc="lower center", bbox_to_anchor=(.5, .013),
               ncol=2, frameon=False, fontsize=11, columnspacing=3, handlelength=2.7)
    save_figure(fig, "lambda_rule_grid")
    (OUT/"figure_layout.json").write_text(json.dumps({
        "shape": [2,4], "rows": list(DISPLAY_TARGETS),
        "columns": [f"{sweep}:{act}" for sweep in ("width", "precision")
                    for act in DISPLAY_ACTIVATIONS],
        "widths": list(DISPLAY_WIDTHS), "precision_bits": list(DISPLAY_PRECISIONS),
        "precision_selection": "p=24,32,40,48,53; GELU refined predictions at p=16 exceed the measured lambda interval.",
        "colormap": "viridis", "xscale": "log", "yscale": "log",
        "xlim": [.03, 1.5], "ylim": [1e-16, 10], "divider_figure_x": .5,
        "basic_color": GREEN, "refined_color": RED,
        "prediction_trace": "Predicted lambda choices on corresponding observed curves; y is measured, not predicted. Width-independent basic choice is a vertical dashed line.",
        "style_reference": "expC08 anchor_rule_width_mean.png / anchor_rule_precision_mean.png",
        "points": displayed}, indent=2)+"\n")


def summarize(prediction, rows):
    summary, selected, displayed_precision = {}, [], []
    for sweep in ("width", "precision"):
        for act in ACTIVATIONS:
            values = []
            for key, pr in prediction[sweep].items():
                a, fn, _ = key.split("|")
                if a != act or any(pr[arm]["status"] != "root" for arm in ("basic", "refined")):
                    continue
                x, y = curve(rows, sweep, a, fn, pr["N"], pr["p"])
                values.append([score(x, y, pr[arm]["lambda"]) for arm in ("basic", "refined")])
            ar = np.asarray(values)
            summary[f"{sweep}:{act}"] = {
                "n": len(values), "basic_median": float(np.median(ar[:, 0])),
                "refined_median": float(np.median(ar[:, 1])),
                "basic_p90": float(np.quantile(ar[:, 0], .9)),
                "refined_p90": float(np.quantile(ar[:, 1], .9))}
    for act in DISPLAY_ACTIVATIONS:
        for fn, n in ((f, n) for f in DISPLAY_TARGETS for n in DISPLAY_WIDTHS):
            pr = prediction["width"][f"{act}|{fn}|{n}"]
            x, y = curve(rows, "width", act, fn, n, 53)
            selected.append({"act": act, "fn": fn, **pr,
                             **{f"{arm}_score": score(x, y, pr[arm]["lambda"])
                                for arm in ("basic", "refined")},
                             **{f"{arm}_interpolated_error": error_at(x, y, pr[arm]["lambda"])
                                for arm in ("basic", "refined")}})
    for act in DISPLAY_ACTIVATIONS:
        for fn in DISPLAY_TARGETS:
            for p in DISPLAY_PRECISIONS:
                displayed_precision.append({"act": act, "fn": fn, **prediction["precision"][f"{act}|{fn}|{p}"]})
    counts = {}
    for sweep in ("width", "precision"):
        counts[sweep] = {}
        for arm in ("basic", "refined"):
            statuses = [pr[arm]["status"] for pr in prediction[sweep].values()]
            counts[sweep][arm] = {status: statuses.count(status) for status in sorted(set(statuses))}
    return {"source": "Original anchor_rule_2026-09-09 observed fits; revised predictions",
            "metric": "Five-point median log error; interpolation in log lambda; ratio to minimum of same smoothed 40-point curve; common uncapped cases. Raw interpolated errors use the unsmoothed curve.",
            "summary": summary, "selected": selected, "displayed_precision": displayed_precision,
            "prediction_status_counts": counts}


def main():
    original_paths = [DATA / "anchor_rule_predictions_mean.json"] + [
        DATA / f"anchor_rule_rows_{sweep}.json" for sweep in ("width", "precision")]
    hashes_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original_paths}
    original = json.loads(original_paths[0].read_text())
    rows = {sweep: json.loads((DATA / f"anchor_rule_rows_{sweep}.json").read_text())
            for sweep in ("width", "precision")}
    prediction = make_predictions(original)
    prediction["original_input_sha256"] = hashes_before
    (DATA / "revised_rule_predictions.json").write_text(json.dumps(prediction, indent=2, allow_nan=False) + "\n")
    OUT.mkdir(exist_ok=True)
    draw_figures(prediction, rows)
    statistics = summarize(prediction, rows)
    statistics["original_input_sha256"] = hashes_before
    (OUT / "figure_statistics.json").write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")
    hashes_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("An original input file changed.")
    print(json.dumps({"figure": "lambda_rule_grid", "shape": [2, 4],
                      "widths": DISPLAY_WIDTHS, "precisions": DISPLAY_PRECISIONS,
                      "original_inputs_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
