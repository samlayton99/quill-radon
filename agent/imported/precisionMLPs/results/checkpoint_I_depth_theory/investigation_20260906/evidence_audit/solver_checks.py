"""Correctness and bounded empirical audit of the independent GN utility."""
import copy
import json
import os
import sys
import time
import types
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/i_evidence_fontcache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(REPO / "experiments/expI02_block_prior"))
sys.path.insert(0, str(REPO / "experiments/expI03_investigation/solver_audit"))
import torch
from torch import nn
import qi2
import tasks
from qiblocks import gauss_newton as old_gn
import varpro_gn as fixed

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class RedundantVector(nn.Module):
    """Two-output regression with a genuine exact redundant feature column."""
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([0.65, 0.35]))
        self.W = nn.Parameter(torch.zeros(3, 2))
        self.b = nn.Parameter(torch.zeros(2))

    def feats(self, X):
        x = X[:, 0]
        h = torch.tanh(self.theta[0] * x + self.theta[1])
        return torch.stack([h, 2*h, x.square()], dim=1)

    def forward(self, X):
        return self.feats(X) @ self.W + self.b

    def readout(self):
        return [self.W, self.b]

    def nonlinear_params(self):
        return [self.theta]

    def update_ranges(self, X, momentum=None):
        pass

    @torch.no_grad()
    def solve_readout(self, X, Y, rcond=1e-14):
        F = self.feats(X)
        A = torch.cat([F, torch.ones(len(X), 1)], dim=1)
        fit = fixed.fit_head(A, Y.reshape(len(Y), -1), rcond)
        fixed.assign_head(self, fit)
        return int(fit.keep.sum())


def correctness_checks():
    g = torch.Generator().manual_seed(5)
    U, _ = torch.linalg.qr(torch.randn(24, 5, generator=g))
    V, _ = torch.linalg.qr(torch.randn(5, 5, generator=g))
    A = U @ torch.diag(torch.tensor([8., 3., 1., .2, .05])) @ V.T
    Y = torch.randn(24, 2, generator=g)
    dA = torch.randn(24, 5, 3, generator=g)
    fit = fixed.fit_head(A, Y, rcond=.05)
    J = fixed.reduced_jacobian(fit, Y, dA, "full")
    jk = fixed.reduced_jacobian(fit, Y, dA, "kaufman")
    errors = []
    for eps in [1e-4, 1e-5, 1e-6]:
        fd = torch.stack([(fixed.fit_head(A + eps*dA[:, :, j], Y, .05).residual
                           - fixed.fit_head(A - eps*dA[:, :, j], Y, .05).residual)/(2*eps)
                          for j in range(3)], dim=-1)
        errors.append(dict(eps=eps, rel=float((J-fd).norm()/fd.norm())))
    assert errors[-1]["rel"] < 1e-7
    assert int(fit.keep.sum()) == 3
    exact_truncated = dict(rank=3, residual_norm=float(fit.residual.norm()), finite_difference=errors,
                           kaufman_relative_difference=float((jk-J).norm()/J.norm()))

    X = torch.linspace(-1, 1, 31).view(-1, 1)
    model = RedundantVector()
    theta, features, _ = fixed.feature_problem(model, X)
    Y2 = vector_target(X)
    A2 = features(theta)
    fit2 = fixed.fit_head(A2, Y2)
    dA2 = torch.func.jacfwd(features)(theta)
    J2 = fixed.reduced_jacobian(fit2, Y2, dA2)
    eps = 1e-5
    eye = torch.eye(theta.numel())
    fd2 = torch.stack([(fixed.fit_head(features(theta+eps*e), Y2).residual
                      - fixed.fit_head(features(theta-eps*e), Y2).residual)/(2*eps) for e in eye], dim=-1)
    fd_error = float((J2-fd2).norm()/fd2.norm())
    assert fd_error < 1e-7
    assert int(fit2.keep.sum()) == 3 and A2.shape[1] == 4
    bias_net = qi2.make_block(3, 4, 8, 1, 1, depth=1)
    old_has = any(p is bias_net.layers[0].bV for p in bias_net.nonlinear_params())
    new_has = any(p is bias_net.layers[0].bV for _, p in fixed.nonlinear_named_parameters(bias_net))
    assert not old_has and new_has
    return dict(truncated_nonzero_spectrum=exact_truncated,
                exactly_redundant_vector=dict(rank=3, columns=4, finite_difference_relative_error=fd_error,
                                             nonzero_residual_norm=float(fit2.residual.norm())),
                shallow_bias=dict(old_includes=old_has, corrected_includes=new_has))


def vector_target(X):
    x = X[:, 0]
    h = torch.tanh(1.8*x-.2)
    return torch.stack([h+.2*x*x+.04*torch.sin(6*x), -.7*h+.1*x*x+.06*torch.cos(5*x)], dim=1)


def run_old(model, X, Y, Xt, Yt, iters, rcond):
    """Observe old solver only at its initial/accepted iteration endpoints."""
    traces = dict(step=[], train=[], test=[], rank=[], accepted=[])
    capture = [False]
    orig_solve, orig_ranges = model.solve_readout, model.update_ranges

    def ranges(self, XX, momentum=None):
        orig_ranges(XX, momentum)
        capture[0] = True

    def solve(self, XX, YY, rcond=1e-14):
        rank = orig_solve(XX, YY, rcond)
        if capture[0]:
            with torch.no_grad():
                traces["step"].append(len(traces["step"]))
                traces["train"].append(float((self(X).reshape_as(Y)-Y).norm()/Y.norm()))
                traces["test"].append(float((self(Xt).reshape_as(Yt)-Yt).norm()/Yt.norm()))
                traces["rank"].append(rank)
                traces["accepted"].append(None)
            capture[0] = False
        return rank
    model.solve_readout = types.MethodType(solve, model)
    model.update_ranges = types.MethodType(ranges, model)
    old_gn(model, X, Y, Xt, Yt, iters=iters, chunk=32, rcond=rcond)
    traces["final"] = traces["test"][-1]
    return traces


def comparisons():
    cases = {}
    D = tasks.analytic("fast_waves", 3, 192, n_test=1024, seed=3)
    net = qi2.make_block(3, 3, 8, 2, 32, seed=3)
    qi2.calibrate(net, D["Xtr"])
    cases["scalar_qi_full_tail"] = (net, D["Xtr"], D["Ytr"].view(-1, 1), D["Xte"], D["Yte"].view(-1, 1), 1e-14)
    D = tasks.analytic("fast_waves", 3, 768, n_test=1024, seed=3)
    net = qi2.make_block(3, 3, 8, 2, 32, seed=3)
    qi2.calibrate(net, D["Xtr"])
    cases["scalar_qi_truncated"] = (net, D["Xtr"], D["Ytr"].view(-1, 1), D["Xte"], D["Yte"].view(-1, 1), 1e-6)
    X = torch.linspace(-1, 1, 65).view(-1, 1)
    Xt = torch.linspace(-.997, .997, 256).view(-1, 1)
    cases["redundant_vector"] = (RedundantVector(), X, vector_target(X), Xt, vector_target(Xt), 1e-14)
    result = {}
    for case, (net, X, Y, Xt, Yt, rcond) in cases.items():
        rows = {}
        for mode in ["old", "kaufman", "full"]:
            model = copy.deepcopy(net)
            t0 = time.monotonic()
            if mode == "old":
                rows[mode] = run_old(model, X, Y, Xt, Yt, 12, rcond)
            else:
                rows[mode] = fixed.gauss_newton(model, X, Y, Xt, Yt, iters=12, jacobian=mode, rcond=rcond)
            rows[mode]["rcond"] = rcond
            rows[mode]["seconds"] = time.monotonic()-t0
            train = rows[mode]["train"]
            assert all(b <= a + 1e-10*max(a, 1e-15) for a, b in zip(train, train[1:])), (case, mode, train)
            print(case, mode, "train", train[-1], "test", rows[mode]["test"][-1],
                  "rank", rows[mode]["rank"], "seconds", rows[mode]["seconds"], flush=True)
        result[case] = rows
    return result


def plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    styles = dict(old=("Original GN", "#b2182b", "o--"),
                  kaufman=("Corrected Kaufman GN", "#2166ac", "s-"),
                  full=("Exact reduced-residual GN", "#1b7837", "^-"))
    for col, (case, modes) in enumerate(results.items()):
        for mode, row in modes.items():
            label, color, style = styles[mode]
            for i, metric in enumerate(["train", "test"]):
                axes[i, col].semilogy(row["step"], row[metric], style, color=color, label=label, markersize=4)
        axes[0, col].set_title(case.replace("_", " "))
        axes[1, col].set_xlabel("GN iteration")
        for i in range(2):
            axes[i, col].set_ylim(1e-3, 1e3 if i == 1 and col == 0 else 1)
            axes[i, col].grid(alpha=.2)
    axes[0, 0].set_ylabel("Train relative L2")
    axes[1, 0].set_ylabel("Independent test relative L2")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / "solver_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.monotonic()
    result = dict(checks=correctness_checks())
    print(json.dumps(result["checks"], indent=2), flush=True)
    result["comparisons"] = comparisons()
    result["elapsed_seconds"] = time.monotonic()-t0
    (OUT / "solver_checks.json").write_text(json.dumps(result, indent=2)+"\n")
    plots(result["comparisons"])
    print("total_seconds", result["elapsed_seconds"], flush=True)
