"""Meaningful regression tests for the independent diagnostic GN correction."""
import torch
from solver_checks import RedundantVector, correctness_checks, fixed, vector_target


def test_reduced_jacobian_matches_resolved_finite_differences():
    evidence = correctness_checks()
    assert evidence["truncated_nonzero_spectrum"]["finite_difference"][-1]["rel"] < 1e-7
    assert evidence["exactly_redundant_vector"]["finite_difference_relative_error"] < 1e-7


def test_test_targets_do_not_change_training_or_parameter_selection():
    X = torch.linspace(-1, 1, 41).view(-1, 1)
    Xt = torch.linspace(-.99, .99, 61).view(-1, 1)
    Y, Yt = vector_target(X), vector_target(Xt)
    first, second = RedundantVector(), RedundantVector()
    a = fixed.gauss_newton(first, X, Y, Xt, Yt, iters=8)
    b = fixed.gauss_newton(second, X, Y, Xt, -10*Yt, iters=8)
    assert a["train"] == b["train"]
    assert a["rank"] == b["rank"]
    assert torch.equal(first.theta, second.theta)
    assert a["test"] != b["test"]
    assert all(v <= u for u, v in zip(a["train"], a["train"][1:]))


def test_head_parameter_views_are_excluded():
    class HeadView(RedundantVector):
        def readout(self):
            return [self.W.view(-1), self.b]
    model = HeadView()
    assert [name for name, _ in fixed.nonlinear_named_parameters(model)] == ["theta"]
