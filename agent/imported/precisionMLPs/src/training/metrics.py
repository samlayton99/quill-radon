"""Uniform metrics collection across all experiments.

MetricsCollector computes a fixed set of metrics at every eval step,
ensuring cross-experiment comparability.

Metrics logged:
- step, train_loss, eval_loss, eval_linf, eval_rel_l2
- gamma_mean/median/max/min, lambda_mean/median/max
- readout_weight_max, readout_weight_l2
- feature_rank, feature_stable_rank, phi_smin, phi_smax, phi_cond
"""

from __future__ import annotations

import json
from collections import defaultdict

import torch
import torch.nn as nn

from src.data.dataset import Dataset


class MetricsCollector:
    """Collects and stores metrics during training."""

    def __init__(self, model: nn.Module, dataset: Dataset, h: float | None = None,
                 rank_tol: float = 1e-12):
        self._model = model
        self._dataset = dataset
        self._history: dict[str, list] = defaultdict(list)
        self._rank_tol = rank_tol
        # h is only used to report the dimensionless lambda = gamma * h.
        self._h = h if h is not None else self._infer_h(model, dataset)

    @staticmethod
    def _infer_h(model: nn.Module, dataset: Dataset) -> float:
        """Infer the grid spacing h for lambda = gamma * h reporting.

        Prefer the actual spacing between adjacent centers (exact for the QI
        uniform grid, regardless of halo), since the construction defines
        h = 2/N with centers exactly h apart. Fall back to domain-width / width
        only when centers are unavailable.
        """
        if hasattr(model, "get_centers"):
            c = model.get_centers().detach().reshape(-1)
            if c.numel() >= 2:
                diffs = torch.diff(torch.sort(c).values)
                return float(diffs.median())
        a, b = float(dataset.x_eval.min()), float(dataset.x_eval.max())
        cfg = getattr(model, "config", None)
        width = getattr(cfg, "width", None) or dataset.x_train.shape[0]
        return (b - a) / max(width, 1)

    @torch.no_grad()
    def collect(self, step: int, train_loss: float | None = None) -> dict[str, float]:
        """Compute all metrics at current model state."""
        model = self._model
        ds = self._dataset

        # Eval metrics
        eval_pred = model(ds.x_eval)
        resid = eval_pred - ds.y_eval
        eval_loss = float((resid ** 2).mean())
        eval_linf = float(resid.abs().max())
        eval_rel_l2 = float(torch.linalg.norm(resid) / torch.linalg.norm(ds.y_eval))

        # Train loss (if not provided, compute)
        if train_loss is None:
            train_pred = model(ds.x_train)
            train_loss = float(((train_pred - ds.y_train) ** 2).mean())

        # Gamma / lambda stats
        gamma = model.get_gamma() if hasattr(model, "get_gamma") else None
        if gamma is not None and gamma.numel() > 0:
            g = gamma.detach().reshape(-1).abs()
            gamma_mean = float(g.mean())
            gamma_median = float(g.median())
            gamma_max = float(g.max())
            gamma_min = float(g.min())
            lam = g * self._h
            lambda_mean = float(lam.mean())
            lambda_median = float(lam.median())
            lambda_max = float(lam.max())
        else:
            gamma_mean = gamma_median = gamma_max = gamma_min = float("nan")
            lambda_mean = lambda_median = lambda_max = float("nan")

        # Readout weight stats
        if hasattr(model, "get_readout_weights"):
            v = model.get_readout_weights().detach().reshape(-1)
            readout_weight_max = float(v.abs().max())
            readout_weight_l2 = float(torch.linalg.norm(v))
        else:
            readout_weight_max = readout_weight_l2 = float("nan")

        # Feature rank diagnostics (SVD of Phi on eval grid)
        if hasattr(model, "features"):
            Phi = model.features(ds.x_eval).detach()
            # Compute singular values
            s = torch.linalg.svdvals(Phi)
            smax = float(s.max())
            smin = float(s.min())
            phi_cond = smax / smin if smin > 0 else float("inf")
            # Effective rank: count of sigma_i > tol * sigma_max
            tol = self._rank_tol * smax
            feature_rank = int((s > tol).sum())
            # Stable rank = ||Phi||_F^2 / ||Phi||_2^2
            frob2 = float((s ** 2).sum())
            feature_stable_rank = frob2 / (smax ** 2) if smax > 0 else 0.0
        else:
            smax = smin = phi_cond = float("nan")
            feature_rank = -1
            feature_stable_rank = float("nan")

        metrics = {
            "step": step,
            "train_loss": float(train_loss),
            "eval_loss": eval_loss,
            "eval_linf": eval_linf,
            "eval_rel_l2": eval_rel_l2,
            "gamma_mean": gamma_mean,
            "gamma_median": gamma_median,
            "gamma_max": gamma_max,
            "gamma_min": gamma_min,
            "lambda_mean": lambda_mean,
            "lambda_median": lambda_median,
            "lambda_max": lambda_max,
            "readout_weight_max": readout_weight_max,
            "readout_weight_l2": readout_weight_l2,
            "feature_rank": feature_rank,
            "feature_stable_rank": feature_stable_rank,
            "phi_smin": smin,
            "phi_smax": smax,
            "phi_cond": phi_cond,
        }

        for k, val in metrics.items():
            self._history[k].append(val)
        return metrics

    def to_dict(self) -> dict[str, list]:
        """Return full metric history."""
        return dict(self._history)

    def to_jsonl(self, path: str) -> None:
        """Write metric history as JSON-lines file (one step per line)."""
        history = self._history
        if not history:
            return
        keys = list(history.keys())
        n_steps = len(history["step"])
        with open(path, "w") as f:
            for i in range(n_steps):
                row = {k: history[k][i] for k in keys}
                f.write(json.dumps(row) + "\n")


@torch.no_grad()
def compute_eval_metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Standalone eval metrics: mse, linf, rel_l2."""
    pred = model(x)
    resid = pred - y
    return {
        "mse": float((resid ** 2).mean()),
        "linf": float(resid.abs().max()),
        "rel_l2": float(torch.linalg.norm(resid) / torch.linalg.norm(y)),
    }
