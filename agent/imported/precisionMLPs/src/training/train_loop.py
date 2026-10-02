"""Multi-stage training loop.

Supports Adam, SGD (standard step) and LBFGS (closure-based step).

Usage:
    result = run_training(config, model, dataset)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from src.config.schema import ExperimentConfig, OptimizerStageConfig
from src.data.dataset import Dataset
from src.training.losses import get_loss_fn
from src.training.optimizers import build_optimizer, build_scheduler
from src.training.metrics import MetricsCollector
from src.construction.initialize import initialize_with_readout_solve


@dataclass
class TrainResult:
    """Result of a complete training run."""
    loss_history: list[float] = field(default_factory=list)
    eval_metrics: dict[str, list] = field(default_factory=dict)
    eval_steps: list[int] = field(default_factory=list)
    stage_boundaries: list[int] = field(default_factory=list)
    final_metrics: dict[str, float] = field(default_factory=dict)


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> float:
    """Standard training step (Adam, SGD). Returns loss value."""
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model, x, y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train_step_lbfgs(model: nn.Module, optimizer: torch.optim.LBFGS,
                     x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> float:
    """LBFGS training step with closure."""
    last = {"loss": 0.0}

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, x, y)
        loss.backward()
        last["loss"] = float(loss.detach())
        return loss

    optimizer.step(closure)
    return last["loss"]


def run_training(config: ExperimentConfig, model: nn.Module, dataset: Dataset,
                 verbose: bool = False) -> TrainResult:
    """Execute full multi-stage training.

    For each stage in config.training.stages:
    1. Build optimizer and scheduler.
    2. Select step function (standard or LBFGS closure).
    3. Run training steps, evaluate at config.training.eval_interval.
    4. Optionally solve readout periodically via readout_solve_every.
    """
    loss_fn = get_loss_fn(config.training.loss, **config.training.loss_kwargs)
    collector = MetricsCollector(model, dataset)
    result = TrainResult(eval_metrics={})
    global_step = 0
    eval_interval = max(1, config.training.eval_interval)

    # Initial eval before training
    m = collector.collect(step=global_step)
    result.eval_steps.append(global_step)
    if verbose:
        print(f"[step {global_step:6d}] eval_linf={m['eval_linf']:.3e} rel_l2={m['eval_rel_l2']:.3e}")

    x_train, y_train = dataset.x_train, dataset.y_train

    for stage_idx, stage in enumerate(config.training.stages):
        result.stage_boundaries.append(global_step)
        optimizer = build_optimizer(stage, model)
        scheduler = build_scheduler(optimizer, stage, stage.steps)
        is_lbfgs = stage.name.lower() == "lbfgs"
        step_fn = train_step_lbfgs if is_lbfgs else train_step

        for step_in_stage in range(stage.steps):
            loss_val = step_fn(model, optimizer, x_train, y_train, loss_fn)
            result.loss_history.append(loss_val)
            global_step += 1

            if scheduler is not None:
                scheduler.step()

            # Periodic readout re-solve (exact least-squares refit of the
            # readout given the current inner layer). Skip when the readout is
            # frozen -- re-solving would silently overwrite a parameter the user
            # asked to hold fixed. Use the configured solver/penalty.
            rse = config.training.readout_solve_every
            if (rse > 0 and global_step % rse == 0
                    and model.readout.weight.requires_grad):
                method = config.init.readout_solve
                if method == "none":
                    method = "lstsq"
                initialize_with_readout_solve(
                    model, x_train, y_train,
                    method=method, ridge_alpha=config.init.ridge_alpha,
                )

            if global_step % eval_interval == 0:
                m = collector.collect(step=global_step, train_loss=loss_val)
                result.eval_steps.append(global_step)
                if verbose:
                    print(f"[stage {stage_idx} step {global_step:6d}] "
                          f"loss={loss_val:.3e} eval_linf={m['eval_linf']:.3e} "
                          f"rel_l2={m['eval_rel_l2']:.3e}")

    # Final eval at end
    m = collector.collect(step=global_step)
    if global_step not in result.eval_steps:
        result.eval_steps.append(global_step)
    result.final_metrics = m
    result.eval_metrics = collector.to_dict()
    return result
