"""Optimizer dispatch using PyTorch built-in optimizers.

Supports: Adam, SGD, LBFGS. No custom optimizers.
LBFGS uses PyTorch's built-in closure pattern with strong Wolfe line search,
replacing the need for SSBroyden or custom second-order methods.

Usage:
    optimizer = build_optimizer(stage_config, model)
    scheduler = build_scheduler(optimizer, stage_config, total_steps)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.config.schema import OptimizerStageConfig


def build_optimizer(config: OptimizerStageConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Build optimizer for a training stage.

    Only passes parameters with requires_grad=True to the optimizer.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters; all are frozen.")

    name = config.name.lower()
    kwargs = dict(config.kwargs) if config.kwargs else {}

    if name == "adam":
        return torch.optim.Adam(trainable, lr=config.learning_rate, **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(trainable, lr=config.learning_rate, **kwargs)
    elif name == "lbfgs":
        return torch.optim.LBFGS(
            trainable,
            lr=config.learning_rate,
            max_iter=kwargs.pop("max_iter", 20),
            history_size=kwargs.pop("history_size", 100),
            tolerance_grad=kwargs.pop("tolerance_grad", 1e-16),
            tolerance_change=kwargs.pop("tolerance_change", 1e-16),
            line_search_fn=kwargs.pop("line_search_fn", "strong_wolfe"),
            **kwargs,
        )
    elif name == "sgd":
        return torch.optim.SGD(trainable, lr=config.learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {config.name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OptimizerStageConfig,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build a cosine LR scheduler, or None if no schedule applies.

    LBFGS is always excluded: it sets its own step size via a strong-Wolfe line
    search, so cosine-annealing its lr (e.g. from 1.0 down to min_learning_rate)
    fights the line search rather than helping. We skip it even when
    use_cosine_schedule is left at its default True.
    """
    if config.name.lower() == "lbfgs":
        return None
    if config.use_cosine_schedule and total_steps > 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.min_learning_rate,
        )
    return None
