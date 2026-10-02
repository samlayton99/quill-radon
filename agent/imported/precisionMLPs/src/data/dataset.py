"""Dataset construction from config.

Eval points are always dense equispaced for consistent L_inf measurement.
Noise is applied only to training data.

Seed semantics (important when interpreting multi-seed runs): the seed only
changes the data for ``sampling="uniform"`` and for the noise paths
(x_noise_std / y_noise_std). For ``equispaced``, ``chebyshev`` and ``qi_grid``
with no noise, the training data is identical across seeds, so any seed-to-seed
variation comes from model initialization and training stochasticity, not data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.config.schema import ExperimentConfig
from src.data.targets import TargetFn, get_target
from src.data.sampling import get_sampling_fn, equispaced


@dataclass
class Dataset:
    """Train/eval data. All tensors float64, shape [n, 1]."""
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_eval: torch.Tensor
    y_eval: torch.Tensor
    target_fn: TargetFn


def build_dataset(config: ExperimentConfig, seed: int = 0) -> Dataset:
    """Build train/eval dataset from experiment config.

    1. Look up target function via config.target.
    2. Sample x_train using config.sampling strategy.
    3. Compute y_train = target.fn(x_train).
    4. If config.x_noise_std > 0: perturb x_train, recompute y on perturbed x.
    5. If config.y_noise_std > 0: add Gaussian noise to y_train.
    6. x_eval = equispaced(config.n_eval, config.domain). No noise on eval.
    """
    target = get_target(config.target)
    sampling_fn = get_sampling_fn(config.sampling)

    # Sample training points
    if config.sampling == "uniform":
        x_train = sampling_fn(config.n_train, config.domain, seed=seed)
    elif config.sampling == "qi_grid":
        # qi_grid uses N instead of n_points; interpret n_train as N
        x_train = sampling_fn(config.n_train, config.domain, halo=0)
    else:
        x_train = sampling_fn(config.n_train, config.domain)

    # Optionally perturb x (breaks uniform grid)
    if config.x_noise_std > 0.0:
        gen = torch.Generator().manual_seed(seed + 1000)
        noise = torch.randn(x_train.shape, generator=gen, dtype=torch.float64) * config.x_noise_std
        x_train = x_train + noise

    y_train = target.fn(x_train)

    # Optionally add y-noise
    if config.y_noise_std > 0.0:
        gen = torch.Generator().manual_seed(seed + 2000)
        noise = torch.randn(y_train.shape, generator=gen, dtype=torch.float64) * config.y_noise_std
        y_train = y_train + noise

    # Eval always clean and equispaced
    x_eval = equispaced(config.n_eval, config.domain)
    y_eval = target.fn(x_eval)

    return Dataset(x_train=x_train, y_train=y_train,
                   x_eval=x_eval, y_eval=y_eval, target_fn=target)
