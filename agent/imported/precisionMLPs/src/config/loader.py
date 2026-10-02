"""YAML config loading and sweep expansion.

Usage:
    config = load_config("experiments/expC01_lambda_tradeoff/config.yaml")
    configs = expand_sweep(config)
"""

from __future__ import annotations

import yaml
from dataclasses import asdict, fields, is_dataclass, replace
from itertools import product
from copy import deepcopy
from typing import Any

from src.config.schema import ExperimentConfig


def load_config(yaml_path: str, overrides: dict | None = None) -> ExperimentConfig:
    """Load YAML, merge with defaults, apply overrides.

    Nested YAML keys map to sub-dataclass fields:
        model:
          width: 128
    becomes config.model.width = 128.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    config = ExperimentConfig()
    _apply_dict(config, data)

    if overrides:
        for dotted_key, value in overrides.items():
            _set_nested(config, dotted_key, value)

    return config


def config_to_yaml(config: ExperimentConfig, path: str | None = None) -> str:
    """Serialize config to YAML. Write to file if path given."""
    data = _to_dict(config)
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    if path is not None:
        with open(path, "w") as f:
            f.write(text)
    return text


def expand_sweep(config: ExperimentConfig) -> list[ExperimentConfig]:
    """Cartesian product over config.sweep axes.

    config.sweep = {"construction.lambda_star": [0.25, 0.30, 0.35], "model.width": [32, 64]}
    returns 6 configs. If sweep is empty, returns [config].
    """
    sweep = config.sweep or {}
    if not sweep:
        return [config]

    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]

    result = []
    for combo in product(*values):
        cfg = deepcopy(config)
        cfg.sweep = {}  # clear sweep on child
        for k, v in zip(keys, combo):
            _set_nested(cfg, k, v)
        result.append(cfg)
    return result


def _set_nested(obj: Any, dotted_key: str, value: Any) -> None:
    """Set a nested field: _set_nested(config, "model.width", 128)."""
    parts = dotted_key.split(".")
    target = obj
    for p in parts[:-1]:
        target = getattr(target, p)
    # Handle list-valued fields that accept scalars too (e.g., single width)
    setattr(target, parts[-1], value)


def _apply_dict(obj: Any, data: dict) -> None:
    """Recursively apply a nested dict onto a dataclass instance in-place."""
    if not is_dataclass(obj):
        raise TypeError(f"Cannot apply dict to non-dataclass: {type(obj)}")
    field_map = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in field_map:
            raise KeyError(f"Unknown config field '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict(current, value)
        elif is_dataclass(current) and value is None:
            pass
        elif isinstance(value, list) and value and isinstance(value[0], dict) and key == "stages":
            # Special-case list[OptimizerStageConfig]
            from src.config.schema import OptimizerStageConfig
            stages = []
            for item in value:
                stage = OptimizerStageConfig()
                _apply_dict(stage, item)
                stages.append(stage)
            setattr(obj, key, stages)
        else:
            # Normalize tuples (domain is tuple[float, float] in schema)
            if isinstance(value, list) and key == "domain":
                value = tuple(value)
            setattr(obj, key, value)


def _to_dict(obj: Any) -> Any:
    """Convert dataclass (recursively) to plain dict/list/scalar for YAML."""
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
