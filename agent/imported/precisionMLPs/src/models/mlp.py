"""Single-hidden-layer tanh MLP for quasi-interpolant experiments.

QIMlp implements:  f(x) = readout(tanh(inner_layer(x)))

The inner layer type (GammaLinear, GammaExpLinear, StandardLinear) determines
the parameterization. The readout is always nn.Linear(width, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config.schema import ModelConfig
from src.models.layers import get_layer, GammaLinear, GammaExpLinear, StandardLinear


class QIMlp(nn.Module):
    """Single-hidden-layer tanh MLP.

    Architecture:
        x: [batch, 1] -> inner_layer: [batch, width] -> tanh -> readout: [batch, 1]
    """

    def __init__(self, config: ModelConfig, **layer_kwargs):
        super().__init__()
        self.config = config
        self.inner_layer = get_layer(config.layer_type, config.width, **layer_kwargs)
        self.readout = nn.Linear(config.width, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: [batch, 1] -> [batch, 1]."""
        return self.readout(torch.tanh(self.inner_layer(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature matrix Phi = tanh(inner_layer(x)). Shape [n_points, width]."""
        return torch.tanh(self.inner_layer(x))

    def get_gamma(self) -> torch.Tensor:
        """Effective gamma values, shape [width]. Handles all layer types."""
        if isinstance(self.inner_layer, GammaLinear):
            return self.inner_layer.gamma.data.squeeze(0)
        elif isinstance(self.inner_layer, GammaExpLinear):
            return self.inner_layer.get_gamma().data.squeeze(0)
        elif isinstance(self.inner_layer, StandardLinear):
            return self.inner_layer.linear.weight.data.squeeze(1)
        raise TypeError(f"Unknown layer type: {type(self.inner_layer)}")

    def get_centers(self) -> torch.Tensor:
        """Center positions, shape [width]."""
        if isinstance(self.inner_layer, (GammaLinear, GammaExpLinear)):
            return self.inner_layer.centers.data.squeeze(0)
        elif isinstance(self.inner_layer, StandardLinear):
            w = self.inner_layer.linear.weight.data.squeeze(1)
            b = self.inner_layer.linear.bias.data
            return -b / w
        raise TypeError(f"Unknown layer type: {type(self.inner_layer)}")

    def get_readout_weights(self) -> torch.Tensor:
        """Outer weights v_k, shape [width]."""
        return self.readout.weight.data.squeeze(0)

    def get_readout_bias(self) -> float:
        """Readout bias (scalar)."""
        return self.readout.bias.data.item()

    def get_lambda(self, h: float) -> torch.Tensor:
        """lambda = gamma * h for each neuron, shape [width]."""
        return self.get_gamma() * h
