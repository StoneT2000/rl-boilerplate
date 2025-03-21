"""MLP class"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import numpy as np

from rl.models.types import NetworkConfig


@dataclass
class MLPArchConfig:
    features: list[int]
    """hidden units in each layer. The number of output features is features[-1]."""
    activation: nn.Module | str | None = "tanh"
    """internal activation"""
    output_activation: nn.Module | str | None = None
    """activation after final layer, default is None meaning no activation"""
    use_layer_norm: bool = False
    """whether to apply layer norm after each linear layer"""


@dataclass
class MLPConfig(NetworkConfig):
    type = "mlp"
    arch_cfg: MLPArchConfig


def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP(nn.Module):
    """
    Args:
        sample_input - batch of sample input to determine the input dimension
        features - hidden units in each layer
        activation - internal activation
        output_activation - activation after final layer, default is None
    """
    def __init__(self, config: MLPConfig, sample_input: torch.Tensor, device: torch.device | None = None):
        super().__init__()
        layers = []
        assert sample_input.ndim == 2, "sample_input must be a tensor with shape (batch_size, input_dim)"
        features = [sample_input.shape[1]] + config.arch_cfg.features
        for i in range(len(features)-1):
            layers.append(layer_init(nn.Linear(features[i], features[i+1], device=device)))
            if config.arch_cfg.use_layer_norm:
                layers.append(nn.LayerNorm(features[i+1], device=device))
            if config.arch_cfg.activation is not None and i < len(features) - 2:
                layers.append(config.arch_cfg.activation())
        if config.arch_cfg.output_activation is not None:
            if config.arch_cfg.use_layer_norm:
                layers.append(nn.LayerNorm(features[-1], device=device))
            layers.append(config.arch_cfg.output_activation())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)