"""
File for building torch neural networks from configs
"""
from dataclasses import asdict
from typing import Callable

import torch
import torch.nn as nn
from dacite import from_dict

from rl.models.mlp import MLP, MLPConfig
from rl.models.types import NetworkConfig

ACTIVATIONS = dict(relu=nn.ReLU, gelu=nn.GELU, tanh=nn.Tanh, sigmoid=nn.Sigmoid, log_softmax=nn.LogSoftmax)


def activation_to_fn(activation: str | None) -> Callable | None:
    if activation is None:
        return None
    if activation in ACTIVATIONS:
        return ACTIVATIONS[activation]
    else:
        raise ValueError(f"{activation} is not handled as an activation. Handled activations are {list(ACTIVATIONS.keys())}")


def build_network_from_cfg(sample_input: torch.Tensor, cfg: NetworkConfig, device: torch.device | None = None) -> nn.Module: # type: ignore
    """build any nn.Module from a sample input and a config. We check cfg.type to see what nn module to build. String type used to enable exporting to human readable config formats"""
    if cfg.type == "mlp":
        cfg = from_dict(data_class=MLPConfig, data=asdict(cfg))
        cfg.arch_cfg.activation = activation_to_fn(cfg.arch_cfg.activation) # type: ignore
        cfg.arch_cfg.output_activation = activation_to_fn(cfg.arch_cfg.output_activation) # type: ignore
        return MLP(sample_input, **asdict(cfg.arch_cfg), device=device)