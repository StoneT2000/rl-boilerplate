from rl.models.mlp import MLP
from rl.models.types import NetworkConfig
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class NatureCNNArchConfig:
    state_features: list[int]
    """hidden units in each layer for processing state data."""
    activation: nn.Module | str | None = "relu"
    """internal activation"""
    output_activation: nn.Module | str | None = None
    """activation after final layer, default is None meaning no activation"""

@dataclass
class NatureCNNConfig(NetworkConfig):
    type = "nature_cnn"
    arch_cfg: NatureCNNArchConfig


class NatureCNN(nn.Module):
    def __init__(self, config: NatureCNNConfig, sample_obs: torch.Tensor, device: torch.device | None = None):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = sample_obs["rgb"].shape[1:3]
        assert image_size == (128, 128), "nature_cnn type currently only supports 128x128 images"
        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
                device=device,
            ),
            config.arch_cfg.activation(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0,
                device=device,
            ),
            config.arch_cfg.activation(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0,
                device=device,
            ),
            config.arch_cfg.activation(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2)).shape[1]
            layers = [nn.Linear(n_flatten, feature_size, device=device)]
            if config.arch_cfg.activation is not None:
                layers.append(config.arch_cfg.activation())
            extractors["rgb"] = nn.Sequential(cnn, nn.Sequential(*layers))

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            features = [state_size] + config.arch_cfg.state_features
            linear_layers = []
            for i in range(len(features)-1):
                linear_layers.append(nn.Linear(features[i], features[i+1], device=device))
                if config.arch_cfg.activation is not None and i < len(features) - 2:
                    linear_layers.append(config.arch_cfg.activation())
            if config.arch_cfg.output_activation is not None:
                linear_layers.append(config.arch_cfg.output_activation())
            extractors["state"] = nn.Sequential(*linear_layers)

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)